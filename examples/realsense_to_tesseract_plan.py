import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import time
import os
import sys
from scipy.interpolate import BPoly, interp1d

from tesseract_robotics.tesseract_common import GeneralResourceLocator, FilesystemPath, Isometry3d, Translation3d, Quaterniond, ManipulatorInfo
from tesseract_robotics.tesseract_environment import Environment, AddLinkCommand, AnyPoly_wrap_EnvironmentConst
from tesseract_robotics.tesseract_scene_graph import Link, Visual, Collision, Joint, JointType_FIXED, Material
from tesseract_robotics.tesseract_geometry import ConvexMesh, Box, Mesh, createMeshFromPath, createConvexMeshFromPath
from tesseract_robotics.tesseract_common import VectorVector3d, AnyPoly
from tesseract_robotics.tesseract_command_language import CartesianWaypoint, WaypointPoly, \
    MoveInstructionType_FREESPACE, MoveInstruction, InstructionPoly, StateWaypoint, StateWaypointPoly, \
    CompositeInstruction, MoveInstructionPoly, CartesianWaypointPoly, ProfileDictionary, \
    AnyPoly_as_CompositeInstruction, CompositeInstructionOrder_ORDERED, DEFAULT_PROFILE_KEY, \
    AnyPoly_wrap_CompositeInstruction, JointWaypoint, JointWaypointPoly, \
    InstructionPoly_as_MoveInstructionPoly, WaypointPoly_as_StateWaypointPoly, \
    MoveInstructionPoly_wrap_MoveInstruction, StateWaypointPoly_wrap_StateWaypoint, \
    CartesianWaypointPoly_wrap_CartesianWaypoint, JointWaypointPoly_wrap_JointWaypoint, \
    AnyPoly_wrap_ProfileDictionary
from tesseract_robotics.tesseract_task_composer import TaskComposerPluginFactory, \
    TaskComposerDataStorage, TaskComposerContext
from tesseract_robotics_viewer import TesseractViewer

# Trajectory interpolation timestep
TRAJECTORY_DT = 0.01  # 100hz


def reset_device():
    """
    Forces a hardware reset on the first connected RealSense device.
    """
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("No device connected. Please check the cable.")
        return False

    dev = devices[0]
    print(f"Resetting device: {dev.get_info(rs.camera_info.name)}")
    dev.hardware_reset()
    return True


def capture_pointcloud_from_realsense(num_frames=30, depth_min=0.1, depth_max=1.5):
    """
    Captures a point cloud from RealSense camera.
    
    Returns:
        Open3D PointCloud object or None if capture fails
    """
    # Reset device first
    if not reset_device():
        return None
    
    print("Waiting 3 seconds for device re-initialization...")
    time.sleep(3)
    
    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    print("Starting RealSense pipeline...")
    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"Failed to start RealSense: {e}")
        return None
    
    try:
        # Get depth scale
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        
        # Warmup frames
        print(f"Warming up ({num_frames} frames)...")
        for _ in range(num_frames):
            pipeline.wait_for_frames()
        
        # Capture frame
        print("Capturing frame...")
        frames = pipeline.wait_for_frames()
        
        # Align depth to color
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            print("Failed to capture frames!")
            return None
        
        # Get intrinsics
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        
        # Convert to numpy
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())[:, :, ::-1]  # BGR to RGB
        
        # Create Open3D RGBD
        o3d_depth = o3d.geometry.Image(depth_image)
        o3d_color = o3d.geometry.Image(color_image.copy())
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth,
            depth_scale=1.0/depth_scale,
            depth_trunc=depth_max,
            convert_rgb_to_intensity=False
        )
        
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            depth_intrin.width, depth_intrin.height,
            depth_intrin.fx, depth_intrin.fy,
            depth_intrin.ppx, depth_intrin.ppy
        )
        
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        print(f"Captured point cloud with {len(pcd.points)} points")
        
        # Filter by depth
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        mask = (points[:, 2] > depth_min) & (points[:, 2] < depth_max)
        pcd.points = o3d.utility.Vector3dVector(points[mask])
        pcd.colors = o3d.utility.Vector3dVector(colors[mask])
        
        print(f"After depth filtering: {len(pcd.points)} points")
        return pcd
        
    except RuntimeError as e:
        print(f"RealSense error: {e}")
        return None
    finally:
        pipeline.stop()


def process_pointcloud(pcd, voxel_size=0.01):
    """Process point cloud: downsample, remove outliers, estimate normals."""
    print("Processing PointCloud...")
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd_clean, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_clean.estimate_normals()
    print(f"  Processed: {len(pcd_clean.points)} points")
    return pcd_clean


def segment_objects(pcd, eps=0.05, min_points=30, remove_plane=True, plane_threshold=0.02):
    """
    Segment point cloud into individual objects using DBSCAN clustering.
    
    Returns:
        List of (point_cloud, mesh) tuples for each detected object
    """
    print(f"Segmenting objects (eps={eps}, min_points={min_points})...")
    
    pcd_work = o3d.geometry.PointCloud(pcd)
    
    # Remove dominant plane (table/floor)
    if remove_plane:
        print("  Removing dominant plane...")
        plane_model, inliers = pcd_work.segment_plane(
            distance_threshold=plane_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        if len(inliers) > 0:
            print(f"  Plane removed: {len(inliers)} points")
            pcd_work = pcd_work.select_by_index(inliers, invert=True)
    
    if len(pcd_work.points) < min_points:
        print("  Not enough points after plane removal!")
        return []
    
    # DBSCAN clustering
    labels = np.array(pcd_work.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    max_label = labels.max()
    print(f"  Found {max_label + 1} objects")
    
    if max_label < 0:
        return []
    
    # Extract each cluster and generate convex hull mesh
    objects = []
    points = np.asarray(pcd_work.points)
    colors = np.asarray(pcd_work.colors) if pcd_work.has_colors() else None
    
    for i in range(max_label + 1):
        cluster_mask = labels == i
        cluster_points = points[cluster_mask]
        
        if len(cluster_points) < 4:  # Need at least 4 points for convex hull
            continue
        
        # Create point cloud for this object
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(cluster_points)
        if colors is not None:
            obj_pcd.colors = o3d.utility.Vector3dVector(colors[cluster_mask])
        
        # Generate convex hull mesh
        mesh, _ = obj_pcd.compute_convex_hull()
        
        if len(mesh.vertices) > 0:
            objects.append((obj_pcd, mesh))
            print(f"    Object {i}: {len(cluster_points)} pts -> {len(mesh.vertices)} vertices")
    
    return objects


def create_tesseract_convex_mesh(o3d_mesh):
    """
    Convert Open3D mesh to Tesseract ConvexMesh geometry.
    """
    # Convert vertices to VectorVector3d
    vertices = VectorVector3d()
    for v in np.asarray(o3d_mesh.vertices):
        vertices.append(np.array(v, dtype=np.float64))
    
    # Convert faces to the Tesseract format: [num_vertices, v1, v2, v3, ...]
    triangles = np.asarray(o3d_mesh.triangles)
    face_data = []
    for tri in triangles:
        face_data.append(3)  # 3 vertices per triangle
        face_data.extend(tri.tolist())
    
    faces = np.array(face_data, dtype=np.int32)
    
    # Create ConvexMesh
    convex_mesh = ConvexMesh(vertices, faces)
    return convex_mesh


def create_tesseract_mesh(o3d_mesh):
    """
    Convert Open3D mesh to Tesseract Mesh geometry (for visualization).
    """
    # Convert vertices to VectorVector3d
    vertices = VectorVector3d()
    for v in np.asarray(o3d_mesh.vertices):
        vertices.append(np.array(v, dtype=np.float64))
    
    # Convert faces to the Tesseract format: [num_vertices, v1, v2, v3, ...]
    triangles = np.asarray(o3d_mesh.triangles)
    face_data = []
    for tri in triangles:
        face_data.append(3)  # 3 vertices per triangle
        face_data.extend(tri.tolist())
    
    faces = np.array(face_data, dtype=np.int32)
    
    # Create Mesh (for visual)
    mesh = Mesh(vertices, faces)
    return mesh


def add_obstacles_to_environment(t_env, objects, output_dir, camera_transform, use_box=True):
    """
    Add detected objects as obstacles to the Tesseract environment.
    
    Args:
        t_env: Tesseract environment
        objects: List of (point_cloud, mesh) tuples
        output_dir: Directory to save mesh files
        camera_transform: Transform from camera frame to robot base frame
        use_box: If True, use axis-aligned bounding box instead of convex mesh
    """
    print(f"\nAdding {len(objects)} obstacles to Tesseract environment...")
    
    # Define colors for different objects
    colors = [
        [1.0, 0.0, 0.0, 1.0],  # Red
        [0.0, 1.0, 0.0, 1.0],  # Green
        [0.0, 0.0, 1.0, 1.0],  # Blue
        [1.0, 1.0, 0.0, 1.0],  # Yellow
        [1.0, 0.0, 1.0, 1.0],  # Magenta
        [0.0, 1.0, 1.0, 1.0],  # Cyan
        [1.0, 0.5, 0.0, 1.0],  # Orange
        [0.5, 0.0, 1.0, 1.0],  # Purple
    ]
    
    for i, (obj_pcd, obj_mesh) in enumerate(objects):
        # Save mesh file (for reference)
        mesh_path = os.path.join(output_dir, f"obstacle_{i}.ply")
        o3d.io.write_triangle_mesh(mesh_path, obj_mesh)
        
        # Transform mesh vertices to robot frame
        vertices = np.asarray(obj_mesh.vertices)
        
        # Apply camera-to-robot transform
        transformed_vertices = (camera_transform[:3, :3] @ vertices.T).T + camera_transform[:3, 3]
        
        if use_box:
            # Use axis-aligned bounding box (simpler and more reliable for viewer)
            min_pt = transformed_vertices.min(axis=0)
            max_pt = transformed_vertices.max(axis=0)
            center = (min_pt + max_pt) / 2
            size = max_pt - min_pt
            
            # Add a small margin
            margin = 0.01
            size = size + margin
            
            # Create Box geometry
            geometry = Box(size[0], size[1], size[2])
            
            # Position is at the center of the box
            position = center
        else:
            # Use ConvexMesh - save transformed mesh to file and load with proper resource
            transformed_mesh = o3d.geometry.TriangleMesh()
            transformed_mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
            transformed_mesh.triangles = obj_mesh.triangles
            transformed_mesh.compute_vertex_normals()
            
            # Save transformed mesh to file
            transformed_mesh_path = os.path.join(output_dir, f"obstacle_{i}_transformed.ply")
            o3d.io.write_triangle_mesh(transformed_mesh_path, transformed_mesh)
            
            # Load as Tesseract geometry with proper resource for viewer
            scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
            visual_meshes = createMeshFromPath(transformed_mesh_path, scale, True, True)
            convex_meshes = createConvexMeshFromPath(transformed_mesh_path, scale, False, False)
            
            if len(visual_meshes) > 0 and len(convex_meshes) > 0:
                visual_geometry = visual_meshes[0]
                collision_geometry = convex_meshes[0]
            else:
                print(f"  Warning: Failed to load mesh for obstacle_{i}")
                continue
            
            position = (0.0, 0.0, 0.0)  # Mesh already transformed
        
        # Create Link
        link_name = f"obstacle_{i}"
        obstacle_link = Link(link_name)
        
        # Material with color
        material = Material(f"obstacle_{i}_material")
        color = colors[i % len(colors)]
        material.color = np.array(color, dtype=np.float64)
        
        # Visual
        visual = Visual()
        if use_box:
            visual.geometry = geometry
        else:
            visual.geometry = visual_geometry
        visual.material = material
        obstacle_link.visual.push_back(visual)
        
        # Collision
        collision = Collision()
        if use_box:
            collision.geometry = geometry
        else:
            collision.geometry = collision_geometry
        obstacle_link.collision.push_back(collision)
        
        # Joint (fixed to world)
        obstacle_joint = Joint(f"obstacle_{i}_joint")
        obstacle_joint.parent_link_name = "base_cuboid"
        obstacle_joint.child_link_name = link_name
        obstacle_joint.type = JointType_FIXED
        
        # Set position transform
        obstacle_joint.parent_to_joint_origin_transform = Isometry3d.Identity() * Translation3d(
            float(position[0]), float(position[1]), float(position[2])
        )
        
        # Add to environment
        t_env.applyCommand(AddLinkCommand(obstacle_link, obstacle_joint))
        
        if use_box:
            print(f"  Added obstacle_{i} (Box): size=[{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}] at [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
        else:
            print(f"  Added obstacle_{i} (ConvexMesh): {len(obj_mesh.vertices)} vertices")
    
    return True


def plan_motion(t_env, locator, config_dir, manip_info, wp1, wp2):
    """
    Plan motion from wp1 to wp2 using Task Composer.
    
    Args:
        t_env: Tesseract environment
        locator: Resource locator
        config_dir: Directory containing task composer config
        manip_info: Manipulator info
        wp1: Start CartesianWaypoint
        wp2: End CartesianWaypoint
    
    Returns:
        CompositeInstruction with planned trajectory or None if failed
    """
    print("\n--- Motion Planning ---")
    
    # Create the task composer plugin factory and load the plugins
    task_composer_filename = os.path.join(config_dir, "task_composer_plugins.yaml")
    config_path = FilesystemPath(task_composer_filename)
    factory = TaskComposerPluginFactory(config_path, locator)
    
    # Create the task composer node. Using TrajOptPipeline for planning.
    task = factory.createTaskComposerNode("TrajOptPipeline")
    
    # Get the output keys for the task
    output_key = task.getOutputKeys().get("program")
    input_key = task.getInputKeys().get("planning_input")
    
    # Create a profile dictionary
    profiles = ProfileDictionary()
    
    # Create the input command program instructions
    start_instruction = MoveInstruction(CartesianWaypointPoly_wrap_CartesianWaypoint(wp1), MoveInstructionType_FREESPACE, "DEFAULT")
    plan_f1 = MoveInstruction(CartesianWaypointPoly_wrap_CartesianWaypoint(wp2), MoveInstructionType_FREESPACE, "DEFAULT")
    
    # Create the input command program
    program = CompositeInstruction("DEFAULT")
    program.setManipulatorInfo(manip_info)
    program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(start_instruction))
    program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(plan_f1))
    
    # Create AnyPoly objects
    program_anypoly = AnyPoly_wrap_CompositeInstruction(program)
    environment_anypoly = AnyPoly_wrap_EnvironmentConst(t_env)
    profiles_anypoly = AnyPoly_wrap_ProfileDictionary(profiles)
    
    # Create the task data
    task_data = TaskComposerDataStorage()
    task_data.setData(input_key, program_anypoly)
    task_data.setData("environment", environment_anypoly)
    task_data.setData("profiles", profiles_anypoly)
    
    # Create an executor to run the task
    task_executor = factory.createTaskComposerExecutor("TaskflowExecutor")
    
    # Run the task and wait for completion
    print("Running motion planner...")
    future = task_executor.run(task.get(), task_data)
    future.wait()
    
    if not future.context.isSuccessful():
        print("Planning task failed!")
        return None
    
    # Retrieve the output
    results = AnyPoly_as_CompositeInstruction(future.context.data_storage.getData(output_key))
    
    print("Planning successful!")
    
    # Print waypoints
    for instr in results:
        if instr.isMoveInstruction():
            move_instr = InstructionPoly_as_MoveInstructionPoly(instr)
            wp = WaypointPoly_as_StateWaypointPoly(move_instr.getWaypoint())
            print(f"  Joint Positions: {wp.getPosition().flatten()} time: {wp.getTime()}")
    
    return results


def interpolate_and_save_trajectory(results, output_path, dt=TRAJECTORY_DT):
    """
    Interpolate trajectory to fixed timestep and save to file.
    
    Args:
        results: CompositeInstruction with planned trajectory
        output_path: Path to save interpolated trajectory
        dt: Timestep for interpolation
    """
    times = []
    positions = []
    velocities = []
    
    for instr in results:
        if instr.isMoveInstruction():
            move_instr = InstructionPoly_as_MoveInstructionPoly(instr)
            wp = WaypointPoly_as_StateWaypointPoly(move_instr.getWaypoint())
            times.append(wp.getTime())
            positions.append(np.asarray(wp.getPosition()).flatten())
            velocities.append(np.asarray(wp.getVelocity()).flatten())
    
    print(f"\nNumber of waypoints: {len(times)}")
    
    if len(velocities) > 0 and len(velocities[0]) > 0:
        print("Velocities are present. Using BPoly for interpolation.")
        
        # Stack positions and velocities: (N, 2, 6)
        yi = np.stack([positions, velocities], axis=1)
        
        # Create the interpolator
        interpolator = BPoly.from_derivatives(times, yi)
        
        # Define fixed timestamps
        t_start = times[0]
        t_end = times[-1]
        fixed_timestamps = np.arange(t_start, t_end + dt, dt)
        
        # Evaluate
        interpolated_positions = interpolator(fixed_timestamps)
    else:
        print("Velocities are NOT present. Using linear interpolation.")
        
        interpolator = interp1d(times, positions, axis=0, kind='linear')
        
        t_start = times[0]
        t_end = times[-1]
        fixed_timestamps = np.arange(t_start, t_end + dt, dt)
        
        interpolated_positions = interpolator(fixed_timestamps)
    
    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        for t, pos in zip(fixed_timestamps, interpolated_positions):
            pos_deg = np.rad2deg(pos)
            pos_str = ", ".join(f"{angle:.6f}" for angle in pos_deg)
            f.write(f"{t:.6f}, {pos_str}\n")
    
    print(f"Interpolated trajectory saved to {output_path}")
    print(f"Total points: {len(fixed_timestamps)}")
    
    return fixed_timestamps, interpolated_positions


def main():
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "test_output")
    urdf_dir = os.path.join(os.path.dirname(script_dir), "urdf")
    config_dir = os.path.join(os.path.dirname(script_dir), "config")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("RealSense to Tesseract - Object Detection & Motion Planning")
    print("=" * 60)
    
    # Step 1: Capture point cloud from RealSense
    print("\n--- Step 1: Capture Point Cloud ---")
    pcd = capture_pointcloud_from_realsense(
        num_frames=30,
        depth_min=0.1,
        depth_max=1.5
    )
    
    if pcd is None or len(pcd.points) == 0:
        print("RealSense capture failed. Using dummy objects for testing...")
        # Create dummy objects
        objects = []
        for i, (pos, size) in enumerate([
            ([0.4, 0.1, 0.1], [0.1, 0.1, 0.1]),
            ([0.4, -0.1, 0.1], [0.08, 0.08, 0.15]),
        ]):
            box = o3d.geometry.TriangleMesh.create_box(*size)
            box.translate(pos)
            pcd_dummy = box.sample_points_uniformly(500)
            objects.append((pcd_dummy, box))
        print(f"Created {len(objects)} dummy obstacles")
    else:
        # Save raw point cloud
        pcd_path = os.path.join(output_dir, "raw_pointcloud.ply")
        o3d.io.write_point_cloud(pcd_path, pcd)
        
        # Step 2: Process point cloud
        print("\n--- Step 2: Process Point Cloud ---")
        pcd_processed = process_pointcloud(pcd, voxel_size=0.01)
        
        # Step 3: Segment objects
        print("\n--- Step 3: Segment Objects ---")
        objects = segment_objects(
            pcd_processed,
            eps=0.05,
            min_points=30,
            remove_plane=True,
            plane_threshold=0.02
        )
    
    if len(objects) == 0:
        print("No objects detected!")
        return
    
    print(f"\nDetected {len(objects)} objects")
    
    # Step 4: Initialize Tesseract Environment
    print("\n--- Step 4: Initialize Tesseract Environment ---")
    locator = GeneralResourceLocator()
    t_env = Environment()
    
    # Load robot URDF
    rm_65_b_urdf_fname = FilesystemPath(os.path.join(urdf_dir, "rm_65_b.urdf"))
    rm_65_b_srdf_fname = FilesystemPath(os.path.join(urdf_dir, "rm_65_b.srdf"))
    
    if not t_env.init(rm_65_b_urdf_fname, rm_65_b_srdf_fname, locator):
        print("Failed to initialize Tesseract environment!")
        return
    print("Robot loaded successfully")
    
    # Step 5: Add obstacles to environment (using mesh, not box)
    print("\n--- Step 5: Add Obstacles (Mesh) ---")
    
    # Define camera-to-robot transform
    camera_transform = np.array([
        [0, -1, 0, 0.4],
        [-1, 0, 0, -0.2],
        [0, 0, -1, 1.0],
        [0, 0, 0, 1]
    ], dtype=np.float64)
    
    add_obstacles_to_environment(t_env, objects, output_dir, camera_transform, use_box=False)
    
    # Set initial robot state
    joint_names = ["joint%d" % (i+1) for i in range(6)]
    initial_joint_pos = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float64)
    t_env.setState(joint_names, initial_joint_pos)
    
    # Setup manipulator info
    manip_info = ManipulatorInfo()
    manip_info.tcp_frame = "dummy_tcp"
    manip_info.manipulator = "manipulator"
    manip_info.working_frame = "base_cuboid"
    
    # Step 6: Start Viewer
    print("\n--- Step 6: Start Viewer ---")
    viewer = None
    try:
        viewer = TesseractViewer()
        viewer.update_environment(t_env, [0, 0, 0])
        viewer.start_serve_background()
        viewer.update_joint_positions(joint_names, initial_joint_pos)
        print("Viewer started at http://localhost:8000")
    except Exception as e:
        print(f"Viewer failed: {e}")
    
    # Step 7: Collision check
    print("\n--- Step 7: Collision Check ---")
    from tesseract_robotics.tesseract_collision import ContactResultMap, ContactRequest, ContactTestType_ALL, ContactResultVector
    
    contact_manager = t_env.getDiscreteContactManager()
    contact_manager.setActiveCollisionObjects(t_env.getLinkNames())
    
    contacts = ContactResultMap()
    request = ContactRequest(ContactTestType_ALL)
    contact_manager.contactTest(contacts, request)
    
    if contacts.size() > 0:
        print(f"Collisions detected: {contacts.size()}")
        flattened = ContactResultVector()
        contacts.flattenCopyResults(flattened)
        for c in flattened:
            print(f"  {c.link_names[0]} <-> {c.link_names[1]}")
    else:
        print("No collisions in current state")
    
    # Step 8: Motion Planning
    print("\n--- Step 8: Motion Planning ---")
    
    # Define waypoints
    wp1 = CartesianWaypoint(Isometry3d.Identity() * Translation3d(0.38, 0.3, 0.45) * Quaterniond(0, 0, 1, 0))
    wp2 = CartesianWaypoint(Isometry3d.Identity() * Translation3d(0.39, -0.3, 0.30) * Quaterniond(0, 0, 1, 0))
    
    print(f"Planning from: (0.38, 0.3, 0.45) to (0.39, -0.3, 0.43)")
    
    # Plan motion
    results = plan_motion(t_env, locator, config_dir, manip_info, wp1, wp2)
    
    if results is None:
        print("Motion planning failed!")
    else:
        # Interpolate and save trajectory
        trajectory_path = os.path.join(output_dir, "planned_trajectory.txt")
        interpolate_and_save_trajectory(results, trajectory_path)
        
        # Update viewer with trajectory
        if viewer is not None:
            viewer.update_trajectory(results)
            viewer.plot_trajectory(results, manip_info)
            print("Trajectory visualized in viewer")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Objects detected: {len(objects)}")
    print(f"Output directory: {output_dir}")
    print(f"Viewer: http://localhost:8000")
    if results is not None:
        print(f"Motion planning: SUCCESS")
    else:
        print(f"Motion planning: FAILED")
    print("=" * 60)
    
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
