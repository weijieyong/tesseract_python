"""
Test script for capturing point cloud from RealSense and converting to mesh.
"""
import numpy as np
import open3d as o3d
import os
import time

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("WARNING: pyrealsense2 not available. Install with: uv pip install pyrealsense2")


def reset_realsense_device():
    """
    Forces a hardware reset on the first connected RealSense device.
    """
    if not REALSENSE_AVAILABLE:
        return False
    
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("No RealSense device connected.")
        return False

    dev = devices[0]
    print(f"Resetting device: {dev.get_info(rs.camera_info.name)}")
    dev.hardware_reset()
    return True


def capture_pointcloud_from_realsense(num_frames=30, depth_min=0.1, depth_max=2.0):
    """
    Captures a point cloud from RealSense camera.
    
    Args:
        num_frames: Number of warmup frames before capture
        depth_min: Minimum depth in meters (filter out closer points)
        depth_max: Maximum depth in meters (filter out farther points)
    
    Returns:
        Open3D PointCloud object or None if capture fails
    """
    if not REALSENSE_AVAILABLE:
        print("RealSense not available!")
        return None
    
    # Reset device first
    if not reset_realsense_device():
        return None
    
    print("Waiting 3 seconds for device re-initialization...")
    time.sleep(3)
    
    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable depth and color streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    print("Starting RealSense pipeline...")
    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"Failed to start RealSense: {e}")
        return None
    
    try:
        # Get depth scale for converting to meters
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth scale: {depth_scale}")
        
        # Warmup frames
        print(f"Warming up ({num_frames} frames)...")
        for i in range(num_frames):
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
        print(f"Intrinsics: {depth_intrin.width}x{depth_intrin.height}, fx={depth_intrin.fx:.2f}, fy={depth_intrin.fy:.2f}")
        
        # Convert to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Convert BGR to RGB
        color_image = color_image[:, :, ::-1]
        
        # Create Open3D images
        o3d_depth = o3d.geometry.Image(depth_image)
        o3d_color = o3d.geometry.Image(color_image.copy())
        
        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth,
            depth_scale=1.0/depth_scale,
            depth_trunc=depth_max,
            convert_rgb_to_intensity=False
        )
        
        # Create camera intrinsics
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            depth_intrin.width, depth_intrin.height,
            depth_intrin.fx, depth_intrin.fy,
            depth_intrin.ppx, depth_intrin.ppy
        )
        
        # Generate point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        
        print(f"Captured point cloud with {len(pcd.points)} points")
        
        # Check point cloud bounds before transformation
        points = np.asarray(pcd.points)
        if len(points) > 0:
            print(f"Point cloud bounds (before transform):")
            print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
            print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
            print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
        
        # Filter by depth range (Z is depth in camera frame)
        colors = np.asarray(pcd.colors)
        
        # Filter based on Z coordinate (depth) - before any transformation
        mask = (points[:, 2] > depth_min) & (points[:, 2] < depth_max)
        filtered_points = points[mask]
        filtered_colors = colors[mask]
        
        print(f"After depth filtering ({depth_min}m - {depth_max}m): {len(filtered_points)} points")
        
        if len(filtered_points) == 0:
            print("WARNING: No points after filtering! Returning unfiltered point cloud.")
            # Return original without filtering
            return pcd
        
        pcd.points = o3d.utility.Vector3dVector(filtered_points)
        pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
        
        return pcd
        
    except RuntimeError as e:
        print(f"RealSense error: {e}")
        return None
    finally:
        pipeline.stop()
        print("RealSense pipeline stopped.")


def process_pointcloud(pcd, voxel_size=0.02):
    """
    Process point cloud: downsample, remove outliers, estimate normals.
    This mirrors the processing in the original script.
    """
    print("\nProcessing PointCloud...")
    print(f"  Original points: {len(pcd.points)}")
    
    # Downsample
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"  After voxel downsampling (voxel_size={voxel_size}): {len(pcd_down.points)} points")
    
    # Remove outliers
    pcd_clean, ind = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"  After outlier removal: {len(pcd_clean.points)} points")
    
    # Estimate normals (needed for mesh generation)
    pcd_clean.estimate_normals()
    print("  Normals estimated")
    
    return pcd_clean


def segment_objects(pcd, eps=0.02, min_points=100, remove_plane=True, plane_threshold=0.01):
    """
    Segment point cloud into individual objects using DBSCAN clustering.
    
    Args:
        pcd: Input point cloud
        eps: DBSCAN epsilon (max distance between points in cluster)
        min_points: Minimum points to form a cluster
        remove_plane: If True, remove the largest plane (e.g., table/floor) first
        plane_threshold: Distance threshold for plane segmentation
    
    Returns:
        List of point clouds, one per detected object
    """
    print(f"\nSegmenting objects (eps={eps}, min_points={min_points})...")
    
    pcd_work = o3d.geometry.PointCloud(pcd)
    
    # Optionally remove the dominant plane (table/floor)
    if remove_plane:
        print("  Detecting and removing dominant plane...")
        plane_model, inliers = pcd_work.segment_plane(
            distance_threshold=plane_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        
        if len(inliers) > 0:
            print(f"  Plane found with {len(inliers)} points (removed)")
            # Keep only non-plane points (the objects)
            pcd_work = pcd_work.select_by_index(inliers, invert=True)
            print(f"  Remaining points after plane removal: {len(pcd_work.points)}")
    
    if len(pcd_work.points) < min_points:
        print("  Not enough points remaining after plane removal!")
        return []
    
    # DBSCAN clustering
    print("  Running DBSCAN clustering...")
    labels = np.array(pcd_work.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    
    max_label = labels.max()
    print(f"  Found {max_label + 1} object clusters")
    
    if max_label < 0:
        print("  No clusters found!")
        return []
    
    # Extract each cluster as a separate point cloud
    object_pcds = []
    points = np.asarray(pcd_work.points)
    colors = np.asarray(pcd_work.colors) if pcd_work.has_colors() else None
    
    for i in range(max_label + 1):
        cluster_mask = labels == i
        cluster_points = points[cluster_mask]
        
        if len(cluster_points) < min_points:
            continue
        
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(cluster_points)
        
        if colors is not None:
            obj_pcd.colors = o3d.utility.Vector3dVector(colors[cluster_mask])
        
        # Estimate normals for each object
        obj_pcd.estimate_normals()
        
        object_pcds.append(obj_pcd)
        print(f"    Object {i}: {len(cluster_points)} points")
    
    return object_pcds


def generate_mesh_convex_hull(pcd):
    """
    Generate mesh using Convex Hull method.
    Guaranteed to produce a closed, watertight mesh suitable for collision checking.
    """
    mesh, _ = pcd.compute_convex_hull()
    
    if len(mesh.vertices) == 0:
        return None
    
    return mesh


def generate_object_meshes(object_pcds, method="convex_hull"):
    """
    Generate individual meshes for each segmented object.
    
    Args:
        object_pcds: List of point clouds (one per object)
        method: Mesh generation method ("convex_hull", "ball_pivoting", "alpha_shape")
    
    Returns:
        List of meshes
    """
    print(f"\nGenerating {method} meshes for {len(object_pcds)} objects...")
    
    meshes = []
    for i, obj_pcd in enumerate(object_pcds):
        if len(obj_pcd.points) < 4:  # Need at least 4 points for a 3D hull
            print(f"  Object {i}: Skipped (too few points)")
            continue
        
        if method == "convex_hull":
            mesh = generate_mesh_convex_hull(obj_pcd)
        elif method == "alpha_shape":
            # Alpha shape provides tighter fit than convex hull
            alpha = 0.03  # Adjust based on object size
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(obj_pcd, alpha)
        elif method == "ball_pivoting":
            radii = [0.005, 0.01, 0.02, 0.04]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                obj_pcd, o3d.utility.DoubleVector(radii))
        else:
            mesh = generate_mesh_convex_hull(obj_pcd)
        
        if mesh is not None and len(mesh.vertices) > 0:
            meshes.append(mesh)
            print(f"  Object {i}: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        else:
            print(f"  Object {i}: Failed to generate mesh")
    
    return meshes


def generate_mesh_ball_pivoting(pcd):
    """
    Generate mesh using Ball Pivoting Algorithm.
    Better shape preservation but may not be watertight.
    """
    print("\nGenerating Ball Pivoting Mesh...")
    
    # Multiple radii for different scales of detail
    radii = [0.005, 0.01, 0.02, 0.04]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Triangles: {len(mesh.triangles)}")
    
    if len(mesh.vertices) == 0:
        print("  WARNING: Ball pivoting produced no mesh!")
        return None
    
    return mesh


def generate_mesh_poisson(pcd, depth=8):
    """
    Generate mesh using Poisson Surface Reconstruction.
    Good for smooth surfaces, but requires good normal estimation.
    """
    print(f"\nGenerating Poisson Mesh (depth={depth})...")
    
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth)
    
    # Remove low-density vertices (noise)
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.1)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Triangles: {len(mesh.triangles)}")
    
    if len(mesh.vertices) == 0:
        print("  WARNING: Poisson reconstruction produced no mesh!")
        return None
    
    return mesh


def save_and_verify_mesh(mesh, output_path):
    """
    Save mesh to file and verify it can be loaded back.
    """
    print(f"\nSaving mesh to {output_path}...")
    success = o3d.io.write_triangle_mesh(output_path, mesh)
    
    if not success:
        print("  ERROR: Failed to save mesh!")
        return False
    
    # Verify by loading back
    print("Verifying saved mesh...")
    loaded_mesh = o3d.io.read_triangle_mesh(output_path)
    
    if len(loaded_mesh.vertices) != len(mesh.vertices):
        print(f"  WARNING: Vertex count mismatch! Original: {len(mesh.vertices)}, Loaded: {len(loaded_mesh.vertices)}")
        return False
    
    print(f"  Mesh saved and verified successfully!")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.2f} KB")
    return True


def visualize_objects(pcd, object_meshes, show=True):
    """
    Visualize point cloud and individual object meshes.
    """
    if not show:
        print("\nVisualization skipped (set show=True to enable)")
        return
    
    print("\nPreparing visualization...")
    
    geometries = []
    
    # Original point cloud
    pcd_vis = o3d.geometry.PointCloud(pcd)
    geometries.append(pcd_vis)
    
    # Generate distinct colors for each object
    num_objects = len(object_meshes)
    colors = [
        [1, 0, 0],    # Red
        [0, 1, 0],    # Green
        [0, 0, 1],    # Blue
        [1, 1, 0],    # Yellow
        [1, 0, 1],    # Magenta
        [0, 1, 1],    # Cyan
        [1, 0.5, 0],  # Orange
        [0.5, 0, 1],  # Purple
    ]
    
    # Add each object mesh with a different color
    for i, mesh in enumerate(object_meshes):
        mesh_vis = o3d.geometry.TriangleMesh(mesh)
        color = colors[i % len(colors)]
        mesh_vis.paint_uniform_color(color)
        mesh_vis.compute_vertex_normals()
        geometries.append(mesh_vis)
        print(f"  Object {i}: color={color}")
    
    print("\nOpening visualization window...")
    print("  Controls: Left-click + drag to rotate, scroll to zoom, middle-click to pan")
    print("  Point cloud shown with original colors, meshes overlaid with distinct colors")
    o3d.visualization.draw_geometries(geometries)


def main():
    print("=" * 60)
    print("RealSense Point Cloud to Mesh Capture")
    print("=" * 60)
    
    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "test_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Capture point cloud from RealSense
    print("\n--- Step 1: Capture Point Cloud ---")
    pcd = capture_pointcloud_from_realsense(
        num_frames=30,      # Warmup frames
        depth_min=0.1,      # Min depth 10cm
        depth_max=1.5       # Max depth 1.5m
    )
    
    if pcd is None or len(pcd.points) == 0:
        print("Failed to capture point cloud from RealSense!")
        return False
    
    # Save raw point cloud
    pcd_path = os.path.join(output_dir, "raw_pointcloud.ply")
    o3d.io.write_point_cloud(pcd_path, pcd)
    print(f"\nRaw point cloud saved to: {pcd_path}")
    
    # Step 2: Process point cloud
    print("\n--- Step 2: Process Point Cloud ---")
    pcd_processed = process_pointcloud(pcd, voxel_size=0.01)
    
    # Save processed point cloud
    pcd_processed_path = os.path.join(output_dir, "processed_pointcloud.ply")
    o3d.io.write_point_cloud(pcd_processed_path, pcd_processed)
    print(f"Processed point cloud saved to: {pcd_processed_path}")
    
    # Step 3: Segment into individual objects
    print("\\n--- Step 3: Segment Objects ---")
    object_pcds = segment_objects(
        pcd_processed,
        eps=0.05,           # Max distance between points in a cluster (5cm)
        min_points=30,      # Minimum points to form an object
        remove_plane=True,  # Remove table/floor plane
        plane_threshold=0.02  # Plane detection threshold (2cm)
    )
    
    if len(object_pcds) == 0:
        print("No objects detected! Try adjusting segmentation parameters.")
        return False
    
    # Save individual object point clouds
    for i, obj_pcd in enumerate(object_pcds):
        obj_path = os.path.join(output_dir, f"object_{i}_pointcloud.ply")
        o3d.io.write_point_cloud(obj_path, obj_pcd)
    
    # Step 4: Generate meshes for each object
    print("\n--- Step 4: Generate Object Meshes ---")
    object_meshes = generate_object_meshes(object_pcds, method="convex_hull")
    
    # Save individual object meshes
    for i, mesh in enumerate(object_meshes):
        mesh_path = os.path.join(output_dir, f"object_{i}_mesh.ply")
        save_and_verify_mesh(mesh, mesh_path)
    
    # Also create a combined mesh file for convenience
    if len(object_meshes) > 0:
        combined_mesh = o3d.geometry.TriangleMesh()
        for mesh in object_meshes:
            combined_mesh += mesh
        combined_path = os.path.join(output_dir, "all_objects_combined.ply")
        save_and_verify_mesh(combined_mesh, combined_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Objects detected: {len(object_pcds)}")
    print(f"Meshes generated: {len(object_meshes)}")
    print(f"\nFiles generated:")
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        print(f"  - {f} ({os.path.getsize(fpath) / 1024:.2f} KB)")
    
    print("\nObject mesh details:")
    for i, mesh in enumerate(object_meshes):
        print(f"  Object {i}: {len(mesh.vertices):5d} vertices, {len(mesh.triangles):5d} triangles")
    
    # Recommendation
    print("\n" + "-" * 60)
    print("For Tesseract collision checking:")
    print("  - Each object_X_mesh.ply is a separate convex hull")
    print("  - Add each as a separate collision object in the environment")
    print("  - Or use all_objects_combined.ply for a single mesh")
    print("-" * 60)
    
    # Optional visualization
    try:
        visualize_objects(pcd_processed, object_meshes, show=True)
    except Exception as e:
        print(f"\nVisualization not available: {e}")
    
    print("\nTest completed successfully!")
    return True


if __name__ == "__main__":
    main()
