import re
import traceback
import os
import numpy as np
import numpy.testing as nptest
from scipy.interpolate import BPoly

from tesseract_robotics.tesseract_common import GeneralResourceLocator
from tesseract_robotics.tesseract_environment import Environment, AnyPoly_wrap_EnvironmentConst, AddLinkCommand
from tesseract_robotics.tesseract_common import FilesystemPath, Isometry3d, Translation3d, Quaterniond, \
    ManipulatorInfo, AnyPoly, AnyPoly_wrap_double
from tesseract_robotics.tesseract_command_language import CartesianWaypoint, WaypointPoly, \
    MoveInstructionType_FREESPACE, MoveInstruction, InstructionPoly, StateWaypoint, StateWaypointPoly, \
    CompositeInstruction, MoveInstructionPoly, CartesianWaypointPoly, ProfileDictionary, \
        AnyPoly_as_CompositeInstruction, CompositeInstructionOrder_ORDERED, DEFAULT_PROFILE_KEY, \
        AnyPoly_wrap_CompositeInstruction, DEFAULT_PROFILE_KEY, JointWaypoint, JointWaypointPoly, \
        InstructionPoly_as_MoveInstructionPoly, WaypointPoly_as_StateWaypointPoly, \
        MoveInstructionPoly_wrap_MoveInstruction, StateWaypointPoly_wrap_StateWaypoint, \
        CartesianWaypointPoly_wrap_CartesianWaypoint, JointWaypointPoly_wrap_JointWaypoint, \
        AnyPoly_wrap_ProfileDictionary

from tesseract_robotics.tesseract_task_composer import TaskComposerPluginFactory, \
    TaskComposerDataStorage, TaskComposerContext

from tesseract_robotics_viewer import TesseractViewer

from tesseract_robotics.tesseract_scene_graph import Link, Joint, Visual, Collision, JointType_FIXED
from tesseract_robotics.tesseract_geometry import Box

# This example demonstrates using the Tesseract Planning Task Composer to create a simple robot motion plan from
# an input request program. The composer is a high level interface for creating motion plans. It is designed to
# be used by users who do not want to deal with the details of the Tesseract Planning Library. The composer automatically
# runs a sequence of planning steps to generate an output plan with minimal configuration. "Profiles" are used to
# configure the planning steps. Profiles are a dictionary of key value pairs that are used to configure the planning
# steps. The various planners have default configurations that should work for most use cases. There are numerous
# configurations available for the task composer that execute different sequences of planning steps. This example 
# demonstrates using the "freespace" planner, which is for moving the robot to a desired pose in free space while
# avoiding collisions. The freespace planner first uses OMPL to find a collision free path, and then uses TrajOpt
# to refine the path. Finally, the TimeOptimalTrajectoryGeneration time parametrization algorithm is used to generate
# timestamps for the trajectory.

# The task composer requires a configuration YAML file to be passed in. A default configuration file is provided
# in the Tesseract Planning Library. This configuration file can be copied and modified to suit the user's needs.

# An environment is initialized using URDF and SRDF files. These files need to be configured for the scene, and
# to use the correct collision and kinematics plugins. See the collision and kinematics examples for more details on
# how to do this.

# This example uses the GeneralResourceLocator to find resources on the file system. The GeneralResourceLocator
# uses the TESSERACT_RESOURCE_PATH environmental variable.
#
# TESSERACT_RESOURCE_PATH must be set to the directory containing the `tesseract_support` package. This can be done
# by running:
#
# git clone https://github.com/tesseract-robotics/tesseract.git
# export TESSERACT_RESOURCE_PATH="$(pwd)/tesseract/"
#
# or on Windows
#
# git clone https://github.com/tesseract-robotics/tesseract.git
# set TESSERACT_RESOURCE_PATH=%cd%\tesseract\

# The environmental variable TESSERACT_TASK_COMPOSER_CONFIG_FILE must be set to the location of the configuration file.
# This can be done by running:
#
# git clone https://github.com/tesseract-robotics/tesseract-planning.git
# export TESSERACT_TASK_COMPOSER_CONFIG_FILE="$(pwd)/tesseract_planning/tesseract_task_composer/config/task_composer_plugins_no_trajopt_ifopt.yaml"
#
# or on Windows
#
# git clone https://github.com/tesseract-robotics/tesseract-planning.git
# set TESSERACT_TASK_COMPOSER_CONFIG_FILE=%cd%\tesseract_planning\tesseract_task_composer\config\task_composer_plugins_no_trajopt_ifopt.yaml
#
# This file can be distributed with the application, and modified as needed.

OMPL_DEFAULT_NAMESPACE = "OMPLMotionPlannerTask"
TRAJOPT_DEFAULT_NAMESPACE = "TrajOptMotionPlannerTask"
TRAJECTORY_DT = 0.01  # change this depends on your hardware 100hz


# Initialize the resource locator and environment
locator = GeneralResourceLocator()
# Get the script directory and construct relative paths to URDF files
script_dir = os.path.dirname(os.path.abspath(__file__))
urdf_dir = os.path.join(os.path.dirname(script_dir), "urdf")
config_dir = os.path.join(os.path.dirname(script_dir), "config")
rm_65_b_urdf_fname = FilesystemPath(os.path.join(urdf_dir, "rm_65_b.urdf"))
rm_65_b_srdf_fname = FilesystemPath(os.path.join(urdf_dir, "rm_65_b.srdf"))
task_composer_filename = str(FilesystemPath(os.path.join(config_dir, "task_composer_plugins.yaml")))
# task_composer_filename = str(FilesystemPath(os.path.join(config_dir, "task_composer_plugins_no_trajopt_ifopt.yaml")))
t_env = Environment()

# locator_fn must be kept alive by maintaining a reference
assert t_env.init(rm_65_b_urdf_fname, rm_65_b_srdf_fname, locator)

# Add a fixed obstacle so the planner must route around it
BOX_HEIGHT = 0.8
BOX_WIDTH = 0.1
BOX_DEPTH = 0.6
box_obstacle = Box(BOX_DEPTH, BOX_WIDTH, BOX_HEIGHT)
# box_obstacle = Box(0.3, 0.1, 0.4) # small
box_link = Link("box_obstacle")
box_visual = Visual()
box_visual.geometry = box_obstacle
box_link.visual.push_back(box_visual)
box_collision = Collision()
box_collision.geometry = box_obstacle
box_link.collision.push_back(box_collision)

box_joint = Joint("box_obstacle_joint")
box_joint.parent_link_name = "base_cuboid"
box_joint.child_link_name = box_link.getName()
box_joint.type = JointType_FIXED
box_joint.parent_to_joint_origin_transform = Isometry3d.Identity() * Translation3d(0.9, 0.0, BOX_HEIGHT/2)
# box_joint.parent_to_joint_origin_transform = Isometry3d.Identity() * Translation3d(0.45, 0.0, 0.3)

t_env.applyCommand(AddLinkCommand(box_link, box_joint))

# Fill in the manipulator information. This is used to find the kinematic chain for the manipulator. This must
# match the SRDF, although the exact tcp_frame can differ if a tool is used.
manip_info = ManipulatorInfo()
manip_info.tcp_frame = "dummy_tcp"
manip_info.manipulator = "manipulator"
manip_info.working_frame = "base_cuboid"

# Create a viewer and set the environment so the results can be displayed later
viewer = TesseractViewer()
viewer.update_environment(t_env, [0,0,0])

# Set the initial state of the robot
joint_names = ["joint%d" % (i+1) for i in range(6)]
viewer.update_joint_positions(joint_names, np.array([1,-.2,.01,.3,-.5,1]))

# Start the viewer
viewer.start_serve_background()

# Set the initial state of the robot
t_env.setState(joint_names, np.ones(6)*0.1)

# Create the input command program waypoints
wp1 = CartesianWaypoint(Isometry3d.Identity() * Translation3d(0.38, 0.3, 0.45) * Quaterniond(0, 0, 1, 0))
wp2 = CartesianWaypoint(Isometry3d.Identity() * Translation3d(0.39, -0.3, 0.43) * Quaterniond(0, 0, 1, 0))
# wp3 = CartesianWaypoint(Isometry3d.Identity() * Translation3d(0.38, 0.14, 0.73) * Quaterniond(0, 0, 1, 0))

# Create the input command program instructions. Note the use of explicit construction of the CartesianWaypointPoly
# using the *_wrap_CartesianWaypoint functions. This is required because the Python bindings do not support implicit
# conversion from the CartesianWaypoint to the CartesianWaypointPoly.
start_instruction = MoveInstruction(CartesianWaypointPoly_wrap_CartesianWaypoint(wp1), MoveInstructionType_FREESPACE, "DEFAULT")
plan_f1 = MoveInstruction(CartesianWaypointPoly_wrap_CartesianWaypoint(wp2), MoveInstructionType_FREESPACE, "DEFAULT")
# plan_f2 = MoveInstruction(CartesianWaypointPoly_wrap_CartesianWaypoint(wp3), MoveInstructionType_FREESPACE, "DEFAULT")

# Create the input command program. Note the use of *_wrap_MoveInstruction functions. This is required because the
# Python bindings do not support implicit conversion from the MoveInstruction to the MoveInstructionPoly.
program = CompositeInstruction("DEFAULT")
program.setManipulatorInfo(manip_info)
program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(start_instruction))
program.appendMoveInstruction(MoveInstructionPoly_wrap_MoveInstruction(plan_f1))
# program.appendMoveInstruction(MoveInstructionPoly(plan_f2))

# Create the task composer plugin factory and load the plugins
config_path = FilesystemPath(task_composer_filename)
factory = TaskComposerPluginFactory(config_path, locator)

# Create the task composer node. In this case the FreespacePipeline is used. Many other are available.
task = factory.createTaskComposerNode("TrajOptPipeline")

# Get the output keys for the task
output_key = task.getOutputKeys().get("program")
input_key = task.getInputKeys().get("planning_input")

# Create a profile dictionary. Profiles can be customized by adding to this dictionary and setting the profiles
# in the instructions.
profiles = ProfileDictionary()

# Create an AnyPoly containing the program. This explicit step is required because the Python bindings do not
# support implicit conversion from the CompositeInstruction to the AnyPoly.
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
future = task_executor.run(task.get(), task_data)
future.wait()

if not future.context.isSuccessful():
    print("Planning task failed")
    exit(1)

# Retrieve the output, converting the AnyPoly back to a CompositeInstruction
results = AnyPoly_as_CompositeInstruction(future.context.data_storage.getData(output_key))

# Display the output
# Print out the resulting waypoints
for instr in results:
    assert instr.isMoveInstruction()
    move_instr1 = InstructionPoly_as_MoveInstructionPoly(instr)
    wp1 = move_instr1.getWaypoint()
    assert wp1.isStateWaypoint()
    wp = WaypointPoly_as_StateWaypointPoly(wp1)
    print(f"Joint Positions: {wp.getPosition().flatten()} time: {wp.getTime()}")

# Update the viewer with the results to animate the trajectory
# Open web browser to http://localhost:8000 to view the results
viewer.update_trajectory(results)
viewer.plot_trajectory(results, manip_info)

input("press enter to exit")

# Analyze the trajectory
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

print(f"Number of waypoints: {len(times)}")
print(f"Times: {times}")

if len(velocities) > 0 and len(velocities[0]) > 0:
    print("Velocities are present. Using BPoly for interpolation.")
    
    # Prepare data for BPoly
    # yi must be shape (n_points, n_dims) if only function values
    # or (n_points, n_derivatives, n_dims) if derivatives are included.
    # Here we have position and velocity, so n_derivatives = 2.
    
    # positions: (N, 6)
    # velocities: (N, 6)
    
    # Stack them: (N, 2, 6)
    yi = np.stack([positions, velocities], axis=1)
    
    # Create the interpolator
    interpolator = BPoly.from_derivatives(times, yi)
    
    # Define fixed timestamps
    dt = TRAJECTORY_DT # 100 Hz
    t_start = times[0]
    t_end = times[-1]
    fixed_timestamps = np.arange(t_start, t_end + dt, dt)
    
    # Evaluate
    interpolated_positions = interpolator(fixed_timestamps)
    
    # Save/Print
    fixed_output_path = os.path.join(script_dir, "joint_positions_fixed_dt.txt")
    with open(fixed_output_path, "w", encoding="utf-8") as f:
        for t, pos in zip(fixed_timestamps, interpolated_positions):
            pos_deg = np.rad2deg(pos)
            pos_str = ", ".join(f"{angle:.6f}" for angle in pos_deg)
            # print(f"Time {t:.3f}: {pos_str}")
            f.write(f"{t:.6f}, {pos_str}\n")
            
    print(f"Interpolated trajectory saved to {fixed_output_path}")
    print(f"Total points: {len(fixed_timestamps)}")

else:
    print("Velocities are NOT present. Using linear interpolation.")
    from scipy.interpolate import interp1d
    
    interpolator = interp1d(times, positions, axis=0, kind='linear')
    
    dt = TRAJECTORY_DT
    t_start = times[0]
    t_end = times[-1]
    fixed_timestamps = np.arange(t_start, t_end + dt, dt)
    
    interpolated_positions = interpolator(fixed_timestamps)
    
    fixed_output_path = os.path.join(script_dir, "joint_positions_fixed_dt.txt")
    with open(fixed_output_path, "w", encoding="utf-8") as f:
        for t, pos in zip(fixed_timestamps, interpolated_positions):
            pos_deg = np.rad2deg(pos)
            pos_str = ", ".join(f"{angle:.6f}" for angle in pos_deg)
            f.write(f"{t:.6f}, {pos_str}\n")
            
    print(f"Interpolated trajectory saved to {fixed_output_path}")
    print(f"Total points: {len(fixed_timestamps)}")