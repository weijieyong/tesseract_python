from tesseract_robotics.tesseract_common import FilesystemPath, GeneralResourceLocator, Isometry3d, Translation3d, \
    TransformMap, Quaterniond
from tesseract_robotics.tesseract_environment import Environment
from tesseract_robotics.tesseract_kinematics import KinGroupIKInput, KinGroupIKInputs
import numpy as np

# Example of using kinematics to solve for forward and inverse kinematics. A tesseract environment is created
# using URDF and SRDF files. The kinematics solver is configured using the SRDF file and plugin configuration files.

# Initialize Environment with a robot from URDF file
# The URDF and SRDF file must be configured. The kinematics solver also requires plugin configuration,
# which is specified in the SRDF file. For this example, the plugin configuration file is `abb_irb2400_plugins.yaml`
# and is located in the same directory as the SRDF file. This example uses the OPW kinematics solver, which is
# a solver for industrial 6-dof robots with spherical wrists. The kinematic parameters for the robot must
# be specified in the plugin configuration file in addition to the URDF file for the plugin to work.
# The other main solver is the KDL solver, which is used by the lbr_iiwa_14_r820 robot also included in the
# tesseract_support package. The KDL solver is a numerical solver and does not require additional configuration.

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

from tesseract_robotics_viewer import TesseractViewer

locator = GeneralResourceLocator()
env = Environment()
urdf_path = FilesystemPath("/home/artc/weijie/tesseract_python/urdf/rm_65_b.urdf")
srdf_path = FilesystemPath("/home/artc/weijie/tesseract_python/urdf/rm_65_b.srdf")
assert env.init(urdf_path, srdf_path, locator)

viewer = TesseractViewer()
viewer.update_environment(env, [0,0,0])

robot_joint_names = [f"joint{i+1}" for i in range(6)]

# Get the kinematics solver. The name "manipulator" is specified in the SRDF file
kin_group = env.getKinematicGroup("manipulator")
print(f"Kinematic group name: {kin_group.getName()}")

# Solve forward kinematics at a specific joint position
robot_joint_pos = np.deg2rad(np.array([0, -45, 90, 0 , 90, 0], dtype=np.float64))
viewer.update_joint_positions(robot_joint_names, robot_joint_pos)
viewer.start_serve_background()
fwdkin_result = kin_group.calcFwdKin(robot_joint_pos)
#fwdkin_result is a TransformMap, which is a dictionary of link names to Isometry3d. For this robot, we are
#interested in the transform of the "tool0" link
tool0_transform = fwdkin_result["Link6"]
# Print the transform as a translation and quaternion
print("Tool0 transform at joint position " + str(robot_joint_pos) + " is: ")
q = Quaterniond(tool0_transform.rotation())
print("Translation: " + str(tool0_transform.translation().flatten()))
print(f"Rotation: {q.w()} {q.x()} {q.y()} {q.z()}")



# Solve inverse kinematics at a specific tool0 pose
tool0_transform2 = Isometry3d.Identity() * Translation3d(0.38005958, -0.00003496, 0.74205694) * Quaterniond(-4.591528909320721e-06, 4.502811352206949e-05, 0.9999999989755488, -5.377145301590243e-07)

# Create a KinGroupIKInput and KinGroupIKInputs object. The KinGroupIKInputs object is a list of KinGroupIKInput
ik = KinGroupIKInput()
ik.pose = tool0_transform2
ik.tip_link_name = "Link6"
ik.working_frame = "base_cuboid"
iks = KinGroupIKInputs()
iks.append(ik)
# Solve IK
ik_result = kin_group.calcInvKin(iks, robot_joint_pos)

# if len(ik_result) > 0:
#     viewer.update_joint_positions(robot_joint_names, ik_result[0])

# Print the result
print(f"Found {len(ik_result)} solutions")
for i in range(len(ik_result)):
    print("Solution " + str(i) + ": " + str(ik_result[i].flatten()))



input("press enter to exit")