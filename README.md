# Tesseract Python

## Quick Start

Run example scripts:
```bash
uv run examples/my_tesseract_planning_example_no_composer.py
# or
uv run examples/my_tesseract_kinematics_example.py
```

## Project Goal

- Enable motion planning for the Realman [RM65-B](https://www.realman-robotics.com/rm65-ae1.html) robot arm
- No ROS required

## Status

- URDF loads successfully
- SRDF and plugins.yaml created manually (referenced ABB IRB2400)

## Current Blockers

- [ ] IK solution: expected pose is different from actual result
- [ ] This leads to the following error when running OMPL planner:

```bash
Traceback (most recent call last):
  File "/home/artc/weijie/tesseract_python/examples/my_tesseract_planning_example_no_composer.py", line 139, in <module>
    response=ompl_planner.solve(request)
  File "/home/artc/weijie/tesseract_python/.venv/lib/python3.10/site-packages/tesseract_robotics/tesseract_motion_planners_ompl/tesseract_motion_planners_ompl_python.py", line 603, in solve
    return _tesseract_motion_planners_ompl_python.OMPLMotionPlanner_solve(self, request)
RuntimeError: In OMPLPlannerFreespaceConfig: All start states are either in collision or outside limits
```