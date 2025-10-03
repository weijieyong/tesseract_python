# Tesseract Python

Lightweight workspace for testing Tesseract motion planning with the Realman [RM65-B](https://www.realman-robotics.com/rm65-ae1.html) arm—no ROS in the loop.

## Preview

![](media/plan_side_by_side.gif)

## Running The Samples

Assumes [`uv`](https://github.com/astral-sh/uv) is available.

```bash
# demo using the Tesseract Planning Task Composer
uv run examples/my_tesseract_planning_example_composer.py
# without Tesseract Planning Task Composer
uv run examples/my_tesseract_planning_example_no_composer.py
# solving kinematics
uv run examples/my_tesseract_kinematics_example.py
```

## Highlights

- RM65-B URDF loads with dummy gripper and TCP
- SRDF, KDL plugin config created manually
- Obstacle-aware motion plans execute successfully in viewer

## Notes

- ~~OMPL error (`All start states are either in collision or outside limits`) resolved by adjusting EEF orientation to stay within joint limits~~
