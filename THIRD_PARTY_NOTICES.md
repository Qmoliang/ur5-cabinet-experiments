# Third-party notices

## UR5 visualization assets

UR5e visual meshes are derived from the MuJoCo Menagerie Universal Robots model and ROS Industrial resources. The scene retains the experiment's robot kinematics and certificate spheres; using these meshes does not replace the experimental robot model with a certified UR5e model.

- [MuJoCo Menagerie source](https://github.com/google-deepmind/mujoco_menagerie/tree/main/universal_robots_ur5e)
- Original license: experiment/assets/ur5e/LICENSE (also retained in the frozen runtime asset directories).
- Copyright 2018, ROS Industrial Consortium. The full included BSD license text controls permitted use.

## Numerical software

The frozen known-geometry runtime retains metadata and the license for quadprog 0.1.13. The platform binary is omitted from this public snapshot; install the dependency separately. Its license remains at experiment/runtimes/known/external_neo/vendor/quadprog-0.1.13.dist-info/LICENSE. Other runtime dependencies, including MuJoCo, NumPy, SciPy, OSQP and associated libraries, retain their respective licenses.

## Research methods and scope

LiuQP and NEO identify the research methods under comparison. This project is an experimental implementation and adaptation; it is not an official project page for either paper. Timing observations here should not be described as published cross-paper benchmarks.

The cabinet videos and posters are rendered from this project's recorded MuJoCo states. No footage from another project's demonstration is used.

## Repository license

No new blanket license is assigned to the original project code in this snapshot. Existing third-party license notices remain applicable. Select an appropriate repository license before offering broader reuse rights.


## Research document rendering

The library uses Marked 17.0.5 (MIT), MathJax 3.2.2 (Apache 2.0), and Mermaid 12.0.0 (MIT). Full original licenses and bundled dependency notices are retained. See [document rendering notices](docs/library/licenses.html). Original document contents retain their existing rights.
