"""Run the v4.2g task with spheres matched to the 10 mm CenterVox ellipsoids."""

from run_strict_goal_v4_2e_matched_sphere import main


if __name__ == "__main__":
    main(
        scene_version="camera_quarter",
        protocol_version="v4_2g",
        centervox_size=0.010,
    )
