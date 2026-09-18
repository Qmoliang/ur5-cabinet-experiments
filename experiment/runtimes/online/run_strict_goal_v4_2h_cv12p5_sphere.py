"""Run v4.2h with spheres matched to the 12.5 mm CenterVox ellipsoids."""

from run_strict_goal_v4_2e_matched_sphere import main


if __name__ == "__main__":
    main(
        scene_version="camera_quarter",
        protocol_version="v4_2h",
        centervox_size=0.0125,
    )
