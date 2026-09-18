"""Run the v4.2g 10 mm CenterVox task with the frozen thin ellipsoid rule."""

from run_strict_goal_v4_2e_reachable_thin_ellipsoid import main


if __name__ == "__main__":
    main(
        scene_version="camera_quarter",
        protocol_version="v4_2g",
        centervox_size=0.010,
    )
