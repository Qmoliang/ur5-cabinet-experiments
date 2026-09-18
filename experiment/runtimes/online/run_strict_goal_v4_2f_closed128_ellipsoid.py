"""Run the v4.2f 128 mm passage with the frozen thin ellipsoid proxy."""

from run_strict_goal_v4_2e_reachable_thin_ellipsoid import main


if __name__ == "__main__":
    main(
        scene_version="camera_quarter_closed128",
        protocol_version="v4_2f",
    )
