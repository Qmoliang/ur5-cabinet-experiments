"""Fail-fast verification of the final robust sphere/ellipsoid evidence."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_geometry_ablation"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    sphere = load(RESULTS / "robust_slot_sphere_direct" / "summary.json")
    ellipsoid = load(
        RESULTS / "robust_slot_ellipsoid_full_robot_direct" / "summary.json"
    )
    robustness = load(RESULTS / "robustness_sphere_dither" / "aggregate.json")
    refined_sphere = load(
        RESULTS
        / "resolution_sensitivity_sphere_z0075_20260828"
        / "robust_slot_sphere_direct"
        / "summary.json"
    )
    barrier = load(
        RESULTS / "robustness_sphere_dither" / "barrier_certificate.json"
    )
    assert sphere["target_source"] == ellipsoid["target_source"] == "final_goal_only"
    assert sphere["guidance"] == ellipsoid["guidance"] == "direct"
    assert sphere["objective_dither_std"] == ellipsoid["objective_dither_std"] == 0.0
    assert not sphere["success"]
    assert sphere["final_error_m"] > 0.20
    assert ellipsoid["success"]
    assert ellipsoid["robot_certificate_geometry"] == "full-ellipsoid"
    assert ellipsoid["final_error_m"] < 0.018
    assert sphere["solved_qp_fraction"] == ellipsoid["solved_qp_fraction"] == 1.0
    assert ellipsoid["minimum_certificate_clearance_m"] > 0.0
    assert sphere["minimum_mujoco_world_contact_distance_m"] is None
    assert ellipsoid["minimum_mujoco_world_contact_distance_m"] is None
    assert barrier["sphere_model_is_closed"]
    assert barrier["sphere_modeled_opening_m"] < 0.0
    assert barrier["ellipsoid_model_is_open"]
    assert barrier["ellipsoid_modeled_opening_m"] > 0.0
    assert barrier["critical_proxy_equals_attachment_error_m"] == 0.0
    assert robustness["trials"] == 12
    assert robustness["successes"] == 0
    assert robustness["minimum_solved_qp_fraction"] == 1.0
    assert refined_sphere["success"]
    assert refined_sphere["obstacle_proxies"] == 969
    assert refined_sphere["final_error_m"] < 0.018
    assert (ROOT / "results" / "shelf_prescribed" / "summary.json").exists()
    assert (ROOT / "results" / "cage_prescribed" / "summary.json").exists()
    figure = ROOT / "figures" / "geometry_ablation" / "fig2_robust_slot_full_ellipsoid.png"
    assert figure.stat().st_size > 100_000
    with Image.open(figure) as image:
        image.verify()
        assert image.width >= 2500 and image.height >= 2000
    print(
        json.dumps(
            {
                "status": "verified",
                "sphere_final_error_mm": 1000.0 * sphere["final_error_m"],
                "ellipsoid_final_error_mm": 1000.0 * ellipsoid["final_error_m"],
                "sphere_opening_mm": 1000.0 * barrier["sphere_modeled_opening_m"],
                "ellipsoid_opening_mm": 1000.0 * barrier["ellipsoid_modeled_opening_m"],
                "perturbed_sphere_successes": robustness["successes"],
                "perturbed_sphere_trials": robustness["trials"],
                "refined_sphere_proxies": refined_sphere["obstacle_proxies"],
                "refined_sphere_final_error_mm": 1000.0
                * refined_sphere["final_error_m"],
                "unit_tests_expected": 13,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
