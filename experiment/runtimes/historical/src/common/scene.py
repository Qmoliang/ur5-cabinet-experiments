"""Only the accepted drawer scene; its MJCF is frozen, not regenerated."""
import json
from pathlib import Path
from robot import BoxObstacle, SceneDefinition

ROOT = Path(__file__).resolve().parents[2]
SAFETY_MARGIN = 0.006
NEAR_DISTANCE = 0.040
CONTACT_DISTANCE = 0.0
CAMERA_HZ = 30.0
INITIAL_CALIBRATED_FREE_PADDING = 0.005
# Historical internal default, not the formal run's goal gate. The frozen
# entry passes success_tolerance=0.001 and success_hold_cycles=50 explicitly.
SUCCESS_TOLERANCE = 0.018


def formal_drawer_camera_quarter_scene():
    value = json.loads((ROOT / 'assets/scene.json').read_text(encoding='utf-8'))
    return SceneDefinition(
        name=value['name'], description=value['description'],
        boxes=tuple(BoxObstacle(b['name'], tuple(b['center']), tuple(b['half_size']), b['material'])
                    for b in value['boxes']),
        q0=tuple(value['q0']), waypoints=tuple(tuple(p) for p in value['waypoints']),
        duration=value['duration'])


def _protocol_scene(version):
    if version != 'camera_quarter':
        raise ValueError('Only the accepted camera_quarter drawer scene is packaged')
    return formal_drawer_camera_quarter_scene()
