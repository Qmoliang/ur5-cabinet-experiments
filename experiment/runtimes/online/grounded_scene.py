from pathlib import Path
import json
from model import BoxObstacle,SceneDefinition
def load_scene():
    x=json.loads((Path(__file__).resolve().parent/'assets/scene.json').read_text(encoding='utf-8'))
    return SceneDefinition(name=x['name'],description=x['description'],boxes=tuple(BoxObstacle(b['name'],tuple(b['center']),tuple(b['half_size']),b['material']) for b in x['boxes']),q0=tuple(x['q0']),waypoints=tuple(tuple(p) for p in x['waypoints']),duration=x['duration'])
