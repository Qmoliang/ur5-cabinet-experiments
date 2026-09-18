"""Value types required by the original C++ occupancy wrapper."""
from dataclasses import dataclass

UNKNOWN = 0
FREE = 1
OCCUPIED = 2

@dataclass(frozen=True)
class MapUpdateStats:
    frame_index: int
    rays: int
    free_updates: int
    occupied_updates: int
    free_voxels: int
    occupied_voxels: int
    unknown_queries: int

@dataclass(frozen=True)
class SweptVolumeCertificate:
    safe: bool
    occupied_voxels: int
    unknown_voxels: int
    checked_voxels: int
