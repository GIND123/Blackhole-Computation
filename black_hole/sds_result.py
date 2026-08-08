"""Storage container for one saved scalar evolution.

The container is kept separate from the solver so that post-processing can
read archived evolutions without importing Dedalus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .reproducibility import reproducibility_metadata


@dataclass
class SdSSimulationResult:
    """In-memory output from one SdS scalar evolution."""

    rho: np.ndarray
    areal_radius: np.ndarray
    signal_times: np.ndarray
    observer_rho: np.ndarray
    observer_areal_radius: np.ndarray
    signals: np.ndarray
    snapshot_times: np.ndarray
    u_snapshots: np.ndarray
    constraint_linf: np.ndarray
    constraint_l2: np.ndarray
    metadata: dict

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata.setdefault("reproducibility", reproducibility_metadata())
        np.savez_compressed(
            path,
            rho=self.rho,
            areal_radius=self.areal_radius,
            signal_times=self.signal_times,
            observer_rho=self.observer_rho,
            observer_areal_radius=self.observer_areal_radius,
            signals=self.signals,
            snapshot_times=self.snapshot_times,
            u_snapshots=self.u_snapshots,
            constraint_linf=self.constraint_linf,
            constraint_l2=self.constraint_l2,
            metadata=np.array(json.dumps(self.metadata, sort_keys=True)),
        )


def load_sds_result(path: Path) -> SdSSimulationResult:
    """Load a saved SdS scalar result without enabling pickle."""

    with np.load(path, allow_pickle=False) as data:
        return SdSSimulationResult(
            rho=data["rho"],
            areal_radius=data["areal_radius"],
            signal_times=data["signal_times"],
            observer_rho=data["observer_rho"],
            observer_areal_radius=data["observer_areal_radius"],
            signals=data["signals"],
            snapshot_times=data["snapshot_times"],
            u_snapshots=data["u_snapshots"],
            constraint_linf=data["constraint_linf"],
            constraint_l2=data["constraint_l2"],
            metadata=json.loads(data["metadata"].item()),
        )
