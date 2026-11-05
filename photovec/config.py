"""Single configuration module: budget profiles, detail presets, run config."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Dict, Optional, Tuple

STAGES = ("load", "preprocess", "foliage", "extract", "fuse", "vectorise", "export")


@dataclass(frozen=True)
class BudgetProfile:
    """Everything that scales with the time budget."""
    name: str
    limit_s: Optional[float]            # None = unlimited (cancel only)
    work_long_side: int                 # working resolution (long side, px)
    edge_scales: Tuple[float, ...]      # Gaussian sigmas for multi-scale edges
    tonal_levels: int                   # tonal-contour levels
    ridge_sigmas: Tuple[float, ...]     # Sato sigmas for centre lines
    ridge_bright_too: bool              # also detect bright thin ridges
    foliage_levels: int                 # coarse tonal levels inside foliage
    curve_tol: float                    # Bezier fit tolerance (px)
    stage_fractions: Dict[str, float] = field(default_factory=dict)


_FRAC = dict(load=.06, preprocess=.04, foliage=.06, extract=.45,
             fuse=.13, vectorise=.16, export=.10)

PROFILES: Dict[str, BudgetProfile] = {
    "15": BudgetProfile("Fast", 15.0, 1152, (1.4,), 2, (1.5,), False, 2, 1.8, _FRAC),
    "30": BudgetProfile("Balanced", 30.0, 1792, (0.9, 2.2), 3, (1.2, 2.0), False, 3, 1.5, _FRAC),
    "unlimited": BudgetProfile("Max quality", None, 3072, (0.8, 1.6, 3.0), 4,
                               (1.0, 1.6, 2.4), True, 4, 1.1, _FRAC),
}


@dataclass(frozen=True)
class DetailPreset:
    edge_pct: float        # gradient percentile used as Canny high threshold
    ridge_pct: float       # ridge response percentile (hysteresis high)
    tonal_min_area: float  # min contour area as fraction of image area
    min_chain_len: float   # px at 1280 long side
    lsd_min_len: float     # px at 1280 long side
    smooth_std: float      # local-std below which edges are suppressed


DETAILS: Dict[str, DetailPreset] = {
    "clean": DetailPreset(95.0, 98.5, 4e-4, 22.0, 30.0, 3.5),
    "photocad": DetailPreset(91.0, 97.0, 1.2e-4, 12.0, 20.0, 2.8),
    "maximum": DetailPreset(85.0, 94.5, 4e-5, 7.0, 12.0, 2.0),
}


@dataclass
class RunConfig:
    budget: str = "30"                  # "15" | "30" | "unlimited"
    mode: str = "general"               # stage 1 implements general (auto -> general)
    line_aware: Optional[bool] = None   # None = auto-detect strong straight structure
    foliage: str = "stylized"           # stylized | simplify | remove | keep_raw
    detail: str = "photocad"            # clean | photocad | maximum
    stroke_frac: float = 0.00058        # stroke width / drawing width
    margin_frac: float = 0.015          # page margin / drawing long side
    simplify: float = 1.0               # multiplier on fitting tolerances
    seed: int = 1234
    units_per_px: float = 1.0           # DXF scale (mm per px when calibrated)
    formats: Tuple[str, ...] = ("svg", "dxf", "pdf", "png")

    def validate(self) -> None:
        if self.budget not in PROFILES:
            raise ValueError(f"budget must be one of {list(PROFILES)}")
        if self.foliage not in ("stylized", "simplify", "remove", "keep_raw"):
            raise ValueError("foliage must be stylized|simplify|remove|keep_raw")
        if self.detail not in DETAILS:
            raise ValueError(f"detail must be one of {list(DETAILS)}")
        if self.mode not in ("facade", "general", "auto"):
            raise ValueError("mode must be facade|general|auto")

    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig":
        import yaml
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        known = {f.name for f in fields(cls)}
        bad = set(raw) - known
        if bad:
            raise ValueError(f"unknown config keys: {sorted(bad)}")
        if "formats" in raw:
            raw["formats"] = tuple(raw["formats"])
        cfg = cls(**raw)
        cfg.validate()
        return cfg
