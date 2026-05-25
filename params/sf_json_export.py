# sf_json_export.py
import os, json
from typing import Dict, Any
import numpy as np

def _asarr(x):
    return np.asarray(x, dtype=float)

def _aslist(x):
    a = _asarr(x)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a.tolist()

def _load_maybe(path, fallback=0.0):
    return _asarr(np.load(path)) if os.path.exists(path) else _asarr(fallback)

def build_complete_json_block_2d(emap, cat, *, energy_str="13.6 TeV") -> Dict[str, Any]:
    """
    Build the complete JSON block for ONE (histname, category) in 2D,
    consolidating anything produced earlier (era/env .npy). Safe if files are missing.
    """
    assert emap.dim == 2, "This builder is for 2D maps."

    axx, axy = emap.axis_x, emap.axis_y

    # Nominal SF & stat straight from the current EfficiencyMap (nominal run)
    sf_nom = _asarr(emap.eff[{'map': 'sf'}].values())
    stat_abs = _asarr(emap.unc_eff[{'map': 'unc_sf'}].values())

    # Optional pieces (load from disk if present)
    era_path = os.path.join(emap.plot_dir, f"{emap.histname}__era__{cat}.npy")
    era_abs  = _load_maybe(era_path, 0.0)

    env_met_path   = os.path.join(emap.plot_dir, f"{emap.histname}__env_met__{cat}.npy")
    env_njets_path = os.path.join(emap.plot_dir, f"{emap.histname}__env_njets__{cat}.npy")
    env_npv_path   = os.path.join(emap.plot_dir, f"{emap.histname}__env_npv__{cat}.npy")

    env_met   = _load_maybe(env_met_path,   0.0)
    env_njets = _load_maybe(env_njets_path, 0.0)
    env_npv   = _load_maybe(env_npv_path,   0.0)

    # Correlation absolute from config (corr_abs = |1 - corr_coeff| * SF_nominal)
    corr_frac = 0.0
    try:
        cfg = emap.config.get('correlation', {})
        v = cfg.get(str(emap.year), cfg.get(emap.year, None))
        if v is not None:
            corr_frac = abs(1.0 - float(v))
    except Exception:
        pass
    corr_abs = corr_frac * sf_nom

    # Variations
    def pm(base, delta):
        return base - delta, base + delta

    statDown,  statUp  = pm(sf_nom, stat_abs)
    eraDown,   eraUp   = pm(sf_nom, era_abs)
    metDown,   metUp   = pm(sf_nom, env_met)
    njetsDown, njetsUp = pm(sf_nom, env_njets)
    npvDown,   npvUp   = pm(sf_nom, env_npv)
    corrDown,  corrUp  = pm(sf_nom, corr_abs)

    # Total = sqrt(stat^2 + era^2 + max_env^2 + corr^2)
    env_abs = np.maximum.reduce([env_met, env_njets, env_npv])
    total_abs = np.sqrt(stat_abs**2 + era_abs**2 + env_abs**2 + corr_abs**2)
    totalDown, totalUp = pm(sf_nom, total_abs)

    # Lumi fractions meta (present only if spliteras was run; harmless if empty)
    lumi_fracs = {}
    if getattr(emap, "mode", "") == "spliteras":
        lumi_fracs = {k: float(v) for k, v in getattr(emap, "lumi_fractions", {}).items()}

    return {
        "version": 1,
        "histname": emap.histname,
        "category": cat,
        "year": str(emap.year),
        "mode": emap.mode,  # will be "standard" here; fine
        "axes": {
            "x": {"name": axx.name, "label": axx.label, "edges": _aslist(axx.edges)},
            "y": {"name": axy.name, "label": axy.label, "edges": _aslist(axy.edges)},
        },
        "meta": {
            "lumi_fractions": lumi_fracs,
            "energy": energy_str,
            "units": "dimensionless SF",
            "provenance": {
                "stat_source": "unc_sf",
                "era_method": "lumi-weighted diff (Σ w_i * SF_i) - SF_tot",
                "envelopes": ["met", "njets", "npv"],
                "corr_fraction": float(corr_frac),
                "loaded_files": {
                    "era": os.path.basename(era_path) if os.path.exists(era_path) else None,
                    "env_met": os.path.basename(env_met_path) if os.path.exists(env_met_path) else None,
                    "env_njets": os.path.basename(env_njets_path) if os.path.exists(env_njets_path) else None,
                    "env_npv": os.path.basename(env_npv_path) if os.path.exists(env_npv_path) else None,
                }
            },
        },
        "maps": {
            "nominal": { "sf": _aslist(sf_nom) },
            "components": {
                "stat":      { "abs": _aslist(stat_abs) },
                "era":       { "abs": _aslist(era_abs)  },
                "met_env":   { "abs": _aslist(env_met)   },
                "njets_env": { "abs": _aslist(env_njets) },
                "npv_env":   { "abs": _aslist(env_npv)   },
                "corr":      { "abs": _aslist(corr_abs)  },
            },
            "variations": {
                "statDown":  { "sf": _aslist(statDown)  },
                "statUp":    { "sf": _aslist(statUp)    },
                "eraDown":   { "sf": _aslist(eraDown)   },
                "eraUp":     { "sf": _aslist(eraUp)     },
                "metDown":   { "sf": _aslist(metDown)   },
                "metUp":     { "sf": _aslist(metUp)     },
                "njetsDown": { "sf": _aslist(njetsDown) },
                "njetsUp":   { "sf": _aslist(njetsUp)   },
                "npvDown":   { "sf": _aslist(npvDown)   },
                "npvUp":     { "sf": _aslist(npvUp)     },
                "corrDown":  { "sf": _aslist(corrDown)  },
                "corrUp":    { "sf": _aslist(corrUp)    },
                "totalDown": { "sf": _aslist(totalDown) },
                "totalUp":   { "sf": _aslist(totalUp)   },
            },
        },
    }

class PerVariableAccumulator:
    """Collect all categories for ONE variable (histname) and write one JSON file."""
    def __init__(self, histname: str):
        self.histname = histname
        self._cats: Dict[str, Dict[str, Any]] = {}

    def add_cat_block(self, cat: str, block: Dict[str, Any]):
        self._cats[cat] = block

    def save(self, outdir: str):
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f"{self.histname}__complete.json")
        with open(path, "w") as f:
            json.dump(self._cats, f, indent=2)
        return path
