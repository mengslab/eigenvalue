
from __future__ import annotations
import numpy as np
from scipy import stats

def _as_arrays(time_points, values):
    t = np.asarray(time_points, dtype=float)
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    order = np.argsort(t)
    return t[order], y[order]

def _median_dt(t):
    if len(t) < 2:
        return 1.0
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(np.median(dt)) if len(dt) else 1.0

def _candidate_periods(t, n_grid=160):
    dt = _median_dt(t)
    total = max(float(t[-1] - t[0]), dt * 8)
    pmin = max(2.0 * dt, 2.0)
    pmax = max(min(48.0, total * 0.95), pmin + dt * 2)
    return np.linspace(pmin, pmax, n_grid)

def _phase_time_to_deg(phase_time, period):
    if not np.isfinite(phase_time) or not np.isfinite(period) or period <= 0:
        return np.nan
    return float(((phase_time / period) * 360.0 + 180.0) % 360.0 - 180.0)

def _rain_template(phase_frac, peak_frac):
    rise = np.clip(phase_frac / max(peak_frac, 1e-6), 0, 1)
    fall = np.clip((1 - phase_frac) / max(1 - peak_frac, 1e-6), 0, 1)
    return np.where(phase_frac <= peak_frac, rise, fall)

def rain_method(time_points, values, peak_border=(0.2, 0.8), period_grid=None):
    t, y = _as_arrays(time_points, values)
    if len(t) < 6:
        return {
            "method": "RAIN",
            "period_h": np.nan,
            "amplitude": np.nan,
            "phase_deg": np.nan,
            "p_value": np.nan,
            "score": np.nan,
            "notes": "Too few points for nonparametric rise/fall screening",
            "fitted": np.full_like(y, np.nan),
        }

    periods = _candidate_periods(t) if period_grid is None else np.asarray(period_grid, dtype=float)
    phase_offsets = np.linspace(0, periods.max() if len(periods) else t.ptp(), 24)
    peak_fracs = np.linspace(float(peak_border[0]), float(peak_border[1]), 7)

    best = None
    y_center = y - np.mean(y)
    for p in periods:
        phase_offsets = np.linspace(0, p, 24, endpoint=False)
        for phase in phase_offsets:
            phase_frac = ((t - phase) % p) / p
            for peak_frac in peak_fracs:
                template = _rain_template(phase_frac, peak_frac)
                tau, pval = stats.kendalltau(y_center, template)
                tau = float(tau) if np.isfinite(tau) else -np.inf
                pval = float(pval) if np.isfinite(pval) else np.nan
                fitted = np.mean(y) + np.std(y_center) * (template - np.mean(template)) / max(np.std(template), 1e-9)
                score = abs(tau)
                if best is None or score > best["score"]:
                    best = {
                        "period_h": float(p),
                        "amplitude": float((np.nanmax(fitted) - np.nanmin(fitted)) / 2.0),
                        "phase_deg": _phase_time_to_deg(phase, p),
                        "p_value": pval,
                        "score": score,
                        "notes": f"Rank-based rise/fall umbrella screening; peak fraction {peak_frac:.2f}",
                        "fitted": fitted,
                    }

    return {"method": "RAIN", **best}
