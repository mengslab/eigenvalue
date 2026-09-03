
from __future__ import annotations
import numpy as np
import pandas as pd
from src.rhythm_methods import METHOD_ORDER

def _profile(time_points, values):
    t = np.asarray(time_points, dtype=float)
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]; y = y[mask]
    if len(t) < 3:
        return {"n_points": int(len(t)), "regular": False, "short": True, "noisy": False, "damped": False, "asymmetric": False, "multifrequency": False, "median_dt": np.nan, "cv_dt": np.nan, "snr_proxy": np.nan}
    order = np.argsort(t)
    t = t[order]; y = y[order]
    dt = np.diff(t)
    pos = dt[np.isfinite(dt) & (dt > 0)]
    median_dt = float(np.median(pos)) if len(pos) else np.nan
    cv_dt = float(np.std(pos) / max(np.mean(pos), 1e-12)) if len(pos) else np.nan
    regular = bool(np.isfinite(cv_dt) and cv_dt < 0.05)
    short = len(y) < 24
    yc = y - np.mean(y)
    snr_proxy = float(np.std(yc) / max(np.std(np.diff(y)) if len(y) > 1 else 1e-12, 1e-12))
    noisy = snr_proxy < 1.25
    env = np.abs(yc)
    damped = False
    if len(env) >= 10:
        slope = np.polyfit(np.arange(len(env)), np.log(env + 1e-6), 1)[0]
        damped = bool(abs(slope) > 0.01)
    s = np.std(yc) + 1e-9
    skew = float(np.mean(((y - np.mean(y))/s)**3))
    asymmetric = abs(skew) > 0.5
    power = np.abs(np.fft.rfft(yc))
    multifrequency = bool(np.sum(power > (np.max(power) * 0.45 if len(power) else 1)) >= 2) if len(power) else False
    return {"n_points": int(len(y)), "regular": regular, "short": short, "noisy": noisy, "damped": damped, "asymmetric": asymmetric, "multifrequency": multifrequency, "median_dt": median_dt, "cv_dt": cv_dt, "snr_proxy": snr_proxy}

def recommend_methods(time_points, values):
    p = _profile(time_points, values)
    scores = {m: 0.0 for m in METHOD_ORDER}
    why = {m: [] for m in METHOD_ORDER}
    if p["regular"]:
        for m in ["Cosinor","Harmonic regression","FFT"]:
            scores[m] += 1.4
            why[m].append("regular sampling supports this method")
    else:
        scores["Lomb–Scargle"] += 2.5; why["Lomb–Scargle"].append("uneven sampling strongly favors Lomb–Scargle")
        scores["RAIN"] += 2.6; why["RAIN"].append("RAIN is designed for nonparametric rise/fall testing under irregular or non-ideal sampling")
        scores["FFT"] -= 0.8; why["FFT"].append("irregular sampling weakens FFT assumptions")
        scores["JTK-like"] += 0.8; why["JTK-like"].append("rank-based screening tolerates irregularity")

    if p["short"]:
        for m in ["Matrix pencil","Bayesian harmonic","Cosinor"]:
            scores[m] += 2.0
        why["Matrix pencil"].append("short series benefit from compact oscillatory decomposition")
        why["Bayesian harmonic"].append("short series benefit from shrinkage and uncertainty quantification")
        why["Cosinor"].append("simple sinusoidal fit remains interpretable on short data")
        scores["FFT"] -= 0.6; why["FFT"].append("short series reduce frequency resolution")
    if p["noisy"]:
        scores["Bayesian harmonic"] += 2.2; why["Bayesian harmonic"].append("posterior shrinkage helps noisy data")
        scores["RAIN"] += 1.9; why["RAIN"].append("nonparametric rise/fall testing is robust to non-Gaussian noise and outliers")
        scores["JTK-like"] += 1.0; why["JTK-like"].append("rank-style screening is comparatively robust")
        scores["Autocorrelation"] += 0.6; why["Autocorrelation"].append("coarse rhythm support remains useful in noise")
    if p["damped"]:
        scores["Matrix pencil"] += 2.8; why["Matrix pencil"].append("damped envelope favors oscillatory decomposition")
        scores["Wavelet"] += 1.1; why["Wavelet"].append("time-local power helps nonstationary signals")
    if p["asymmetric"]:
        scores["RAIN"] += 2.8; why["RAIN"].append("RAIN is well suited for asymmetric and sawtooth-like waveforms")
        scores["JTK-like"] += 0.8; why["JTK-like"].append("template screening may capture skewed signals")
        scores["Wavelet"] += 0.5; why["Wavelet"].append("time-local shape changes can aid interpretation")
    if p["multifrequency"]:
        scores["Harmonic regression"] += 1.8; why["Harmonic regression"].append("multiple harmonics detected")
        scores["Matrix pencil"] += 1.7; why["Matrix pencil"].append("multiple frequency content detected")
        scores["FFT"] += 0.8; why["FFT"].append("frequency content suggests spectral comparison")

    # baseline scientific utility
    base = {
        "Matrix pencil": ("captures damped oscillatory components", 1.0),
        "Cosinor": ("interpretable mesor/amplitude/phase fit", 1.0),
        "Harmonic regression": ("captures multi-harmonic rhythmic structure", 1.0),
        "Lomb–Scargle": ("strong default for period discovery", 0.8),
        "FFT": ("fast spectral overview", 0.5),
        "Wavelet": ("time-local rhythm support", 0.8),
        "Autocorrelation": ("coarse rhythm validation", 0.5),
        "JTK-like": ("screening-style rhythm ranking", 0.8),
        "ARSER-like": ("regression-style screening", 0.8),
        "Bayesian harmonic": ("uncertainty-aware harmonic inference", 0.9),
        "RAIN": ("nonparametric rise/fall umbrella screening for arbitrary rhythmic shapes", 1.1),
    }
    for m,(reason,val) in base.items():
        scores[m] += val
        why[m].append(reason)
    rows=[]
    for m in METHOD_ORDER:
        rows.append({"method": m, "recommendation_score": round(scores[m], 3), "recommended": scores[m] >= 2.0, "why": "; ".join(why[m])})
    df = pd.DataFrame(rows).sort_values(["recommendation_score","method"], ascending=[False,True]).reset_index(drop=True)
    profile_bits = []
    if p["regular"]: profile_bits.append("evenly sampled")
    else: profile_bits.append("unevenly sampled")
    if p["damped"]: profile_bits.append("damped/nonstationary")
    if p["short"]: profile_bits.append("short")
    if p["noisy"]: profile_bits.append("noisy")
    if p["asymmetric"]: profile_bits.append("asymmetric")
    if p["multifrequency"]: profile_bits.append("multi-frequency")
    summary = "Signal profile: " + ", ".join(profile_bits) + ". Top recommendations: " + ", ".join([f"{r.method} ({r.recommendation_score:.2f})" for r in df.head(3).itertuples()]) + "."
    return df, summary, p
