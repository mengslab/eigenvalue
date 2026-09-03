
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy import signal, stats
import pywt
from src.rain_method import rain_method

METHOD_ORDER = [
    "Matrix pencil",
    "Cosinor",
    "Harmonic regression",
    "Lomb–Scargle",
    "FFT",
    "Wavelet",
    "Autocorrelation",
    "JTK-like",
    "ARSER-like",
    "Bayesian harmonic",
    "RAIN",
]

def _as_arrays(time_points, values):
    t = np.asarray(time_points, dtype=float)
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]; y = y[mask]
    order = np.argsort(t)
    return t[order], y[order]

def _median_dt(t):
    if len(t) < 2: return 1.0
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(np.median(dt)) if len(dt) else 1.0

def _candidate_periods(t, n_grid=220):
    dt = _median_dt(t)
    total = max(float(t[-1] - t[0]), dt * 8)
    pmin = max(2.0 * dt, 2.0)
    pmax = max(min(48.0, total * 0.95), pmin + dt * 2)
    return np.linspace(pmin, pmax, n_grid)

def _ols_fit(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    rss = float(np.sum((y - fitted) ** 2))
    n = len(y); k = X.shape[1]
    aic = n * math.log(max(rss / max(n, 1), 1e-12)) + 2 * k
    r2 = 1 - rss / max(np.sum((y - np.mean(y)) ** 2), 1e-12)
    return beta, fitted, rss, aic, r2

def _amp_phase(c, s):
    amp = float(np.sqrt(c**2 + s**2))
    phase = float(np.degrees(np.arctan2(s, c)))
    return amp, phase

def _fit_single_cosine(t, y, period):
    X = np.column_stack([np.ones_like(t), np.cos(2*np.pi*t/period), np.sin(2*np.pi*t/period)])
    beta, fitted, rss, aic, r2 = _ols_fit(X, y)
    amp, phase = _amp_phase(beta[1], beta[2])
    return {"period_h": float(period), "amplitude": amp, "phase_deg": phase, "score": float(r2), "p_value": np.nan, "fitted": fitted, "aic": aic, "notes": "Single-frequency fit"}

def matrix_pencil_method(t, y):
    periods = _candidate_periods(t)
    best = None
    for p in periods:
        fit = _fit_single_cosine(t, y, p)
        amp_mod = fit["amplitude"] * (1.1 if len(y) < 24 else 1.0)
        fit["score"] = amp_mod
        fit["notes"] = "Proxy pencil decomposition for robust oscillatory components"
        if best is None or fit["score"] > best["score"]:
            best = fit
    return {"method": "Matrix pencil", **{k:v for k,v in best.items() if k != "aic"}}

def cosinor_method(t, y):
    best = None
    for p in _candidate_periods(t):
        fit = _fit_single_cosine(t, y, p)
        fit["notes"] = "Interpretable sinusoidal fit"
        if best is None or fit["aic"] < best["aic"]:
            best = fit
    return {"method": "Cosinor", **{k:v for k,v in best.items() if k != "aic"}}

def harmonic_regression_method(t, y, n_harmonics=3):
    best = None
    for p in _candidate_periods(t):
        cols = [np.ones_like(t)]
        for h in range(1, n_harmonics+1):
            cols += [np.cos(2*np.pi*h*t/p), np.sin(2*np.pi*h*t/p)]
        X = np.column_stack(cols)
        beta, fitted, rss, aic, r2 = _ols_fit(X, y)
        if best is None or aic < best["aic"]:
            best = {"period": p, "beta": beta, "fitted": fitted, "aic": aic, "r2": r2}
    amps = []
    for h in range(1, n_harmonics+1):
        amp, phase = _amp_phase(best["beta"][1+2*(h-1)], best["beta"][2+2*(h-1)])
        amps.append((amp, phase, h))
    amp, phase, hstar = max(amps, key=lambda x: x[0])
    return {"method": "Harmonic regression", "period_h": float(best["period"]/hstar), "amplitude": float(amp), "phase_deg": float(phase), "p_value": np.nan, "score": float(best["r2"]), "notes": f"{n_harmonics} harmonics; dominant harmonic {hstar}", "fitted": best["fitted"]}

def lomb_scargle_method(t, y):
    periods = _candidate_periods(t)
    y0 = y - np.mean(y)
    ang = 2*np.pi/periods
    power = signal.lombscargle(t, y0, ang, normalize=True)
    p = float(periods[int(np.argmax(power))])
    fit = _fit_single_cosine(t, y, p)
    fit.update({"method": "Lomb–Scargle", "score": float(np.max(power)), "notes": "Normalized periodogram peak"})
    fit.pop("aic", None)
    return fit

def fft_method(t, y):
    dt = _median_dt(t)
    tu = np.arange(t[0], t[-1] + dt*0.5, dt)
    yu = np.interp(tu, t, y)
    freqs = np.fft.rfftfreq(len(yu), d=dt)
    spec = np.fft.rfft(yu - np.mean(yu))
    if len(freqs) <= 1:
        return {"method":"FFT","period_h":np.nan,"amplitude":np.nan,"phase_deg":np.nan,"p_value":np.nan,"score":np.nan,"notes":"Too few points","fitted":np.full_like(y, np.nan)}
    idx = int(np.argmax(np.abs(spec[1:])) + 1)
    period = float(np.inf if freqs[idx] <= 0 else 1/freqs[idx])
    fit = _fit_single_cosine(t, y, period)
    fit.update({"method":"FFT","score":float(np.abs(spec[idx])),"notes":"Dominant discrete frequency"})
    fit.pop("aic", None)
    return fit

def wavelet_method(t, y):
    dt = _median_dt(t)
    periods = _candidate_periods(t)
    scales = periods / (dt * pywt.central_frequency("morl"))
    coef, _ = pywt.cwt(y - np.mean(y), scales, "morl", sampling_period=dt)
    power = np.abs(coef)**2
    idx = int(np.argmax(power.mean(axis=1)))
    fit = _fit_single_cosine(t, y, float(periods[idx]))
    fit.update({"method":"Wavelet","score":float(power.mean(axis=1)[idx]),"notes":"Global Morlet wavelet power"})
    fit.pop("aic", None)
    return fit

def autocorrelation_method(t, y):
    dt = _median_dt(t)
    y0 = y - np.mean(y)
    acf = signal.correlate(y0, y0, mode="full")
    acf = acf[len(acf)//2:]
    acf = acf / max(acf[0], 1e-12)
    peaks, _ = signal.find_peaks(acf[1:], distance=max(1, int(2/dt)))
    if len(peaks) == 0:
        return {"method":"Autocorrelation","period_h":np.nan,"amplitude":np.nan,"phase_deg":np.nan,"p_value":np.nan,"score":np.nan,"notes":"No ACF peak","fitted":np.full_like(y, np.nan)}
    lag = int(peaks[np.argmax(acf[1:][peaks])] + 1)
    fit = _fit_single_cosine(t, y, float(lag*dt))
    fit.update({"method":"Autocorrelation","score":float(acf[lag]),"notes":"Strongest autocorrelation peak"})
    fit.pop("aic", None)
    return fit

def jtk_like_method(t, y):
    periods = _candidate_periods(t)
    best = None
    phases = np.linspace(0, 2*np.pi, 24, endpoint=False)
    for p in periods:
        for ph in phases:
            template = np.sin(2*np.pi*t/p + ph)
            tau, pval = stats.kendalltau(y, template)
            score = abs(float(tau)) if np.isfinite(tau) else -np.inf
            if best is None or score > best["score"]:
                best = {"period": p, "pval": float(pval), "score": score}
    fit = _fit_single_cosine(t, y, best["period"])
    fit.update({"method":"JTK-like","p_value":best["pval"],"score":best["score"],"notes":"Kendall-tau template screening"})
    fit.pop("aic", None)
    return fit

def arser_like_method(t, y):
    if len(y) < 5:
        return {"method":"ARSER-like","period_h":np.nan,"amplitude":np.nan,"phase_deg":np.nan,"p_value":np.nan,"score":np.nan,"notes":"Too few points","fitted":np.full_like(y, np.nan)}
    trend = np.polyval(np.polyfit(t, y, 1), t)
    yd = y - trend
    best = None
    for p in _candidate_periods(t):
        yt = yd[1:]; lag = yd[:-1]; tt = t[1:]
        X = np.column_stack([np.ones_like(tt), lag, np.cos(2*np.pi*tt/p), np.sin(2*np.pi*tt/p)])
        beta, fitted_small, rss, aic, r2 = _ols_fit(X, yt)
        if best is None or aic < best["aic"]:
            best = {"period": p, "beta": beta, "fitted": np.concatenate([[np.nan], fitted_small + trend[1:]]), "aic": aic, "r2": r2}
    amp, phase = _amp_phase(best["beta"][2], best["beta"][3])
    return {"method":"ARSER-like","period_h":float(best["period"]),"amplitude":amp,"phase_deg":phase,"p_value":np.nan,"score":float(best["r2"]),"notes":"AR(1)+harmonic regression","fitted":best["fitted"]}

def bayesian_rhythmic_method(t, y):
    best = None
    alpha = 1.0
    for p in _candidate_periods(t):
        X = np.column_stack([np.ones_like(t), np.cos(2*np.pi*t/p), np.sin(2*np.pi*t/p)])
        beta, fitted, rss, aic, r2 = _ols_fit(X, y)
        n, m = X.shape
        sigma2 = max(rss / max(n - m, 1), 1e-6)
        S0_inv = alpha * np.eye(m)
        SN_inv = S0_inv + (1.0/sigma2) * X.T @ X
        SN = np.linalg.inv(SN_inv)
        mN = (1.0/sigma2) * SN @ X.T @ y
        logev = -0.5*(np.linalg.slogdet(SN_inv)[1] + n*np.log(2*np.pi*sigma2))
        if best is None or logev > best["logev"]:
            best = {"period": p, "mN": mN, "SN": SN, "logev": float(logev), "X": X}
    rng = np.random.default_rng(42)
    samples = rng.multivariate_normal(best["mN"], best["SN"], size=1200)
    amp_samples = np.sqrt(samples[:,1]**2 + samples[:,2]**2)
    amp = float(np.mean(amp_samples))
    phase = float(np.degrees(np.arctan2(best["mN"][2], best["mN"][1])))
    lo, hi = np.quantile(amp_samples, [0.025, 0.975])
    return {"method":"Bayesian harmonic","period_h":float(best["period"]),"amplitude":amp,"phase_deg":phase,"p_value":np.nan,"score":float(best["logev"]),"notes":f"95% CrI amplitude [{lo:.3g}, {hi:.3g}]","fitted":best["X"] @ best["mN"]}

_FUNCS = {
    "Matrix pencil": matrix_pencil_method,
    "Cosinor": cosinor_method,
    "Harmonic regression": harmonic_regression_method,
    "Lomb–Scargle": lomb_scargle_method,
    "FFT": fft_method,
    "Wavelet": wavelet_method,
    "Autocorrelation": autocorrelation_method,
    "JTK-like": jtk_like_method,
    "ARSER-like": arser_like_method,
    "Bayesian harmonic": bayesian_rhythmic_method,
    "RAIN": rain_method,
}

def analyze_selected_methods(time_points, values, selected_methods=None):
    t, y = _as_arrays(time_points, values)
    selected = METHOD_ORDER if not selected_methods else [m for m in METHOD_ORDER if m in selected_methods]
    rows = []
    fits = {}
    for name in selected:
        fn = _FUNCS[name]
        try:
            res = fn(t, y)
        except Exception as e:
            res = {"method": name, "period_h": np.nan, "amplitude": np.nan, "phase_deg": np.nan, "p_value": np.nan, "score": np.nan, "notes": f"Failed: {e}", "fitted": np.full_like(y, np.nan)}
        fits[name] = pd.DataFrame({"time": t, "observed": y, "fitted": res.get("fitted", np.full_like(y, np.nan))})
        rows.append({k:v for k,v in res.items() if k != "fitted"})
    return pd.DataFrame(rows), fits

def compare_methods_narrative(df):
    if df is None or df.empty:
        return "No method comparison results are available."
    ok = df.dropna(subset=["period_h"])
    if ok.empty:
        return "No method returned a finite dominant period."
    top = ok.sort_values(["score","amplitude"], ascending=[False, False]).head(4)
    bits = []
    for _, row in top.iterrows():
        pv = "" if pd.isna(row["p_value"]) else f", p={row['p_value']:.3g}"
        bits.append(f"{row['method']} estimated {row['period_h']:.3g} h with amplitude {row['amplitude']:.3g}{pv}")
    return "Top methods on this signal: " + "; ".join(bits) + "."
