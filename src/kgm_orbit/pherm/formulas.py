import numpy as np

def finite(x):
    a = np.asarray(x, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError('Non-finite input')
    return a

def claim_weight(scores, tau):
    v = finite(scores)
    tau = float(finite(tau))
    if v.ndim != 1 or not v.size or tau < 0 or np.any(v <= tau):
        raise ValueError('Require nonempty accepted scores strictly above tau')
    return float(np.mean(v - tau))

def aggregate(z, weights, directions, observed):
    z = np.asarray(z, dtype=float)
    (w, d) = (finite(weights), finite(directions))
    mask = np.asarray(observed)
    if z.ndim != 2 or w.ndim != 1 or d.shape != w.shape or (z.shape[1] != w.size) or (mask.shape != z.shape) or (mask.dtype != np.bool_) or (not w.size) or np.any(w <= 0) or (not np.isin(d,
        [-1, 1]).all()):
        raise ValueError('Invalid aggregation shapes, weights, directions or mask')
    if not np.isfinite(z[mask]).all():
        raise ValueError('Observed expression must be finite')
    measured_weight = mask @ w
    if np.any(measured_weight <= 0):
        raise ValueError('No directional gene measured; refuse module score')
    raw = (np.where(mask, z, 0) * (w * d)).sum(axis=1) / measured_weight
    coverage = measured_weight / w.sum()
    return dict(raw=raw, coverage=coverage, score=coverage * np.tanh(raw), confidence=float(w.mean()))
