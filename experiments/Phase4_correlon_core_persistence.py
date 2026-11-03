import math
import numpy as np
import pandas as pd

SCENARIOS = [
    'null', 'rank1_persistent', 'rank2_degenerate',
    'rank1_switching', 'transient_burst', 'common_driver'
]


def ar1_noise(T, d, rho, rng):
    x = np.zeros((T, d)); e = rng.normal(size=(T, d))
    for t in range(1, T): x[t] = rho*x[t-1] + e[t]
    x /= np.std(x, axis=0, keepdims=True) + 1e-12
    return x


def ar1_scalar(T, rho, rng):
    z = np.zeros(T); e = rng.normal(size=T)
    for t in range(1, T): z[t] = rho*z[t-1] + e[t]
    z /= np.std(z) + 1e-12
    return z


def orthovec(d, k, rng):
    q, _ = np.linalg.qr(rng.normal(size=(d, k)))
    return q[:, :k]


def make_data(sc, seed, T=900, d=8, q=8, strength=1.2):
    rng = np.random.default_rng(seed)
    X = .8*ar1_noise(T, d, .7, rng)
    Y = .8*ar1_noise(T, q, .7, rng)
    ux = orthovec(d, 2, rng); vy = orthovec(q, 2, rng)
    if sc == 'null': return X, Y
    if sc in ('rank1_persistent', 'common_driver'):
        z = ar1_scalar(T, .92, rng)
        X += strength*z[:, None]*ux[:, 0]
        Y += strength*z[:, None]*vy[:, 0]
    elif sc == 'rank2_degenerate':
        z1 = ar1_scalar(T, .92, rng); z2 = ar1_scalar(T, .92, rng)
        X += strength*(z1[:, None]*ux[:, 0] + z2[:, None]*ux[:, 1])
        Y += strength*(z1[:, None]*vy[:, 0] + z2[:, None]*vy[:, 1])
    elif sc == 'rank1_switching':
        z = ar1_scalar(T, .92, rng); m = T//2
        X[:m] += strength*z[:m, None]*ux[:, 0]; Y[:m] += strength*z[:m, None]*vy[:, 0]
        X[m:] += strength*z[m:, None]*ux[:, 1]; Y[m:] += strength*z[m:, None]*vy[:, 1]
    elif sc == 'transient_burst':
        z = ar1_scalar(T, .92, rng); a, b = T//3, 2*T//3
        X[a:b] += strength*z[a:b, None]*ux[:, 0]
        Y[a:b] += strength*z[a:b, None]*vy[:, 0]
    return X, Y


def invsqrt(C, eps=1e-5):
    w, V = np.linalg.eigh(C)
    return (V*(1/np.sqrt(np.maximum(w, 0)+eps))) @ V.T


def modes(X, Y, win=140, step=40):
    out = []
    for st in range(0, len(X)-win+1, step):
        xx = X[st:st+win] - X[st:st+win].mean(0)
        yy = Y[st:st+win] - Y[st:st+win].mean(0)
        Cxx = xx.T@xx/(win-1); Cyy = yy.T@yy/(win-1); Cxy = xx.T@yy/(win-1)
        Q = invsqrt(Cxx) @ Cxy @ invsqrt(Cyy)
        U, S, Vt = np.linalg.svd(Q, full_matrices=False)
        out.append((S, U[:, 0], Vt.T[:, 0]))
    return out


def metrics(ms, max_lag=3):
    s1 = np.array([m[0][0] for m in ms]); s2 = np.array([m[0][1] for m in ms])
    lag_p = []
    for lag in range(1, min(max_lag, len(ms)-1)+1):
        gx = np.mean([np.dot(ms[i][1], ms[i+lag][1])**2 for i in range(len(ms)-lag)])
        gy = np.mean([np.dot(ms[i][2], ms[i+lag][2])**2 for i in range(len(ms)-lag)])
        lag_p.append(math.sqrt(gx*gy))
    joint = np.array([
        math.sqrt((np.dot(ms[i][1], ms[i+1][1])**2)*(np.dot(ms[i][2], ms[i+1][2])**2))
        for i in range(len(ms)-1)
    ])
    return {
        'strength': float(np.median(s1)),
        'gap': float(np.median(s1-s2)),
        'rel_gap': float(np.median((s1-s2)/s1)),
        'persistence_mean': float(np.mean(lag_p)),
        'persistence_q10': float(np.quantile(joint, .10)),
        'persistence_min': float(np.min(joint)),
    }


def null_medians(X, Y, seed, nnull=6):
    rng = np.random.default_rng(seed+7777); T = len(Y); vals = []
    for sh in rng.integers(T//4, 3*T//4, size=nnull):
        vals.append(metrics(modes(X, np.roll(Y, int(sh), axis=0))))
    return {k: float(np.median([v[k] for v in vals])) for k in vals[0]}


def ci95(x):
    x = np.asarray(x); se = np.std(x, ddof=1)/np.sqrt(len(x))
    return float(np.mean(x)-1.96*se), float(np.mean(x)+1.96*se)


def run(nseeds=100):
    rows = []
    for si, sc in enumerate(SCENARIOS):
        for seed in range(nseeds):
            sid = seed + si*10000
            X, Y = make_data(sc, sid)
            obs = metrics(modes(X, Y)); nul = null_medians(X, Y, sid)
            rows.append({
                'scenario': sc, 'seed': seed,
                **{f'obs_{k}': v for k, v in obs.items()},
                **{f'null_{k}': v for k, v in nul.items()},
                'T_iso': obs['gap']-nul['gap'],
                'T_pers': obs['persistence_mean']-nul['persistence_mean'],
                'T_floor': obs['persistence_q10']-nul['persistence_q10'],
            })
    raw = pd.DataFrame(rows)
    out = []
    for sc, g in raw.groupby('scenario'):
        il, ih = ci95(g.T_iso); pl, ph = ci95(g.T_pers); fl, fh = ci95(g.T_floor)
        out.append({
            'scenario': sc,
            'T_iso_mean': g.T_iso.mean(), 'T_iso_CI95_lo': il, 'T_iso_CI95_hi': ih,
            'T_pers_mean': g.T_pers.mean(), 'T_pers_CI95_lo': pl, 'T_pers_CI95_hi': ph,
            'T_floor_mean': g.T_floor.mean(), 'T_floor_CI95_lo': fl, 'T_floor_CI95_hi': fh,
            'obs_rel_gap_mean': g.obs_rel_gap.mean(),
            'obs_persistence_q10_mean': g.obs_persistence_q10.mean(),
            'obs_persistence_min_mean': g.obs_persistence_min.mean(),
        })
    summary = pd.DataFrame(out)
    raw.to_csv('results/Phase4_correlon_core_100seeds_raw.csv', index=False)
    summary.to_csv('results/Phase4_correlon_core_summary.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    run()
