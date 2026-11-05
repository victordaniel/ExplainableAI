"""Vectorised permutation-sampling estimator of interventional Shapley values
(the quantity estimated by shap.PermutationExplainer with an Independent masker).
Each permutation is paired with its reverse (antithetic sampling)."""
import numpy as np


def shapley(f, X, bg, n_perm=4, seed=0, chunk=20):
    """f: rows -> scores; X: (n, d) points to explain; bg: (b, d) background.
    Returns (n, d) Shapley values of f relative to the background mean."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    b = len(bg)
    assert n_perm % 2 == 0
    out = np.zeros((n, d))
    for s in range(0, n, chunk):
        Xs = X[s:s + chunk]
        m = len(Xs)
        perms = []
        for _ in range(m):
            ps = [rng.permutation(d) for _ in range(n_perm // 2)]
            perms.append(ps + [p[::-1] for p in ps])
        perms = np.array(perms)                                  # (m, P, d)
        # Z[i, p, k] = background with the first k features of perm p set to x_i
        Z = np.broadcast_to(bg, (m, n_perm, d + 1, b, d)).copy()
        for k in range(1, d + 1):
            feat = perms[:, :, k - 1]                            # (m, P)
            # features already switched on stay on for all later steps
            for kk in range(k, d + 1):
                ii, pp = np.meshgrid(np.arange(m), np.arange(n_perm), indexing='ij')
                Z[ii, pp, kk, :, feat] = Xs[ii, feat][..., None]
        v = f(Z.reshape(-1, d)).reshape(m, n_perm, d + 1, b).mean(-1)   # (m, P, d+1)
        dv = np.diff(v, axis=2)                                  # marginal contribution at step k
        phi = np.zeros((m, n_perm, d))
        ii, pp = np.meshgrid(np.arange(m), np.arange(n_perm), indexing='ij')
        for k in range(d):
            phi[ii, pp, perms[:, :, k]] = dv[:, :, k]
        out[s:s + m] = phi.mean(1)
    return out
