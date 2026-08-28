import numpy as np
import pandas as pd
import scipy.sparse as sp


def adata_df(adata):
    return (
        pd.DataFrame(check_sparse(adata.X), columns=adata.var_names)
        .melt(id_vars=[], value_vars=adata.var_names)
        .reset_index(drop=True)
    )


def merge_samples(adata, sim):
    source = adata_df(adata)
    simulated = adata_df(sim)
    return pd.concat(
        {"real": source, "simulated": simulated}, names=["source"]
    ).reset_index(level="source")


def check_sparse(X):
    """Dense `ndarray` view of `X`, sparse or not.

    Always returns a plain `ndarray` rather than `.todense()`'s `np.matrix`,
    so downstream reductions like `.mean(axis=0)` come back 1-D instead of
    shaped `(1, n)`.
    """
    if sp.issparse(X):
        return np.asarray(X.todense())
    return np.asarray(X)


def pseudobulk(X, group_codes, n_groups):
    """Mean of dense `X` within each group, `(n_groups, n_features)`.

    Groups with no members return NaN rather than 0, so an absent group does
    not read as a silenced feature.
    """
    out = np.full((n_groups, X.shape[1]), np.nan)
    for g in range(n_groups):
        rows = group_codes == g
        if rows.any():
            out[g] = X[rows].mean(axis=0)
    return out


def prepare_dense(real, simulated):
    real_ = real.copy()
    simulated_ = simulated.copy()
    real_.X = check_sparse(real_.X)
    simulated_.X = check_sparse(simulated_.X)
    return real_, simulated_


def concat_real_sim(real, simulated):
    real_, simulated_ = prepare_dense(real, simulated)
    real_.obs["source"] = "real"
    simulated_.obs["source"] = "simulated"
    return real_.concatenate(simulated_, join="outer", batch_key=None)
