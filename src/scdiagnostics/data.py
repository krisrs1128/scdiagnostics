import numpy as np
import pandas as pd


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
    if not isinstance(X, np.ndarray):
        X = X.todense()
    return X


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
