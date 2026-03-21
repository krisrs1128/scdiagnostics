import scanpy as sc
import numpy as np
import pandas as pd
import altair as alt
from .data import check_sparse, concat_real_sim


def plot_umap(
    adata,
    color=None,
    shape=None,
    facet=None,
    opacity=0.6,
    n_comps=20,
    n_neighbors=15,
    transform=lambda x: np.log1p(x),
    **kwargs,
):
    mapping = {"x": "UMAP1", "y": "UMAP2", "color": color, "shape": shape}
    mapping = {k: v for k, v in mapping.items() if v is not None}

    adata_ = adata.copy()
    adata_.X = check_sparse(adata_.X)
    Z = transform(adata_.X)
    if Z.shape[1] == adata_.X.shape[1]:
        adata_.X = transform(adata_.X)
    else:
        adata_ = adata_[:, : Z.shape[1]]
        adata_.X = Z
        adata_.var_names = [f"transform_{k}" for k in range(Z.shape[1])]

    # umap on the top PCA dimensions
    sc.pp.pca(adata_, n_comps=n_comps)
    sc.pp.neighbors(adata_, n_neighbors=n_neighbors, n_pcs=n_comps)
    sc.tl.umap(adata_, **kwargs)

    # get umap embeddings
    umap_df = pd.DataFrame(adata_.obsm["X_umap"], columns=["UMAP1", "UMAP2"])
    umap_df = pd.concat([umap_df, adata_.obs.reset_index(drop=True)], axis=1)

    # encode and visualize
    chart = alt.Chart(umap_df).mark_point(opacity=opacity).encode(**mapping)
    if facet is not None:
        chart = chart.facet(column=alt.Facet(facet))
    return chart


def plot_pca(
    adata,
    color=None,
    shape=None,
    facet=None,
    opacity=0.6,
    plot_dims=[0, 1],
    transform=lambda x: np.log1p(x),
    **kwargs,
):
    mapping = {"x": "PCA1", "y": "PCA2", "color": color, "shape": shape}
    mapping = {k: v for k, v in mapping.items() if v is not None}

    adata_ = adata.copy()
    adata_.X = check_sparse(adata_.X)
    adata_.X = transform(adata_.X)

    # get PCA scores
    sc.pp.pca(adata_, **kwargs)
    pca_df = pd.DataFrame(adata_.obsm["X_pca"][:, plot_dims], columns=["PCA1", "PCA2"])
    pca_df = pd.concat([pca_df, adata_.obs.reset_index(drop=True)], axis=1)

    # plot
    chart = alt.Chart(pca_df).mark_point(opacity=opacity).encode(**mapping)
    if facet is not None:
        chart = chart.facet(column=alt.Facet(facet))
    return chart


def compare_umap(real, simulated, transform=lambda x: x, **kwargs):
    adata = concat_real_sim(real, simulated)
    return plot_umap(adata, facet="source", transform=transform, **kwargs)


def compare_pca(real, simulated, transform=lambda x: x, **kwargs):
    adata = concat_real_sim(real, simulated)
    return plot_pca(adata, facet="source", transform=transform, **kwargs)
