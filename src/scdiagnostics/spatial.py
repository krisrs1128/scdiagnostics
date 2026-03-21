from scipy.interpolate import RBFInterpolator
import scipy.sparse as sp
import altair as alt
import numpy as np
import pandas as pd


def make_grid_obs(adata, basis_cols, n_grid=30):
    s1 = adata.obs["spatial1"].values
    s2 = adata.obs["spatial2"].values
    g1 = np.linspace(s1.min(), s1.max(), n_grid)
    g2 = np.linspace(s2.min(), s2.max(), n_grid)
    G1, G2 = np.meshgrid(g1, g2, indexing="ij")

    grid_base = pd.DataFrame(
        {
            "spatial1": G1.ravel(),
            "spatial2": G2.ravel(),
            "bin1": pd.cut(G1.ravel(), bins=n_grid, labels=False, include_lowest=True),
            "bin2": pd.cut(G2.ravel(), bins=n_grid, labels=False, include_lowest=True),
        }
    )
    obs_bins = (
        pd.DataFrame(
            {
                "bin1": pd.cut(s1, bins=n_grid, labels=False, include_lowest=True),
                "bin2": pd.cut(s2, bins=n_grid, labels=False, include_lowest=True),
            }
        )
        .dropna()
        .astype(int)
        .drop_duplicates()
    )
    occupied = set(map(tuple, obs_bins[["bin1", "bin2"]].to_numpy()))
    keep = pd.Series(list(zip(grid_base["bin1"], grid_base["bin2"]))).isin(occupied)
    grid_base = grid_base.loc[keep, ["spatial1", "spatial2"]].reset_index(drop=True)

    cell_types = list(adata.obs["cell_type"].unique())
    n_pts = len(grid_base)
    grid_obs = pd.DataFrame(
        {
            "spatial1": np.tile(grid_base["spatial1"].values, len(cell_types)),
            "spatial2": np.tile(grid_base["spatial2"].values, len(cell_types)),
            "cell_type": np.repeat(cell_types, n_pts),
        }
    )

    train_coords = adata.obs[["spatial1", "spatial2"]].values
    grid_coords = grid_obs[["spatial1", "spatial2"]].values
    interp = RBFInterpolator(train_coords, adata.obs[basis_cols].values, neighbors=50)
    grid_basis = interp(grid_coords)
    for i, col in enumerate(basis_cols):
        grid_obs[col] = grid_basis[:, i]
    return grid_obs


def dense_gene_counts(adata, gene_idx):
    if sp.issparse(adata.X):
        return adata.X[:, gene_idx].toarray().ravel()
    return np.array(adata.X)[:, gene_idx]


def binned_obs_df(adata, gene_idx, n_obs_bins, value_col):
    s1 = adata.obs["spatial1"].values
    s2 = adata.obs["spatial2"].values
    return pd.DataFrame(
        {
            "spatial1": pd.cut(s1, bins=n_obs_bins)
            .map(lambda b: round(b.mid, 2))
            .astype(float),
            "spatial2": pd.cut(s2, bins=n_obs_bins)
            .map(lambda b: round(b.mid, 2))
            .astype(float),
            value_col: dense_gene_counts(adata, gene_idx),
        }
    )


def fitted_surface_df(sim, adata, basis_cols, gene_idx, pred_key, value_col, n_grid):
    grid_obs = make_grid_obs(adata, basis_cols, n_grid)
    n_ct = adata.obs["cell_type"].nunique()
    n_pts = len(grid_obs) // n_ct
    pred = np.log1p(sim.predict(obs=grid_obs)[pred_key][:, gene_idx])
    pred_avg = pred.reshape(n_ct, n_pts).mean(axis=0)
    ref = grid_obs.iloc[:n_pts]
    return pd.DataFrame(
        {
            "spatial1": ref["spatial1"].round(2).values,
            "spatial2": ref["spatial2"].round(2).values,
            value_col: pred_avg,
        }
    )


def surface_heatmaps(
    fitted_df, obs_df, value_col, color_title, fitted_title, obs_title
):
    vmax = max(fitted_df[value_col].max(), obs_df[value_col].max())
    scale = alt.Scale(domain=[0, vmax], scheme="viridis")

    def heatmap(df, title):
        return (
            alt.Chart(df)
            .mark_rect()
            .encode(
                x=alt.X("spatial1:O", title="spatial1"),
                y=alt.Y("spatial2:O", title="spatial2"),
                color=alt.Color(f"{value_col}:Q", scale=scale, title=color_title),
            )
            .properties(width=300, height=300, title=title)
        )

    return heatmap(fitted_df, fitted_title) | heatmap(obs_df, obs_title)


def plot_mean_surface(sim, adata, basis_cols, gene_idx, n_grid=30, n_obs_bins=30):
    fitted_df = fitted_surface_df(
        sim,
        adata,
        basis_cols,
        gene_idx,
        pred_key="mean",
        value_col="mu",
        n_grid=n_grid,
    )
    obs_df = (
        binned_obs_df(adata, gene_idx, n_obs_bins, value_col="mu")
        .groupby(["spatial1", "spatial2"], as_index=False)["mu"]
        .mean()
        .assign(mu=lambda d: np.log1p(d["mu"]))
    )
    plots = surface_heatmaps(
        fitted_df,
        obs_df,
        value_col="mu",
        color_title="log(mu+1)",
        fitted_title=f"Fitted mean(x) - Gene {gene_idx}",
        obs_title=f"Observed bin mean - Gene {gene_idx}",
    )
    return fitted_df, plots


def plot_dispersion_surface(sim, adata, basis_cols, gene_idx, n_grid=10, n_obs_bins=10):
    fitted_df = fitted_surface_df(
        sim,
        adata,
        basis_cols,
        gene_idx,
        pred_key="dispersion",
        value_col="dispersion",
        n_grid=n_grid,
    )
    obs_df = (
        binned_obs_df(adata, gene_idx, n_obs_bins, value_col="count")
        .groupby(["spatial1", "spatial2"], as_index=False)["count"]
        .agg(mu="mean", var="var")
        .assign(
            dispersion=lambda d: np.where(
                d["var"] > d["mu"], d["mu"] ** 2 / (d["var"] - d["mu"]), np.nan
            )
        )
        .assign(dispersion=lambda d: np.log1p(d["dispersion"]))
        .dropna(subset=["dispersion"])
    )
    plots = surface_heatmaps(
        fitted_df,
        obs_df,
        value_col="dispersion",
        color_title="log(dispersion+1)",
        fitted_title=f"Fitted dispersion(x) - Gene {gene_idx}",
        obs_title=f"Observed bin dispersion - Gene {gene_idx}",
    )
    return fitted_df, plots
