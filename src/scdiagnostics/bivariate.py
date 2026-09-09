import numpy as np
import pandas as pd
import altair as alt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from .data import prepare_dense, merge_wide_samples


def compare_gene_pairs(adata, sim, var_names=None, max_plot=10,
                       transform=np.log1p, width=100, height=100, **kwargs):
    """Pairs scatterplot comparing gene-gene relationships

    These pairs of scatterplots are useful for checking whether the copula has
    learned the right bivariate relationships. Defaults to showing just 10 genes
    at a time.
    """
    if var_names is None:
        var_names = adata.var_names[:max_plot]
    var_names = list(var_names)

    combined = merge_wide_samples(adata[:, var_names], sim[:, var_names])
    combined[var_names] = transform(combined[var_names])
    alt.data_transformers.enable("vegafusion")

    base = (
        alt.Chart(combined)
        .mark_circle(opacity=0.5, size=15)
        .encode(
            x=alt.X(alt.repeat("column"), type="quantitative"),
            y=alt.Y(alt.repeat("row"), type="quantitative"),
            color=alt.Color("source:N"),
        )
        .properties(width=width, height=height)
    )

    plot = base.repeat(row=var_names, column=var_names).properties(**kwargs)
    plot.show()
    return plot, combined


def _melt_corr(corr_df, value_name="correlation"):
    return corr_df.reset_index(names="gene_y").melt(
        id_vars="gene_y", var_name="gene_x", value_name=value_name
    )


def compare_correlation(adata, sim, var_names=None, max_plot=20,
                        transform=np.log1p, method="average", width=250,
                        height=250, **kwargs):
    """Real vs. simulated gene-gene correlation heatmap

    We sort the genes using average linkage hierarchical clustering on the real
    correlation matrix.
    """
    if var_names is None:
        var_names = adata.var_names[:max_plot]
    var_names = list(var_names)

    real_, sim_ = prepare_dense(adata[:, var_names], sim[:, var_names])
    real_corr = pd.DataFrame(transform(real_.X), columns=var_names).corr()
    sim_corr = pd.DataFrame(transform(sim_.X), columns=var_names).corr()

    if len(var_names) < 2:
        order = var_names
    else:
        distance = squareform(1 - real_corr.to_numpy(), checks=False)
        order = [var_names[i] for i in leaves_list(linkage(distance, method=method))]

    real_ordered = real_corr.loc[order, order]
    sim_ordered = sim_corr.loc[order, order]

    scale = alt.Scale(domain=[-1, 1], scheme="redblue")

    def heatmap(df, title):
        return (
            alt.Chart(_melt_corr(df))
            .mark_rect()
            .encode(
                x=alt.X("gene_x:N", sort=order, title=None),
                y=alt.Y("gene_y:N", sort=order, title=None),
                color=alt.Color("correlation:Q", scale=scale, title="Pearson r"),
            )
            .properties(width=width, height=height, title=title)
        )

    plot = (heatmap(real_ordered, "Real") | heatmap(sim_ordered, "Simulated")).properties(**kwargs)
    plot.show()
    return plot, {"real": real_ordered, "simulated": sim_ordered}
