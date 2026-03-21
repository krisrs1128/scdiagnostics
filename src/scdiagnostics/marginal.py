import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from .data import prepare_dense, merge_samples


def compare_summary(real, simulated, summary_fun, labels=None):
    df = pd.DataFrame({"real": summary_fun(real), "simulated": summary_fun(simulated)})

    identity = pd.DataFrame(
        {
            "real": [df["real"].min(), df["real"].max()],
            "simulated": [df["real"].min(), df["real"].max()],
        }
    )
    chart = alt.Chart(identity).mark_line(color="#dedede").encode(
        x="real", y="simulated"
    ) + alt.Chart(df).mark_circle().encode(x="real", y="simulated")

    if labels is not None:
        df["label"] = labels
        chart = chart + alt.Chart(df[df["label"] != ""]).mark_text(
            dx=6, dy=-6, align="left"
        ).encode(x="real:Q", y="simulated:Q", text="label:N")

    return chart


def compare_means(real, simulated, transform=lambda x: x, labels=None):
    real_, simulated_ = prepare_dense(real, simulated)
    summary = lambda a: np.asarray(transform(a.X).mean(axis=0)).flatten()
    return compare_summary(real_, simulated_, summary, labels)


def compare_variances(real, simulated, transform=lambda x: x, labels=None):
    real_, simulated_ = prepare_dense(real, simulated)
    summary = lambda a: np.asarray(np.var(transform(a.X), axis=0)).flatten()
    return compare_summary(real_, simulated_, summary, labels)


def compare_standard_deviation(real, simulated, transform=lambda x: x, labels=None):
    real_, simulated_ = prepare_dense(real, simulated)
    summary = lambda a: np.asarray(np.std(transform(a.X), axis=0)).flatten()
    return compare_summary(real_, simulated_, summary, labels)


def compare_histogram2(sim_data, real_data, idx):
    sim = sim_data[:, idx]
    real = real_data[:, idx]
    b = np.linspace(min(min(sim), min(real)), max(max(sim), max(real)), 50)

    plt.hist([real, sim], b, label=["Real", "Simulated"], histtype="bar")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


def compare_ecdf(adata, sim, var_names=None, max_plot=10, n_cols=5, transform=lambda x: x, **kwargs):
    if var_names is None:
        var_names = adata.var_names[:max_plot]

    combined = merge_samples(adata[:, var_names], sim[:, var_names])
    combined["value"] = transform(combined["value"])
    alt.data_transformers.enable("vegafusion")

    plot = (
        alt.Chart(combined)
        .transform_window(
            ecdf="cume_dist()", sort=[{"field": "value"}], groupby=["variable", "source"]
        )
        .mark_line(
            interpolate="step-after",
        )
        .encode(
            x="value:Q",
            y="ecdf:Q",
            color="source:N",
            facet=alt.Facet(
                "variable", sort=alt.EncodingSortField("value"), columns=n_cols
            ),
        )
        .properties(**kwargs)
    )
    plot.show()
    return plot, combined


def compare_boxplot(adata, sim, var_names=None, max_plot=20, transform=lambda x: x, **kwargs):
    if var_names is None:
        var_names = adata.var_names[:max_plot]

    combined = merge_samples(adata[:, var_names], sim[:, var_names])
    combined["value"] = transform(combined["value"])
    alt.data_transformers.enable("vegafusion")

    plot = (
        alt.Chart(combined)
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("value:Q").scale(zero=False),
            y=alt.Y(
                "variable:N",
                sort=alt.EncodingSortField("mid_box_value", order="descending"),
            ),
            facet="source:N",
        )
        .properties(**kwargs)
    )
    plot.show()
    return plot, combined


def compare_histogram(adata, sim, var_names=None, max_plot=20, transform=lambda x: x, **kwargs):
    if var_names is None:
        var_names = adata.var_names[:max_plot]

    combined = merge_samples(adata[:, var_names], sim[:, var_names])
    combined["value"] = transform(combined["value"])
    alt.data_transformers.enable("vegafusion")

    plot = (
        alt.Chart(combined)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X("value:Q").bin(maxbins=20),
            y=alt.Y("count()").stack(None),
            color="source:N",
            facet=alt.Facet(
                "variable", sort=alt.EncodingSortField("bin_maxbins_20_value")
            ),
        )
        .properties(**kwargs)
    )
    plot.show()
    return plot, combined


def compare_moments(real, simulated, log_scale=True, labels=None, label_threshold=None):
    """Compare real vs. simulated means and standard deviations gene-wise.

    Parameters
    ----------
    log_scale : bool
        If True, log-transform the gene-level statistics before plotting
        (useful when counts span several orders of magnitude).
    labels : array-like of str, optional
        Gene labels aligned with the variables. Non-empty strings are
        rendered as text marks next to the corresponding point.
    label_threshold : float, optional
        If provided, genes where the absolute difference between real and
        simulated exceeds this value in either the mean or SD are labeled
        with their gene name in both panels.
    """
    real_, simulated_ = prepare_dense(real, simulated)
    transform = np.log if log_scale else lambda x: x

    def gene_summary(stat_fn):
        def summary(a):
            return transform(np.asarray(stat_fn(a.X)).flatten())
        return summary

    means_summary = gene_summary(lambda X: X.mean(axis=0))
    sd_summary = gene_summary(lambda X: np.std(X, axis=0))

    if label_threshold is not None:
        means_real = means_summary(real_)
        means_sim = means_summary(simulated_)
        sd_real = sd_summary(real_)
        sd_sim = sd_summary(simulated_)
        exceeds = (np.abs(means_real - means_sim) > label_threshold) | (np.abs(sd_real - sd_sim) > label_threshold)
        labels = np.where(exceeds, real_.var_names, "")

    return (
        compare_summary(real_, simulated_, means_summary, labels).properties(title="Means: Real vs. Simulated")
        | compare_summary(real_, simulated_, sd_summary, labels).properties(title="Std. Dev.: Real vs. Simulated")
    )