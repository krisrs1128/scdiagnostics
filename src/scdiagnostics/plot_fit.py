"""Diagnostic plots for assessing fitted single-cell generative models.

The plotting functions in this module compare observed expression with predictions 
or samples from a fitted model.
"""

import altair as alt
import numpy as np
import pandas as pd
import scipy.sparse as sp

alt.data_transformers.enable("vegafusion")
alt.renderers.enable("png")


def plot_gene_correlations(sim, num_genes=5):
    """Plot pairwise expression relationships in a simulated sample.

    Expression is transformed with ``log1p`` before plotting. The first
    ``num_genes`` genes in the sampled data are shown as a scatterplot matrix.
    """
    sampled = sim.sample()
    expression = (
        sampled.X.toarray() if sp.issparse(sampled.X) else np.asarray(sampled.X)
    )
    expression = pd.DataFrame(
        np.log1p(expression),
        columns=sampled.var_names,
        index=sampled.obs.index,
    )

    sampled_subset = pd.concat([expression.iloc[:, :num_genes], sampled.obs], axis=1)
    genes = list(sampled.var_names[:num_genes])
    return (
        alt.Chart(sampled_subset)
        .mark_circle(opacity=1, size=1)
        .encode(
            alt.X(alt.repeat("column"), type="quantitative"),
            alt.Y(alt.repeat("row"), type="quantitative"),
        )
        .properties(width=40, height=40)
        .repeat(row=genes, column=genes)
        .configure_axis(labelFontSize=9, titleFontSize=9)
    )


_DISTRIBUTIONS = {"negbin", "poisson", "bernoulli", "gaussian"}


def _validate_distribution(distribution):
    distribution = distribution.lower()
    if distribution not in _DISTRIBUTIONS:
        options = ", ".join(sorted(_DISTRIBUTIONS))
        raise ValueError(f"distribution must be one of: {options}")
    return distribution


def _covariate_info(sce, covariate):
    if covariate not in sce.obs.columns:
        raise KeyError(f"{covariate!r} is not a column in sce.obs")

    values = sce.obs[covariate]
    if values.isna().any():
        raise ValueError(f"sce.obs[{covariate!r}] contains missing values")
    numeric = pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(
        values
    )
    if numeric:
        return values, True, None

    values = values.astype(str)
    return values, False, list(dict.fromkeys(values))


def _standard_deviation(predictions, gene_ix, distribution):
    """Return the fitted standard deviation."""
    mean = np.asarray(predictions["mean"][:, gene_ix])
    if distribution == "negbin":
        dispersion = np.asarray(predictions["dispersion"][:, gene_ix])
        return np.sqrt(mean + mean**2 / dispersion)
    if distribution == "poisson":
        return np.sqrt(mean)
    if distribution == "gaussian":
        return np.asarray(predictions["sdev"][:, gene_ix])
    return np.sqrt(np.clip(mean * (1 - mean), 0, None))


def _numeric_fit_chart(observed, mean, sdev, covariate, distribution):
    x = alt.X(field=covariate, type="quantitative", title=covariate)
    y = alt.Y("expression:Q", title="Expression")
    points = alt.Chart(observed).mark_circle(size=20, opacity=0.35).encode(x=x, y=y)
    fitted = pd.DataFrame({covariate: observed[covariate], "mean": mean})

    if distribution == "bernoulli":
        fitted = fitted.groupby(covariate, as_index=False, sort=True).mean()
        return points + alt.Chart(fitted).mark_circle(size=20, color="red").encode(
            x=x, y=alt.Y("mean:Q", title="Expression")
        )

    fitted["second_moment"] = sdev**2 + fitted["mean"] ** 2
    fitted = fitted.groupby(covariate, as_index=False, sort=True).mean()
    fitted["sdev"] = np.sqrt(
        np.clip(fitted["second_moment"] - fitted["mean"] ** 2, 0, None)
    )
    fitted["lower"] = fitted["mean"] - fitted["sdev"]
    fitted["upper"] = fitted["mean"] + fitted["sdev"]
    band = alt.Chart(fitted).mark_area(opacity=0.3, color="orange").encode(
        x=x, y=alt.Y("lower:Q", title="Expression"), y2="upper:Q"
    )
    line = alt.Chart(fitted).mark_line(color="red").encode(
        x=x, y=alt.Y("mean:Q", title="Expression")
    )
    return band + line + points


def _categorical_fit_chart(observed, mean, sdev, covariate, category_order):
    observed = observed.copy()
    observed["jitter"] = np.random.default_rng(0).uniform(-0.5, 0.5, len(observed))
    fitted = pd.DataFrame(
        {covariate: observed[covariate], "mean": mean, "sdev": sdev}
    )
    fitted["second_moment"] = fitted["sdev"] ** 2 + fitted["mean"] ** 2
    fitted = fitted.groupby(covariate, sort=False, observed=True, as_index=False).mean()
    fitted["sdev"] = np.sqrt(
        np.clip(fitted["second_moment"] - fitted["mean"] ** 2, 0, None)
    )
    fitted["lower"] = fitted["mean"] - fitted["sdev"]
    fitted["upper"] = fitted["mean"] + fitted["sdev"]
    fitted["jitter"] = 0.0
    x = alt.X(
        field=covariate, type="nominal", title=covariate, sort=category_order,
    )
    jitter = alt.XOffset(
        "jitter:Q", scale=alt.Scale(domain=[-0.5, 0.5])
    )

    error_bars = alt.Chart(fitted).mark_rule(color="red").encode(
        x=x,
        xOffset=jitter,
        y=alt.Y("lower:Q", title="Expression"),
        y2="upper:Q",
    )
    lower_caps = alt.Chart(fitted).mark_tick(
        color="red", orient="horizontal", size=24
    ).encode(x=x, xOffset=jitter, y="lower:Q")
    upper_caps = alt.Chart(fitted).mark_tick(
        color="red", orient="horizontal", size=24
    ).encode(x=x, xOffset=jitter, y="upper:Q")
    fitted_means = alt.Chart(fitted).mark_point(color="red", filled=True, size=55).encode(
        x=x, xOffset=jitter, y=alt.Y("mean:Q", title="Expression")
    )
    points = alt.Chart(observed).mark_circle(size=24, opacity=0.35).encode(
        x=x,
        xOffset=jitter,
        y=alt.Y("expression:Q", title="Expression"),
    )
    return points + error_bars + lower_caps + upper_caps + fitted_means


def plot_fit_by_covariate(
    sim,
    sce,
    distribution="negbin",
    covariate="pseudotime",
    num_genes=5,
):
    """Plot fitted gene expression against a numeric or categorical covariate.

    Numeric covariates show observations and fitted values; negative-binomial,
    Poisson, and Gaussian fits also show a one-standard-deviation band.
    Categorical covariates show jittered observations with the model's
    category-level mean and one-standard-deviation error bars.

    Parameters
    ----------
    distribution : {"negbin", "poisson", "bernoulli", "gaussian"}
        Distribution used by the fitted model.
    covariate : str
        Column in ``sce.obs``. Its dtype determines which plot is produced.
    """
    distribution = _validate_distribution(distribution)
    values, numeric, category_order = _covariate_info(sce, covariate)

    expression = sce.X.toarray() if sp.issparse(sce.X) else np.asarray(sce.X)
    predictions = sim.predict(sce.obs)
    charts = []
    for ix in range(min(num_genes, sce.n_vars)):
        mean = np.asarray(predictions["mean"][:, ix])
        observed = pd.DataFrame(
            {covariate: values.to_numpy(), "expression": expression[:, ix]}
        )
        sdev = _standard_deviation(predictions, ix, distribution)
        if numeric:
            chart = _numeric_fit_chart(
                observed, mean, sdev, covariate, distribution
            )
        else:
            chart = _categorical_fit_chart(
                observed, mean, sdev, covariate, category_order
            )
        charts.append(
            chart.properties(
                title=f"Gene {ix}: {sce.var_names[ix]}", width=300, height=200
            )
        )

    return alt.vconcat(*charts)


def plot_sampled_expression_by_covariate(
    sim,
    sce,
    covariate="pseudotime",
    num_genes=5,
    jitter=False,
):
    """Compare real and simulated expression by a numeric or categorical covariate.

    A sample is drawn from the fitted model at ``sce.obs``. Numeric covariates
    use an overlaid scatter plot. Categorical covariates use horizontally
    jittered points within each category. Color distinguishes real and simulated
    expression.

    Parameters
    ----------
    covariate : str
        Column in ``sce.obs``. Its dtype determines which plot is produced.
    jitter : bool
        Add small vertical jitter to expression values. Horizontal jitter is
        always applied for categorical covariates.
    """
    values, numeric, category_order = _covariate_info(sce, covariate)

    real_X = sce.X.toarray() if sp.issparse(sce.X) else np.asarray(sce.X)
    sampled = sim.sample(sce.obs)
    sim_X = sampled.X.toarray() if sp.issparse(sampled.X) else np.asarray(sampled.X)
    if real_X.shape != sim_X.shape:
        raise ValueError("real and simulated expression matrices must have equal shapes")

    rng = np.random.default_rng(0)
    charts = []
    for ix in range(min(num_genes, sce.n_vars)):
        real_values = real_X[:, ix]
        simulated_values = sim_X[:, ix]
        if jitter:
            real_values = real_values + rng.normal(0, 0.1, len(values))
            simulated_values = simulated_values + rng.normal(0, 0.1, len(values))

        combined = pd.DataFrame(
            {
                covariate: np.tile(values.to_numpy(), 2),
                "expression": np.concatenate([real_values, simulated_values]),
                "type": np.repeat(["real", "simulated"], len(values)),
            }
        )
        x = alt.X(
            field=covariate,
            type="quantitative" if numeric else "nominal",
            title=covariate,
            sort=None if numeric else category_order,
        )
        if not numeric:
            combined["x_jitter"] = rng.uniform(-0.5, 0.5, len(combined))

        encoding = {
            "x": x,
            "y": alt.Y("expression:Q", title="Expression"),
            "color": alt.Color("type:N", title=None),
        }
        if not numeric:
            encoding["xOffset"] = alt.XOffset(
                "x_jitter:Q", scale=alt.Scale(domain=[-0.5, 0.5])
            )
        charts.append(
            alt.Chart(combined)
            .mark_circle(size=24, opacity=0.35)
            .encode(**encoding)
            .properties(
                title=f"Gene {ix}: {sce.var_names[ix]}", width=300, height=200
            )
        )

    return alt.vconcat(*charts)
