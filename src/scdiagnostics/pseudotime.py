"""Plot observed expression and model fitted mean/sd along one-dimensional pseudotime."""

import altair as alt
import pandas as pd


def plot_fitted_curve(
    observed,
    fitted,
    *,
    x,
    y,
    mean="mean",
    group=None,
    model=None,
    lower=None,
    upper=None,
    n_bins=10,
    transform=None,
    y_title=None,
    width=300,
    height=220,
    columns=2,
):
    """Plot observations and model fits against one-dimensional time or pseudotime.

    Parameters
    ----------
    observed, fitted : pandas.DataFrame
        Two pandas DFs. Observed should have columns `x`, `y`, and optionally a
        `group`.  Fitted should have `x`, `mean`, optionally `group`, `model`,
        and the intervals.  Each `model` will get a curve.
    x, y, mean : str
        Column names for the feature to plot against, the response, and the
        fitted mean.
    group, model : str, optional
        Column names to facet by (group) and to use for separate lines (model).
    lower, upper : str, optional
        Column names for the intervals.
    n_bins : int or None
        Equal-width bins used for averaging. Defaults to None.
    transform : callable, optional
        Transform applied after averaging within bins.  E.g., np.log1p plots
        log(1 + E[Y|x]).
    y_title : str, optional
        Axis label.
    columns : int, default 2
        Number of facet columns.

    Returns
    -------
    altair.LayerChart or altair.FacetChart
        Observations, fitted curves, and optional intervals and binned means.
    """
    obs, fit = _prepare_tables(
        observed, fitted, x=x, y=y, mean=mean, group=group, model=model,
        lower=lower, upper=upper,
    )
    tables = [obs, fit]
    if n_bins is not None:
        tables.append(_bin_observations(obs, n_bins))
    data = pd.concat(tables, ignore_index=True)
    if transform is not None:
        data = _transform_response(data, transform)

    chart = _curve_layers(
        data, x_title=x, y_title=y_title or y, model_title=model or "Model",
        show_intervals=lower is not None, show_bins=n_bins is not None,
    ).properties(width=width, height=height)
    if group is not None:
        chart = chart.facet(facet=alt.Facet("_group:N", title=group), columns=columns)
        chart = chart.resolve_scale(y="independent")
    return chart


def _prepare_tables(observed, fitted, *, x, y, mean, group, model, lower, upper):
    """Copy input columns into the shared internal schema used by plot layers."""
    obs = pd.DataFrame({
        "_x": observed[x].to_numpy(),
        "_y": observed[y].to_numpy(),
        "_group": observed[group].to_numpy() if group is not None else "All",
        "_kind": "observed",
    })
    fit = pd.DataFrame({
        "_x": fitted[x].to_numpy(),
        "_y": fitted[mean].to_numpy(),
        "_group": fitted[group].to_numpy() if group is not None else "All",
        "_model": fitted[model].to_numpy() if model is not None else "Fitted mean",
        "_kind": "fitted",
    })
    if lower is not None:
        fit["_lower"] = fitted[lower].to_numpy()
        fit["_upper"] = fitted[upper].to_numpy()
    return obs, fit


def _bin_observations(observed, n_bins):
    """Average time and response in equal-width bins within each group."""
    tables = []
    for label, part in observed.groupby("_group", sort=False, observed=True):
        time_bins = pd.cut(part["_x"], n_bins)
        means = part.groupby(time_bins, observed=True)[["_x", "_y"]].mean()
        tables.append(means.assign(_group=label, _kind="binned"))
    if not tables:
        return observed.iloc[:0].copy()
    return pd.concat(tables, ignore_index=True)


def _transform_response(data, transform):
    """Transform response columns, leaving absent layer-specific values alone."""
    data = data.copy()
    for column in ("_y", "_lower", "_upper"):
        data[column] = data[column].astype(float)
        valid = data[column].notna()
        data.loc[valid, column] = transform(data.loc[valid, column].to_numpy())
    return data


def _curve_layers(data, *, x_title, y_title, model_title, show_intervals, show_bins):
    """Compose layers back-to-front using one shared dataset for faceting."""
    base = alt.Chart(data).encode(
        x=alt.X("_x:Q", title=x_title, scale=alt.Scale(zero=False)),
        y=alt.Y("_y:Q", title=y_title, scale=alt.Scale(zero=False)),
    )
    fitted = base.transform_filter(alt.datum._kind == "fitted")
    color = alt.Color("_model:N", title=model_title)
    observations = base.transform_filter(alt.datum._kind == "observed").mark_circle(
        color="black", opacity=0.25, size=15,
    )
    curves = fitted.mark_line().encode(color=color, order="_x:Q")

    layers = []
    if show_intervals:
        bands = fitted.mark_area(opacity=0.12).encode(
            y=alt.Y("_lower:Q", title=y_title, stack=None, scale=alt.Scale(zero=False)),
            y2="_upper:Q",
            color=color,
        )
        layers.append(bands)
    layers.extend([observations, curves])
    if show_bins:
        binned_means = (
            base.transform_filter(alt.datum._kind == "binned")
            .mark_point(shape="diamond", filled=True, color="#6E9CB6", size=55)
            .encode(tooltip=[
                alt.Tooltip("_x:Q", title=x_title),
                alt.Tooltip("_y:Q", title="Observed bin mean"),
            ])
        )
        layers.append(binned_means)
    return alt.layer(*layers)
