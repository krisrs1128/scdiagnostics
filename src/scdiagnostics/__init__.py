from .dimred import plot_umap, plot_pca, compare_pca, compare_umap, overlay_expression
from .marginal import (
    compare_boxplot,
    compare_ecdf,
    compare_histogram,
    compare_means,
    compare_moments,
    compare_standard_deviation,
    compare_variances,
    compare_histogram2,
)
from .spatial import plot_dispersion_surface, plot_mean_surface, plot_spatial
from .plot_fit import (
    plot_gene_correlations,
    plot_fit_by_covariate,
    plot_sampled_expression_by_covariate,
)
