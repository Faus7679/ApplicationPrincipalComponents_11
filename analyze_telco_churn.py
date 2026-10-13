"""Run correlation and principal-component analysis on the Telco churn data."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_PLOTS = ("tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen")
ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"


def load_dataset(path):
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Input must be a .csv, .xls, or .xlsx file.")


def _prepare_features(data):
    required = {TARGET_COLUMN, ID_COLUMN}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    predictors = data.drop(columns=[ID_COLUMN, TARGET_COLUMN]).copy()
    if "TotalCharges" in predictors.columns:
        charges = pd.to_numeric(predictors["TotalCharges"], errors="coerce")
        if "tenure" in predictors.columns:
            tenure = pd.to_numeric(predictors["tenure"], errors="coerce")
            charges = charges.mask(charges.isna() & tenure.eq(0), 0)
        predictors["TotalCharges"] = charges

    numeric_columns = predictors.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        column for column in predictors.columns if column not in numeric_columns
    ]
    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(
                                drop="first",
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                categorical_columns,
            )
        )

    transformer = ColumnTransformer(transformers, verbose_feature_names_out=False)
    matrix = transformer.fit_transform(predictors)
    names = transformer.get_feature_names_out()
    features = pd.DataFrame(matrix, columns=names, index=data.index)
    return predictors, features


def _plot_numeric_scatter(data, output_path, sample_size):
    columns = [column for column in NUMERIC_PLOTS if column in data.columns]
    if len(columns) < 2:
        return None

    values = data.loc[:, columns].copy()
    for column in columns:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    churn = data[TARGET_COLUMN].astype(str)
    if len(data) > sample_size:
        sample = data.sample(sample_size, random_state=11).index
        values = values.loc[sample]
        churn = churn.loc[sample]

    categories = sorted(churn.dropna().unique())
    palette = dict(zip(categories, ("#2878B5", "#E45756", "#72B7B2", "#F2CF5B")))
    fig, axes = plt.subplots(len(columns), len(columns), figsize=(11, 10))
    for row, y_column in enumerate(columns):
        for col, x_column in enumerate(columns):
            axis = axes[row, col]
            if row == col:
                axis.hist(values[x_column].dropna(), bins=24, color="#2878B5", alpha=0.8)
            else:
                for category in categories:
                    mask = churn.eq(category) & values[x_column].notna() & values[y_column].notna()
                    axis.scatter(
                        values.loc[mask, x_column],
                        values.loc[mask, y_column],
                        s=7,
                        alpha=0.28,
                        label=category,
                        color=palette[category],
                        linewidths=0,
                    )
            if row == len(columns) - 1:
                axis.set_xlabel(x_column)
            if col == 0:
                axis.set_ylabel(y_column)
            if row == 0 and col != 0:
                axis.set_title(x_column)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title=TARGET_COLUMN, loc="upper right")
    fig.suptitle("Numeric scatter plots, colored by churn", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return columns


def _plot_correlation(correlation, output_path):
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(correlation)), correlation.columns, rotation=35, ha="right")
    axis.set_yticks(range(len(correlation)), correlation.index)
    for row in range(len(correlation)):
        for col in range(len(correlation)):
            axis.text(col, row, f"{correlation.iloc[row, col]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=axis, label="Pearson correlation")
    axis.set_title("Correlation matrix of numeric variables")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_variance(pca, output_path, threshold):
    individual = pca.explained_variance_ratio_
    cumulative = np.cumsum(individual)
    count = int(np.searchsorted(cumulative, threshold) + 1)
    components = np.arange(1, len(individual) + 1)
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(components, individual, color="#2878B5", alpha=0.75, label="Individual")
    axis.plot(components, cumulative, color="#E45756", marker="o", markersize=3, label="Cumulative")
    axis.axhline(threshold, color="black", linestyle="--", label=f"{threshold:.0%} threshold")
    axis.axvline(count, color="gray", linestyle=":")
    axis.set(xlabel="Principal component", ylabel="Explained variance ratio", ylim=(0, 1.02))
    axis.set_title("PCA scree and cumulative variance")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return count


def _plot_scores(scores, churn, output_path, sample_size):
    if len(scores) > sample_size:
        sample = scores.sample(sample_size, random_state=11).index
        scores = scores.loc[sample]
        churn = churn.loc[sample]
    categories = sorted(churn.astype(str).dropna().unique())
    palette = dict(zip(categories, ("#2878B5", "#E45756", "#72B7B2", "#F2CF5B")))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, x_pc, y_pc in (
        (axes[0], "PC1", "PC2"),
        (axes[1], "PC1", "PC3"),
    ):
        for category in categories:
            mask = churn.astype(str).eq(category)
            axis.scatter(
                scores.loc[mask, x_pc],
                scores.loc[mask, y_pc],
                s=9,
                alpha=0.35,
                color=palette[category],
                label=category,
                linewidths=0,
            )
        axis.set_xlabel(f"{x_pc} ({scores.attrs[x_pc]:.1%} variance)")
        axis.set_ylabel(f"{y_pc} ({scores.attrs[y_pc]:.1%} variance)")
        axis.set_title(f"{x_pc} vs {y_pc}")
    axes[1].legend(title=TARGET_COLUMN)
    fig.suptitle("PCA score plots colored by churn")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_loadings(loadings, output_path):
    components = min(3, loadings.shape[1])
    fig, axes = plt.subplots(1, components, figsize=(6 * components, 7), squeeze=False)
    for index, axis in enumerate(axes[0]):
        component = loadings.iloc[:, index]
        selected = component.abs().nlargest(10).sort_values()
        axis.barh(selected.index, component.loc[selected.index], color="#2878B5")
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_title(f"PC{index + 1} top absolute loadings")
        axis.set_xlabel("Loading (sign is arbitrary)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def analyze_dataset(data, output_dir, variance_threshold=0.80, sample_size=3000):
    if not 0 < variance_threshold < 1:
        raise ValueError("variance_threshold must be between 0 and 1.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictors, features = _prepare_features(data)
    if features.shape[0] < 3 or features.shape[1] < 3:
        raise ValueError("PCA score plots require at least 3 rows and 3 encoded features.")

    numeric_columns = predictors.select_dtypes(include="number").columns.tolist()
    correlation = predictors[numeric_columns].corr()
    correlation.to_csv(output_dir / "numeric_correlations.csv")
    if len(correlation.columns) >= 2:
        _plot_correlation(correlation, output_dir / "numeric_correlations.png")
    _plot_numeric_scatter(data, output_dir / "numeric_scatter_matrix.png", sample_size)

    pca = PCA(svd_solver="full")
    transformed = pca.fit_transform(features)
    variance = pd.DataFrame(
        {
            "component": [f"PC{index}" for index in range(1, len(pca.explained_variance_ratio_) + 1)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    variance.to_csv(output_dir / "pca_explained_variance.csv", index=False)
    component_count = _plot_variance(
        pca, output_dir / "pca_scree_plot.png", variance_threshold
    )

    scores = pd.DataFrame(
        transformed[:, :3],
        columns=["PC1", "PC2", "PC3"],
        index=data.index,
    )
    for index, column in enumerate(scores.columns):
        scores.attrs[column] = pca.explained_variance_ratio_[index]
    _plot_scores(scores, data[TARGET_COLUMN], output_dir / "pca_score_plots.png", sample_size)

    loadings = pd.DataFrame(
        pca.components_[:component_count].T,
        index=features.columns,
        columns=[f"PC{index}" for index in range(1, component_count + 1)],
    )
    loadings.to_csv(output_dir / "pca_loadings.csv")
    _plot_loadings(loadings, output_dir / "pca_top_loadings.png")

    distances = (scores[["PC1", "PC2"]].to_numpy() ** 2 / pca.explained_variance_[:2]).sum(axis=1)
    outlier_cutoff = float(np.quantile(distances, 0.99))
    outlier_count = int(np.count_nonzero(distances > outlier_cutoff))
    top_loadings = {}
    for component in loadings.columns:
        top_loadings[component] = loadings[component].abs().nlargest(6).index.tolist()

    summary = {
        "rows": len(data),
        "predictor_count": len(predictors.columns),
        "encoded_feature_count": features.shape[1],
        "components_for_threshold": component_count,
        "threshold": variance_threshold,
        "pc1_variance": float(pca.explained_variance_ratio_[0]),
        "pc2_variance": float(pca.explained_variance_ratio_[1]),
        "pc3_variance": float(pca.explained_variance_ratio_[2]),
        "pc1_pc2_cumulative": float(np.sum(pca.explained_variance_ratio_[:2])),
        "outlier_cutoff": outlier_cutoff,
        "outlier_count": outlier_count,
        "outlier_percent": outlier_count / len(data) * 100,
        "top_loadings": top_loadings,
    }
    summary_lines = [
        "# Analysis summary",
        "",
        f"- Records: {summary['rows']:,}",
        f"- Original predictors: {summary['predictor_count']}",
        f"- Encoded, standardized PCA features: {summary['encoded_feature_count']}",
        f"- Components for {variance_threshold:.0%} cumulative variance: {component_count}",
        f"- PC1 / PC2 / PC3 variance: {summary['pc1_variance']:.2%} / {summary['pc2_variance']:.2%} / {summary['pc3_variance']:.2%}",
        f"- PC1 and PC2 cumulative variance: {summary['pc1_pc2_cumulative']:.2%}",
        f"- Top 1% PC1-PC2 distance screen: {outlier_count} records ({summary['outlier_percent']:.2f}%)",
        "",
        "Largest absolute loadings for each retained component (feature: loading):",
    ]
    for component, names in top_loadings.items():
        signed_names = ", ".join(
            f"{name} ({loadings.loc[name, component]:+.3f})" for name in names
        )
        summary_lines.append(f"- {component}: {signed_names}")
    (output_dir / "analysis_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Telco churn .csv or .xlsx file")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_output"))
    parser.add_argument("--variance-threshold", type=float, default=0.80)
    parser.add_argument("--sample-size", type=int, default=3000)
    args = parser.parse_args()
    summary = analyze_dataset(
        load_dataset(args.data),
        args.output_dir,
        variance_threshold=args.variance_threshold,
        sample_size=args.sample_size,
    )
    print(f"Analysis artifacts written to {args.output_dir.resolve()}")
    print((args.output_dir / "analysis_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
