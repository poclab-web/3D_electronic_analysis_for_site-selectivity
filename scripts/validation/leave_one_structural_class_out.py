#!/usr/bin/env python3
"""Reproduce the five-class holdout and interpretation-stability analyses.

The script reads the repository's hash-verified frozen inputs but writes only
to the user-specified output directory. Diketone observations are never used
for scaling, feature selection, alpha selection, or model fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODELS = tuple("ABCDE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def correlation(left: np.ndarray, right: np.ndarray, method: str) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    result = pearsonr(left, right) if method == "pearson" else spearmanr(left, right)
    return float(result.statistic)


def primary_checks(prediction: dict[str, float], cm, dm) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group, stage, expected in cm.PRIMARY_DIKETONE_CHECKS:
        simulation = dm.simulate_full(prediction, group)
        values = simulation["intermediate_abs"] if stage == "initial" else simulation["final_frac"]
        predicted = max(values, key=values.get)
        rows.append({
            "group": group,
            "stage": stage,
            "expected": expected,
            "predicted": predicted,
            "correct": bool(predicted == expected),
            "expected_percent": float(values[expected]),
        })
    return rows


def expanded_diketone_evaluation(label: str, prediction: dict[str, float], dm) -> dict[str, object]:
    primary_correct = 0
    initial_site_correct = 0
    initial_face_correct = 0
    final_axes_correct = 0
    for group in dm.GROUPS:
        simulation = dm.simulate_full(prediction, group)
        intermediate = simulation["intermediate_abs"]
        expected_face = dm.INITIAL_IDENTITY_TARGETS[group]
        initial_site_correct += int((intermediate["1"] + intermediate["2"]) > (intermediate["3"] + intermediate["4"]))
        initial_face_correct += int(max(("1", "2"), key=lambda item: intermediate[item]) == expected_face)
        primary_correct += int(max(intermediate, key=intermediate.get) == expected_face)
        if group in dm.FINAL_TARGETS:
            final = simulation["final_frac"]
            primary_correct += int(max(final, key=final.get) == dm.FINAL_IDENTITY_TARGETS[group])
            target = dm.FINAL_TARGETS[group]
            final_axes_correct += int(dm.final_axis_percent(final, target["first_axis_major"]) >= 50)
            final_axes_correct += int(dm.final_axis_percent(final, target["second_axis_major"]) >= 50)
    _, details = dm.evaluate_predictions(label, prediction)
    errors = np.asarray([row["abs_error_percent"] for row in details if pd.notna(row["observed_percent"])], dtype=float)
    return {
        "condition": label,
        "primary_correct": int(primary_correct),
        "primary_total": 8,
        "initial_site_correct": int(initial_site_correct),
        "initial_site_total": 6,
        "initial_face_within_expected_site_correct": int(initial_face_correct),
        "initial_face_total": 6,
        "final_axes_correct": int(final_axes_correct),
        "final_axes_total": 4,
        "semiquant_rmse_percent": float(np.sqrt(np.mean(errors**2))),
        "semiquant_mae_percent": float(np.mean(errors)),
    }


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repository_root / "libs"))
    import current_model as cm
    import diketone_metrics as dm

    verified_manifest = cm.verify_input_manifest().copy()
    payload = cm.ensure_frozen_inputs()
    metadata = payload["meta"].copy()
    raw = {block: np.asarray(payload["raw_blocks"][block], dtype=float) for block in cm.BLOCKS}
    coordinates = {block: np.asarray(payload["coords"][block], dtype=int) for block in cm.BLOCKS}
    response = pd.to_numeric(metadata[cm.TARGET], errors="coerce").to_numpy(dtype=float)
    train_manifest = pd.read_csv(cm.TRAIN_ROWS_PATH)
    train = train_manifest["row_index"].to_numpy(dtype=int)
    class_labels = metadata.loc[train, "entry"].astype(str).str.extract(r"^([A-E])")[0].to_numpy()
    if pd.isna(class_labels).any() or tuple(sorted(set(class_labels))) != MODELS:
        raise RuntimeError("Training entries do not define the expected A-E structural classes.")
    class_assignments = train_manifest.copy()
    class_assignments.insert(1, "structural_class", class_labels)
    class_assignments.to_csv(output_dir / "class_assignments.csv", index=False)

    diketone_pattern = rf"[a-f](?:{'|'.join(cm.DIKETONE_ENTRY_SUFFIXES)})"
    diketone = np.flatnonzero(metadata["entry"].astype(str).str.fullmatch(diketone_pattern).to_numpy())
    diketone_entries = metadata.loc[diketone, "entry"].astype(str).tolist()

    prediction_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    diketone_model_rows: list[dict[str, object]] = []
    diketone_check_rows: list[dict[str, object]] = []
    pathway_prediction_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    effect_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []

    for fold_index, heldout_class in enumerate(MODELS):
        test_mask = class_labels == heldout_class
        outer_test = train[test_mask]
        outer_train = train[~test_mask]
        _, candidate_rmse = cm.inner_path_scores(raw, coordinates, response, outer_train)
        best_index = int(np.argmin(candidate_rmse))
        alpha = float(cm.ALPHAS[best_index])
        features, names, _ = cm.build_features(raw, coordinates, outer_train)
        model = Lasso(alpha=alpha, fit_intercept=True, max_iter=cm.LASSO_FIT_MAX_ITER, tol=cm.LASSO_FIT_TOL)
        model.fit(features[outer_train], response[outer_train])
        predicted = model.predict(features[outer_test])
        for row_index, observed, estimate in zip(outer_test, response[outer_test], predicted):
            prediction_rows.append({
                "fold_id": fold_index,
                "heldout_class": heldout_class,
                "row_index": int(row_index),
                "entry": str(metadata.at[row_index, "entry"]),
                "name": str(metadata.at[row_index, "name"]),
                "observed_ddg_kcal_mol": float(observed),
                "predicted_ddg_kcal_mol": float(estimate),
                "residual_kcal_mol": float(estimate - observed),
            })
        fold_rows.append({
            "heldout_class": heldout_class,
            "train_n": int(len(outer_train)),
            "test_n": int(len(outer_test)),
            "selected_alpha": alpha,
            "inner_loocv_rmse_kcal_mol": float(candidate_rmse[best_index]),
            "nonzero_features": int(np.count_nonzero(model.coef_)),
            "r2": float(r2_score(response[outer_test], predicted)),
            "rmse_kcal_mol": float(math.sqrt(mean_squared_error(response[outer_test], predicted))),
            "mae_kcal_mol": float(mean_absolute_error(response[outer_test], predicted)),
            "bias_kcal_mol": float(np.mean(predicted - response[outer_test])),
            "pearson_r": correlation(response[outer_test], predicted, "pearson"),
            "spearman_rho": correlation(response[outer_test], predicted, "spearman"),
        })
        coefficient_rows.extend({"model": heldout_class, "feature": name, "coefficient": float(value), "selected": bool(value != 0)} for name, value in zip(names, model.coef_))
        centered = features[train] - np.mean(features[outer_train], axis=0)
        signed_effect = np.sign(model.coef_) * np.sqrt(np.mean((centered * model.coef_[None, :]) ** 2, axis=0))
        effect_rows.extend({"model": heldout_class, "feature": name, "signed_effect_rms_kcal_mol": float(value)} for name, value in zip(names, signed_effect))
        for block in cm.BLOCKS:
            block_mask = np.asarray([name.startswith(f"{block}:") for name in names])
            contributions = centered[:, block_mask] @ model.coef_[block_mask]
            block_rows.extend({"model": heldout_class, "entry": str(metadata.at[row_index, "entry"]), "block": block, "centered_contribution_kcal_mol": float(value)} for row_index, value in zip(train, contributions))

        diketone_values = model.predict(features[diketone])
        prediction = dict(zip(diketone_entries, map(float, diketone_values)))
        pathway_prediction_rows.append({"heldout_class": heldout_class, **prediction})
        checks = primary_checks(prediction, cm, dm)
        diketone_check_rows.extend({"heldout_class": heldout_class, **row} for row in checks)
        evaluation = expanded_diketone_evaluation(heldout_class, prediction, dm)
        diketone_model_rows.append({
            "heldout_class": heldout_class,
            "selected_alpha": alpha,
            "nonzero_features": int(np.count_nonzero(model.coef_)),
            **{key: value for key, value in evaluation.items() if key != "condition"},
        })

    predictions = pd.DataFrame(prediction_rows)
    folds = pd.DataFrame(fold_rows)
    pathway_predictions = pd.DataFrame(pathway_prediction_rows)
    ensemble_prediction = pathway_predictions.drop(columns="heldout_class").median(axis=0).to_dict()
    ensemble_evaluation = expanded_diketone_evaluation("median_barrier_ensemble", ensemble_prediction, dm)
    diketone_models = pd.DataFrame(diketone_model_rows)

    pooled = {
        "heldout_class": "pooled",
        "train_n": None,
        "test_n": int(len(predictions)),
        "selected_alpha": None,
        "inner_loocv_rmse_kcal_mol": None,
        "nonzero_features": None,
        "r2": float(r2_score(predictions["observed_ddg_kcal_mol"], predictions["predicted_ddg_kcal_mol"])),
        "rmse_kcal_mol": float(math.sqrt(mean_squared_error(predictions["observed_ddg_kcal_mol"], predictions["predicted_ddg_kcal_mol"]))),
        "mae_kcal_mol": float(mean_absolute_error(predictions["observed_ddg_kcal_mol"], predictions["predicted_ddg_kcal_mol"])),
        "bias_kcal_mol": float(np.mean(predictions["residual_kcal_mol"])),
        "pearson_r": correlation(predictions["observed_ddg_kcal_mol"].to_numpy(), predictions["predicted_ddg_kcal_mol"].to_numpy(), "pearson"),
        "spearman_rho": correlation(predictions["observed_ddg_kcal_mol"].to_numpy(), predictions["predicted_ddg_kcal_mol"].to_numpy(), "spearman"),
    }
    fold_metrics = pd.concat([folds, pd.DataFrame([pooled])], ignore_index=True)

    full_features, full_names, _ = cm.build_features(raw, coordinates, train)
    full_model = Lasso(alpha=0.01, fit_intercept=True, max_iter=cm.LASSO_FIT_MAX_ITER, tol=cm.LASSO_FIT_TOL)
    full_model.fit(full_features[train], response[train])
    coefficient_rows.extend({"model": "full", "feature": name, "coefficient": float(value), "selected": bool(value != 0)} for name, value in zip(full_names, full_model.coef_))
    full_centered = full_features[train] - np.mean(full_features[train], axis=0)
    full_effect = np.sign(full_model.coef_) * np.sqrt(np.mean((full_centered * full_model.coef_[None, :]) ** 2, axis=0))
    effect_rows.extend({"model": "full", "feature": name, "signed_effect_rms_kcal_mol": float(value)} for name, value in zip(full_names, full_effect))
    for block in cm.BLOCKS:
        block_mask = np.asarray([name.startswith(f"{block}:") for name in full_names])
        contributions = full_centered[:, block_mask] @ full_model.coef_[block_mask]
        block_rows.extend({"model": "full", "entry": str(metadata.at[row_index, "entry"]), "block": block, "centered_contribution_kcal_mol": float(value)} for row_index, value in zip(train, contributions))

    coefficients = pd.DataFrame(coefficient_rows)
    effects = pd.DataFrame(effect_rows)
    block_contributions = pd.DataFrame(block_rows)
    coefficient_matrix = coefficients.pivot(index="feature", columns="model", values="coefficient")
    selected = coefficient_matrix.ne(0)
    full_set = set(selected.index[selected["full"]])
    overlap_rows = []
    for model_name in MODELS:
        model_set = set(selected.index[selected[model_name]])
        overlap = len(full_set & model_set)
        overlap_rows.append({
            "model": model_name,
            "selected_n": len(model_set),
            "overlap_with_full_n": overlap,
            "full_feature_recall": overlap / len(full_set),
            "model_precision_vs_full": overlap / len(model_set),
            "jaccard_vs_full": overlap / len(full_set | model_set),
        })
    overlap_frame = pd.DataFrame(overlap_rows)
    pairwise_frame = pd.DataFrame({
        "left": left,
        "right": right,
        "jaccard": len(set(selected.index[selected[left]]) & set(selected.index[selected[right]])) / len(set(selected.index[selected[left]]) | set(selected.index[selected[right]])),
    } for left, right in itertools.combinations(MODELS, 2))

    sign_comparisons = []
    reversed_features = set()
    for feature in full_set:
        reference_sign = np.sign(coefficient_matrix.at[feature, "full"])
        for model_name in MODELS:
            value = coefficient_matrix.at[feature, model_name]
            if value != 0:
                agreement = np.sign(value) == reference_sign
                sign_comparisons.append(bool(agreement))
                if not agreement:
                    reversed_features.add(feature)
    class_selected_count = selected[list(MODELS)].sum(axis=1)
    repeated_features = class_selected_count[class_selected_count >= 2].index
    invariant_count = 0
    for feature in repeated_features:
        signs = {int(np.sign(coefficient_matrix.at[feature, model_name])) for model_name in MODELS if coefficient_matrix.at[feature, model_name] != 0}
        invariant_count += int(len(signs) == 1)

    effect_matrix = effects.pivot(index="feature", columns="model", values="signed_effect_rms_kcal_mol")
    spatial_rows = []
    block_similarity_rows = []
    for block in cm.BLOCKS:
        feature_names = [name for name in effect_matrix.index if name.startswith(f"{block}:grid:")]
        reference = effect_matrix.loc[feature_names, "full"].to_numpy(dtype=float)
        contribution_matrix = block_contributions[block_contributions["block"] == block].pivot(index="entry", columns="model", values="centered_contribution_kcal_mol")
        contribution_reference = contribution_matrix["full"].to_numpy(dtype=float)
        for model_name in MODELS:
            values = effect_matrix.loc[feature_names, model_name].to_numpy(dtype=float)
            contributions = contribution_matrix[model_name].to_numpy(dtype=float)
            spatial_rows.append({"block": block, "model": model_name, "pearson_vs_full": correlation(reference, values, "pearson"), "selected_grid_n": int(np.count_nonzero(values))})
            block_similarity_rows.append({"block": block, "model": model_name, "pearson_vs_full": correlation(contribution_reference, contributions, "pearson"), "rmse_vs_full_kcal_mol": float(np.sqrt(np.mean((contributions - contribution_reference) ** 2)))})

    full_frequencies = selected.loc[sorted(full_set), list(MODELS)].mean(axis=1)
    stability_summary = {
        "full_model_selected_features": int(len(full_set)),
        "full_features_selected_in_all_five_models": int((full_frequencies == 1).sum()),
        "full_features_selected_in_at_least_four_models": int((full_frequencies >= 0.8).sum()),
        "full_features_selected_in_at_least_three_models": int((full_frequencies >= 0.6).sum()),
        "full_feature_selected_instances": int(len(sign_comparisons)),
        "sign_agreement_instances": int(sum(sign_comparisons)),
        "sign_agreement_fraction": float(np.mean(sign_comparisons)),
        "full_features_with_any_sign_reversal": int(len(reversed_features)),
        "features_selected_in_at_least_two_class_models": int(len(repeated_features)),
        "class_model_sign_invariant_features": int(invariant_count),
        "class_model_sign_invariant_fraction": float(invariant_count / len(repeated_features)),
    }

    predictions.to_csv(output_dir / "outer_predictions.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    pathway_predictions.to_csv(output_dir / "diketone_pathway_predictions.csv", index=False)
    diketone_models.to_csv(output_dir / "diketone_model_metrics.csv", index=False)
    pd.DataFrame(diketone_check_rows).to_csv(output_dir / "diketone_primary_checks.csv", index=False)
    pd.DataFrame([ensemble_evaluation]).to_csv(output_dir / "diketone_ensemble_metrics.csv", index=False)
    coefficients.to_csv(output_dir / "feature_coefficients.csv", index=False)
    effects.to_csv(output_dir / "feature_signed_effect_rms.csv", index=False)
    block_contributions.to_csv(output_dir / "block_contributions_common83.csv", index=False)
    overlap_frame.to_csv(output_dir / "feature_overlap_vs_full.csv", index=False)
    pairwise_frame.to_csv(output_dir / "pairwise_feature_jaccard.csv", index=False)
    pd.DataFrame(spatial_rows).to_csv(output_dir / "spatial_effect_map_similarity.csv", index=False)
    pd.DataFrame(block_similarity_rows).to_csv(output_dir / "block_contribution_similarity.csv", index=False)
    (output_dir / "feature_stability_summary.json").write_text(json.dumps(stability_summary, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2), constrained_layout=True)
    colors = dict(zip(MODELS, ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377"]))
    for model_name in MODELS:
        subset = predictions[predictions["heldout_class"] == model_name]
        axes[0].scatter(subset["observed_ddg_kcal_mol"], subset["predicted_ddg_kcal_mol"], s=18, alpha=0.85, label=model_name, color=colors[model_name])
    limits = [-1.2, 3.5]
    axes[0].plot(limits, limits, color="0.35", linewidth=1, linestyle="--")
    axes[0].set(xlim=limits, ylim=limits, xlabel="Observed ΔΔG$^{‡}$ (kcal mol$^{-1}$)", ylabel="Predicted ΔΔG$^{‡}$ (kcal mol$^{-1}$)", title="A  Class-holdout predictions")
    axes[0].legend(title="Held out", fontsize=7, title_fontsize=7, frameon=False, ncol=5, loc="upper left")
    accuracy_names = ["Primary\nchecks", "Initial\nsite", "Initial\nface", "Final\naxes", "Median\nensemble"]
    accuracy_values = [diketone_models["primary_correct"].sum() / 40, diketone_models["initial_site_correct"].sum() / 30, diketone_models["initial_face_within_expected_site_correct"].sum() / 30, diketone_models["final_axes_correct"].sum() / 20, ensemble_evaluation["primary_correct"] / 8]
    axes[1].bar(range(5), np.asarray(accuracy_values) * 100, color=["#4477AA"] * 4 + ["#EE6677"])
    axes[1].set(xticks=range(5), xticklabels=accuracy_names, ylim=(0, 105), ylabel="Correct (%)", title="B  Diketone evaluation")
    axes[1].tick_params(axis="x", labelsize=7)
    block_similarity = pd.DataFrame(block_similarity_rows)
    heatmap = block_similarity.pivot(index="block", columns="model", values="pearson_vs_full").loc[list(cm.BLOCKS), list(MODELS)]
    image = axes[2].imshow(heatmap, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axes[2].set(xticks=range(5), xticklabels=MODELS, yticks=range(3), yticklabels=["Electronic", "Electrostatic", "Orbital"], xlabel="Held-out class", title="C  Block-contribution correlation")
    for row in range(3):
        for column in range(5):
            axes[2].text(column, row, f"{heatmap.iloc[row, column]:.2f}", ha="center", va="center", color="white" if heatmap.iloc[row, column] < 0.65 else "black", fontsize=7)
    fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04, label="Pearson r vs full model")
    fig.savefig(output_dir / "figure_S13_class_holdout_validation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    specification = {
        "analysis": "leave-one-structural-class-out validation with existing inner observation-level LOOCV",
        "classes": {model_name: int(np.sum(class_labels == model_name)) for model_name in MODELS},
        "candidate_alphas": list(cm.ALPHAS),
        "full_model_reference_alpha": 0.01,
        "descriptor_blocks": list(cm.BLOCKS),
        "frozen_input_files_verified": int(len(verified_manifest)),
        "frozen_input_manifest_sha256": sha256(cm.INPUT_MANIFEST_PATH),
        "software": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn_version, "matplotlib": matplotlib.__version__},
        "pooled_metrics": pooled,
        "diketone_median_barrier_ensemble": ensemble_evaluation,
        "feature_stability": stability_summary,
    }
    (output_dir / "analysis_specification.json").write_text(json.dumps(specification, indent=2) + "\n", encoding="utf-8")

    generated = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "output_manifest.csv")
    pd.DataFrame({"file": [path.name for path in generated], "size_bytes": [path.stat().st_size for path in generated], "sha256": [sha256(path) for path in generated]}).to_csv(output_dir / "output_manifest.csv", index=False)
    print(json.dumps({"pooled": pooled, "ensemble": ensemble_evaluation, "stability": stability_summary}, indent=2))


if __name__ == "__main__":
    main()

