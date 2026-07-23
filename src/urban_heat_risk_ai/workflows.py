"""Data-dependent CLI orchestration.

Every public function in this module is reached only after an explicit CLI
subcommand.  Importing it reads no data and creates no paths.  All model matrices
are selected by the version-controlled allow-list and checked by the central
leakage guard immediately before preprocessing/model fitting.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import pandas as pd

from urban_heat_risk_ai.artifacts import (
    ArtifactStore,
    FinalTestLock,
    FrozenHashes,
    authorize_final_test,
    experiment_metadata,
    sha256_file,
    software_versions,
)
from urban_heat_risk_ai.baselines import (
    configured_simple_baselines,
    fit_neural_baseline,
    fit_random_forest,
    fit_regularized_linear,
    select_elastic_net_parameters,
)
from urban_heat_risk_ai.components import (
    COMPONENT_A_TARGET,
    COMPONENT_B_TARGET,
    COMPONENT_C_TARGET,
    COMPONENT_D_TARGET,
    COMPONENT_TARGET_NAMES,
    ComponentModelBundle,
    DisagreementWarningRule,
    ReconstructionBounds,
    build_component_targets,
    calculate_component_utci,
    compare_direct_and_component,
    fit_component_inner_fold,
    fit_component_models,
    learn_disagreement_warning_threshold,
    median_selected_tree_count,
    reconstruct_components,
)
from urban_heat_risk_ai.config import ProjectConfig, load_project_config
from urban_heat_risk_ai.conformal import (
    SplitConformalCalibrator,
    conformal_coverage_report,
    fit_split_conformal,
)
from urban_heat_risk_ai.direct_xgb import (
    FittedDirectModel,
    IndexFold,
    TuningResult,
    fit_fixed_model,
    refit_outer_model,
    tune_inner_cv,
)
from urban_heat_risk_ai.errors import ArtifactIntegrityError, DataRequiredError
from urban_heat_risk_ai.explain import (
    FeatureRangeProfile,
    MultivariateAnomalyDetector,
    save_partial_dependence_plots,
    save_shap_global_plots,
    save_shap_local_waterfall,
)
from urban_heat_risk_ai.features import (
    FeaturePolicy,
    LeakageGuard,
    build_predictor_frame,
    derive_prespecified_interactions,
    make_preprocessor,
    satellite_eligibility_mask,
)
from urban_heat_risk_ai.metrics import (
    compose_spatial_day_blocks,
    derive_utci_categories,
    evaluate_prespecified_subgroups,
    full_metric_report,
    paired_block_bootstrap_improvement,
    time_of_day_groups,
)
from urban_heat_risk_ai.schema import load_observations, validate_schema
from urban_heat_risk_ai.splits import (
    BlockedFold,
    SpatioTemporalBlockedSplit,
    assert_split_invariants,
    read_or_create_split_manifest,
    validate_manifest_matches_data,
)

LOGGER = logging.getLogger(__name__)
TARGET = "calculated_utci_c"


def _load_project(args: Any) -> tuple[ProjectConfig, FeaturePolicy]:
    project = load_project_config(args.config, args.features_config)
    return project, FeaturePolicy.from_mapping(project.features)


def _validate_observations(
    args: Any,
    *,
    predictor_set: str = "core",
    require_labels: bool = True,
    require_split_role: bool = True,
) -> tuple[ProjectConfig, FeaturePolicy, pd.DataFrame, Any]:
    project, policy = _load_project(args)
    frame = load_observations(args.data)
    report = validate_schema(
        frame,
        policy,
        predictor_set=predictor_set,
        model_config=project.model,
        require_spatial_block=False,
        strict_unknown_columns=True,
        require_labels=require_labels,
        require_split_role=require_split_role,
    )
    return project, policy, frame, report


def _ensure_safe_destination(destination: Path, *protected: Path) -> None:
    resolved_destination = destination.expanduser().resolve()
    for source in protected:
        if resolved_destination == source.expanduser().resolve():
            raise ArtifactIntegrityError(
                f"Refusing to overwrite protected input {source} with output {destination}."
            )


def _attach_manifest(data: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Attach manifest assignments to a copy while preserving source row order."""

    validate_manifest_matches_data(manifest, data)
    if manifest["sample_id"].astype(str).duplicated().any():
        raise ValueError("Split manifest contains duplicate sample_id values.")
    keyed = manifest.copy()
    keyed["sample_id"] = keyed["sample_id"].astype(str)
    keyed = keyed.set_index("sample_id", drop=True)
    sample_ids = data["sample_id"].astype(str)
    result = data.copy()
    assignment_columns = [
        column
        for column in manifest.columns
        if column
        not in {
            "sample_id",
            "site_id",
            "date",
            "weather_event_id",
            "manifest_version",
        }
    ]
    for column in assignment_columns:
        result[column] = sample_ids.map(keyed[column])
        if result[column].isna().any():
            raise ValueError(f"Manifest assignment {column!r} did not map to every data row.")
    return result


def _read_attached(
    args: Any,
    project: ProjectConfig,
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = read_or_create_split_manifest(
        data,
        args.manifest,
        project.model["splits"],
        create_if_missing=False,
        source_data_path=args.data,
        target_column=TARGET,
    )
    return _attach_manifest(data, manifest), manifest


def _role(frame: pd.DataFrame, name: str, project: ProjectConfig) -> pd.DataFrame:
    role_mask = frame["split_role"].astype("string").eq(name)
    quality_config = project.model.get("validation", {})
    allowed_quality = {
        str(value).strip().lower()
        for value in quality_config.get("training_quality_flags", ("pass",))
    }
    quality = frame["quality_flag"].astype("string").str.strip().str.lower()
    selected_mask = role_mask & quality.isin(allowed_quality)
    excluded = int((role_mask & ~selected_mask).sum())
    if excluded:
        LOGGER.warning(
            "Excluded %d %s rows whose quality_flag is outside %s",
            excluded,
            name,
            sorted(allowed_quality),
        )
    selected = frame.loc[selected_mask].copy()
    if selected.empty:
        raise ValueError(
            f"Split manifest contains no quality-eligible {name!r} observations."
        )
    if name == "final_test":
        final_config = project.model["splits"].get("final_test", {})
        minimum_sites = int(
            final_config.get("minimum_geographically_separated_sites", 4)
        )
        block_column = str(
            project.model["splits"]
            .get("spatial", {})
            .get("block_column", "spatial_block_id")
        )
        unique_sites = selected["site_id"].astype(str).nunique()
        unique_blocks = selected[block_column].astype(str).nunique()
        if unique_sites < minimum_sites or unique_blocks < minimum_sites:
            raise ValueError(
                "Quality filtering leaves too few geographically separated final-test "
                f"sites/blocks; at least {minimum_sites} are required "
                f"(observed sites={unique_sites}, blocks={unique_blocks})."
            )
        threshold = float(final_config.get("hot_date_threshold_c", 32.0))
        minimum_dates = int(final_config.get("minimum_complete_hot_dates", 1))
        date_column = str(
            project.model["splits"]
            .get("temporal", {})
            .get("date_column", "date")
        )
        hot_dates = selected.loc[
            pd.to_numeric(selected[TARGET], errors="coerce").ge(threshold), date_column
        ].astype(str)
        if hot_dates.nunique() < minimum_dates:
            raise ValueError(
                "Quality filtering leaves too few complete hot final-test dates; "
                f"at least {minimum_dates} at >= {threshold:g} C are required."
            )
    return selected


def _block_groups(frame: pd.DataFrame, project: ProjectConfig) -> np.ndarray:
    split_config = project.model["splits"]
    block_column = str(
        split_config.get("spatial", {}).get("block_column", "spatial_block_id")
    )
    date_column = str(
        split_config.get("temporal", {}).get("date_column", "date")
    )
    return compose_spatial_day_blocks(frame[block_column], frame[date_column])


def _processor_factory(
    policy: FeaturePolicy,
    project: ProjectConfig,
    predictor_set: str,
    predictors: tuple[str, ...],
    model_kind: str,
) -> Any:
    return lambda: make_preprocessor(
        policy,
        predictor_set=predictor_set,
        predictors=predictors,
        model_kind=model_kind,
        preprocessing_config=project.model.get("preprocessing", {}),
    )


def _range_value(spec: Any, key: str, default: Any) -> Any:
    return spec.get(key, default) if isinstance(spec, dict) else default


def _direct_tuning_config(
    project: ProjectConfig, *, n_trials_override: int | None = None
) -> dict[str, Any]:
    """Adapt the checked-in nested YAML contract to direct_xgb's compact API."""

    direct = project.model["direct_xgb"]
    tuning = project.model.get("tuning", {})
    spaces = tuning.get("search_space", {})
    n_estimators = spaces.get("n_estimators", {})
    max_depth = spaces.get("max_depth", {})
    learning_rate = spaces.get("learning_rate", {})
    subsample = spaces.get("subsample", {})
    colsample = spaces.get("colsample_bytree", {})
    child = spaces.get("min_child_weight", {})
    alpha = spaces.get("reg_alpha", {})
    regular_lambda = spaces.get("reg_lambda", {})
    gamma = spaces.get("gamma", {})
    huber = spaces.get("huber_slope", {})
    outer_refit = direct.get("outer_refit", {})
    trial_count = int(
        tuning.get("n_trials", 100)
        if n_trials_override is None
        else n_trials_override
    )
    if trial_count <= 0:
        raise ValueError("Optuna n_trials must be positive.")
    return {
        "seed": int(tuning.get("seed", project.seed)),
        "n_jobs": int(direct.get("n_jobs", 1)),
        "n_trials": trial_count,
        "timeout_seconds": tuning.get("timeout_seconds"),
        "pruner": tuning.get("pruner", "MedianPruner"),
        "early_stopping_rounds": int(direct.get("early_stopping_rounds", 50)),
        "high_utci_threshold_c": float(tuning.get("high_utci_threshold_c", 32.0)),
        "minimum_refit_trees": int(outer_refit.get("minimum_trees", 300)),
        "maximum_refit_trees": int(outer_refit.get("maximum_trees", 1500)),
        "search": {
            "objectives": list(
                direct.get(
                    "objective_candidates",
                    ["reg:absoluteerror", "reg:pseudohubererror"],
                )
            ),
            "n_estimators_min": int(_range_value(n_estimators, "low", 300)),
            "n_estimators_max": int(_range_value(n_estimators, "high", 1500)),
            "n_estimators_step": int(_range_value(n_estimators, "step", 50)),
            "max_depth_min": int(_range_value(max_depth, "low", 3)),
            "max_depth_max": int(_range_value(max_depth, "high", 8)),
            "learning_rate_min": float(_range_value(learning_rate, "low", 0.01)),
            "learning_rate_max": float(_range_value(learning_rate, "high", 0.10)),
            "subsample_min": float(_range_value(subsample, "low", 0.6)),
            "subsample_max": float(_range_value(subsample, "high", 1.0)),
            "colsample_min": float(_range_value(colsample, "low", 0.6)),
            "colsample_max": float(_range_value(colsample, "high", 1.0)),
            "min_child_weight_min": float(_range_value(child, "low", 0.1)),
            "min_child_weight_max": float(_range_value(child, "high", 30.0)),
            "reg_alpha_min": float(_range_value(alpha, "low", 1.0e-8)),
            "reg_alpha_max": float(_range_value(alpha, "high", 10.0)),
            "reg_lambda_min": float(_range_value(regular_lambda, "low", 1.0e-3)),
            "reg_lambda_max": float(_range_value(regular_lambda, "high", 100.0)),
            "gamma_min": float(_range_value(gamma, "low", 0.0)),
            "gamma_max": float(_range_value(gamma, "high", 5.0)),
            "huber_slope_min": float(_range_value(huber, "low", 0.5)),
            "huber_slope_max": float(_range_value(huber, "high", 5.0)),
        },
    }


def _tuning_summary(result: TuningResult) -> dict[str, Any]:
    return {
        "best_params": dict(result.best_params),
        "objective_value_grouped_mae": result.objective_value,
        "selected_tree_count": result.selected_tree_count,
        "best_trial_number": result.best_trial_number,
        "fold_scores": [asdict(score) for score in result.fold_scores],
    }


def _index_folds(folds: tuple[BlockedFold, ...] | list[BlockedFold]) -> list[IndexFold]:
    return [
        IndexFold(
            train=np.asarray(fold.train, dtype=int),
            validation=np.asarray(fold.validation, dtype=int),
            name=f"fold_{fold.fold}",
        )
        for fold in folds
    ]


def _frozen_outer_folds(
    observations: pd.DataFrame, project: ProjectConfig
) -> tuple[BlockedFold, ...]:
    """Reconstruct train/embargo rows from authoritative manifest fold labels."""

    config = project.model["splits"]
    spatial = config.get("spatial", {})
    temporal = config.get("temporal", {})
    block_column = str(spatial.get("block_column", "spatial_block_id"))
    date_column = str(temporal.get("date_column", "date"))
    event_column = temporal.get("weather_event_column")
    declared = pd.to_numeric(observations["outer_fold"], errors="coerce")
    if declared.isna().any():
        raise ArtifactIntegrityError("Manifest outer_fold values must be whole integers.")
    blocks = observations[block_column].astype(str).to_numpy()
    dates = observations[date_column].astype(str).to_numpy()
    events = (
        observations[str(event_column)].astype("string").to_numpy()
        if event_column and str(event_column) in observations.columns
        else None
    )
    positions = np.arange(len(observations), dtype=int)
    embargo_config = config.get("embargo", {})
    embargo_days = int(
        embargo_config.get("days", embargo_config.get("temporal_days", 0))
        if isinstance(embargo_config, dict)
        else 0
    )
    parsed_dates = pd.to_datetime(dates, errors="raise")
    folds: list[BlockedFold] = []
    for fold_number in range(int(config.get("outer_folds", 5))):
        validation_mask = declared.eq(fold_number).to_numpy()
        if not validation_mask.any():
            raise ValueError(
                f"Frozen fold {fold_number} has no observations in this eligible subset."
            )
        validation_blocks = set(blocks[validation_mask])
        validation_dates = set(dates[validation_mask])
        validation_events = (
            {
                str(value)
                for value in events[validation_mask]
                if pd.notna(value) and str(value).strip()
            }
            if events is not None
            else set()
        )
        restricted = np.isin(blocks, list(validation_blocks)) | np.isin(
            dates, list(validation_dates)
        )
        if events is not None and validation_events:
            restricted |= np.isin(events.astype(str), list(validation_events))
        if embargo_days:
            for held_date in pd.to_datetime(sorted(validation_dates)):
                restricted |= np.abs((parsed_dates - held_date).days) <= embargo_days
        train = positions[~restricted]
        validation = positions[validation_mask]
        embargo = positions[~validation_mask & restricted]
        assert_split_invariants(
            observations,
            train,
            validation,
            embargo=embargo,
            spatial_block_column=block_column,
            date_column=date_column,
            event_column=str(event_column) if events is not None else None,
        )
        folds.append(
            BlockedFold(
                fold=fold_number,
                train=train,
                validation=validation,
                embargo=embargo,
                validation_spatial_blocks=tuple(sorted(validation_blocks)),
                validation_dates=tuple(sorted(validation_dates)),
                validation_events=tuple(sorted(validation_events)),
            )
        )
    return tuple(folds)


def _inner_folds_for_outer(
    observations: pd.DataFrame,
    outer: BlockedFold,
    project: ProjectConfig,
) -> tuple[BlockedFold, ...]:
    """Create inner folds only within one frozen outer-training partition."""

    subset = observations.iloc[outer.train]
    local_splitter = SpatioTemporalBlockedSplit.from_config(
        project.model["splits"], inner=True
    )
    global_folds: list[BlockedFold] = []
    for local in local_splitter.split_with_embargo(subset):
        global_folds.append(
            BlockedFold(
                fold=local.fold,
                train=outer.train[local.train],
                validation=outer.train[local.validation],
                embargo=outer.train[local.embargo],
                validation_spatial_blocks=local.validation_spatial_blocks,
                validation_dates=local.validation_dates,
                validation_events=local.validation_events,
            )
        )
    return tuple(global_folds)


def _nested_direct_tune(
    observations: pd.DataFrame,
    *,
    project: ProjectConfig,
    policy: FeaturePolicy,
    predictor_set: str,
    n_trials_override: int | None,
    heat_weighted_experiment: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.Series, dict[str, Any]]:
    """Nested blocked OOF evaluation followed by all-development selection."""

    predictors = policy.allowed_predictors(predictor_set)  # fixed declared order
    guard = LeakageGuard(policy, predictor_set)
    guard.validate(predictors)
    predictor_frame = build_predictor_frame(observations, policy, predictor_set)
    target = pd.to_numeric(observations[TARGET], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(target).all():
        raise ValueError("Development UTCI labels must all be finite before tuning.")
    groups = _block_groups(observations, project)
    split_config = project.model["splits"]
    tuning_config = _direct_tuning_config(
        project, n_trials_override=n_trials_override
    )
    factory = _processor_factory(
        policy, project, predictor_set, predictors, "xgboost"
    )
    oof = pd.Series(np.nan, index=observations.index, dtype=float)
    outer_summaries: list[dict[str, Any]] = []
    studies: dict[str, Any] = {}
    for outer in _frozen_outer_folds(observations, project):
        inner_folds = _index_folds(
            list(_inner_folds_for_outer(observations, outer, project))
        )
        tuned = tune_inner_cv(
            predictor_frame,
            target,
            folds=inner_folds,
            preprocessor_factory=factory,
            config=tuning_config,
            validation_groups=groups,
            heat_weighted_experiment=heat_weighted_experiment,
            predictor_guard=guard.validate,
        )
        fitted = refit_outer_model(
            predictor_frame.iloc[outer.train],
            target[outer.train],
            predictors=predictors,
            preprocessor_factory=factory,
            tuning=tuned,
            seed=project.seed + outer.fold,
            n_jobs=int(project.model["direct_xgb"].get("n_jobs", 1)),
            heat_weighted_experiment=heat_weighted_experiment,
            predictor_guard=guard.validate,
        )
        oof.iloc[outer.validation] = fitted.predict(
            predictor_frame.iloc[outer.validation]
        )
        outer_summaries.append(
            {
                "outer_fold": outer.fold,
                "n_train": len(outer.train),
                "n_validation": len(outer.validation),
                "n_embargo": len(outer.embargo),
                "selection": _tuning_summary(tuned),
            }
        )
        studies[f"outer_{outer.fold}"] = tuned.study

    all_inner = tuple(
        SpatioTemporalBlockedSplit.from_config(split_config, inner=True).split_with_embargo(
            observations
        )
    )
    final_selection = tune_inner_cv(
        predictor_frame,
        target,
        folds=_index_folds(list(all_inner)),
        preprocessor_factory=factory,
        config=tuning_config,
        validation_groups=groups,
        heat_weighted_experiment=heat_weighted_experiment,
        predictor_guard=guard.validate,
    )
    studies["all_development"] = final_selection.study
    summary = {
        "predictor_set": predictor_set,
        "predictor_version": policy.version,
        "n_development_rows": len(observations),
        "n_oof_rows": int(oof.notna().sum()),
        "model_variant": (
            "secondary_heat_weighted"
            if heat_weighted_experiment and heat_weighted_experiment.get("enabled")
            else "primary_unweighted"
        ),
        "nested_outer_folds": outer_summaries,
        "final_selection": _tuning_summary(final_selection),
    }
    return summary, oof, studies


def run_validate(args: Any) -> int:
    """Validate an explicit real table and write nothing."""

    require_labels = args.mode == "training"
    _, _, _, report = _validate_observations(
        args,
        predictor_set="core",
        require_labels=require_labels,
        require_split_role=require_labels,
    )
    print(report.format_text())
    return 0 if report.is_valid else 2


def run_make_splits(args: Any) -> int:
    """Create or explicitly overwrite one deterministic manifest."""

    project, _, data, report = _validate_observations(args)
    print(report.format_text())
    report.raise_for_errors()
    _ensure_safe_destination(
        args.manifest, args.data, project.model_path, project.features_path
    )
    manifest = read_or_create_split_manifest(
        data,
        args.manifest,
        project.model["splits"],
        create_if_missing=True,
        overwrite=bool(args.overwrite_manifest),
        source_data_path=args.data,
        target_column=TARGET,
    )
    LOGGER.info(
        "Split manifest ready: %s (%d observations)", args.manifest, len(manifest)
    )
    return 0


def run_tune(args: Any) -> int:
    """Run nested blocked tuning and persist only real-data runtime results."""

    predictor_set = str(args.predictor_set)
    project, policy, data, report = _validate_observations(
        args, predictor_set=predictor_set
    )
    print(report.format_text())
    report.raise_for_errors()
    attached, _ = _read_attached(args, project, data)
    development = _role(attached, "development", project)
    _ensure_safe_destination(
        args.output_dir,
        args.data,
        args.manifest,
        project.model_path,
        project.features_path,
    )
    _refuse_existing(args.output_dir, "Tuning output directory")

    selected_models: dict[str, Any] = {}
    studies: dict[str, Any] = {}
    prediction_table: pd.DataFrame
    if predictor_set == "core":
        summary, oof, model_studies = _nested_direct_tune(
            development,
            project=project,
            policy=policy,
            predictor_set="core",
            n_trials_override=args.n_trials,
        )
        selected_models["core"] = summary
        studies.update({f"core_{name}": study for name, study in model_studies.items()})
        prediction_table = pd.DataFrame(
            {
                "sample_id": development["sample_id"].to_numpy(),
                "split_role": "development",
                "calculated_utci_c": development[TARGET].to_numpy(),
                "direct_core": oof.to_numpy(),
            }
        )
        heat_experiment = project.model.get("heat_weighted_experiment", {})
        if isinstance(heat_experiment, dict) and heat_experiment.get("enabled", False):
            heat_summary, heat_oof, heat_studies = _nested_direct_tune(
                development,
                project=project,
                policy=policy,
                predictor_set="core",
                n_trials_override=args.n_trials,
                heat_weighted_experiment=heat_experiment,
            )
            selected_models["core_heat_weighted_secondary"] = heat_summary
            prediction_table["direct_core_heat_weighted_secondary"] = heat_oof.to_numpy()
            studies.update(
                {f"core_heat_weighted_{name}": study for name, study in heat_studies.items()}
            )
    else:
        eligibility = satellite_eligibility_mask(
            development,
            project.model,
            feature_config=project.features,
        )
        eligible = development.loc[eligibility].copy()
        if eligible.empty:
            raise ValueError(
                "No development observations satisfy the frozen satellite age/quality rules."
            )
        enhanced_summary, enhanced_oof, enhanced_studies = _nested_direct_tune(
            eligible,
            project=project,
            policy=policy,
            predictor_set="satellite_enhanced",
            n_trials_override=args.n_trials,
        )
        core_summary, core_oof, core_studies = _nested_direct_tune(
            eligible,
            project=project,
            policy=policy,
            predictor_set="core",
            n_trials_override=args.n_trials,
        )
        selected_models["satellite_enhanced"] = enhanced_summary
        selected_models["core_eligible"] = core_summary
        studies.update(
            {
                f"enhanced_{name}": study
                for name, study in enhanced_studies.items()
            }
        )
        studies.update(
            {f"core_eligible_{name}": study for name, study in core_studies.items()}
        )
        common = enhanced_oof.notna() & core_oof.notna()
        prediction_table = pd.DataFrame(
            {
                "sample_id": eligible["sample_id"].to_numpy(),
                "split_role": "development_satellite_eligible",
                "satellite_eligible": True,
                "calculated_utci_c": eligible[TARGET].to_numpy(),
                "direct_satellite_enhanced": enhanced_oof.where(common).to_numpy(),
                "direct_core_eligible": core_oof.where(common).to_numpy(),
                "baseline_satellite_lst_alone": pd.to_numeric(
                    eligible["satellite_lst_c"], errors="coerce"
                ).to_numpy(dtype=float),
            }
        )

    hashes = FrozenHashes.from_files(
        dataset=args.data,
        model_config=project.model_path,
        feature_allowlist=project.features_path,
        split_manifest=args.manifest,
    )
    store = ArtifactStore(args.output_dir).initialize()
    prediction_name = "development_oof_predictions.parquet"
    store.write_dataframe(prediction_name, prediction_table)
    payload = {
        "schema_version": "1.0",
        "predictor_set": predictor_set,
        "selected_models": selected_models,
        "oof_predictions": prediction_name,
        "hashes": asdict(hashes),
        "runtime_overrides": {"n_trials": args.n_trials},
    }
    store.write_json("best_trial.json", payload)
    for name, study in studies.items():
        store.write_joblib(f"studies/{name}.joblib", study)
    store.copy_file(project.model_path, "configs/model.yaml")
    store.copy_file(project.features_path, "configs/features.yaml")
    store.copy_file(args.manifest, f"splits/{Path(args.manifest).name}")
    store.write_json("software_versions.json", software_versions())
    LOGGER.info("Nested tuning artifacts written to %s", args.output_dir)
    return 0


def _current_hashes(args: Any, project: ProjectConfig) -> FrozenHashes:
    return FrozenHashes.from_files(
        dataset=args.data,
        model_config=project.model_path,
        feature_allowlist=project.features_path,
        split_manifest=args.manifest,
    )


def _read_tuning_result(
    path: Path,
    *,
    current_hashes: FrozenHashes,
    required_predictor_set: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if not path.is_file():
        raise ArtifactIntegrityError(f"Tuning result does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("predictor_set") != required_predictor_set:
        raise ArtifactIntegrityError(
            f"Tuning result {path} is for {payload.get('predictor_set')!r}, not "
            f"{required_predictor_set!r}."
        )
    expected = payload.get("hashes", {})
    observed = asdict(current_hashes)
    mismatches = {
        key: (expected.get(key), observed[key])
        for key in observed
        if expected.get(key) != observed[key]
    }
    if mismatches:
        raise ArtifactIntegrityError(
            f"Tuning inputs do not match current frozen inputs: {mismatches}"
        )
    prediction_path = path.parent / str(payload.get("oof_predictions", ""))
    if not prediction_path.is_file():
        raise ArtifactIntegrityError(
            f"Tuning OOF prediction artifact is missing: {prediction_path}"
        )
    return payload, pd.read_parquet(prediction_path)


def _selected_params(payload: dict[str, Any], model_name: str) -> dict[str, Any]:
    try:
        params = payload["selected_models"][model_name]["final_selection"][
            "best_params"
        ]
    except (KeyError, TypeError) as exc:
        raise ArtifactIntegrityError(
            f"Tuning result has no final selected settings for {model_name!r}."
        ) from exc
    if not isinstance(params, dict) or int(params.get("n_estimators", 0)) <= 0:
        raise ArtifactIntegrityError(
            f"Selected settings for {model_name!r} lack a positive tree count."
        )
    return dict(params)


def _fit_direct_final(
    observations: pd.DataFrame,
    *,
    project: ProjectConfig,
    policy: FeaturePolicy,
    predictor_set: str,
    params: dict[str, Any],
    variant: str,
    heat_weighted_experiment: dict[str, Any] | None = None,
) -> FittedDirectModel:
    predictors = policy.allowed_predictors(predictor_set)
    guard = LeakageGuard(policy, predictor_set)
    guard.validate(predictors)
    matrix = build_predictor_frame(observations, policy, predictor_set)
    return fit_fixed_model(
        matrix,
        pd.to_numeric(observations[TARGET], errors="raise").to_numpy(dtype=float),
        predictors=predictors,
        preprocessor_factory=_processor_factory(
            policy, project, predictor_set, predictors, "xgboost"
        ),
        params=params,
        seed=project.seed,
        n_jobs=int(project.model["direct_xgb"].get("n_jobs", 1)),
        model_variant=variant,
        predictor_guard=guard.validate,
        heat_weighted_experiment=heat_weighted_experiment,
    )


def _component_prediction_frame(
    bundle: ComponentModelBundle,
    observations: pd.DataFrame,
    project: ProjectConfig,
) -> pd.DataFrame:
    raw = bundle.predict_raw(observations)
    reconstruction_config = project.model.get("components", {}).get(
        "reconstruction", {}
    )
    bounds = ReconstructionBounds(
        air_temperature_c=tuple(
            reconstruction_config.get("local_air_temperature_range_c", (-50.0, 65.0))
        ),
        relative_humidity_percent=tuple(
            reconstruction_config.get("relative_humidity_range_pct", (0.0, 100.0))
        ),
        pedestrian_wind_m_s=tuple(
            reconstruction_config.get("pedestrian_wind_range_m_s", (0.0, 40.0))
        ),
        mean_radiant_temperature_c=tuple(
            reconstruction_config.get("mrt_range_c", (-70.0, 120.0))
        ),
    )
    reconstruction = reconstruct_components(
        background_air_temperature_c=pd.to_numeric(
            observations["background_air_temperature_c"], errors="coerce"
        ).to_numpy(dtype=float),
        background_wind_m_s=pd.to_numeric(
            observations["background_wind_speed_m_s"], errors="coerce"
        ).to_numpy(dtype=float),
        predicted_air_temperature_delta_c=raw[COMPONENT_A_TARGET],
        predicted_local_vapor_pressure_kpa=raw[COMPONENT_B_TARGET],
        predicted_wind_log_adjustment=raw[COMPONENT_C_TARGET],
        predicted_mrt_delta_c=raw[COMPONENT_D_TARGET],
        bounds=bounds,
    )
    wind = project.model.get("wind_profile", {})
    applicability = wind.get("applicability", {})
    roughness = float(wind.get("roughness_length_m", 0.10))
    component_utci = calculate_component_utci(
        reconstruction,
        measurement_height_m=pd.to_numeric(
            observations["measurement_height_m"], errors="coerce"
        ).to_numpy(dtype=float),
        roughness_length_m=np.full(len(observations), roughness),
        displacement_height_m=float(wind.get("displacement_height_m", 0.0)),
        neutral_stability=bool(applicability.get("assume_neutral_stability", True)),
        minimum_measurement_height_m=float(
            applicability.get("minimum_source_height_m", 0.10)
        ),
        maximum_measurement_height_m=float(
            applicability.get("maximum_source_height_m", 10.0)
        ),
        sensitivity_roughness_lengths_m=tuple(
            wind.get("roughness_sensitivity_values_m", ())
        ),
        utci_limits=project.model.get("utci", {}).get(
            "explicit_applicability_limits", {}
        ),
    )
    return pd.concat(
        [
            raw,
            reconstruction.to_frame(index=observations.index),
            component_utci.to_frame(index=observations.index),
        ],
        axis=1,
    )


def _component_base_params(project: ProjectConfig) -> dict[str, Any]:
    """Prespecified label-independent starting settings for component early stopping."""

    search = project.model.get("tuning", {}).get("search_space", {})
    cap = int(search.get("n_estimators", {}).get("high", 1500))
    return {
        "objective": "reg:absoluteerror",
        "n_estimators": cap,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "gamma": 0.0,
        "tree_method": "hist",
        "eval_metric": "mae",
        "device": "cpu",
    }


def _select_component_tree_counts(
    observations: pd.DataFrame,
    *,
    inner_folds: list[IndexFold],
    project: ProjectConfig,
    policy: FeaturePolicy,
) -> dict[str, dict[str, Any]]:
    """Select each component's tree count on inner validation folds only."""

    predictors = policy.allowed_predictors("core")
    guard = LeakageGuard(policy, "core")
    guard.validate(predictors)
    matrix = build_predictor_frame(observations, policy, "core")
    targets = build_component_targets(observations)
    base = _component_base_params(project)
    selected: dict[str, dict[str, Any]] = {}
    for component_offset, name in enumerate(COMPONENT_TARGET_NAMES):
        y = targets[name].to_numpy(dtype=float)
        counts: list[int] = []
        for fold_offset, fold in enumerate(inner_folds):
            processor = _processor_factory(
                policy, project, "core", predictors, "component_xgb"
            )()
            x_train = processor.fit_transform(matrix.iloc[fold.train], y[fold.train])
            x_validation = processor.transform(matrix.iloc[fold.validation])
            fitted = fit_component_inner_fold(
                x_train,
                y[fold.train],
                x_validation,
                y[fold.validation],
                params=base,
                random_seed=project.seed + component_offset * 100 + fold_offset,
                n_jobs=int(project.model["direct_xgb"].get("n_jobs", 1)),
                early_stopping_rounds=int(
                    project.model["direct_xgb"].get("early_stopping_rounds", 50)
                ),
            )
            counts.append(fitted.best_tree_count)
        params = dict(base)
        params["n_estimators"] = median_selected_tree_count(counts)
        selected[name] = params
    return selected


def _fit_component_bundle(
    observations: pd.DataFrame,
    *,
    project: ProjectConfig,
    policy: FeaturePolicy,
    selected_params_by_component: dict[str, dict[str, Any]],
    seed_offset: int = 0,
) -> ComponentModelBundle:
    predictors = policy.allowed_predictors("core")
    guard = LeakageGuard(policy, "core")
    guard.validate(predictors)
    return fit_component_models(
        observations,
        predictors=predictors,
        preprocessor_factory=_processor_factory(
            policy, project, "core", predictors, "component_xgb"
        ),
        fixed_selected_params_by_component=selected_params_by_component,
        predictor_guard=guard.validate,
        random_seed=project.seed + seed_offset,
        n_jobs=int(project.model["direct_xgb"].get("n_jobs", 1)),
    )


def _development_folds(
    development: pd.DataFrame, project: ProjectConfig
) -> tuple[BlockedFold, ...]:
    return _frozen_outer_folds(development, project)


def _component_oof(
    development: pd.DataFrame,
    *,
    project: ProjectConfig,
    policy: FeaturePolicy,
    folds: tuple[BlockedFold, ...],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    result = pd.DataFrame(index=development.index)
    nested_folds = {
        fold.fold: SimpleNamespace(
            outer=fold,
            inner=_inner_folds_for_outer(development, fold, project),
        )
        for fold in folds
    }
    for fold in folds:
        nested_outer = nested_folds[fold.fold].outer
        if not np.array_equal(
            np.sort(nested_outer.train), np.sort(fold.train)
        ) or not np.array_equal(
            np.sort(nested_outer.validation), np.sort(fold.validation)
        ):
            raise ArtifactIntegrityError(
                f"Nested component fold {fold.fold} differs from the frozen outer split."
            )
        selected = _select_component_tree_counts(
            development,
            inner_folds=_index_folds(list(nested_folds[fold.fold].inner)),
            project=project,
            policy=policy,
        )
        bundle = _fit_component_bundle(
            development.iloc[fold.train],
            project=project,
            policy=policy,
            selected_params_by_component=selected,
            seed_offset=fold.fold,
        )
        predicted = _component_prediction_frame(
            bundle, development.iloc[fold.validation], project
        )
        for column in predicted.columns:
            values = predicted[column]
            if column == "wind_speed_10m_sensitivity_by_roughness_m":
                values = values.map(lambda item: json.dumps(item, sort_keys=True))
            if column not in result.columns:
                if pd.api.types.is_bool_dtype(values.dtype):
                    result[column] = pd.Series(
                        pd.NA, index=result.index, dtype="boolean"
                    )
                elif pd.api.types.is_numeric_dtype(values.dtype):
                    result[column] = np.nan
                else:
                    result[column] = pd.Series(None, index=result.index, dtype="object")
            result.iloc[fold.validation, result.columns.get_loc(column)] = values.to_numpy()
    all_inner = tuple(
        SpatioTemporalBlockedSplit.from_config(
            project.model["splits"], inner=True
        ).split_with_embargo(development)
    )
    final_selected = _select_component_tree_counts(
        development,
        inner_folds=_index_folds(list(all_inner)),
        project=project,
        policy=policy,
    )
    return result, final_selected


def _baseline_oof(
    development: pd.DataFrame,
    *,
    project: ProjectConfig,
    policy: FeaturePolicy,
    folds: tuple[BlockedFold, ...],
) -> tuple[pd.DataFrame, dict[str, float]]:
    predictors = policy.allowed_predictors("core")
    guard = LeakageGuard(policy, "core")
    guard.validate(predictors)
    y = pd.to_numeric(development[TARGET], errors="raise").to_numpy(dtype=float)
    groups = _block_groups(development, project)
    output = pd.DataFrame(index=development.index)
    for name in (
        "baseline_background_air_temperature",
        "baseline_heat_index",
        "baseline_regularized_linear",
        "baseline_random_forest",
    ):
        output[name] = np.nan
    neural_config = project.model.get("baselines", {}).get("neural_network", {})
    neural_enabled = isinstance(neural_config, dict) and neural_config.get(
        "enabled", False
    )
    if neural_enabled:
        output["baseline_neural_network"] = np.nan
    simple = configured_simple_baselines(project.model.get("baselines", {}))
    satellite_eligible: pd.Series | None = None
    if set(policy.satellite_enhanced_additions).issubset(development.columns):
        satellite_eligible = satellite_eligibility_mask(
            development, project.model, feature_config=project.features
        )
        output["baseline_satellite_lst_alone"] = np.nan
    linear_choices: list[tuple[float, float]] = []
    nested_by_fold = {
        fold.fold: SimpleNamespace(
            outer=fold,
            inner=_inner_folds_for_outer(development, fold, project),
        )
        for fold in folds
    }
    for fold in folds:
        validation = development.iloc[fold.validation]
        output.iloc[
            fold.validation,
            output.columns.get_loc("baseline_background_air_temperature"),
        ] = simple["background_air_temperature"].predict(validation)
        output.iloc[
            fold.validation, output.columns.get_loc("baseline_heat_index")
        ] = simple["background_heat_index"].predict(validation)
        if satellite_eligible is not None:
            eligible_positions = fold.validation[
                satellite_eligible.iloc[fold.validation].to_numpy(dtype=bool)
            ]
            if len(eligible_positions):
                output.iloc[
                    eligible_positions,
                    output.columns.get_loc("baseline_satellite_lst_alone"),
                ] = simple["satellite_lst_alone"].predict(
                    development.iloc[eligible_positions]
                )
        linear_factory = _processor_factory(
            policy, project, "core", predictors, "linear"
        )
        selection = select_elastic_net_parameters(
            development,
            y,
            predictors=predictors,
            folds=_index_folds(list(nested_by_fold[fold.fold].inner)),
            preprocessor_factory=linear_factory,
            groups=groups,
            seed=project.seed + fold.fold,
        )
        linear_choices.append((selection.alpha, selection.l1_ratio))
        linear = fit_regularized_linear(
            development.iloc[fold.train],
            y[fold.train],
            predictors=predictors,
            preprocessor_factory=linear_factory,
            alpha=selection.alpha,
            l1_ratio=selection.l1_ratio,
            seed=project.seed + fold.fold,
        )
        output.iloc[
            fold.validation, output.columns.get_loc("baseline_regularized_linear")
        ] = linear.predict(validation)
        forest = fit_random_forest(
            development.iloc[fold.train],
            y[fold.train],
            predictors=predictors,
            preprocessor_factory=_processor_factory(
                policy, project, "core", predictors, "random_forest"
            ),
            seed=project.seed + fold.fold,
            n_jobs=int(
                project.model.get("baselines", {})
                .get("random_forest", {})
                .get("n_jobs", 1)
            ),
        )
        output.iloc[
            fold.validation, output.columns.get_loc("baseline_random_forest")
        ] = forest.predict(validation)
        if neural_enabled:
            first_inner = nested_by_fold[fold.fold].inner[0]
            runtime_config = dict(neural_config)
            runtime_config.setdefault("seed", project.seed + fold.fold)
            neural = fit_neural_baseline(
                development.iloc[first_inner.train],
                y[first_inner.train],
                development.iloc[first_inner.validation],
                y[first_inner.validation],
                predictors=predictors,
                preprocessor_factory=_processor_factory(
                    policy, project, "core", predictors, "neural_network"
                ),
                config=runtime_config,
            )
            output.iloc[
                fold.validation,
                output.columns.get_loc("baseline_neural_network"),
            ] = neural.predict(validation)
    final_alpha = float(np.median([choice[0] for choice in linear_choices]))
    final_ratio = float(np.median([choice[1] for choice in linear_choices]))
    return output, {"alpha": final_alpha, "l1_ratio": final_ratio}


def run_train(args: Any) -> int:
    """Fit frozen development-only models and create the final-test hash lock."""

    project, policy, data, report = _validate_observations(args)
    print(report.format_text())
    report.raise_for_errors()
    attached, _ = _read_attached(args, project, data)
    development = _role(attached, "development", project)
    current_hashes = _current_hashes(args, project)
    core_payload, core_oof = _read_tuning_result(
        args.tuning_result,
        current_hashes=current_hashes,
        required_predictor_set="core",
    )
    core_params = _selected_params(core_payload, "core")
    if args.include_satellite_enhanced and args.enhanced_tuning_result is None:
        raise ValueError(
            "--include-satellite-enhanced requires --enhanced-tuning-result from a "
            "separate satellite_enhanced tune run."
        )
    _ensure_safe_destination(
        args.output_dir,
        args.data,
        args.manifest,
        args.tuning_result,
        project.model_path,
        project.features_path,
    )
    _refuse_existing(args.output_dir, "Training output directory")

    LOGGER.info("Fitting primary unweighted core model on development rows only")
    direct_core = _fit_direct_final(
        development,
        project=project,
        policy=policy,
        predictor_set="core",
        params=core_params,
        variant="primary_unweighted",
    )
    folds = _development_folds(development, project)
    oof = core_oof.set_index("sample_id").reindex(
        development["sample_id"].astype(str)
    )
    # Preserve an explicit sample_id column rather than relying on a pandas index.
    oof_result = pd.DataFrame(
        {
            "sample_id": development["sample_id"].astype(str).to_numpy(),
            "split_role": "development",
            "calculated_utci_c": pd.to_numeric(
                development[TARGET], errors="raise"
            ).to_numpy(dtype=float),
            "direct_core": pd.to_numeric(
                oof["direct_core"], errors="coerce"
            ).to_numpy(dtype=float),
        }
    )

    component_bundle: ComponentModelBundle | None = None
    component_selected_params: dict[str, dict[str, Any]] | None = None
    disagreement_rule: DisagreementWarningRule | None = None
    if not args.skip_components:
        LOGGER.info("Fitting component models with the frozen core split rules")
        component_predictions, component_selected_params = _component_oof(
            development,
            project=project,
            policy=policy,
            folds=folds,
        )
        oof_result["component_utci_c"] = component_predictions[
            "component_utci_c"
        ].to_numpy()
        for column in component_predictions.columns:
            output_name = (
                "component_utci_c"
                if column == "component_utci_c"
                else f"component_{column}"
            )
            oof_result[output_name] = component_predictions[column].to_numpy()
        disagreement_config = project.model["components"].get(
            "disagreement_warning", {}
        )
        disagreement_rule = learn_disagreement_warning_threshold(
            oof_result["direct_core"],
            oof_result["component_utci_c"],
            source_partition="development_oof",
            quantile=float(disagreement_config.get("quantile", 0.95)),
            minimum_samples=int(disagreement_config.get("minimum_samples", 20)),
        )
        comparison = compare_direct_and_component(
            oof_result["direct_core"],
            oof_result["component_utci_c"],
            warning_rule=disagreement_rule,
        )
        oof_result["direct_component_absolute_disagreement_c"] = comparison[
            "absolute_disagreement_c"
        ].to_numpy()
        oof_result["direct_component_disagreement_warning"] = comparison[
            "disagreement_warning"
        ].to_numpy(dtype=bool)
        component_bundle = _fit_component_bundle(
            development,
            project=project,
            policy=policy,
            selected_params_by_component=component_selected_params,
        )

    LOGGER.info("Fitting comparison baselines with frozen outer folds")
    baseline_predictions, linear_choice = _baseline_oof(
        development,
        project=project,
        policy=policy,
        folds=folds,
    )
    for column in baseline_predictions:
        oof_result[column] = baseline_predictions[column].to_numpy()
    predictors = policy.allowed_predictors("core")
    linear_model = fit_regularized_linear(
        development,
        development[TARGET],
        predictors=predictors,
        preprocessor_factory=_processor_factory(
            policy, project, "core", predictors, "linear"
        ),
        alpha=linear_choice["alpha"],
        l1_ratio=linear_choice["l1_ratio"],
        seed=project.seed,
    )
    forest_model = fit_random_forest(
        development,
        development[TARGET],
        predictors=predictors,
        preprocessor_factory=_processor_factory(
            policy, project, "core", predictors, "random_forest"
        ),
        seed=project.seed,
        n_jobs=int(
            project.model.get("baselines", {})
            .get("random_forest", {})
            .get("n_jobs", 1)
        ),
    )
    simple_baselines = configured_simple_baselines(
        project.model.get("baselines", {})
    )

    neural_model: Any | None = None
    neural_config = project.model.get("baselines", {}).get("neural_network", {})
    if isinstance(neural_config, dict) and neural_config.get("enabled", False):
        inner = next(
            SpatioTemporalBlockedSplit.from_config(
                project.model["splits"], inner=True
            ).split_with_embargo(development)
        )
        neural_runtime = dict(neural_config)
        neural_runtime.setdefault("seed", project.seed)
        neural_model = fit_neural_baseline(
            development.iloc[inner.train],
            development.iloc[inner.train][TARGET],
            development.iloc[inner.validation],
            development.iloc[inner.validation][TARGET],
            predictors=predictors,
            preprocessor_factory=_processor_factory(
                policy, project, "core", predictors, "neural_network"
            ),
            config=neural_runtime,
        )

    enhanced_models: dict[str, FittedDirectModel] = {}
    enhanced_oof: pd.DataFrame | None = None
    enhanced_feature_ranges: FeatureRangeProfile | None = None
    if args.include_satellite_enhanced:
        enhanced_payload, enhanced_oof = _read_tuning_result(
            args.enhanced_tuning_result,
            current_hashes=current_hashes,
            required_predictor_set="satellite_enhanced",
        )
        eligibility = satellite_eligibility_mask(
            development, project.model, feature_config=project.features
        )
        eligible_development = development.loc[eligibility].copy()
        enhanced_models["satellite_enhanced"] = _fit_direct_final(
            eligible_development,
            project=project,
            policy=policy,
            predictor_set="satellite_enhanced",
            params=_selected_params(enhanced_payload, "satellite_enhanced"),
            variant="satellite_enhanced_eligible_only",
        )
        enhanced_models["core_eligible"] = _fit_direct_final(
            eligible_development,
            project=project,
            policy=policy,
            predictor_set="core",
            params=_selected_params(enhanced_payload, "core_eligible"),
            variant="core_refit_on_satellite_eligible_rows",
        )
        enhanced_feature_ranges = FeatureRangeProfile.fit(
            eligible_development,
            policy.allowed_predictors("satellite_enhanced"),
        )

    heat_weighted_model: FittedDirectModel | None = None
    if "core_heat_weighted_secondary" in core_payload.get("selected_models", {}):
        heat_weighted_model = _fit_direct_final(
            development,
            project=project,
            policy=policy,
            predictor_set="core",
            params=_selected_params(core_payload, "core_heat_weighted_secondary"),
            variant="secondary_heat_weighted",
            heat_weighted_experiment=project.model.get(
                "heat_weighted_experiment", {}
            ),
        )
        if "direct_core_heat_weighted_secondary" in core_oof:
            secondary = core_oof.set_index("sample_id").reindex(
                development["sample_id"].astype(str)
            )
            oof_result["direct_core_heat_weighted_secondary"] = pd.to_numeric(
                secondary["direct_core_heat_weighted_secondary"], errors="coerce"
            ).to_numpy(dtype=float)

    feature_ranges = FeatureRangeProfile.fit(development, predictors)
    anomaly_detector: MultivariateAnomalyDetector | None = None
    anomaly_config = project.model.get("anomaly_detection", {})
    if isinstance(anomaly_config, dict) and anomaly_config.get("enabled", False):
        anomaly_detector = MultivariateAnomalyDetector.fit(
            development,
            predictors=predictors,
            preprocessor_factory=_processor_factory(
                policy, project, "core", predictors, "random_forest"
            ),
            contamination=anomaly_config.get("contamination", "auto"),
            seed=int(anomaly_config.get("random_state", project.seed)),
            n_jobs=int(project.model["direct_xgb"].get("n_jobs", 1)),
        )

    run_id = Path(args.output_dir).name
    lock = FinalTestLock.create(current_hashes, run_id)
    store = ArtifactStore(args.output_dir).initialize()
    store.write_joblib("models/direct_core.joblib", direct_core)
    if component_bundle is not None:
        store.write_joblib("models/components.joblib", component_bundle)
    store.write_joblib("models/regularized_linear.joblib", linear_model)
    store.write_joblib("models/random_forest_500.joblib", forest_model)
    store.write_joblib("models/simple_baselines.joblib", simple_baselines)
    if neural_model is not None:
        store.write_joblib("models/neural_network.joblib", neural_model)
    if heat_weighted_model is not None:
        store.write_joblib(
            "models/direct_core_heat_weighted_secondary.joblib", heat_weighted_model
        )
    for name, model in enhanced_models.items():
        store.write_joblib(f"models/direct_{name}.joblib", model)
    store.write_joblib("diagnostics/feature_ranges.joblib", feature_ranges)
    if enhanced_feature_ranges is not None:
        store.write_joblib(
            "diagnostics/feature_ranges_satellite_enhanced.joblib",
            enhanced_feature_ranges,
        )
    if anomaly_detector is not None:
        store.write_joblib("diagnostics/anomaly_detector.joblib", anomaly_detector)
    if disagreement_rule is not None:
        store.write_json("diagnostics/disagreement_rule.json", asdict(disagreement_rule))
    store.write_dataframe("predictions/development_oof.parquet", oof_result)
    if enhanced_oof is not None:
        store.write_dataframe(
            "predictions/development_satellite_eligible_oof.parquet", enhanced_oof
        )
    store.write_json(
        "selected_settings.json",
        {
            "primary_core": core_params,
            "components": component_selected_params,
            "linear": linear_choice,
            "neural_enabled": neural_model is not None,
            "components_enabled": component_bundle is not None,
            "satellite_enhanced_enabled": bool(enhanced_models),
            "heat_weighted_secondary_enabled": heat_weighted_model is not None,
        },
    )
    store.copy_file(project.model_path, "configs/model.yaml")
    store.copy_file(project.features_path, "configs/features.yaml")
    store.copy_file(args.manifest, f"splits/{Path(args.manifest).name}")
    store.write_json("hashes.json", asdict(current_hashes))
    store.freeze_final_test(
        lock,
        relative=str(
            project.model.get("paths", {}).get(
                "final_test_lock_name", "final_test.lock.json"
            )
        ),
    )
    store.write_json("software_versions.json", software_versions())
    store.write_json(
        "experiment_metadata.json",
        experiment_metadata(
            command="train",
            run_id=run_id,
            seed=project.seed,
            hashes=current_hashes,
            extra={
                "n_development": len(development),
                "calibration_and_final_excluded": True,
                "tuning_result": str(args.tuning_result),
                "enhanced_tuning_result": (
                    str(args.enhanced_tuning_result)
                    if args.enhanced_tuning_result
                    else None
                ),
            },
        ),
    )
    LOGGER.info("Frozen training run written to %s", args.output_dir)
    return 0


def _verify_run_hashes(
    run_dir: Path, current_hashes: FrozenHashes
) -> dict[str, Any]:
    hash_path = run_dir / "hashes.json"
    if not hash_path.is_file():
        raise ArtifactIntegrityError(f"Run hash record is missing: {hash_path}")
    payload = json.loads(hash_path.read_text(encoding="utf-8"))
    expected = FrozenHashes(**payload)
    mismatches = current_hashes.compare(expected)
    if mismatches:
        raise ArtifactIntegrityError(f"Run inputs changed since training: {mismatches}")
    return payload


def _load_joblib(path: Path, description: str) -> Any:
    if not path.is_file():
        raise ArtifactIntegrityError(f"{description} is missing: {path}")
    return joblib.load(path)


def _refuse_existing(path: Path, description: str) -> None:
    if path.exists():
        raise ArtifactIntegrityError(
            f"{description} already exists at {path}; refusing to overwrite a frozen result."
        )


def _read_runtime_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise DataRequiredError(f"Runtime table does not exist: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise DataRequiredError(f"Runtime table must be CSV or Parquet: {path}")


def run_calibrate(args: Any) -> int:
    """Fit the symmetric 90% conformal radius on calibration rows only."""

    project, _, data, report = _validate_observations(args)
    print(report.format_text())
    report.raise_for_errors()
    attached, _ = _read_attached(args, project, data)
    current_hashes = _current_hashes(args, project)
    _verify_run_hashes(args.run_dir, current_hashes)
    calibration = _role(attached, "calibration", project)
    model: FittedDirectModel = _load_joblib(
        args.run_dir / "models/direct_core.joblib", "Primary core model"
    )
    prediction = model.predict(calibration)
    conformal_config = project.model.get("conformal", {})
    calibrator = fit_split_conformal(
        calibration[TARGET],
        prediction,
        partition_role="calibration",
        alpha=float(conformal_config.get("alpha", 0.10)),
        minimum_calibration_rows=int(
            conformal_config.get("minimum_calibration_rows", 20)
        ),
    )
    lower, upper = calibrator.predict_interval(prediction)
    artifact_path = args.run_dir / "calibration/conformal.joblib"
    _refuse_existing(artifact_path, "Conformal calibration")
    store = ArtifactStore(args.run_dir).open_existing()
    store.write_joblib("calibration/conformal.joblib", calibrator)
    store.write_json("calibration/conformal.json", calibrator.to_dict())
    store.write_dataframe(
        "calibration/predictions.parquet",
        pd.DataFrame(
            {
                "sample_id": calibration["sample_id"].astype(str).to_numpy(),
                "calculated_utci_c": calibration[TARGET].to_numpy(),
                "direct_core": prediction,
                "interval_lower_c": lower,
                "interval_upper_c": upper,
            }
        ),
    )
    LOGGER.info(
        "Conformal calibration saved from %d untouched calibration rows",
        len(calibration),
    )
    return 0


def _local_hour_values(frame: pd.DataFrame) -> np.ndarray | None:
    """Recover documented local hours from the operational cyclic pair."""

    if not {"hour_sin", "hour_cos"}.issubset(frame.columns):
        return None
    sine = pd.to_numeric(frame["hour_sin"], errors="coerce").to_numpy(dtype=float)
    cosine = pd.to_numeric(frame["hour_cos"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(sine) & np.isfinite(cosine)
    if not valid.any():
        return None
    hours = np.full(len(frame), np.nan, dtype=float)
    hours[valid] = (
        np.mod(np.arctan2(sine[valid], cosine[valid]), 2.0 * np.pi)
        * 24.0
        / (2.0 * np.pi)
    )
    return hours


def _evaluation_bundle(
    observations: pd.DataFrame,
    predictions: pd.DataFrame,
    project: ProjectConfig,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Evaluate all prediction columns after exact sample-ID alignment."""

    if predictions["sample_id"].astype(str).duplicated().any():
        raise ValueError("Prediction table contains duplicate sample_id values.")
    metadata = observations.copy()
    metadata["sample_id"] = metadata["sample_id"].astype(str)
    prediction_copy = predictions.copy()
    prediction_copy["sample_id"] = prediction_copy["sample_id"].astype(str)
    prediction_ids = set(prediction_copy["sample_id"])
    observation_ids = set(metadata["sample_id"])
    if prediction_ids != observation_ids:
        raise ArtifactIntegrityError(
            "Prediction IDs do not exactly match the requested evaluation partition: "
            f"{len(observation_ids - prediction_ids)} missing and "
            f"{len(prediction_ids - observation_ids)} unknown IDs."
        )
    prediction_columns = [
        column
        for column in prediction_copy.columns
        if column
        in {
            "direct_core",
            "direct_satellite_enhanced",
            "direct_core_eligible",
            "direct_core_heat_weighted_secondary",
            "component_utci_c",
        }
        or column.startswith("baseline_")
    ]
    joined = prediction_copy.merge(
        metadata,
        on="sample_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_prediction", ""),
    )
    if f"{TARGET}_prediction" in joined.columns and TARGET in joined.columns:
        reported = pd.to_numeric(joined[f"{TARGET}_prediction"], errors="coerce")
        observed = pd.to_numeric(joined[TARGET], errors="coerce")
        if not np.allclose(reported, observed, equal_nan=True):
            raise ArtifactIntegrityError(
                "Stored prediction labels differ from the current hashed dataset."
            )
    truth = pd.to_numeric(joined[TARGET], errors="coerce").to_numpy(dtype=float)
    subgroup_config = project.model.get("evaluation", {}).get("subgroup_columns", {})
    sun_column = str(subgroup_config.get("sun_shade", "sun_shade_group"))
    land_column = str(subgroup_config.get("land_cover", "land_cover_class"))
    coast_column = str(subgroup_config.get("coast_distance", "coast_distance_group"))
    time_column = str(subgroup_config.get("time_of_day", "time_of_day_group"))
    local_hours = _local_hour_values(joined)
    metrics: dict[str, Any] = {}
    subgroup_frames: list[pd.DataFrame] = []
    for column in prediction_columns:
        values = pd.to_numeric(joined[column], errors="coerce").to_numpy(dtype=float)
        usable = np.isfinite(truth) & np.isfinite(values)
        if not usable.any():
            metrics[column] = {"available": False, "reason": "no paired finite rows"}
            continue
        metrics[column] = {
            "available": True,
            "support": int(usable.sum()),
            **full_metric_report(truth[usable], values[usable]),
        }
        subgroups = evaluate_prespecified_subgroups(
            truth[usable],
            values[usable],
            sun_shade=(
                joined.loc[usable, sun_column]
                if sun_column in joined
                else None
            ),
            shade_fraction=(
                joined.loc[usable, "estimated_shade_fraction"]
                if "estimated_shade_fraction" in joined
                else None
            ),
            land_cover=(
                joined.loc[usable, land_column]
                if land_column in joined
                else None
            ),
            coast_distance=(
                joined.loc[usable, coast_column]
                if coast_column in joined
                else None
            ),
            coast_distance_km=(
                pd.to_numeric(
                    joined.loc[usable, "distance_to_coast_m"], errors="coerce"
                )
                / 1000.0
                if coast_column not in joined and "distance_to_coast_m" in joined
                else None
            ),
            time_of_day=(
                joined.loc[usable, time_column]
                if time_column in joined
                else None
            ),
            hour_of_day=(
                local_hours[usable]
                if time_column not in joined and local_hours is not None
                else None
            ),
        )
        subgroups.insert(0, "model", column)
        subgroup_frames.append(subgroups)

    # Enforce exact enhanced-versus-eligible-core comparison support.
    if {
        "direct_satellite_enhanced",
        "direct_core_eligible",
    }.issubset(joined.columns):
        enhanced_finite = np.isfinite(
            pd.to_numeric(
                joined["direct_satellite_enhanced"], errors="coerce"
            ).to_numpy(dtype=float)
        )
        core_finite = np.isfinite(
            pd.to_numeric(joined["direct_core_eligible"], errors="coerce").to_numpy(
                dtype=float
            )
        )
        if not np.array_equal(enhanced_finite, core_finite):
            raise ValueError(
                "Enhanced and eligible-core predictions do not cover identical observations."
            )

    if {"direct_core", "component_utci_c"}.issubset(joined.columns):
        direct_values = pd.to_numeric(joined["direct_core"], errors="coerce").to_numpy(
            dtype=float
        )
        component_values = pd.to_numeric(
            joined["component_utci_c"], errors="coerce"
        ).to_numpy(dtype=float)
        common = (
            np.isfinite(truth)
            & np.isfinite(direct_values)
            & np.isfinite(component_values)
        )
        metrics["paired_direct_vs_component_identical_rows"] = {
            "available": bool(common.any()),
            "support": int(common.sum()),
            "direct_core": (
                full_metric_report(truth[common], direct_values[common])
                if common.any()
                else None
            ),
            "component_utci": (
                full_metric_report(truth[common], component_values[common])
                if common.any()
                else None
            ),
            "averaged_estimate": False,
        }

    bootstrap_results: dict[str, Any] = {}
    split_config = project.model["splits"]
    block_column = str(
        split_config.get("spatial", {}).get("block_column", "spatial_block_id")
    )
    date_column = str(
        split_config.get("temporal", {}).get("date_column", "date")
    )
    if {
        "direct_core",
        "baseline_background_air_temperature",
        block_column,
        "site_id",
        date_column,
    }.issubset(joined.columns):
        bootstrap_config = project.model.get("evaluation", {}).get("bootstrap", {})
        n_bootstrap = int(bootstrap_config.get("resamples", 2000))
        confidence = float(bootstrap_config.get("confidence_level", 0.95))
        seed = int(bootstrap_config.get("seed", project.seed))
        model_values = pd.to_numeric(joined["direct_core"], errors="coerce")
        baseline_values = pd.to_numeric(
            joined["baseline_background_air_temperature"], errors="coerce"
        )
        for block_name, location in (
            ("spatial_day", block_column),
            ("site_day", "site_id"),
        ):
            blocks = compose_spatial_day_blocks(joined[location], joined[date_column])
            for weighting in ("observation_weighted", "block_balanced"):
                key = f"{block_name}_{weighting}"
                try:
                    bootstrap_results[key] = paired_block_bootstrap_improvement(
                        truth,
                        model_values,
                        baseline_values,
                        blocks,
                        metric="mae",
                        weighting=weighting,
                        n_bootstrap=n_bootstrap,
                        confidence=confidence,
                        random_seed=seed,
                    ).to_dict()
                except ValueError as exc:
                    bootstrap_results[key] = {
                        "available": False,
                        "reason": str(exc),
                    }
    subgroup_table = (
        pd.concat(subgroup_frames, ignore_index=True)
        if subgroup_frames
        else pd.DataFrame()
    )
    return metrics, subgroup_table, bootstrap_results, joined


def run_evaluate(args: Any) -> int:
    """Evaluate development OOF predictions and create association diagnostics."""

    project, _, data, report = _validate_observations(args)
    print(report.format_text())
    report.raise_for_errors()
    attached, _ = _read_attached(args, project, data)
    current_hashes = _current_hashes(args, project)
    _verify_run_hashes(args.run_dir, current_hashes)
    default_predictions = args.run_dir / "predictions/development_oof.parquet"
    prediction_path = args.predictions or default_predictions
    predictions = _read_runtime_table(prediction_path)
    enhanced_path = (
        args.run_dir / "predictions/development_satellite_eligible_oof.parquet"
    )
    if args.predictions is None and enhanced_path.is_file():
        enhanced = pd.read_parquet(enhanced_path)
        additions = [
            column
            for column in enhanced.columns
            if column not in {"split_role", TARGET}
        ]
        predictions = predictions.merge(
            enhanced.loc[:, additions],
            on="sample_id",
            how="outer",
            validate="one_to_one",
        )
    development = _role(attached, "development", project)
    explain_config = project.model.get("explain", {})
    requested_ids = tuple(
        str(value) for value in getattr(args, "local_explanation_sample_ids", ())
    )
    selected_local_rows: list[tuple[str, pd.DataFrame]] = []
    if requested_ids:
        explanations_enabled = isinstance(explain_config, dict) and explain_config.get(
            "enabled", True
        )
        shap_config = explain_config.get("shap", {}) if explanations_enabled else {}
        waterfalls_enabled = isinstance(shap_config, dict) and shap_config.get(
            "local_waterfall", True
        )
        if not explanations_enabled or not waterfalls_enabled:
            raise ValueError(
                "Local explanation IDs were requested, but local SHAP waterfalls are "
                "disabled by configuration."
            )
        available_ids = development["sample_id"].astype(str)
        for sample_id in requested_ids:
            matches = development.loc[available_ids == sample_id]
            if len(matches) != 1:
                raise ValueError(
                    "Each --local-explanation-sample-id must identify exactly one "
                    f"development observation; {sample_id!r} matched {len(matches)}."
                )
            selected_local_rows.append((sample_id, matches))
    metrics, subgroups, bootstrap, _ = _evaluation_bundle(
        development, predictions, project
    )
    metrics_path = args.run_dir / "evaluation/metrics.json"
    _refuse_existing(metrics_path, "Development evaluation")
    store = ArtifactStore(args.run_dir).open_existing()
    store.write_json("evaluation/paired_block_bootstrap.json", bootstrap)
    if not subgroups.empty:
        store.write_dataframe("evaluation/subgroup_metrics.csv", subgroups)

    direct_model: FittedDirectModel = _load_joblib(
        args.run_dir / "models/direct_core.joblib", "Primary core model"
    )
    range_profile: FeatureRangeProfile = _load_joblib(
        args.run_dir / "diagnostics/feature_ranges.joblib", "Feature range profile"
    )
    store.write_dataframe(
        "evaluation/feature_range_checks.csv", range_profile.check(development)
    )
    anomaly_path = args.run_dir / "diagnostics/anomaly_detector.joblib"
    if anomaly_path.is_file():
        detector: MultivariateAnomalyDetector = joblib.load(anomaly_path)
        anomaly = detector.score(development).copy()
        anomaly.insert(0, "sample_id", development["sample_id"].astype(str).to_numpy())
        store.write_dataframe("evaluation/anomaly_scores.parquet", anomaly)

    if isinstance(explain_config, dict) and explain_config.get("enabled", True):
        plot_root = args.run_dir / "evaluation/plots"
        shap_paths = save_shap_global_plots(
            direct_model,
            development,
            bar_path=plot_root / "shap_global_bar.png",
            beeswarm_path=plot_root / "shap_beeswarm.png",
            seed=project.seed,
        )
        pdp_path = save_partial_dependence_plots(
            direct_model,
            development,
            features=tuple(explain_config.get("partial_dependence_features", ())),
            output_path=plot_root / "partial_dependence.png",
            recompute_derived=lambda frame: derive_prespecified_interactions(
                frame, project.features
            ),
        )
        local_waterfalls: list[dict[str, str]] = []
        for position, (sample_id, matches) in enumerate(
            selected_local_rows, start=1
        ):
            waterfall_path = save_shap_local_waterfall(
                direct_model,
                matches,
                output_path=(
                    plot_root / f"shap_local_waterfall_{position:03d}.png"
                ),
            )
            local_waterfalls.append(
                {"sample_id": sample_id, "path": waterfall_path}
            )
        store.write_json(
            "evaluation/explanations.json",
            {
                "shap": shap_paths,
                "local_waterfalls": local_waterfalls,
                "partial_dependence": pdp_path,
                "interpretation": (
                    "Predictive associations only; these plots are not evidence of causation."
                ),
            },
        )
    # The metrics file is the completion marker. Writing it last keeps failed
    # explanation attempts safely retryable; preceding artifacts are deterministic.
    store.write_json("evaluation/metrics.json", metrics)
    LOGGER.info("Development evaluation written to %s", args.run_dir / "evaluation")
    return 0


def _load_disagreement_rule(run_dir: Path) -> DisagreementWarningRule | None:
    path = run_dir / "diagnostics/disagreement_rule.json"
    if not path.is_file():
        return None
    return DisagreementWarningRule(**json.loads(path.read_text(encoding="utf-8")))


def _all_runtime_predictions(
    observations: pd.DataFrame,
    *,
    run_dir: Path,
    project: ProjectConfig,
    policy: FeaturePolicy,
) -> pd.DataFrame:
    """Predict every fitted comparison without ever averaging direct/component UTCI."""

    result = pd.DataFrame(
        {"sample_id": observations["sample_id"].astype(str).to_numpy()},
        index=observations.index,
    )
    direct: FittedDirectModel = _load_joblib(
        run_dir / "models/direct_core.joblib", "Primary core model"
    )
    result["direct_core"] = direct.predict(observations)
    simple = _load_joblib(
        run_dir / "models/simple_baselines.joblib", "Simple baselines"
    )
    result["baseline_background_air_temperature"] = simple[
        "background_air_temperature"
    ].predict(observations)
    result["baseline_heat_index"] = simple["background_heat_index"].predict(
        observations
    )
    linear = _load_joblib(
        run_dir / "models/regularized_linear.joblib", "Regularized linear baseline"
    )
    forest = _load_joblib(
        run_dir / "models/random_forest_500.joblib", "Random Forest baseline"
    )
    result["baseline_regularized_linear"] = linear.predict(observations)
    result["baseline_random_forest"] = forest.predict(observations)
    neural_path = run_dir / "models/neural_network.joblib"
    if neural_path.is_file():
        result["baseline_neural_network"] = joblib.load(neural_path).predict(observations)

    weighted_path = run_dir / "models/direct_core_heat_weighted_secondary.joblib"
    if weighted_path.is_file():
        result["direct_core_heat_weighted_secondary"] = joblib.load(
            weighted_path
        ).predict(observations)

    component_path = run_dir / "models/components.joblib"
    if component_path.is_file():
        component_bundle: ComponentModelBundle = joblib.load(component_path)
        component = _component_prediction_frame(
            component_bundle, observations, project
        )
        for column in component.columns:
            output_name = (
                "component_utci_c"
                if column == "component_utci_c"
                else f"component_{column}"
            )
            values = component[column]
            if column == "wind_speed_10m_sensitivity_by_roughness_m":
                values = values.map(lambda item: json.dumps(item, sort_keys=True))
            result[output_name] = values.to_numpy()
        warning_rule = _load_disagreement_rule(run_dir)
        if warning_rule is not None:
            comparison = compare_direct_and_component(
                result["direct_core"],
                result["component_utci_c"],
                warning_rule=warning_rule,
            )
            result["direct_component_absolute_disagreement_c"] = comparison[
                "absolute_disagreement_c"
            ].to_numpy()
            result["direct_component_disagreement_warning"] = comparison[
                "disagreement_warning"
            ].to_numpy(dtype=bool)

    satellite_eligible: pd.Series | None = None
    if set(policy.satellite_enhanced_additions).issubset(observations.columns):
        satellite_eligible = satellite_eligibility_mask(
            observations, project.model, feature_config=project.features
        )
        result["satellite_eligible"] = satellite_eligible.to_numpy(dtype=bool)
        result["baseline_satellite_lst_alone"] = np.nan
        result.loc[satellite_eligible, "baseline_satellite_lst_alone"] = pd.to_numeric(
            observations.loc[satellite_eligible, "satellite_lst_c"], errors="coerce"
        ).to_numpy(dtype=float)

    enhanced_path = run_dir / "models/direct_satellite_enhanced.joblib"
    eligible_core_path = run_dir / "models/direct_core_eligible.joblib"
    if enhanced_path.is_file() or eligible_core_path.is_file():
        if not (enhanced_path.is_file() and eligible_core_path.is_file()):
            raise ArtifactIntegrityError(
                "Enhanced and eligible-core model artifacts must exist as a pair."
            )
        eligible = (
            satellite_eligible
            if satellite_eligible is not None
            else satellite_eligibility_mask(
                observations, project.model, feature_config=project.features
            )
        )
        result["satellite_eligible"] = eligible.to_numpy(dtype=bool)
        result["direct_satellite_enhanced"] = np.nan
        result["direct_core_eligible"] = np.nan
        if "baseline_satellite_lst_alone" not in result:
            result["baseline_satellite_lst_alone"] = np.nan
        if eligible.any():
            enhanced_model: FittedDirectModel = joblib.load(enhanced_path)
            eligible_core_model: FittedDirectModel = joblib.load(eligible_core_path)
            eligible_rows = observations.loc[eligible]
            result.loc[eligible, "direct_satellite_enhanced"] = enhanced_model.predict(
                eligible_rows
            )
            result.loc[eligible, "direct_core_eligible"] = eligible_core_model.predict(
                eligible_rows
            )
            result.loc[eligible, "baseline_satellite_lst_alone"] = pd.to_numeric(
                eligible_rows["satellite_lst_c"], errors="coerce"
            ).to_numpy(dtype=float)
    return result.reset_index(drop=True)


def _conformal_subgroups(
    frame: pd.DataFrame, project: ProjectConfig
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "observed_heat_category": derive_utci_categories(frame[TARGET])
    }
    subgroup_config = project.model.get("evaluation", {}).get("subgroup_columns", {})
    sun_column = str(subgroup_config.get("sun_shade", "sun_shade_group"))
    land_column = str(subgroup_config.get("land_cover", "land_cover_class"))
    coast_column = str(subgroup_config.get("coast_distance", "coast_distance_group"))
    time_column = str(subgroup_config.get("time_of_day", "time_of_day_group"))
    if sun_column in frame:
        values["sun_vs_shade"] = frame[sun_column]
    elif "estimated_shade_fraction" in frame:
        shade = pd.to_numeric(frame["estimated_shade_fraction"], errors="coerce")
        values["sun_vs_shade"] = np.where(
            shade.isna(), "missing", np.where(shade >= 0.5, "shade", "sun")
        )
    if land_column in frame:
        values["land_cover"] = frame[land_column]
    if coast_column in frame:
        values["coast_distance"] = frame[coast_column]
    elif "distance_to_coast_m" in frame:
        coast = pd.to_numeric(frame["distance_to_coast_m"], errors="coerce") / 1000.0
        values["coast_distance"] = pd.cut(
            coast,
            bins=[0.0, 5.0, 20.0, np.inf],
            labels=["0_to_5_km", "5_to_20_km", "ge_20_km"],
            right=False,
            include_lowest=True,
        ).astype("string")
    if time_column in frame:
        values["time_of_day"] = frame[time_column]
    else:
        hours = _local_hour_values(frame)
        if hours is not None:
            values["time_of_day"] = time_of_day_groups(hours)
    return values


def run_final_test(args: Any) -> int:
    """Verify frozen hashes, explicitly unlock, and evaluate the final role once."""

    project, policy, data, report = _validate_observations(args)
    print(report.format_text())
    report.raise_for_errors()
    attached, _ = _read_attached(args, project, data)
    current_hashes = _current_hashes(args, project)
    lock_name = str(
        project.model.get("paths", {}).get(
            "final_test_lock_name", "final_test.lock.json"
        )
    )
    lock = FinalTestLock.from_json(args.run_dir / lock_name)
    authorize_final_test(
        unlock_requested=bool(args.unlock_final_test),
        current_hashes=current_hashes,
        lock=lock,
    )
    metrics_path = args.run_dir / "final_test/metrics.json"
    _refuse_existing(metrics_path, "Locked final-test result")
    final = _role(attached, "final_test", project)
    predictions = _all_runtime_predictions(
        final,
        run_dir=args.run_dir,
        project=project,
        policy=policy,
    )
    predictions.insert(1, TARGET, final[TARGET].to_numpy())
    for column in (
        "direct_core",
        "component_utci_c",
        "direct_satellite_enhanced",
        "direct_core_eligible",
    ):
        if column in predictions:
            predictions[f"{column}_category"] = derive_utci_categories(
                predictions[column]
            )

    conformal_report: dict[str, Any] | None = None
    calibrator_path = args.run_dir / "calibration/conformal.joblib"
    if project.model.get("conformal", {}).get("enabled", True) and not calibrator_path.is_file():
        raise ArtifactIntegrityError(
            "Conformal inference is enabled but calibration/conformal.joblib is missing. "
            "Run calibrate on the untouched calibration role before final-test."
        )
    if calibrator_path.is_file():
        calibrator: SplitConformalCalibrator = joblib.load(calibrator_path)
        lower, upper = calibrator.predict_interval(predictions["direct_core"])
        predictions["direct_core_interval_lower_c"] = lower
        predictions["direct_core_interval_upper_c"] = upper
        conformal_report = conformal_coverage_report(
            final[TARGET],
            predictions["direct_core"],
            calibrator,
            subgroup_values=_conformal_subgroups(final, project),
        )
    metrics, subgroups, bootstrap, _ = _evaluation_bundle(
        final, predictions, project
    )
    store = ArtifactStore(args.run_dir).open_existing()
    store.write_dataframe("final_test/predictions.parquet", predictions)
    store.write_json("final_test/metrics.json", metrics)
    store.write_json("final_test/paired_block_bootstrap.json", bootstrap)
    if not subgroups.empty:
        store.write_dataframe("final_test/subgroup_metrics.csv", subgroups)
    if conformal_report is not None:
        store.write_json("final_test/conformal_coverage.json", conformal_report)
    store.write_json(
        "final_test/unlock_record.json",
        {
            "explicit_unlock": True,
            "lock_run_id": lock.run_id,
            "hashes": asdict(current_hashes),
            "one_time_evaluation_intent": True,
        },
    )
    LOGGER.info("Locked final-test evaluation completed once in %s", args.run_dir)
    return 0


def _verify_prediction_config(run_dir: Path, project: ProjectConfig) -> None:
    hash_path = run_dir / "hashes.json"
    if not hash_path.is_file():
        raise ArtifactIntegrityError(f"Run hash record is missing: {hash_path}")
    stored = json.loads(hash_path.read_text(encoding="utf-8"))
    if stored.get("model_config_sha256") != sha256_file(project.model_path):
        raise ArtifactIntegrityError("Prediction model configuration differs from training.")
    if stored.get("feature_allowlist_sha256") != sha256_file(project.features_path):
        raise ArtifactIntegrityError("Prediction feature allow-list differs from training.")


def run_predict(args: Any) -> int:
    """Predict an explicit operational table without searching for any input."""

    predictor_set = "satellite_enhanced" if args.model == "satellite_enhanced" else "core"
    project, _, data, report = _validate_observations(
        args,
        predictor_set=predictor_set,
        require_labels=False,
        require_split_role=False,
    )
    print(report.format_text())
    report.raise_for_errors()
    _verify_prediction_config(args.run_dir, project)
    _ensure_safe_destination(
        args.output, args.data, project.model_path, project.features_path
    )
    _refuse_existing(args.output, "Prediction output")
    model_path = (
        args.run_dir / "models/direct_satellite_enhanced.joblib"
        if args.model == "satellite_enhanced"
        else args.run_dir / "models/direct_core.joblib"
    )
    model: FittedDirectModel = _load_joblib(model_path, f"{args.model} model")
    result = pd.DataFrame({"sample_id": data["sample_id"].astype(str).to_numpy()})
    if args.model == "satellite_enhanced":
        eligible = satellite_eligibility_mask(
            data, project.model, feature_config=project.features
        )
        prediction = np.full(len(data), np.nan, dtype=float)
        if eligible.any():
            prediction[eligible.to_numpy()] = model.predict(data.loc[eligible])
        result["satellite_eligible"] = eligible.to_numpy(dtype=bool)
    else:
        prediction = model.predict(data)
    result["predicted_utci_c"] = prediction
    result["predicted_utci_category"] = derive_utci_categories(prediction)
    if args.model == "core" and (args.run_dir / "models/components.joblib").is_file():
        component_bundle: ComponentModelBundle = joblib.load(
            args.run_dir / "models/components.joblib"
        )
        component = _component_prediction_frame(component_bundle, data, project)
        for column in component.columns:
            output_name = (
                "component_predicted_utci_c"
                if column == "component_utci_c"
                else f"component_{column}"
            )
            values = component[column]
            if column == "wind_speed_10m_sensitivity_by_roughness_m":
                values = values.map(lambda item: json.dumps(item, sort_keys=True))
            result[output_name] = values.to_numpy()
        warning_rule = _load_disagreement_rule(args.run_dir)
        if warning_rule is not None:
            comparison = compare_direct_and_component(
                result["predicted_utci_c"],
                result["component_predicted_utci_c"],
                warning_rule=warning_rule,
            )
            result["direct_component_absolute_disagreement_c"] = comparison[
                "absolute_disagreement_c"
            ].to_numpy()
            result["direct_component_disagreement_warning"] = comparison[
                "disagreement_warning"
            ].to_numpy(dtype=bool)

    calibrator_path = args.run_dir / "calibration/conformal.joblib"
    finite = np.isfinite(prediction)
    if args.model == "core" and calibrator_path.is_file() and finite.any():
        calibrator: SplitConformalCalibrator = joblib.load(calibrator_path)
        lower = np.full(len(data), np.nan, dtype=float)
        upper = np.full(len(data), np.nan, dtype=float)
        finite_lower, finite_upper = calibrator.predict_interval(prediction[finite])
        lower[finite] = finite_lower
        upper[finite] = finite_upper
        result["prediction_interval_lower_c"] = lower
        result["prediction_interval_upper_c"] = upper

    range_path = (
        args.run_dir / "diagnostics/feature_ranges_satellite_enhanced.joblib"
        if args.model == "satellite_enhanced"
        else args.run_dir / "diagnostics/feature_ranges.joblib"
    )
    if range_path.is_file():
        profile: FeatureRangeProfile = joblib.load(range_path)
        counts = np.zeros(len(data), dtype=int)
        for feature, (minimum, maximum) in profile.ranges.items():
            if feature in data:
                values = pd.to_numeric(data[feature], errors="coerce").to_numpy(dtype=float)
                counts += ((values < minimum) | (values > maximum)).astype(int)
        result["feature_extrapolation_count"] = counts
    anomaly_path = args.run_dir / "diagnostics/anomaly_detector.joblib"
    if anomaly_path.is_file():
        anomaly = joblib.load(anomaly_path).score(data)
        result["anomaly_score"] = anomaly["anomaly_score"].to_numpy()
        result["is_anomaly"] = anomaly["is_anomaly"].to_numpy(dtype=bool)

    output_suffix = args.output.suffix.lower()
    if output_suffix not in {".csv", ".parquet", ".pq"}:
        raise ValueError("--output must end in .csv or .parquet.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if output_suffix == ".csv":
        result.to_csv(args.output, index=False)
    else:
        result.to_parquet(args.output, index=False)
    LOGGER.info("Predictions written to %s", args.output)
    return 0
