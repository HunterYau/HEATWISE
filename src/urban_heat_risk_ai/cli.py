"""Command-line interface for explicit, real-data-only workflows."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from urban_heat_risk_ai.config import load_yaml_config
from urban_heat_risk_ai.errors import UrbanHeatRiskError
from urban_heat_risk_ai.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_CONFIG = Path("configs/model.yaml")
DEFAULT_FEATURE_CONFIG = Path("configs/features.yaml")


def _data_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Explicit path to the real CSV or Parquet observation table (never auto-discovered).",
    )


def _manifest_argument(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument(
        "--manifest",
        required=required,
        type=Path,
        help="Explicit split-manifest path.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the side-effect-free CLI parser."""

    parser = argparse.ArgumentParser(
        prog="urban_heat_risk_ai",
        description=(
            "Leakage-safe urban UTCI modelling. Commands never search for data; "
            "observation paths are always explicit."
        ),
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_MODEL_CONFIG, help="Model YAML configuration."
    )
    parser.add_argument(
        "--features-config",
        type=Path,
        default=DEFAULT_FEATURE_CONFIG,
        help="Version-controlled predictor allow-list YAML.",
    )
    parser.add_argument(
        "--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    log_format = parser.add_mutually_exclusive_group()
    log_format.add_argument(
        "--json-logs",
        dest="json_logs",
        action="store_true",
        default=None,
        help="Emit structured JSON logs.",
    )
    log_format.add_argument(
        "--plain-logs",
        dest="json_logs",
        action="store_false",
        help="Override configuration and emit concise plain-text logs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="Validate a real table without modifying it or writing artifacts."
    )
    _data_argument(validate)
    validate.add_argument(
        "--mode",
        choices=["training", "prediction"],
        default="training",
        help="Prediction mode permits absent target/label-only columns.",
    )

    make_splits = subparsers.add_parser(
        "make-splits", help="Create a deterministic real-data split manifest."
    )
    _data_argument(make_splits)
    _manifest_argument(make_splits)
    make_splits.add_argument(
        "--overwrite-manifest",
        action="store_true",
        help="Replace only the explicitly named manifest after re-validating invariants.",
    )

    tune = subparsers.add_parser(
        "tune", help="Run leakage-safe blocked inner tuning on development observations."
    )
    _data_argument(tune)
    _manifest_argument(tune)
    tune.add_argument("--output-dir", required=True, type=Path)
    tune.add_argument("--n-trials", type=int, default=None)
    tune.add_argument(
        "--predictor-set", choices=["core", "satellite_enhanced"], default="core"
    )

    train = subparsers.add_parser(
        "train", help="Fit frozen models on development rows using selected settings."
    )
    _data_argument(train)
    _manifest_argument(train)
    train.add_argument("--tuning-result", required=True, type=Path)
    train.add_argument(
        "--enhanced-tuning-result",
        type=Path,
        default=None,
        help=(
            "Required with --include-satellite-enhanced; result of a separate "
            "satellite_enhanced tune run."
        ),
    )
    train.add_argument("--output-dir", required=True, type=Path)
    train.add_argument(
        "--include-satellite-enhanced",
        action="store_true",
        help="Also fit a separate enhanced model on quality/image-age-eligible rows.",
    )
    train.add_argument(
        "--skip-components",
        action="store_true",
        help="Skip the four configured component models for this runtime run.",
    )

    calibrate = subparsers.add_parser(
        "calibrate", help="Fit split-conformal intervals on the untouched calibration role."
    )
    _data_argument(calibrate)
    _manifest_argument(calibrate)
    calibrate.add_argument("--run-dir", required=True, type=Path)

    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate stored development out-of-fold predictions."
    )
    _data_argument(evaluate)
    _manifest_argument(evaluate)
    evaluate.add_argument("--run-dir", required=True, type=Path)
    evaluate.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Optional explicit OOF prediction table; otherwise use the run artifact.",
    )
    evaluate.add_argument(
        "--local-explanation-sample-id",
        dest="local_explanation_sample_ids",
        action="append",
        default=[],
        help=(
            "Development sample ID for a local SHAP waterfall; repeat for multiple "
            "explicitly selected observations."
        ),
    )

    final_test = subparsers.add_parser(
        "final-test", help="Open the hash-locked final test exactly when explicitly authorized."
    )
    _data_argument(final_test)
    _manifest_argument(final_test)
    final_test.add_argument("--run-dir", required=True, type=Path)
    final_test.add_argument(
        "--unlock-final-test",
        action="store_true",
        help="Confirm that all model and analysis decisions are frozen.",
    )

    predict = subparsers.add_parser(
        "predict", help="Predict UTCI for an explicit operational feature table."
    )
    _data_argument(predict)
    predict.add_argument("--run-dir", required=True, type=Path)
    predict.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Explicit .csv or .parquet destination for runtime predictions.",
    )
    predict.add_argument(
        "--model", choices=["core", "satellite_enhanced"], default="core"
    )

    return parser


def _dispatch(args: argparse.Namespace) -> int:
    # Heavy scientific imports are deliberately delayed so --help is instantaneous
    # and side-effect free.
    from urban_heat_risk_ai import workflows

    handlers = {
        "validate": workflows.run_validate,
        "make-splits": workflows.run_make_splits,
        "tune": workflows.run_tune,
        "train": workflows.run_train,
        "calibrate": workflows.run_calibrate,
        "evaluate": workflows.run_evaluate,
        "final-test": workflows.run_final_test,
        "predict": workflows.run_predict,
    }
    return int(handlers[args.command](args))


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, report expected errors clearly, and return a process code."""

    args = build_parser().parse_args(argv)
    configured_logging: dict[str, object] = {}
    try:
        loaded = load_yaml_config(args.config)
        candidate = loaded.get("logging", {})
        if isinstance(candidate, dict):
            configured_logging = candidate
    except UrbanHeatRiskError:
        # Dispatch reports the actionable configuration error after logging exists.
        pass
    level = args.log_level or str(configured_logging.get("level", "INFO"))
    json_logs = (
        bool(configured_logging.get("structured", False))
        if args.json_logs is None
        else bool(args.json_logs)
    )
    configure_logging(level, json_logs=json_logs)
    try:
        return _dispatch(args)
    except (UrbanHeatRiskError, FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        LOGGER.error("Interrupted; no additional workflow stage will be started.")
        return 130
    except Exception:
        LOGGER.exception("Unexpected failure")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
