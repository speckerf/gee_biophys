# scripts/groundedeo_conversion/convert_to_onnx.py

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import sklearn
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import DoubleTensorType
from sklearn.gaussian_process import GaussianProcessRegressor

logger = logging.getLogger(__name__)


N_FEATURES = 15


def load_model(model_path: Path):
    """Load the original GROUNDED-EO sklearn model."""
    logger.info("Loading model: %s", model_path)
    logger.info("scikit-learn version: %s", sklearn.__version__)

    model = joblib.load(model_path)

    logger.info("Loaded object type: %s", type(model))
    logger.info("Model:\n%s", model)

    return model


def convert_model(
    model,
    output_path: Path,
    n_features: int = N_FEATURES,
) -> None:
    """Convert a fitted sklearn model to ONNX."""
    # skl2onnx uses this example input to determine input dtype and dimensions.
    # Batch dimension remains dynamic.
    X_sample = np.zeros(
        (1, n_features),
        dtype=np.float32,
    )

    logger.info(
        "Converting model to ONNX with %d input features",
        n_features,
    )

    # Important: initialize sklearn's uncertainty-related internals
    model.predict(
        X_sample,
        return_std=True,
    )

    initial_types = [("X", DoubleTensorType([None, n_features]))]

    options = {
        GaussianProcessRegressor: {
            "return_std": True,
        }
    }

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_types,
        options=options,
        target_opset=18,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_bytes(onnx_model.SerializeToString())

    logger.info("Saved ONNX model: %s", output_path)


def predict_onnx(
    model_path: Path,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict mean and standard deviation using an ONNX GPR model."""
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    inputs = session.get_inputs()
    outputs = session.get_outputs()

    logger.info(
        "ONNX inputs: %s",
        [(x.name, x.shape, x.type) for x in inputs],
    )
    logger.info(
        "ONNX outputs: %s",
        [(x.name, x.shape, x.type) for x in outputs],
    )

    if len(inputs) != 1:
        raise RuntimeError(f"Expected one ONNX input, found {len(inputs)}.")

    input_name = inputs[0].name

    result = session.run(
        None,
        {
            input_name: np.asarray(
                X,
                dtype=np.float64,
            ),
        },
    )

    if len(result) != 2:
        raise RuntimeError(f"Expected two ONNX outputs, found {len(result)}.")

    pred_mean = np.asarray(result[0]).squeeze()
    pred_std = np.asarray(result[1]).squeeze()

    return pred_mean, pred_std


def validate_conversion(
    sklearn_model,
    onnx_path: Path,
    X: np.ndarray,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    """Compare sklearn and ONNX mean and uncertainty predictions."""
    X = np.asarray(X, dtype=np.float64)

    logger.info(
        "Validating conversion using %d samples",
        X.shape[0],
    )

    sk_mean, sk_std = sklearn_model.predict(
        X,
        return_std=True,
    )

    onnx_mean, onnx_std = predict_onnx(
        onnx_path,
        X,
    )

    sk_mean = np.asarray(sk_mean).squeeze()
    sk_std = np.asarray(sk_std).squeeze()

    if sk_mean.shape != onnx_mean.shape:
        raise RuntimeError(
            "Mean prediction shapes differ: "
            f"sklearn={sk_mean.shape}, "
            f"onnx={onnx_mean.shape}"
        )

    if sk_std.shape != onnx_std.shape:
        raise RuntimeError(
            "Std prediction shapes differ: "
            f"sklearn={sk_std.shape}, "
            f"onnx={onnx_std.shape}"
        )

    mean_diff = np.abs(sk_mean - onnx_mean)
    std_diff = np.abs(sk_std - onnx_std)

    logger.info(
        "Mean prediction comparison:"
        "\n  max abs difference:    %.10g"
        "\n  mean abs difference:   %.10g"
        "\n  median abs difference: %.10g",
        np.max(mean_diff),
        np.mean(mean_diff),
        np.median(mean_diff),
    )

    logger.info(
        "Std prediction comparison:"
        "\n  max abs difference:    %.10g"
        "\n  mean abs difference:   %.10g"
        "\n  median abs difference: %.10g",
        np.max(std_diff),
        np.mean(std_diff),
        np.median(std_diff),
    )

    np.testing.assert_allclose(
        sk_mean,
        onnx_mean,
        rtol=rtol,
        atol=atol,
    )

    np.testing.assert_allclose(
        sk_std,
        onnx_std,
        rtol=rtol,
        atol=atol,
    )

    logger.info("Validation passed for mean and standard deviation.")


def make_validation_data(
    model,
    n_features: int,
    n_samples: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Create fallback validation inputs.

    If the estimator contains its original training inputs, use their
    empirical feature ranges. Otherwise generate simple random values.

    Prefer supplying real GROUNDED-EO/Sentinel-2 inputs via --validation-data.
    """
    rng = np.random.default_rng(seed)

    # GaussianProcessRegressor stores fitted training predictors as X_train_.
    if hasattr(model, "X_train_"):
        X_train = np.asarray(model.X_train_)

        if X_train.ndim == 2 and X_train.shape[1] == n_features:
            logger.info("Using stored GPR X_train_ to construct validation inputs.")

            n = min(
                n_samples,
                X_train.shape[0],
            )

            indices = rng.choice(
                X_train.shape[0],
                size=n,
                replace=False,
            )

            return X_train[indices]

    logger.warning(
        "Could not obtain training inputs from the model. "
        "Using synthetic random validation data. "
        "For final validation, --validation-data is recommended."
    )

    return rng.random(
        (n_samples, n_features),
        dtype=np.float64,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a GROUNDED-EO sklearn model to ONNX and validate the conversion."
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the original joblib/pickle model.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the resulting ONNX model.",
    )

    parser.add_argument(
        "--n-features",
        type=int,
        default=N_FEATURES,
        help=f"Number of model predictors (default: {N_FEATURES}).",
    )

    parser.add_argument(
        "--validation-data",
        type=Path,
        default=None,
        help=(
            "Optional .npy file containing validation predictors "
            "with shape (n_samples, n_features)."
        ),
    )

    parser.add_argument(
        "--n-validation",
        type=int,
        default=100,
        help=("Number of samples used for automatic validation (default: 100)."),
    )

    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="Relative tolerance for prediction comparison.",
    )

    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for prediction comparison.",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    args = parse_args()

    model = load_model(args.model)

    convert_model(
        model=model,
        output_path=args.output,
        n_features=args.n_features,
    )

    if args.validation_data is not None:
        logger.info(
            "Loading validation data: %s",
            args.validation_data,
        )

        X_validation = np.load(args.validation_data)
    else:
        X_validation = make_validation_data(
            model=model,
            n_features=args.n_features,
            n_samples=args.n_validation,
        )

    if X_validation.ndim != 2:
        raise ValueError(
            f"Validation data must be two-dimensional; got shape {X_validation.shape}."
        )

    if X_validation.shape[1] != args.n_features:
        raise ValueError(
            f"Expected {args.n_features} features, got {X_validation.shape[1]}."
        )

    validate_conversion(
        sklearn_model=model,
        onnx_path=args.output,
        X=X_validation,
        rtol=args.rtol,
        atol=args.atol,
    )

    logger.info("Conversion complete.")


if __name__ == "__main__":
    main()
