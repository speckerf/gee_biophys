import numpy as np
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Custom transformer that converts specified columns (angles) to their cosines
class AngleTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # No fitting necessary for cosine transformation
        return self

    def transform(self, X):
        # X is a numpy array here, not a DataFrame
        return np.cos(np.deg2rad(X))


class NIRvTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # No fitting necessary for cosine transformation
        return self

    def transform(self, X):
        assert X.shape[1] == 10, f"Expected 10 bands, got {X.shape[1]}"
        # problem:
        # X is a numpy array here, not a DataFrame

        # add random number to avoid division by zero
        X = X + np.abs(np.random.rand(*X.shape) * 1e-10)

        NIR = X[["B8"]].values.reshape(-1)
        RED = X[["B4"]].values.reshape(-1)

        # check that NIR + RED is not 0
        if np.any(NIR + RED == 0):
            logger.warning("NIR + RED is 0 for some samples, set to mean sum")
            NIR[NIR + RED == 0] = np.mean(NIR)
            RED[NIR + RED == 0] = np.mean(RED)

        NDVI = (NIR - RED) / (NIR + RED)
        NIRv = NDVI * NIR

        if np.any(NIRv == 0):
            logger.warning(
                "NIRv is 0 for some samples, setting to mean of non-zero values",
            )
            NIRv[NIRv == 0] = np.mean(NIRv[NIRv != 0])

        # divide each row by the corresponding NIRv value
        X_normalized = X / NIRv[:, np.newaxis]
        return X_normalized


# Define the safe logit function
def safe_logit(x):
    # add warning when this clipping was used
    if np.any(x <= 1e-9) or np.any(x >= 1 - 1e-9):
        logger.trace("Clipping values to avoid 0 and 1")
        x = np.clip(x, 1e-9, 1 - 1e-9)  # Clip the values to avoid 0 and 1
    return np.log(x / (1 - x))


# Define the inverse sigmoid function
def safe_inverse_logit(x):
    return 1 / (1 + np.exp(-x))


def identity(x):
    return x


def get_pipeline(model: BaseEstimator, config: dict) -> Pipeline:
    angles = ["tts", "tto", "psi"]
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

    angle_transformer = Pipeline(steps=[("angle_transformer", AngleTransformer())])
    if config["nirv_norm"]:
        band_transformer = Pipeline(
            steps=[
                ("nirv_transformer", NIRvTransformer()),
                ("scaler", StandardScaler()),
            ],
        )
    else:
        band_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("band_transformer", band_transformer, bands),
            ("angle_transformer", angle_transformer, angles),
        ],
        remainder="passthrough",
    )

    if config["transform_target"] == "log1p":
        regressor = TransformedTargetRegressor(
            regressor=model,
            func=np.log1p,
            inverse_func=np.expm1,
        )
    elif config["transform_target"] == "standard":
        regressor = TransformedTargetRegressor(
            regressor=model,
            transformer=StandardScaler(),
        )
    elif config["transform_target"] == "None":
        regressor = TransformedTargetRegressor(
            regressor=model,
            func=identity,
            inverse_func=identity,
        )
    elif config["transform_target"] == "logit":
        regressor = TransformedTargetRegressor(
            regressor=model,
            func=safe_logit,
            inverse_func=safe_inverse_logit,
        )
    else:
        raise ValueError(f"Unknown target transformation: {config['transform_target']}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ],
    )

    return pipeline
