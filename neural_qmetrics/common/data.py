import os

import numpy as np
from sklearn.model_selection import train_test_split

from . import defs


def load_dataset(dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    X_path = defs.DATA_ROOT / dataset_name / "X.npy"
    y_path = defs.DATA_ROOT / dataset_name / "y.npy"

    return np.load(X_path), np.load(y_path)


def load_projection(dataset_name: str, projection_name: str) -> np.ndarray | None:
    X_proj_path = defs.DATA_ROOT / dataset_name / "proj" / projection_name / "X_proj.npy"

    if os.path.exists(X_proj_path):
        return np.load(X_proj_path)
    return None


def load_and_split_dataset(dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    X, y = load_dataset(dataset_name)

    X_high, _, y_high, _ = train_test_split(
        X, y, stratify=y, train_size=min(int(0.9 * X.shape[0]), 5_000), random_state=420
    )

    return X_high, y_high


def save_projection(dataset_name: str, projection_name: str, X_proj, fig=None) -> None:
    dir_path = defs.DATA_ROOT / dataset_name / "proj" / projection_name

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    np.save(defs.DATA_ROOT / dataset_name / "proj" / projection_name / "X_proj.npy", X_proj)

    fig.savefig(defs.DATA_ROOT / dataset_name / "proj" / projection_name / "X_proj.png")
