from __future__ import annotations

import logging
import urllib.request

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from .config import MNIST_URL, PipelineConfig

LOGGER = logging.getLogger(__name__)


def download_raw_dataset(config: PipelineConfig, force_refresh: bool) -> None:
    if config.raw_dataset_path.exists() and not force_refresh:
        LOGGER.info("Using cached raw dataset at %s", config.raw_dataset_path)
        return

    LOGGER.info("Downloading MNIST dataset to %s", config.raw_dataset_path)
    urllib.request.urlretrieve(MNIST_URL, config.raw_dataset_path)


def load_processed_dataset(config: PipelineConfig, force_refresh: bool) -> tuple[np.ndarray, ...]:
    if config.processed_dataset_path.exists() and not force_refresh:
        LOGGER.info("Using cached processed dataset at %s", config.processed_dataset_path)
        with np.load(config.processed_dataset_path) as dataset:
            return dataset["X_train"], dataset["X_test"], dataset["y_train"], dataset["y_test"]

    download_raw_dataset(config, force_refresh=force_refresh)

    LOGGER.info("Transforming raw dataset into flat float32 arrays")
    with np.load(config.raw_dataset_path) as dataset:
        x_train = dataset["x_train"].astype(np.float32) / 255.0
        x_test = dataset["x_test"].astype(np.float32) / 255.0
        y_train = dataset["y_train"].astype(np.uint8)
        y_test = dataset["y_test"].astype(np.uint8)

    X_train = x_train.reshape((x_train.shape[0], -1))
    X_test = x_test.reshape((x_test.shape[0], -1))

    np.savez_compressed(
        config.processed_dataset_path,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    return X_train, X_test, y_train, y_test


def stratified_sample_indices(y: np.ndarray, sample_size: int, random_state: int) -> np.ndarray:
    if sample_size >= len(y):
        return np.arange(len(y))

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=sample_size,
        random_state=random_state,
    )
    sample_indices, _ = next(splitter.split(np.zeros((len(y), 1)), y))
    return sample_indices
