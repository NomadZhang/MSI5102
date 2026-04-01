# MNIST Digit Recognition

Poetry-based ETL and modeling pipeline for handwritten digit recognition on the MNIST dataset.

## What it does

- Downloads the canonical `mnist.npz` dataset and caches a processed flat version locally
- Visualizes sample digits
- Trains three classifiers:
  - k-Nearest Neighbors
  - Logistic Regression
  - Neural Network (`MLPClassifier`)
- Produces 2D PCA and t-SNE visualizations
- Benchmarks raw pixels against several PCA-reduced feature sets
- Writes model metrics and prediction outputs to CSV

## Run

```bash
poetry install
poetry run mnist-pipeline
```

Generated artifacts are written under `artifacts/`.

## Notebook

An equivalent notebook lives at `notebooks/mnist_pipeline.ipynb`.

```bash
poetry run jupyter notebook
```
