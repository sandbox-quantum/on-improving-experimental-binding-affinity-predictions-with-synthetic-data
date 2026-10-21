#!/usr/bin/env python3
"""
pcm_pipeline.py — Self-contained PCM training/inference pipeline.

Implements three model types:
  - RF:  sklearn RandomForestRegressor with tree-std uncertainty
  - MVE: Mean-Variance Estimator (FC network, NLL loss)
  - GMN: Gaussian Mixture Network (ensemble of MVEs)

Includes Bayesian hyperparameter optimization (RF surrogate + EI acquisition)
on a discrete grid.

Usage:
  # Training:
  python pcm_pipeline.py train \\
      --train-csv train.csv \\
      --target-col potency \\
      --smiles-col SMILES \\
      --protein-col sequence \\
      --protein-feature-table esm2_embeddings.parquet \\
      --extra-feature-cols docking_min aqfep_dg \\
      --models GMN MVE RF \\
      --metric spearman \\
      --bo-init 1 --bo-iters 9 \\
      --output-dir workspace/

  # Inference:
  python pcm_pipeline.py predict \\
      --test-csv test.csv \\
      --workspace workspace/ \\
      --output-csv predictions.csv
"""

import argparse
import gzip
import itertools
import logging
import os
import pickle
import sys
import threading
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor as SklearnRF
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU = torch.device("cpu")

# ═══════════════════════════════════════════════════════════════════════════════
# Featurization
# ═══════════════════════════════════════════════════════════════════════════════


def smiles_to_morgan(smiles_list, n_bits=2048, radius=2, chirality=True, counts=True):
    """Generate Morgan (ECFP) fingerprints from SMILES using RDKit.

    Returns numpy array of shape (n_molecules, n_bits).
    """
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits, includeChirality=chirality
    )
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            logger.warning("Could not parse SMILES: %s — filling with zeros", smi)
            fps.append(np.zeros(n_bits, dtype=np.float32))
        else:
            if counts:
                fp = generator.GetCountFingerprint(mol)
                arr = np.zeros(n_bits, dtype=np.float32)
                for idx, val in fp.GetNonzeroElements().items():
                    arr[idx] = val
            else:
                fp = generator.GetFingerprint(mol)
                arr = np.zeros(n_bits, dtype=np.float32)
                for i in range(n_bits):
                    arr[i] = fp.GetBit(i)
            fps.append(arr)
    return np.vstack(fps)


def load_protein_features(df, protein_col, feature_table_path):
    """Load precomputed protein embeddings (e.g. ESM-2) from parquet/csv.

    The feature table has an ID column (first column) matching values in
    df[protein_col], and all other columns are embedding dimensions.
    """
    p = Path(feature_table_path)
    if p.suffix == ".parquet":
        feat_df = pd.read_parquet(p)
    else:
        feat_df = pd.read_csv(p)

    id_col = feat_df.columns[0]
    feat_cols = feat_df.columns[1:]
    lookup = {row[id_col]: row[feat_cols].values.astype(np.float32) for _, row in feat_df.iterrows()}

    n_feat = len(feat_cols)
    result = []
    for pid in df[protein_col]:
        if pid in lookup:
            result.append(lookup[pid])
        else:
            logger.warning("Protein %s not found in feature table — filling NaN", pid)
            result.append(np.full(n_feat, np.nan, dtype=np.float32))
    return np.vstack(result)


def build_features(df, smiles_col, protein_col, protein_feature_table, extra_feature_cols,
                   morgan_bits=2048, morgan_radius=2):
    """Build concatenated feature matrix: Morgan FP + protein embeddings + extra cols."""
    logger.info("Computing Morgan fingerprints (bits=%d, radius=%d) ...", morgan_bits, morgan_radius)
    X_morgan = smiles_to_morgan(df[smiles_col].values, n_bits=morgan_bits, radius=morgan_radius)

    X_protein = None
    if protein_feature_table:
        logger.info("Loading protein features from %s ...", protein_feature_table)
        X_protein = load_protein_features(df, protein_col, protein_feature_table)

    X_extra = None
    if extra_feature_cols:
        logger.info("Using extra feature columns: %s", extra_feature_cols)
        X_extra = df[extra_feature_cols].values.astype(np.float32)

    parts = [X_morgan]
    if X_protein is not None:
        parts.append(X_protein)
    if X_extra is not None:
        parts.append(X_extra)

    return np.hstack(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════


class Preprocessor:
    """Variance filter + feature scaler + target scaler."""

    def __init__(self, feature_scaler_cls=MinMaxScaler, target_scaler_cls=StandardScaler):
        self.feature_scaler_cls = feature_scaler_cls
        self.target_scaler_cls = target_scaler_cls
        self._var_filter = None
        self._feature_scaler = None
        self._target_scaler = None

    def fit_transform(self, X, y):
        # Remove NaN/Inf rows
        mask = np.isnan(X).any(axis=1) | (~np.isfinite(X)).any(axis=1) | np.isnan(y).ravel()
        if mask.any():
            logger.info("Removing %d rows with NaN/Inf", mask.sum())
            X, y = X[~mask], y[~mask]

        self._var_filter = VarianceThreshold()
        X = self._var_filter.fit_transform(X)

        self._feature_scaler = self.feature_scaler_cls()
        X = self._feature_scaler.fit_transform(X)

        y = y.reshape(-1, 1) if y.ndim == 1 else y
        self._target_scaler = self.target_scaler_cls()
        y = self._target_scaler.fit_transform(y)

        return X, y.ravel(), mask

    def transform(self, X, y=None):
        mask = np.isnan(X).any(axis=1) | (~np.isfinite(X)).any(axis=1)
        if y is not None:
            mask = mask | np.isnan(y).ravel()
        if mask.any():
            X = X[~mask]
            if y is not None:
                y = y[~mask]

        X = self._var_filter.transform(X)
        X = self._feature_scaler.transform(X)
        if y is not None:
            y = y.reshape(-1, 1) if y.ndim == 1 else y
            y = self._target_scaler.transform(y).ravel()
            return X, y, mask
        return X, mask

    def inverse_transform_predictions(self, mean, sigma=None):
        mean = mean.reshape(-1, 1)
        mean = self._target_scaler.inverse_transform(mean).ravel()
        if sigma is not None:
            sigma = sigma * self._target_scaler.scale_[0]
        return mean, sigma


# ═══════════════════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════════════════


# ── Random Forest ──────────────────────────────────────────────────────────────

class RFModel:
    """Random Forest with tree-std uncertainty."""

    PARAM_GRID = {
        "max_depth": [5, 10, 20, 40, None],
        "max_features": [0.3, 0.5, 1.0],
        "min_samples_split": [2, 3, 4, 8],
        "n_estimators": [100, 200, 300],
    }

    def __init__(self, **kwargs):
        self.params = kwargs
        self.model = None

    def fit(self, X, y):
        self.model = SklearnRF(
            n_jobs=-1,
            random_state=0,
            **self.params,
        )
        self.model.fit(X, y.ravel())

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        tree_preds = np.array([t.predict(X) for t in self.model.estimators_])
        mean = tree_preds.mean(axis=0)
        sigma = tree_preds.std(axis=0)
        return mean, sigma


# ── MVE (Mean-Variance Estimator) ─────────────────────────────────────────────

class MVENet(nn.Module):
    """FC network with BatchNorm + Dropout + ReLU, outputting mean + variance."""

    def __init__(self, input_dim, hidden_dim=300, num_layers=2, num_common_layers=1,
                 batch_norm=True, dropout_rate=0.1, **kwargs):
        super().__init__()

        def add_layer(layers, in_d, out_d):
            layers.append(nn.Linear(in_d, out_d))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_d))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        common_layers = []
        add_layer(common_layers, input_dim, hidden_dim)

        if num_common_layers >= num_layers:
            for _ in range(num_layers - 2):
                add_layer(common_layers, hidden_dim, hidden_dim)
            common_layers.append(nn.Linear(hidden_dim, 2))
            self.common = nn.Sequential(*common_layers)
            self.mean_branch = None
            self.var_branch = None
        else:
            for _ in range(num_common_layers - 1):
                add_layer(common_layers, hidden_dim, hidden_dim)
            self.common = nn.Sequential(*common_layers)

            mean_hd = hidden_dim // 2
            var_hd = hidden_dim - mean_hd
            mean_layers, var_layers = [], []
            add_layer(mean_layers, hidden_dim, mean_hd)
            add_layer(var_layers, hidden_dim, var_hd)
            for _ in range(num_layers - num_common_layers - 2):
                add_layer(mean_layers, mean_hd, mean_hd)
                add_layer(var_layers, var_hd, var_hd)
            mean_layers.append(nn.Linear(mean_hd, 1))
            var_layers.append(nn.Linear(var_hd, 1))
            self.mean_branch = nn.Sequential(*mean_layers)
            self.var_branch = nn.Sequential(*var_layers)

    def forward(self, x):
        h = self.common(x)
        if self.mean_branch is not None:
            mean = self.mean_branch(h).squeeze(-1)
            var = torch.exp(self.var_branch(h).squeeze(-1))
        else:
            mean, var = h[:, 0], torch.exp(h[:, 1])
        return mean, var


def nll_loss(y, mean, var):
    """Negative log-likelihood loss for Gaussian."""
    keep = ~torch.isnan(y)
    if not torch.any(keep):
        return torch.tensor(0.0, device=y.device)
    nll = (torch.log(var) + (y - mean).pow(2) / var) / 2
    return nll[keep].sum()


class MVEModel:
    """MVE training wrapper."""

    PARAM_GRID = {
        "hidden_dim": [100, 200, 300],
        "num_layers": [2, 3, 4],
        "num_common_layers": [1, 2],
        "epochs": [10, 20, 30, 40, 50],
        "lr": [0.00001, 0.0001, 0.0005, 0.001],
        "dropout_rate": [0.0, 0.1, 0.5],
        "batch_norm": [True, False],
    }

    def __init__(self, **kwargs):
        self.params = kwargs
        self.net = None

    def fit(self, X, y):
        p = self.params
        self.net = MVENet(
            input_dim=X.shape[1],
            hidden_dim=p.get("hidden_dim", 300),
            num_layers=p.get("num_layers", 2),
            num_common_layers=p.get("num_common_layers", 1),
            batch_norm=p.get("batch_norm", True),
            dropout_rate=p.get("dropout_rate", 0.1),
        ).to(DEVICE)

        epochs = p.get("epochs", 30)
        lr = p.get("lr", 0.0001)
        batch_size = min(p.get("batch_size", 100), len(y))

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.net.train()
        for epoch in range(epochs):
            torch.manual_seed(epoch)
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                mean, var = self.net(xb)
                loss = nll_loss(yb, mean, var)
                loss.backward()
                optimizer.step()

    def predict(self, X):
        X_t = torch.tensor(X, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)
        self.net.eval()
        means, variances = [], []
        with torch.inference_mode():
            for (xb,) in loader:
                xb = xb.to(DEVICE)
                m, v = self.net(xb)
                means.append(m.cpu().numpy())
                variances.append(v.cpu().numpy())
        mean = np.concatenate(means)
        sigma = np.sqrt(np.concatenate(variances))
        return mean, sigma


# ── GMN (Gaussian Mixture Network) ────────────────────────────────────────────

class GMNNet(nn.Module):
    """Ensemble of MVE sub-networks with Gaussian mixture pooling."""

    def __init__(self, input_dim, n_estimators=2, hidden_layers=(300, 300),
                 delta=10, batch_norm=True, dropout_rate=0.0, random_state=0, **kwargs):
        super().__init__()
        self.n_estimators = n_estimators
        self.estimators = nn.ModuleList()
        for i in range(n_estimators):
            # Perturb hidden layer sizes
            sizes = []
            for ls in hidden_layers:
                np.random.seed(random_state + i * 5 + len(sizes))
                sizes.append(max(np.random.choice(range(max(50, ls - delta), ls + delta + 1)), 50))

            net = nn.Sequential()
            in_d = input_dim
            for j, hs in enumerate(sizes):
                net.add_module(f"linear{j}", nn.Linear(in_d, hs))
                if batch_norm:
                    net.add_module(f"bn{j}", nn.BatchNorm1d(hs))
                net.add_module(f"relu{j}", nn.ReLU())
                net.add_module(f"drop{j}", nn.Dropout(dropout_rate))
                in_d = hs
            net.add_module(f"output", nn.Linear(in_d, 2))
            self.estimators.append(net)

    def forward(self, x):
        all_means, all_vars = [], []
        for est in self.estimators:
            out = est(x)
            mean, var = out[:, 0], nn.functional.softplus(out[:, 1])
            all_means.append(mean)
            all_vars.append(var)

        means = torch.stack(all_means)   # (K, N)
        vars_ = torch.stack(all_vars)    # (K, N)

        # Gaussian mixture: mu = mean(mu_k), var = mean(var_k + mu_k^2) - mu^2
        pooled_mean = means.mean(dim=0)
        pooled_var = (vars_ + means.pow(2)).mean(dim=0) - pooled_mean.pow(2)
        return pooled_mean, pooled_var


class GMNModel:
    """GMN training wrapper."""

    PARAM_GRID = {
        "n_estimators": [2, 5],
        "hidden_layers": [(100,), (300,), (100, 100), (300, 300), (100, 100, 100), (300, 300, 300)],
        "delta": [10, 50],
        "batch_norm": [True, False],
        "dropout_rate": [0.0, 0.1],
        "epochs": [20, 100],
        "lr": [0.00001, 0.0001, 0.0005, 0.001],
        "valid_size": [0.0, 0.05],
    }

    def __init__(self, **kwargs):
        self.params = kwargs
        self.net = None

    def fit(self, X, y):
        p = self.params
        self.net = GMNNet(
            input_dim=X.shape[1],
            n_estimators=p.get("n_estimators", 2),
            hidden_layers=p.get("hidden_layers", (300, 300)),
            delta=p.get("delta", 10),
            batch_norm=p.get("batch_norm", True),
            dropout_rate=p.get("dropout_rate", 0.0),
            random_state=p.get("random_state", 0),
        ).to(DEVICE)

        epochs = p.get("epochs", 20)
        lr = p.get("lr", 0.001)
        batch_size = min(p.get("batch_size", 100), len(y))

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.net.train()
        for epoch in range(epochs):
            torch.manual_seed(epoch)
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                mean, var = self.net(xb)
                loss = nll_loss(yb, mean, var)
                loss.backward()
                optimizer.step()

    def predict(self, X):
        X_t = torch.tensor(X, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)
        self.net.eval()
        means, variances = [], []
        with torch.inference_mode():
            for (xb,) in loader:
                xb = xb.to(DEVICE)
                m, v = self.net(xb)
                means.append(m.cpu().numpy())
                variances.append(v.cpu().numpy())
        mean = np.concatenate(means)
        sigma = np.sqrt(np.abs(np.concatenate(variances)))
        return mean, sigma


MODEL_REGISTRY = {
    "RF": RFModel,
    "MVE": MVEModel,
    "GMN": GMNModel,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Bayesian Hyperparameter Optimization
# ═══════════════════════════════════════════════════════════════════════════════


def _expected_improvement(mu, sigma, best_score):
    """Expected improvement acquisition function."""
    from scipy.stats import norm
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (mu - best_score) / (sigma + 1e-9)
        ei = (mu - best_score) * norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma < 1e-9] = 0.0
    return ei


def bayesian_optimize(model_cls, X_train, y_train, metric_fn, bo_init=1, bo_iters=9,
                      random_state=0):
    """Bayesian optimization over model_cls.PARAM_GRID.

    Uses an RF surrogate with Expected Improvement over the discrete grid.
    Returns best hyperparams dict and best score.
    """
    param_grid = model_cls.PARAM_GRID
    keys = sorted(param_grid.keys())
    all_combos = list(itertools.product(*[param_grid[k] for k in keys]))

    if not all_combos:
        # No hyperparams to tune
        return {}, 0.0

    logger.info("BO: %d grid points for %s", len(all_combos), model_cls.__name__)

    # Encode grid points as integer indices for surrogate
    param_to_idx = {k: {v: i for i, v in enumerate(param_grid[k])} for k in keys}

    def encode(combo):
        return [param_to_idx[k][v] for k, v in zip(keys, combo)]

    def combo_to_dict(combo):
        return {k: v for k, v in zip(keys, combo)}

    def evaluate(params_dict):
        """Train/val split evaluation."""
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.05, random_state=random_state
        )
        model = model_cls(**params_dict)
        model.fit(X_tr, y_tr)
        mean, _ = model.predict(X_val)
        return metric_fn(y_val, mean)

    # Track evaluated points
    evaluated_indices = set()
    observations_X = []
    observations_y = []

    # Random initialization
    rng = np.random.RandomState(random_state)
    init_indices = rng.choice(len(all_combos), size=min(bo_init, len(all_combos)), replace=False)
    for idx in init_indices:
        combo = all_combos[idx]
        params = combo_to_dict(combo)
        logger.info("BO init: %s", params)
        score = evaluate(params)
        logger.info("  score = %.4f", score)
        observations_X.append(encode(combo))
        observations_y.append(score)
        evaluated_indices.add(idx)

    # Also evaluate default params (first combo)
    if 0 not in evaluated_indices:
        combo = all_combos[0]
        params = combo_to_dict(combo)
        logger.info("BO default: %s", params)
        score = evaluate(params)
        logger.info("  score = %.4f", score)
        observations_X.append(encode(combo))
        observations_y.append(score)
        evaluated_indices.add(0)

    # BO iterations
    for iteration in range(bo_iters):
        obs_X = np.array(observations_X, dtype=np.float32)
        obs_y = np.array(observations_y)
        best_score = obs_y.max()

        # Fit RF surrogate
        surrogate = SklearnRF(n_estimators=100, random_state=random_state, n_jobs=-1)
        surrogate.fit(obs_X, obs_y)

        # Compute EI over remaining grid points
        remaining = [i for i in range(len(all_combos)) if i not in evaluated_indices]
        if not remaining:
            logger.info("BO: exhausted grid after %d iterations", iteration)
            break

        remaining_encoded = np.array([encode(all_combos[i]) for i in remaining], dtype=np.float32)

        # Predict with each tree for mean + std
        tree_preds = np.array([t.predict(remaining_encoded) for t in surrogate.estimators_])
        mu = tree_preds.mean(axis=0)
        sigma = tree_preds.std(axis=0)

        ei = _expected_improvement(mu, sigma, best_score)
        best_ei_idx = remaining[np.argmax(ei)]

        combo = all_combos[best_ei_idx]
        params = combo_to_dict(combo)
        logger.info("BO iter %d/%d: %s", iteration + 1, bo_iters, params)
        score = evaluate(params)
        logger.info("  score = %.4f (best=%.4f)", score, max(best_score, score))
        observations_X.append(encode(combo))
        observations_y.append(score)
        evaluated_indices.add(best_ei_idx)

    # Return best
    best_idx = int(np.argmax(observations_y))
    best_grid_combo = all_combos[list(evaluated_indices)[best_idx]] if best_idx < len(evaluated_indices) else all_combos[0]
    # Safer: reconstruct from observations
    best_combo_encoded = observations_X[best_idx]
    # Decode
    best_params = {}
    for i, k in enumerate(keys):
        val_idx = int(best_combo_encoded[i])
        best_params[k] = param_grid[k][val_idx]

    return best_params, observations_y[best_idx]


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════


def compute_spearman(y_true, y_pred):
    corr, _ = spearmanr(y_true, y_pred)
    return corr if np.isfinite(corr) else 0.0


def compute_metrics(y_true, y_pred):
    """Return dict of Spearman, Pearson, R², RMSE."""
    sp = compute_spearman(y_true, y_pred)
    pearson = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"spearman": sp, "pearson": pearson, "r2": r2, "rmse": rmse}


METRIC_FNS = {
    "spearman": compute_spearman,
    "r2": lambda yt, yp: r2_score(yt, yp),
    "neg_mse": lambda yt, yp: -mean_squared_error(yt, yp),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


class PCMPipeline:
    """Full PCM pipeline: featurize → preprocess → (BO) train → predict."""

    def __init__(self, smiles_col="SMILES", protein_col="sequence", target_col="potency",
                 protein_feature_table=None, extra_feature_cols=None,
                 models=("GMN", "MVE", "RF"), metric="spearman",
                 bo_init=1, bo_iters=9, morgan_bits=2048, morgan_radius=2):
        self.smiles_col = smiles_col
        self.protein_col = protein_col
        self.target_col = target_col
        self.protein_feature_table = protein_feature_table
        self.extra_feature_cols = extra_feature_cols or []
        self.model_names = list(models)
        self.metric = metric
        self.bo_init = bo_init
        self.bo_iters = bo_iters
        self.morgan_bits = morgan_bits
        self.morgan_radius = morgan_radius

        # Fitted components (saved to workspace)
        self.preprocessor = None
        self.best_model = None
        self.best_model_name = None
        self.best_params = None
        self.best_score = None
        # For reproducing featurization at inference
        self._feature_config = None

    def train(self, train_csv):
        df = pd.read_csv(train_csv)
        logger.info("Loaded training data: %d rows from %s", len(df), train_csv)

        X_raw = build_features(
            df, self.smiles_col, self.protein_col,
            self.protein_feature_table, self.extra_feature_cols,
            self.morgan_bits, self.morgan_radius,
        )
        y_raw = df[self.target_col].values.astype(np.float32)

        self.preprocessor = Preprocessor()
        X, y, drop_mask = self.preprocessor.fit_transform(X_raw, y_raw)
        logger.info("After preprocessing: %d samples, %d features", X.shape[0], X.shape[1])

        self._feature_config = {
            "smiles_col": self.smiles_col,
            "protein_col": self.protein_col,
            "protein_feature_table": self.protein_feature_table,
            "extra_feature_cols": self.extra_feature_cols,
            "morgan_bits": self.morgan_bits,
            "morgan_radius": self.morgan_radius,
        }

        metric_fn = METRIC_FNS[self.metric]

        overall_best_score = -np.inf
        overall_best_model = None
        overall_best_name = None
        overall_best_params = None

        for model_name in self.model_names:
            logger.info("=== Optimizing %s ===", model_name)
            model_cls = MODEL_REGISTRY[model_name]

            best_params, best_score = bayesian_optimize(
                model_cls, X, y, metric_fn,
                bo_init=self.bo_init, bo_iters=self.bo_iters,
            )
            logger.info("Best %s: score=%.4f, params=%s", model_name, best_score, best_params)

            if best_score > overall_best_score:
                overall_best_score = best_score
                overall_best_name = model_name
                overall_best_params = best_params

        # Retrain best model on full training data
        logger.info("=== Retraining best model: %s (score=%.4f) ===", overall_best_name, overall_best_score)
        model_cls = MODEL_REGISTRY[overall_best_name]
        self.best_model = model_cls(**overall_best_params)
        self.best_model.fit(X, y)
        self.best_model_name = overall_best_name
        self.best_params = overall_best_params
        self.best_score = overall_best_score

        logger.info("Training complete. Best model: %s", overall_best_name)

    def predict(self, test_csv, output_csv=None):
        df = pd.read_csv(test_csv)
        logger.info("Loaded test data: %d rows from %s", len(df), test_csv)

        cfg = self._feature_config
        X_raw = build_features(
            df, cfg["smiles_col"], cfg["protein_col"],
            cfg["protein_feature_table"], cfg["extra_feature_cols"],
            cfg["morgan_bits"], cfg["morgan_radius"],
        )

        X, drop_mask = self.preprocessor.transform(X_raw)
        mean, sigma = self.best_model.predict(X)
        mean, sigma = self.preprocessor.inverse_transform_predictions(mean, sigma)

        # Build output dataframe
        out_df = df[~drop_mask].copy() if drop_mask.any() else df.copy()
        out_df["prediction"] = mean
        if sigma is not None:
            out_df["uncertainty"] = sigma

        if output_csv:
            out_df.to_csv(output_csv, index=False)
            logger.info("Predictions saved to %s", output_csv)

        return out_df

    def save(self, workspace_dir):
        os.makedirs(workspace_dir, exist_ok=True)
        state = {
            "preprocessor": self.preprocessor,
            "best_model": self.best_model,
            "best_model_name": self.best_model_name,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "feature_config": self._feature_config,
        }
        path = os.path.join(workspace_dir, "pipeline.pkl.gz")
        with gzip.open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info("Pipeline saved to %s", path)

    def load(self, workspace_dir):
        path = os.path.join(workspace_dir, "pipeline.pkl.gz")
        with gzip.open(path, "rb") as f:
            state = pickle.load(f)
        self.preprocessor = state["preprocessor"]
        self.best_model = state["best_model"]
        self.best_model_name = state["best_model_name"]
        self.best_params = state["best_params"]
        self.best_score = state["best_score"]
        self._feature_config = state["feature_config"]
        logger.info("Pipeline loaded from %s (model=%s, score=%.4f)",
                     path, self.best_model_name, self.best_score)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def cli_train(args):
    pipeline = PCMPipeline(
        smiles_col=args.smiles_col,
        protein_col=args.protein_col,
        target_col=args.target_col,
        protein_feature_table=args.protein_feature_table,
        extra_feature_cols=args.extra_feature_cols,
        models=args.models,
        metric=args.metric,
        bo_init=args.bo_init,
        bo_iters=args.bo_iters,
    )
    pipeline.train(args.train_csv)
    pipeline.save(args.output_dir)


def cli_predict(args):
    pipeline = PCMPipeline()
    pipeline.load(args.workspace)
    pipeline.predict(args.test_csv, output_csv=args.output_csv)


def main():
    parser = argparse.ArgumentParser(description="PCM Pipeline (public libraries)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── train ──
    p_train = subparsers.add_parser("train", help="Train PCM pipeline with BO")
    p_train.add_argument("--train-csv", required=True)
    p_train.add_argument("--target-col", default="potency")
    p_train.add_argument("--smiles-col", default="SMILES")
    p_train.add_argument("--protein-col", default="sequence")
    p_train.add_argument("--protein-feature-table", default=None,
                         help="Path to .parquet or .csv with precomputed protein embeddings")
    p_train.add_argument("--extra-feature-cols", nargs="*", default=None)
    p_train.add_argument("--models", nargs="+", default=["GMN", "MVE", "RF"],
                         choices=["GMN", "MVE", "RF"])
    p_train.add_argument("--metric", default="spearman", choices=["spearman", "r2", "neg_mse"])
    p_train.add_argument("--bo-init", type=int, default=1)
    p_train.add_argument("--bo-iters", type=int, default=9)
    p_train.add_argument("--output-dir", required=True)

    # ── predict ──
    p_pred = subparsers.add_parser("predict", help="Run inference with trained pipeline")
    p_pred.add_argument("--test-csv", required=True)
    p_pred.add_argument("--workspace", required=True, help="Directory with pipeline.pkl.gz")
    p_pred.add_argument("--output-csv", required=True)

    args = parser.parse_args()
    if args.command == "train":
        cli_train(args)
    elif args.command == "predict":
        cli_predict(args)


if __name__ == "__main__":
    main()
