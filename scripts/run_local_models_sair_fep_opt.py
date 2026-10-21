#!/usr/bin/env python3
"""
run_local_models_sair_fep_opt.py — Train per-protein local models on SAIR-FEP-OPT.

Reproduces the "AFEP-Linear" and "XGBoost" baselines from Figure 3 of the paper.
For each protein with >= 10 ligands in the test set:
  - Linear regression on AFEP ΔG values
  - XGBoost on Morgan fingerprints (1024 bits, radius 2)

Usage:
    python run_local_models_sair_fep_opt.py \\
        --train-csv <sair_fep_opt_train.csv> \\
        --test-csv  <sair_fep_opt_test.csv> \\
        --output-dir <output_dir> \\
        --min-ligands 10
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy import stats
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


def morgan_fp(smiles: str, nbits: int = 1024, radius: int = 2) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return np.array(fp, dtype=np.float32)


def per_protein_metrics(y_true, y_pred):
    if len(y_true) < 3:
        return {}
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    spearman = float(stats.spearmanr(y_true, y_pred).correlation)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"pearson": pearson, "spearman": spearman, "rmse": rmse}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-ligands", type=int, default=10)
    parser.add_argument("--protein-col", default="sequence")
    parser.add_argument("--smiles-col", default="SMILES")
    parser.add_argument("--target-col", default="potency")
    parser.add_argument("--afep-col", default="aqfep_dg")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    results_rows = []
    proteins = test_df[args.protein_col].unique()
    print(f"Evaluating {len(proteins)} unique proteins in test set...")

    for prot in proteins:
        train_sub = train_df[train_df[args.protein_col] == prot]
        test_sub = test_df[test_df[args.protein_col] == prot]

        if len(test_sub) < args.min_ligands:
            continue

        y_test = test_sub[args.target_col].values

        # ── AFEP-Linear baseline ──────────────────────────────────────────────
        if args.afep_col in train_sub.columns and args.afep_col in test_sub.columns:
            tr_mask = train_sub[args.afep_col].notna()
            te_mask = test_sub[args.afep_col].notna()
            if tr_mask.sum() >= 3 and te_mask.sum() >= 3:
                lr = LinearRegression()
                lr.fit(
                    train_sub.loc[tr_mask, args.afep_col].values.reshape(-1, 1),
                    train_sub.loc[tr_mask, args.target_col].values,
                )
                y_pred_lr = lr.predict(
                    test_sub.loc[te_mask, args.afep_col].values.reshape(-1, 1)
                )
                m = per_protein_metrics(
                    test_sub.loc[te_mask, args.target_col].values, y_pred_lr
                )
                if m:
                    results_rows.append(
                        {"protein": prot, "model": "AFEP-Linear", "n_test": int(te_mask.sum()), **m}
                    )

        # ── XGBoost on Morgan fingerprints ────────────────────────────────────
        train_fps = np.array(
            [fp for fp in train_sub[args.smiles_col].map(morgan_fp) if fp is not None]
        )
        test_fps = np.array(
            [fp for fp in test_sub[args.smiles_col].map(morgan_fp) if fp is not None]
        )

        if len(train_fps) >= 5 and len(test_fps) >= args.min_ligands:
            # Filter out rows with None fps
            tr_valid = [morgan_fp(s) is not None for s in train_sub[args.smiles_col]]
            te_valid = [morgan_fp(s) is not None for s in test_sub[args.smiles_col]]
            y_train_xgb = train_sub.loc[tr_valid, args.target_col].values
            y_test_xgb = test_sub.loc[te_valid, args.target_col].values

            xgb = XGBRegressor(
                max_depth=4, n_estimators=100, learning_rate=0.1, random_state=42
            )
            xgb.fit(train_fps, y_train_xgb)
            y_pred_xgb = xgb.predict(test_fps)

            m = per_protein_metrics(y_test_xgb, y_pred_xgb)
            if m:
                results_rows.append(
                    {"protein": prot, "model": "XGBoost", "n_test": len(test_fps), **m}
                )

    results_df = pd.DataFrame(results_rows)
    out_csv = outdir / "local_model_results.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"Saved {len(results_df)} rows to {out_csv}")

    # Summary
    for model_name in results_df["model"].unique():
        sub = results_df[results_df["model"] == model_name]
        print(f"\n{model_name} ({len(sub)} proteins):")
        for metric in ["pearson", "spearman", "rmse"]:
            vals = sub[metric].dropna()
            print(f"  {metric}: mean={vals.mean():.4f} median={vals.median():.4f}")


if __name__ == "__main__":
    main()
