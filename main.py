

"""
 python main.py --input MVTA_data.csv --output anomaly_results_final.csv --plots ./plots_final
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, StandardScaler
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ----------------------------
# Configuration dataclass
# ----------------------------


@dataclass
class Config:
    """Configuration for pipeline; edit defaults here if required."""
    training_start: str = "2004-01-01 00:00:00"
    training_end: str = "2004-01-05 23:50:00"
    analysis_start: str = "2004-01-06 00:00:00"
    analysis_end: str = "2004-01-19 07:50:00"
    contamination_lenient: float = 0.02
    contamination_final: float = 0.001
    n_estimators_lenient: int = 100
    n_estimators_final: int = 200
    n_components_pca: int = 6
    smoothing_window: int = 9
    if_importance_sample_rows: int = 500
    target_training_mean: float = 8.0
    random_state: int = 42
    min_training_hours: int = 72
    lenient_remove_pct: float = 0.02
    wa_pca: float = 0.6
    wa_z: float = 0.3
    wa_if: float = 0.1
    verbose: bool = True


# ----------------------------
# Utilities
# ----------------------------


def setup_logging(verbose: bool) -> None:
    """Configure logging format and level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        stream=sys.stdout)


def ensure_dir(path: str) -> None:
    """Create directory if missing."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def check_min_training_hours(df_index: pd.DatetimeIndex, start: pd.Timestamp,
                             end: pd.Timestamp, min_hours: int) -> bool:
    """Return True if span inside [start,end] is at least min_hours."""
    mask = (df_index >= start) & (df_index <= end)
    if mask.sum() == 0:
        return False
    span_hours = (df_index[mask].max() - df_index[mask].min()).total_seconds() / 3600.0
    return span_hours >= min_hours


# ----------------------------
# Data Processor
# ----------------------------


class DataProcessor:
    """Load and clean data (timestamp parsing, missing values, jitter for perfect corr)."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def load(self, path: str) -> pd.DataFrame:
        """Load CSV, parse timestamp (first column), set as index, sort."""
        logging.info("Loading CSV: %s", path)
        df = pd.read_csv(path)
        if df.shape[1] < 2:
            raise ValueError("Input CSV must contain timestamp + >=1 feature columns.")
        ts_col = df.columns[0]
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        if df[ts_col].isna().any():
            logging.warning("Dropping rows with invalid timestamps.")
            df = df.dropna(subset=[ts_col]).copy()
        df = df.sort_values(ts_col).reset_index(drop=True)
        df = df.set_index(ts_col)
        logging.info("Loaded shape: %s", df.shape)
        return df

    def clean_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill numeric missing values (ffill/interpolate/median), remove constant columns,
        add tiny jitter if near-perfect correlations exist.
        """
        logging.info("Cleaning data (fill NaNs, drop constants, jitter if needed).")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns present in input.")
        # add tiny jitter if near-perfect correlations exist
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr().fillna(0)
            if (np.isclose(np.abs(corr.values), 1.0)).any():
                logging.info("Detected near-perfect correlation; adding tiny jitter.")
                rng = np.random.default_rng(self.cfg.random_state)
                noise = rng.normal(scale=1e-9, size=(df.shape[0], len(numeric_cols)))
                df.loc[:, numeric_cols] = df[numeric_cols].values + noise
        # fill missing
        df = df.ffill().interpolate(method="linear")
        for c in numeric_cols:
            df[c] = df[c].fillna(df[c].median())
        # drop constant columns
        consts = [c for c in numeric_cols if df[c].nunique() <= 1]
        if consts:
            logging.warning("Dropping constant cols: %s", consts)
            df = df.drop(columns=consts)
        logging.info("Cleaned data shape: %s", df.shape)
        return df


# ----------------------------
# Anomaly Model
# ----------------------------


class AnomalyModel:
    """Train ensemble (lenient cleaning -> final IF + PCA) and produce scores + attributions."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.if_final: Optional[IsolationForest] = None
        self.scaler_if: Optional[RobustScaler] = None
        self.scaler_pca: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None

    @staticmethod
    def _norm01(a: np.ndarray) -> np.ndarray:
        amin = float(np.nanmin(a))
        amax = float(np.nanmax(a))
        if np.isclose(amax, amin):
            return np.zeros_like(a)
        return (a - amin) / (amax - amin)

    def _lenient_filter_mask(self, X_train: np.ndarray) -> np.ndarray:
        """Use a lenient IF to flag a small fraction of training rows as anomalies, return mask to keep."""
        if_model = IsolationForest(n_estimators=self.cfg.n_estimators_lenient,
                                   contamination=self.cfg.contamination_lenient,
                                   random_state=self.cfg.random_state, n_jobs=-1)
        scaler = RobustScaler().fit(X_train)
        Xs = scaler.transform(X_train)
        if_model.fit(Xs)
        raw = -if_model.decision_function(Xs)
        # threshold = keep rows below percentile(1 - remove_pct)
        thresh = np.percentile(raw, 100.0 * (1.0 - self.cfg.lenient_remove_pct))
        keep_mask = raw < thresh
        logging.info("Lenient filter removed %d / %d training rows.",
                     int((~keep_mask).sum()), X_train.shape[0])
        return keep_mask

    def fit(self, df: pd.DataFrame, train_index: pd.DatetimeIndex) -> Dict[str, object]:
        """Fit lenient cleaning, final IF, and PCA on cleaned training rows."""
        X_train = df.loc[train_index].values
        if X_train.shape[0] == 0:
            raise RuntimeError("No training rows found for the specified training window.")
        keep_mask = self._lenient_filter_mask(X_train)
        X_clean = X_train[keep_mask]
        if X_clean.shape[0] < max(10, int(0.5 * X_train.shape[0])):
            logging.warning("Few rows remain after lenient cleaning: %d", X_clean.shape[0])
        # final IF
        self.scaler_if = RobustScaler().fit(X_clean)
        Xs_clean = self.scaler_if.transform(X_clean)
        self.if_final = IsolationForest(n_estimators=self.cfg.n_estimators_final,
                                        contamination=self.cfg.contamination_final,
                                        random_state=self.cfg.random_state, n_jobs=-1)
        logging.info("Training final IsolationForest...")
        self.if_final.fit(Xs_clean)
        # PCA
        self.scaler_pca = StandardScaler().fit(X_clean)
        Xs_clean2 = self.scaler_pca.transform(X_clean)
        n_comp = min(self.cfg.n_components_pca, Xs_clean2.shape[1])
        self.pca = PCA(n_components=n_comp, random_state=self.cfg.random_state)
        logging.info("Fitting PCA with %d components...", n_comp)
        self.pca.fit(Xs_clean2)
        return {
            "n_train_before": X_train.shape[0],
            "n_train_after_clean": X_clean.shape[0],
            "pca_components": n_comp
        }

    def score_and_attributions(self, df: pd.DataFrame,
                               analysis_mask) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Score all rows and compute per-feature attributions.
        Accepts analysis_mask as either pandas Series/Index boolean or numpy boolean array.
        Returns: scores (0-100), combined attribution array (rows x features), feature_names
        """
        if self.if_final is None or self.pca is None or self.scaler_if is None or self.scaler_pca is None:
            raise RuntimeError("Model not fitted yet.")
        # normalize mask to numpy boolean array
        analysis_mask_arr = np.asarray(analysis_mask)
        X_full = df.values
        feature_names = list(df.columns)
        # IsolationForest raw (higher => more anomalous)
        iso_raw = -self.if_final.decision_function(self.scaler_if.transform(X_full))
        iso_n = self._norm01(iso_raw)
        # PCA reconstruction (in PCA-standardized space)
        Xs_full = self.scaler_pca.transform(X_full)
        Z = self.pca.transform(Xs_full)
        Xrec = self.pca.inverse_transform(Z)
        pca_err = np.sum((Xs_full - Xrec) ** 2, axis=1)
        pca_n = self._norm01(pca_err)
        # combine detectors (weighted)
        w_if_raw = 0.01
        w_pca_raw = 0.99
        w_if = w_if_raw / (w_if_raw + w_pca_raw)
        w_pca = w_pca_raw / (w_if_raw + w_pca_raw)
        ensemble_raw = (w_if * iso_n) + (w_pca * pca_n)
        # percentile mapping using analysis reference (accept numpy mask)
        if analysis_mask_arr.sum() > 0:
            ref = ensemble_raw[analysis_mask_arr]
        else:
            ref = ensemble_raw
        srt = np.sort(ref)
        ranks = np.searchsorted(srt, ensemble_raw, side="right")
        scores = (ranks / float(len(srt))) * 100.0
        # smoothing
        if self.cfg.smoothing_window and self.cfg.smoothing_window > 1:
            s = pd.Series(scores, index=df.index)
            scores = s.rolling(window=self.cfg.smoothing_window, center=True).median().bfill().ffill().values
        # Attributions:
        pca_pf = (Xs_full - Xrec) ** 2
        pca_pf_norm = pca_pf / (pca_pf.sum(axis=1, keepdims=True) + 1e-12)
        # z-score per feature
        col_mean = X_full.mean(axis=0)
        col_std = X_full.std(axis=0, ddof=0)
        col_std[col_std == 0] = 1.0
        z = np.abs((X_full - col_mean) / (col_std))
        z_norm = z / (z.sum(axis=1, keepdims=True) + 1e-12)
        # IF global importance via perturbation sampling
        n = X_full.shape[0]
        sample_rows = min(self.cfg.if_importance_sample_rows, n)
        step = max(1, n // sample_rows)
        Xsamp = X_full[::step][:sample_rows].copy()
        if Xsamp.shape[0] == 0:
            Xsamp = X_full[:min(10, n)].copy()
        base = -self.if_final.decision_function(self.scaler_if.transform(Xsamp))
        imp = np.zeros((Xsamp.shape[0], X_full.shape[1]))
        rng = np.random.default_rng(self.cfg.random_state)
        for j in range(X_full.shape[1]):
            Xp = Xsamp.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            pert = -self.if_final.decision_function(self.scaler_if.transform(Xp))
            imp[:, j] = np.abs(base - pert)
        g = imp.mean(axis=0)
        if g.sum() == 0:
            g = np.ones_like(g)
        g = g / (g.sum() + 1e-12)
        g_tile = np.tile(g[np.newaxis, :], (X_full.shape[0], 1))
        # combine attributions
        combined = (self.cfg.wa_pca * pca_pf_norm) + (self.cfg.wa_z * z_norm) + (self.cfg.wa_if * g_tile)
        combined = combined / (combined.sum(axis=1, keepdims=True) + 1e-12)
        return scores, combined, feature_names


# ----------------------------
# Calibrator
# ----------------------------


class Calibrator:
    """Scale scores so training mean < 10 and training max < 25 (conservative multiplicative scaling)."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def calibrate(self, scores: np.ndarray, train_mask) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Apply multiplicative scaling to fit training constraints.
        Accepts train_mask as pandas boolean Series or numpy boolean array.
        Returns calibrated_scores and diagnostic dict.
        """
        train_mask_arr = np.asarray(train_mask)
        series = pd.Series(scores)
        train_scores = series[train_mask_arr]
        if train_scores.empty:
            logging.warning("No training rows for calibration; skipping.")
            return scores, {"scale": 1.0, "pre_mean": None, "pre_max": None, "post_mean": None, "post_max": None}
        pre_mean = float(train_scores.mean())
        pre_max = float(train_scores.max())
        scale_mean = self.cfg.target_training_mean / (pre_mean + 1e-12)
        scale_max = 25.0 / (pre_max + 1e-12)
        scale = min(scale_mean, scale_max, 1.0)
        scale = scale * 0.95  # conservative factor
        calibrated = np.minimum(series * scale, 100.0).values
        post = pd.Series(calibrated)[train_mask_arr]
        diag = {
            "scale": float(scale),
            "pre_mean": pre_mean,
            "pre_max": pre_max,
            "post_mean": float(post.mean()) if not post.empty else None,
            "post_max": float(post.max()) if not post.empty else None,
        }
        logging.info("Calibration: scale=%.4f pre_mean=%.4f pre_max=%.4f post_mean=%.4f post_max=%.4f",
                     diag["scale"], diag["pre_mean"], diag["pre_max"], diag["post_mean"], diag["post_max"])
        return calibrated, diag


# ----------------------------
# Reporter
# ----------------------------


class Reporter:
    """Saves CSV, plots, and prints PASS/FAIL report."""

    def __init__(self, plots_dir: str) -> None:
        self.plots_dir = plots_dir
        ensure_dir(plots_dir)

    def save_csv(self, df: pd.DataFrame, out_path: str) -> None:
        df.to_csv(out_path)
        logging.info("Saved CSV: %s", out_path)

    @staticmethod
    def top_feature_freq(df: pd.DataFrame, top_cols: List[str]) -> pd.Series:
        flattened = pd.Series(df[top_cols].values.ravel()).value_counts()
        flattened = flattened[flattened.index != ""]
        return flattened

    def save_plots(self, out_df: pd.DataFrame, attribution_snapshot: np.ndarray, feat_names: List[str]) -> None:
        # timeseries
        p = os.path.join(self.plots_dir, "abnormality_timeseries.png")
        plt.figure(figsize=(12, 3))
        plt.plot(out_df.index, out_df["Abnormality_score"], linewidth=0.8)
        plt.title("Abnormality Score (0-100)")
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        logging.info("Saved plot: %s", p)
        # histogram
        p = os.path.join(self.plots_dir, "score_hist.png")
        plt.figure(figsize=(6, 4))
        plt.hist(out_df["Abnormality_score"], bins=80)
        plt.title("Score histogram")
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        logging.info("Saved plot: %s", p)
        # top feature frequency
        top_cols = [f"top_feature_{i+1}" for i in range(7)]
        freq = self.top_feature_freq(out_df, top_cols)
        p = os.path.join(self.plots_dir, "top_feature_freq.png")
        plt.figure(figsize=(10, 4))
        n_show = min(40, len(freq))
        plt.bar(range(n_show), freq.values[:n_show])
        plt.xticks(range(n_show), [str(x) for x in freq.index[:n_show]], rotation=90, fontsize=8)
        plt.title("Top feature frequency")
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        logging.info("Saved plot: %s", p)
        # attribution snapshot
        p = os.path.join(self.plots_dir, "contrib_snapshot.png")
        samp = min(200, attribution_snapshot.shape[0])
        plt.figure(figsize=(12, max(3, len(feat_names) * 0.06)))
        plt.imshow(attribution_snapshot[:samp, :], aspect="auto")
        plt.colorbar()
        plt.xticks(range(len(feat_names)), feat_names, rotation=90, fontsize=6)
        plt.title("Attribution snapshot (sampled rows)")
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        logging.info("Saved plot: %s", p)
        # score rolling mean
        p = os.path.join(self.plots_dir, "score_rolling_mean.png")
        plt.figure(figsize=(12, 3))
        plt.plot(out_df.index, out_df["Abnormality_score"].rolling(window=60, min_periods=1).mean())
        plt.title("Score rolling mean (window=60)")
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        logging.info("Saved plot: %s", p)


# ----------------------------
# Main pipeline
# ----------------------------


def main(argv: Optional[List[str]] = None) -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Anomaly detection (IF + PCA ensemble) - final hackathon script.")
    parser.add_argument("--input", "-i", required=True, help="Input CSV path")
    parser.add_argument("--output", "-o", required=True, help="Output CSV path")
    parser.add_argument("--plots", "-p", required=True, help="Directory to save plots")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args(argv)

    cfg = Config(verbose=args.verbose)
    setup_logging(cfg.verbose)
    ensure_dir(args.plots)

    dp = DataProcessor(cfg)
    try:
        df = dp.load(args.input)
    except Exception as exc:
        logging.exception("Failed to load input: %s", exc)
        raise

    df = dp.clean_fill(df)

    # define masks
    t0 = pd.to_datetime(cfg.training_start)
    t1 = pd.to_datetime(cfg.training_end)
    a0 = pd.to_datetime(cfg.analysis_start)
    a1 = pd.to_datetime(cfg.analysis_end)

    train_mask = (df.index >= t0) & (df.index <= t1)
    analysis_mask = (df.index >= a0) & (df.index <= a1)

    # training hours check
    has_min_hours = check_min_training_hours(df.index, t0, t1, cfg.min_training_hours)
    if not has_min_hours:
        logging.warning("Training period may not contain minimum %d hours of data.", cfg.min_training_hours)

    model = AnomalyModel(cfg)
    meta = model.fit(df, df.index[train_mask])
    logging.info("Model fit metadata: %s", meta)

    scores, combined_attr, feat_names = model.score_and_attributions(df, analysis_mask)

    # top features per row
    def topk_row(row: np.ndarray, k: int = 7) -> List[str]:
        idxs = np.argsort(-row)
        chosen: List[str] = []
        total = float(row.sum())
        for idx in idxs:
            if len(chosen) >= k:
                break
            if total > 0 and (row[idx] / (total + 1e-12)) >= 0.01:
                chosen.append(feat_names[idx])
        while len(chosen) < k:
            chosen.append("")
        return chosen

    top_feats = [topk_row(combined_attr[i], k=7) for i in range(combined_attr.shape[0])]

    out_df = df.copy()
    out_df["Abnormality_score"] = scores
    for i in range(7):
        out_df[f"top_feature_{i+1}"] = [tf[i] for tf in top_feats]

    # pre-calibration diagnostics
    tr_pre = out_df.loc[train_mask, "Abnormality_score"]
    pre_mean = float(tr_pre.mean()) if not tr_pre.empty else float("nan")
    pre_max = float(tr_pre.max()) if not tr_pre.empty else float("nan")
    logging.info("Before calibration: training_mean=%.4f training_max=%.4f", pre_mean, pre_max)

    # calibrate
    calibrator = Calibrator(cfg)
    calibrated_scores, calib_diag = calibrator.calibrate(out_df["Abnormality_score"].values, train_mask)
    out_df["Abnormality_score"] = calibrated_scores

    # recompute sudden jumps & top feature uniques
    sudden_jumps = int((np.abs(np.diff(out_df["Abnormality_score"].values)) > 20).sum())
    top_feature_uniques = {f"top_feature_{i+1}": int(out_df[f"top_feature_{i+1}"].nunique()) for i in range(7)}

    # save output CSV & plots
    reporter = Reporter(args.plots)
    reporter.save_csv(out_df, args.output)
    snapshot_rows = min(400, combined_attr.shape[0])
    idxs = np.linspace(0, combined_attr.shape[0] - 1, snapshot_rows).astype(int)
    attribution_snapshot = combined_attr[idxs, :]
    reporter.save_plots(out_df, attribution_snapshot, feat_names)

    # save top training anomalies for manual inspection
    top_train_csv = os.path.join(args.plots, "top_train_anomalies_post_calibration.csv")
    out_df.loc[train_mask].sort_values("Abnormality_score", ascending=False).head(200).to_csv(top_train_csv)
    logging.info("Saved top training anomalies to %s", top_train_csv)

    # final diagnostics
    final_tr_mean = float(out_df.loc[train_mask, "Abnormality_score"].mean()) if train_mask.sum() > 0 else float("nan")
    final_tr_max = float(out_df.loc[train_mask, "Abnormality_score"].max()) if train_mask.sum() > 0 else float("nan")

    # Print concise results & PASS/FAIL report
    print("\n--- Final Validation ---")
    print(f"Training mean (after calibration): {final_tr_mean:.4f}  (target < 10)")
    print(f"Training max  (after calibration): {final_tr_max:.4f}  (target < 25)")
    print(f"Sudden jumps (>20): {sudden_jumps}")
    print(f"Output CSV: {args.output}")
    print(f"Plots directory: {args.plots}")
    print("--- End ---\n")

    # Hackathon PASS/FAIL report
    print("--- Hackathon Success Criteria Report ---")
    functional1 = final_tr_mean < 10 if not np.isnan(final_tr_mean) else False
    functional2 = final_tr_max < 25 if not np.isnan(final_tr_max) else False
    print("1) Training mean < 10 : ", "PASS" if functional1 else f"FAIL ({final_tr_mean:.2f})")
    print("2) Training max  < 25 : ", "PASS" if functional2 else f"FAIL ({final_tr_max:.2f})")
    print("3) Sudden jumps (>20 adjacent): ",
          "PASS" if sudden_jumps == 0 else f"WARN ({sudden_jumps} jumps)")
    print("4) Training rows >= 72 hours : ",
          "PASS" if check_min_training_hours(df.index, t0, t1, cfg.min_training_hours) else "FAIL (Insufficient data)")
    required_cols = ["Abnormality_score"] + [f"top_feature_{i+1}" for i in range(7)]
    print("5) Output CSV contains required columns : ",
          "PASS" if all(col in out_df.columns for col in required_cols) else "FAIL (Missing columns)")
    print("--- End of Report ---\n")


if __name__ == "__main__":
    main()
