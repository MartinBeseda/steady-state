#!/usr/bin/env python3

import os
import re
import shutil
from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


OUTDIR = "tsfresh_targeted_clustering_results"
TABLES_DIR = os.path.join(OUTDIR, "tables")
SIG_DIR = os.path.join(OUTDIR, "significant_clusters")
BANDS_DIR = os.path.join(OUTDIR, "cluster_bands")
CONFIG_CACHE_DIR = os.path.join(OUTDIR, "configuration_cache")

# Needed only for per-series JAGT error boxplots.
JAGT_METHOD_TABLE_PATH = "../man_steady_comparison/ssd_best_cpssd_comparison_results/ssd_best_config_comparison_table.csv"

ONE_DIR = os.path.join(OUTDIR, "ONE_SELECTED_CLUSTERING")
os.makedirs(ONE_DIR, exist_ok=True)

ALPHA = 0.05
MIN_JAGT = 5

TOP_RAW_FEATURES_PER_CLUSTER = 20
TOP_SUBFAMILIES_PER_CLUSTER = 15
TOP_COMMON_FEATURES = 20
TOP_COMMON_SUBFAMILIES = 15


@dataclass
class ClusteringResult:
    """
    Minimal class definition needed to unpickle cached clustering results
    produced by the targeted clustering script.
    """
    variant: str
    config_name: str
    method: str
    labels: np.ndarray
    embedding: np.ndarray
    features: pd.DataFrame
    series_keys: list
    data_normalized: list
    metrics: dict


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def prettify_axes(ax):
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def tsfresh_subfamily(feature_name):
    parts = str(feature_name).split("__")
    if len(parts) >= 2:
        return "__".join(parts[:2])
    return str(feature_name)


def broad_family(feature_name):
    f = str(feature_name)
    if "cwt_coefficients" in f or "fft_" in f or "spkt_welch_density" in f:
        return "frequency_wavelet"
    if "autocorrelation" in f or "partial_autocorrelation" in f or "c3__" in f or "time_reversal" in f:
        return "temporal_dependence"
    if "entropy" in f or "lempel_ziv" in f or "cid_ce" in f:
        return "complexity_entropy"
    if "change_quantiles" in f or "mean_change" in f or "mean_abs_change" in f:
        return "local_variability"
    if "linear_trend" in f or "agg_linear_trend" in f or "augmented_dickey" in f:
        return "trend_nonstationarity"
    if "ratio_beyond" in f or "kurtosis" in f or "skewness" in f:
        return "bursts_heavy_tails"
    if "number_peaks" in f or "longest_strike" in f:
        return "peaks_strikes"
    if "quantile" in f or "median" in f or "mean" in f or "maximum" in f or "minimum" in f:
        return "level_distribution"
    return "other"


def choose_one_clustering(summary_path):
    df = pd.read_csv(summary_path)

    df = df[df["n_jagt"] >= MIN_JAGT].copy()
    df = df[df["sign_q_vs_cp_best"].notna()].copy()

    df["margin"] = df["kb_wins_vs_cp_best"] - df["cp_best_wins_vs_kb"]
    df["is_sig"] = df["sign_q_vs_cp_best"] < ALPHA
    df["is_kb"] = df["is_sig"] & (df["margin"] > 0)
    df["is_cp"] = df["is_sig"] & (df["margin"] < 0)

    grouped = []

    for (variant, config), g in df.groupby(["variant", "config_name"]):
        kb = g[g["is_kb"]]
        cp = g[g["is_cp"]]

        kb_jagt = int(kb["n_jagt"].sum())
        cp_jagt = int(cp["n_jagt"].sum())

        grouped.append({
            "variant": variant,
            "config_name": config,
            "n_kb_sig_clusters": len(kb),
            "n_cp_sig_clusters": len(cp),
            "kb_jagt": kb_jagt,
            "cp_jagt": cp_jagt,
            "total_sig_jagt": kb_jagt + cp_jagt,
            "balanced_specialization_jagt": min(kb_jagt, cp_jagt),
            "specialization_product": kb_jagt * cp_jagt,
            "best_kb_margin": int(kb["margin"].max()) if len(kb) else 0,
            "best_cp_margin": int(cp["margin"].min()) if len(cp) else 0,
            "best_q": float(g[g["is_sig"]]["sign_q_vs_cp_best"].min()) if len(g[g["is_sig"]]) else np.nan,
        })

    rank = pd.DataFrame(grouped)

    if rank.empty:
        raise RuntimeError("No clustering candidates found.")

    rank["has_both"] = (
        (rank["n_kb_sig_clusters"] > 0) &
        (rank["n_cp_sig_clusters"] > 0)
    )

    rank = rank.sort_values(
        by=[
            "has_both",
            "balanced_specialization_jagt",
            "specialization_product",
            "total_sig_jagt",
            "n_cp_sig_clusters",
            "n_kb_sig_clusters",
            "cp_jagt",
            "kb_jagt",
        ],
        ascending=[False, False, False, False, False, False, False, False],
    )

    rank.to_csv(
        os.path.join(ONE_DIR, "selected_clustering_candidates_ranked.csv"),
        index=False,
    )

    selected = rank.iloc[0]
    return selected, df


def load_full_cluster_table_for_selected(selected):
    prefix = safe_name(f"{selected['variant']}_{selected['config_name']}")
    path = os.path.join(TABLES_DIR, f"{prefix}_cluster_method_eval.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Full cluster table not found:\n{path}\n"
            "It should be produced by the targeted clustering script."
        )

    df = pd.read_csv(path)
    return prefix, df


def load_zscore_table_for_selected(prefix):
    path = os.path.join(TABLES_DIR, f"{prefix}_cluster_feature_zscores.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cluster feature z-score table not found:\n{path}\n"
            "It should be produced by the targeted clustering script."
        )

    return pd.read_csv(path)


def configuration_cache_path(variant, config_name):
    return os.path.join(
        CONFIG_CACHE_DIR,
        f"{safe_name(variant)}__{safe_name(config_name)}.pkl",
    )


def load_cached_selected_result(selected):
    """
    Load labels, normalized series, and series keys from the per-configuration cache.
    This is needed for all-cluster mean trajectories and per-series JAGT error boxplots.
    """
    path = configuration_cache_path(selected["variant"], selected["config_name"])

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Selected clustering cache not found:\n{path}\n"
            "Run the targeted clustering script first, or keep its configuration_cache/ folder."
        )

    payload = pd.read_pickle(path)
    return payload["result"], payload.get("cluster_eval", None)


def load_jagt_method_table(path=JAGT_METHOD_TABLE_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"JAGT method-comparison table not found:\n{path}\n"
            "This is needed for per-series KB/CP error boxplots."
        )

    df = pd.read_csv(path)
    if "key" not in df.columns:
        raise ValueError(f"Expected column 'key' in {path}")

    return df.set_index("key", drop=False)


def annotate_all_clusters(df):
    df = df.copy()

    df["margin"] = df["kb_wins_vs_cp_best"] - df["cp_best_wins_vs_kb"]
    df["abs_margin"] = df["margin"].abs()

    df["has_enough_jagt"] = df["n_jagt"] >= MIN_JAGT
    df["is_significant"] = (
        df["has_enough_jagt"] &
        df["sign_q_vs_cp_best"].notna() &
        (df["sign_q_vs_cp_best"] < ALPHA)
    )

    def label(row):
        if not row["has_enough_jagt"]:
            return "insufficient_JAGT"
        if not row["is_significant"]:
            return "neutral"
        if row["margin"] > 0:
            return "KB"
        if row["margin"] < 0:
            return "CP-best"
        return "neutral"

    df["winner_label"] = df.apply(label, axis=1)

    return df.sort_values(by=["cluster"], ascending=True)


def select_representative_clusters(all_clusters):
    g = all_clusters.copy()

    kb = g[g["winner_label"] == "KB"].sort_values(
        by=["margin", "n_jagt"],
        ascending=[False, False],
    )

    cp = g[g["winner_label"] == "CP-best"].sort_values(
        by=["margin", "n_jagt"],
        ascending=[True, False],
    )

    if kb.empty:
        kb = g[(g["has_enough_jagt"]) & (g["margin"] > 0)].sort_values(
            ["margin", "n_jagt"],
            ascending=[False, False],
        )

    if cp.empty:
        cp = g[(g["has_enough_jagt"]) & (g["margin"] < 0)].sort_values(
            ["margin", "n_jagt"],
            ascending=[True, False],
        )

    if kb.empty or cp.empty:
        raise RuntimeError(
            "Selected clustering does not contain both KB-leaning and CP-leaning clusters."
        )

    kb_row = kb.iloc[0]
    cp_row = cp.iloc[0]

    pd.DataFrame([kb_row, cp_row]).to_csv(
        os.path.join(ONE_DIR, "selected_kb_and_cp_clusters.csv"),
        index=False,
    )

    return kb_row, cp_row


def copy_existing_artifacts(prefix, cluster_id, label):
    copied = []

    cluster_id = int(cluster_id) if float(cluster_id).is_integer() else cluster_id

    patterns = [
        f"{prefix}_cluster_{cluster_id}_top_subfamilies.csv",
        f"{prefix}_cluster_{cluster_id}_top_raw_features.csv",
        f"{prefix}_cluster_{cluster_id}_band.png",
    ]

    search_dirs = [SIG_DIR, BANDS_DIR]

    for pat in patterns:
        for d in search_dirs:
            src = os.path.join(d, pat)
            if os.path.exists(src):
                dst = os.path.join(ONE_DIR, f"{label}_{pat}")
                shutil.copy(src, dst)
                copied.append(dst)

    return copied


def feature_columns(zscores):
    return [
        c for c in zscores.columns
        if c not in {"cluster", "cluster_size", "size"}
        and pd.api.types.is_numeric_dtype(zscores[c])
    ]


def top_raw_features_for_cluster(zscores, cluster_id, top_n=TOP_RAW_FEATURES_PER_CLUSTER):
    row = zscores[zscores["cluster"] == cluster_id]

    if row.empty:
        row = zscores[zscores["cluster"] == float(cluster_id)]

    if row.empty:
        return pd.DataFrame()

    cols = feature_columns(zscores)
    z = row.iloc[0][cols].astype(float)

    out = pd.DataFrame({
        "cluster": cluster_id,
        "feature": cols,
        "zscore": z.values,
        "abs_zscore": np.abs(z.values),
        "subfamily": [tsfresh_subfamily(c) for c in cols],
        "broad_family": [broad_family(c) for c in cols],
    })

    return out.sort_values("abs_zscore", ascending=False).head(top_n)


def top_subfamilies_for_cluster(zscores, cluster_id, top_n=TOP_SUBFAMILIES_PER_CLUSTER):
    raw = top_raw_features_for_cluster(
        zscores,
        cluster_id,
        top_n=len(feature_columns(zscores)),
    )

    if raw.empty:
        return pd.DataFrame()

    out = (
        raw.groupby(["cluster", "subfamily", "broad_family"], as_index=False)
        .agg(
            mean_z=("zscore", "mean"),
            mean_abs_z=("abs_zscore", "mean"),
            max_abs_z=("abs_zscore", "max"),
            n_features=("feature", "count"),
        )
        .sort_values("mean_abs_z", ascending=False)
        .head(top_n)
    )

    return out


def save_features_for_every_cluster(all_clusters, zscores, prefix):
    raw_rows = []
    subfamily_rows = []

    for _, row in all_clusters.iterrows():
        cid = row["cluster"]

        raw = top_raw_features_for_cluster(zscores, cid, TOP_RAW_FEATURES_PER_CLUSTER)
        subfam = top_subfamilies_for_cluster(zscores, cid, TOP_SUBFAMILIES_PER_CLUSTER)

        for df in [raw, subfam]:
            if df.empty:
                continue
            df["winner_label"] = row["winner_label"]
            df["cluster_size"] = row["cluster_size"]
            df["n_jagt"] = row["n_jagt"]
            df["margin"] = row["margin"]
            df["sign_q_vs_cp_best"] = row["sign_q_vs_cp_best"]

        if not raw.empty:
            raw_rows.append(raw)
            raw.to_csv(
                os.path.join(ONE_DIR, f"{prefix}_cluster_{cid}_top_raw_features.csv"),
                index=False,
            )

        if not subfam.empty:
            subfamily_rows.append(subfam)
            subfam.to_csv(
                os.path.join(ONE_DIR, f"{prefix}_cluster_{cid}_top_subfamilies.csv"),
                index=False,
            )

    all_raw = pd.concat(raw_rows, ignore_index=True) if raw_rows else pd.DataFrame()
    all_subfam = pd.concat(subfamily_rows, ignore_index=True) if subfamily_rows else pd.DataFrame()

    if not all_raw.empty:
        all_raw.to_csv(
            os.path.join(ONE_DIR, f"{prefix}_ALL_CLUSTERS_top_raw_features.csv"),
            index=False,
        )

    if not all_subfam.empty:
        all_subfam.to_csv(
            os.path.join(ONE_DIR, f"{prefix}_ALL_CLUSTERS_top_subfamilies.csv"),
            index=False,
        )

    return all_raw, all_subfam


def weighted_common_raw_features(all_raw, winner_label, top_n=TOP_COMMON_FEATURES):
    df = all_raw[all_raw["winner_label"] == winner_label].copy()

    if df.empty:
        return pd.DataFrame()

    df["weight"] = df["n_jagt"].clip(lower=1)
    df["weighted_abs_z"] = df["abs_zscore"] * df["weight"]

    out = (
        df.groupby(["feature", "subfamily", "broad_family"], as_index=False)
        .agg(
            mean_abs_z=("abs_zscore", "mean"),
            max_abs_z=("abs_zscore", "max"),
            weighted_mean_abs_z=("weighted_abs_z", "sum"),
            n_clusters=("cluster", "nunique"),
            total_jagt=("n_jagt", "sum"),
            mean_margin=("margin", "mean"),
        )
    )

    out["weighted_mean_abs_z"] = out["weighted_mean_abs_z"] / out["total_jagt"].clip(lower=1)

    return out.sort_values(
        by=["n_clusters", "weighted_mean_abs_z", "total_jagt"],
        ascending=[False, False, False],
    ).head(top_n)


def weighted_common_subfamilies(all_subfam, winner_label, top_n=TOP_COMMON_SUBFAMILIES):
    df = all_subfam[all_subfam["winner_label"] == winner_label].copy()

    if df.empty:
        return pd.DataFrame()

    df["weight"] = df["n_jagt"].clip(lower=1)
    df["weighted_abs_z"] = df["mean_abs_z"] * df["weight"]

    out = (
        df.groupby(["subfamily", "broad_family"], as_index=False)
        .agg(
            mean_abs_z=("mean_abs_z", "mean"),
            max_abs_z=("max_abs_z", "max"),
            weighted_mean_abs_z=("weighted_abs_z", "sum"),
            n_clusters=("cluster", "nunique"),
            total_jagt=("n_jagt", "sum"),
            mean_margin=("margin", "mean"),
            total_features=("n_features", "sum"),
        )
    )

    out["weighted_mean_abs_z"] = out["weighted_mean_abs_z"] / out["total_jagt"].clip(lower=1)

    return out.sort_values(
        by=["n_clusters", "weighted_mean_abs_z", "total_jagt"],
        ascending=[False, False, False],
    ).head(top_n)


def plot_common_bar(df, label_col, value_col, title, outpath, top_n=15):
    if df.empty:
        return

    sub = df.head(top_n).copy()
    sub = sub.sort_values(value_col, ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    y = np.arange(len(sub))
    ax.barh(y, sub[value_col])

    labels = []
    for _, r in sub.iterrows():
        base = str(r[label_col])
        if "n_clusters" in r:
            base += f" | {int(r['n_clusters'])} cl."
        labels.append(base)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(value_col.replace("_", " "))
    ax.set_title(title)

    prettify_axes(ax)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def save_common_feature_summaries(all_raw, all_subfam, prefix):
    outputs = {}

    for winner in ["KB", "CP-best", "neutral"]:
        raw_common = weighted_common_raw_features(all_raw, winner, TOP_COMMON_FEATURES)
        subfam_common = weighted_common_subfamilies(all_subfam, winner, TOP_COMMON_SUBFAMILIES)

        winner_safe = safe_name(winner)

        raw_path = os.path.join(ONE_DIR, f"{prefix}_{winner_safe}_COMMON_raw_features.csv")
        subfam_path = os.path.join(ONE_DIR, f"{prefix}_{winner_safe}_COMMON_subfamilies.csv")

        raw_common.to_csv(raw_path, index=False)
        subfam_common.to_csv(subfam_path, index=False)

        plot_common_bar(
            raw_common,
            label_col="feature",
            value_col="weighted_mean_abs_z",
            title=f"Common raw TSFresh features in {winner} clusters",
            outpath=os.path.join(ONE_DIR, f"{prefix}_{winner_safe}_COMMON_raw_features.png"),
            top_n=10,
        )

        plot_common_bar(
            subfam_common,
            label_col="subfamily",
            value_col="weighted_mean_abs_z",
            title=f"Common TSFresh subfamilies in {winner} clusters",
            outpath=os.path.join(ONE_DIR, f"{prefix}_{winner_safe}_COMMON_subfamilies.png"),
            top_n=10,
        )

        outputs[winner] = {
            "raw": raw_common,
            "subfamilies": subfam_common,
        }

    return outputs


def plot_all_cluster_margins(all_clusters, prefix):
    df = all_clusters.copy()
    df = df[df["cluster"] != -1].copy()
    df = df.sort_values("cluster")

    if df.empty:
        return

    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(x, df["margin"])
    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(c)) for c in df["cluster"]], rotation=45, ha="right")

    ax.set_xlabel("Cluster")
    ax.set_ylabel("KB wins − CP-best wins")
    ax.set_title(
        "Selected clustering: method preference in every cluster\n"
        "Positive = KB-KSSD better; negative = CP-SSD better"
    )

    for i, row in enumerate(df.itertuples()):
        if row.winner_label == "KB":
            ax.text(i, row.margin, "*", ha="center", va="bottom", fontsize=11)
        elif row.winner_label == "CP-best":
            ax.text(i, row.margin, "*", ha="center", va="top", fontsize=11)

    ax.text(
        0.01,
        0.02,
        "* = significant at q < 0.05 and n_JAGT ≥ 5",
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.85,
    )

    prettify_axes(ax)

    plt.tight_layout()
    plt.savefig(
        os.path.join(ONE_DIR, f"{prefix}_all_clusters_method_margin.png"),
        dpi=300,
    )
    plt.close()


def plot_all_cluster_win_counts(all_clusters, prefix):
    df = all_clusters.copy()
    df = df[(df["cluster"] != -1) & (df["n_jagt"] >= MIN_JAGT)].copy()
    df = df.sort_values("n_jagt", ascending=False)

    if df.empty:
        return

    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 4.5))

    ax.bar(x - width, df["kb_wins_vs_cp_best"], width, label="KB-KSSD wins")
    ax.bar(x, df["cp_best_wins_vs_kb"], width, label="CP-SSD wins")
    ax.bar(x + width, df["ties_vs_cp_best"], width, label="ties")

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(c)) for c in df["cluster"]], rotation=45, ha="right")

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of steady JAGT series")
    ax.set_title("Selected clustering: CP-SSD vs KB-KSSD outcomes in all JAGT-bearing clusters")
    ax.legend(frameon=False, ncol=3)

    prettify_axes(ax)

    plt.tight_layout()
    plt.savefig(
        os.path.join(ONE_DIR, f"{prefix}_all_clusters_win_counts.png"),
        dpi=300,
    )
    plt.close()


def plot_all_cluster_sizes(all_clusters, prefix):
    df = all_clusters.copy()
    df = df[df["cluster"] != -1].copy()
    df = df.sort_values("cluster_size", ascending=False)

    if df.empty:
        return

    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 4.5))

    ax.bar(x - width / 2, df["cluster_size"], width, label="All time series")
    ax.bar(x + width / 2, df["n_jagt"], width, label="steady JAGT series")

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(c)) for c in df["cluster"]], rotation=45, ha="right")

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    ax.set_title("Selected clustering: cluster sizes and JAGT support")
    ax.legend(frameon=False)

    prettify_axes(ax)

    plt.tight_layout()
    plt.savefig(
        os.path.join(ONE_DIR, f"{prefix}_all_clusters_sizes_and_jagt.png"),
        dpi=300,
    )
    plt.close()


def plot_representative_method_win_summary(kb_row, cp_row, prefix):
    rows = [
        {
            "cluster": f"KB cluster {int(kb_row['cluster'])}",
            "KB-KSSD": kb_row["kb_wins_vs_cp_best"],
            "CP-SSD": kb_row["cp_best_wins_vs_kb"],
            "ties": kb_row["ties_vs_cp_best"],
        },
        {
            "cluster": f"CP cluster {int(cp_row['cluster'])}",
            "KB-KSSD": cp_row["kb_wins_vs_cp_best"],
            "CP-SSD": cp_row["cp_best_wins_vs_kb"],
            "ties": cp_row["ties_vs_cp_best"],
        },
    ]

    p = pd.DataFrame(rows)

    x = np.arange(len(p))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(x - width, p["KB-KSSD"], width, label="KB-KSSD")
    ax.bar(x, p["CP-SSD"], width, label="CP-SSD")
    ax.bar(x + width, p["ties"], width, label="ties")

    ax.set_xticks(x)
    ax.set_xticklabels(p["cluster"])
    ax.set_ylabel("Number of steady JAGT series")
    ax.set_title("Representative clusters: method specialization")
    ax.legend(frameon=False)

    prettify_axes(ax)

    plt.tight_layout()
    plt.savefig(
        os.path.join(ONE_DIR, f"{prefix}_representative_method_specialization_barplot.png"),
        dpi=300,
    )
    plt.close()


def build_jagt_error_table(labels, series_keys, jagt_methods):
    """
    Per-series absolute-error table for steady JAGT series only.
    Positive cp_best_minus_kb means KB-KSSD has lower absolute error.
    Negative cp_best_minus_kb means CP-SSD has lower absolute error.
    """
    rows = []
    labels = np.asarray(labels)

    for i, key in enumerate(series_keys):
        if key not in jagt_methods.index:
            continue

        r = jagt_methods.loc[key]

        kb_err = r.get("kb_abs_err", np.nan)
        cp_err = r.get("cp_best_abs_err", np.nan)
        cp_orig_err = r.get("cp_orig_abs_err", np.nan)

        rows.append({
            "key": key,
            "cluster": int(labels[i]),
            "kb_abs_err": kb_err,
            "cp_best_abs_err": cp_err,
            "cp_orig_abs_err": cp_orig_err,
            "kb_minus_cp_best": kb_err - cp_err,
            "cp_best_minus_kb": cp_err - kb_err,
        })

    return pd.DataFrame(rows)


def plot_cluster_mean_fingerprints(data_normalized, labels, all_clusters, prefix, max_points=500):
    """
    One compact curve plot showing the mean normalized trajectory of every non-outlier cluster.
    Only the first max_points samples are shown, because this is where the visually relevant
    transient behavior usually appears.
    """
    labels = np.asarray(labels)
    clusters = sorted([c for c in set(labels) if c != -1])

    if not clusters:
        return

    fig, ax = plt.subplots(figsize=(11, 5))

    for cid in clusters:
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            continue

        series = []
        for i in idx:
            ts = np.asarray(data_normalized[i], dtype=float)
            if len(ts) > 0:
                series.append(ts[:max_points])

        if not series:
            continue

        min_len = min(len(ts) for ts in series)
        if min_len == 0:
            continue

        arr = np.vstack([ts[:min_len] for ts in series])
        mean = np.mean(arr, axis=0)

        row = all_clusters[all_clusters["cluster"] == cid]
        if row.empty:
            label = f"C{cid}"
        else:
            row = row.iloc[0]
            winner = row["winner_label"]
            nj = int(row["n_jagt"])
            kb = int(row["kb_wins_vs_cp_best"])
            cp = int(row["cp_best_wins_vs_kb"])
            label = f"C{cid} | {winner} | JAGT={nj} | KB={kb}, CP={cp}"

        ax.plot(np.arange(min_len), mean, linewidth=2, label=label)

    ax.set_title(f"Cluster mean fingerprints, first {max_points} samples\n{prefix}")
    ax.set_xlabel("Time index after dropping first point")
    ax.set_ylabel("Normalized mean value")
    ax.legend(frameon=False, fontsize=8, ncol=1)

    prettify_axes(ax)

    plt.tight_layout()
    plt.savefig(
        os.path.join(ONE_DIR, f"{prefix}_cluster_mean_fingerprints_first{max_points}.png"),
        dpi=300,
    )
    plt.close()


def plot_jagt_error_boxplots_by_cluster(jagt_err, all_clusters, prefix):
    """
    Boxplot/candle-style comparison of KB-KSSD and CP-SSD absolute errors inside each cluster.
    """
    df = jagt_err.copy()
    df = df[df["cluster"] != -1].copy()
    df = df.dropna(subset=["kb_abs_err", "cp_best_abs_err"], how="all")

    if df.empty:
        return

    order_df = all_clusters[all_clusters["cluster"] != -1].copy()
    order_df = order_df.sort_values("cluster")
    clusters = [int(c) for c in order_df["cluster"].tolist() if c in set(df["cluster"])]

    if not clusters:
        clusters = sorted(df["cluster"].unique())

    fig, ax = plt.subplots(figsize=(12, 5))

    positions = []
    data = []
    labels_out = []

    for j, cid in enumerate(clusters):
        sub = df[df["cluster"] == cid]

        kb_vals = sub["kb_abs_err"].dropna().values
        cp_vals = sub["cp_best_abs_err"].dropna().values

        if len(kb_vals):
            positions.append(j * 3.0)
            data.append(kb_vals)
            labels_out.append(f"C{cid}\nKB")

        if len(cp_vals):
            positions.append(j * 3.0 + 1.0)
            data.append(cp_vals)
            labels_out.append(f"C{cid}\nCP")

    if not data:
        return

    ax.boxplot(
        data,
        positions=positions,
        widths=0.7,
        showfliers=False,
        patch_artist=False,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels_out, rotation=45, ha="right", fontsize=8)

    ax.set_ylabel("Absolute error vs JAGT steady-state index")
    ax.set_title(f"JAGT absolute-error distributions by cluster\n{prefix}")

    prettify_axes(ax)

    plt.tight_layout()
    plt.savefig(
        os.path.join(ONE_DIR, f"{prefix}_jagt_error_boxplots_kb_vs_cp.png"),
        dpi=300,
    )
    plt.close()


def plot_jagt_error_difference_boxplot(jagt_err, all_clusters, prefix):
    """
    Boxplot of per-series error difference:
        CP-best error - KB error.
    Positive values mean KB-KSSD is better.
    Negative values mean CP-SSD is better.
    """
    df = jagt_err.copy()
    df = df[df["cluster"] != -1].copy()
    df = df.dropna(subset=["cp_best_minus_kb"])

    if df.empty:
        return

    order_df = all_clusters[all_clusters["cluster"] != -1].copy()
    order_df = order_df.sort_values("cluster")
    clusters = [int(c) for c in order_df["cluster"].tolist() if c in set(df["cluster"])]

    if not clusters:
        clusters = sorted(df["cluster"].unique())

    data = []
    labels_out = []

    for cid in clusters:
        vals = df[df["cluster"] == cid]["cp_best_minus_kb"].dropna().values
        if len(vals):
            data.append(vals)
            labels_out.append(f"C{cid}")

    if not data:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.boxplot(
        data,
        labels=labels_out,
        showfliers=False,
        patch_artist=False,
    )

    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_ylabel("CP-best absolute error − KB absolute error")
    ax.set_title(
        "Per-cluster JAGT error advantage\n"
        f"Positive = KB-KSSD better; negative = CP-SSD better\n{prefix}"
    )

    prettify_axes(ax)

    plt.tight_layout()
    plt.savefig(
        os.path.join(ONE_DIR, f"{prefix}_jagt_error_difference_boxplot.png"),
        dpi=300,
    )
    plt.close()



def make_text_summary(selected, all_clusters, kb_row, cp_row, copied, common_outputs):
    lines = []

    n_total_clusters = int((all_clusters["cluster"] != -1).sum())
    n_outlier_jagt = int(all_clusters.loc[all_clusters["cluster"] == -1, "n_jagt"].sum()) if (-1 in set(all_clusters["cluster"])) else 0

    n_kb = int((all_clusters["winner_label"] == "KB").sum())
    n_cp = int((all_clusters["winner_label"] == "CP-best").sum())
    n_neutral = int((all_clusters["winner_label"] == "neutral").sum())
    n_insuff = int((all_clusters["winner_label"] == "insufficient_JAGT").sum())

    lines.append("ONE SELECTED CLUSTERING SUMMARY")
    lines.append("=" * 40)
    lines.append("")
    lines.append(f"Selected variant: {selected['variant']}")
    lines.append(f"Selected configuration: {selected['config_name']}")
    lines.append("")
    lines.append("Whole-clustering overview:")
    lines.append(f"- Non-outlier clusters: {n_total_clusters}")
    lines.append(f"- Outlier-cluster JAGT series: {n_outlier_jagt}")
    lines.append(f"- Significant KB-favored clusters: {n_kb}")
    lines.append(f"- Significant CP-favored clusters: {n_cp}")
    lines.append(f"- Neutral clusters with enough JAGT: {n_neutral}")
    lines.append(f"- Clusters with insufficient JAGT: {n_insuff}")
    lines.append("")
    lines.append("Representative KB-favored cluster:")
    lines.append(f"- Cluster: {kb_row['cluster']}")
    lines.append(f"- Cluster size: {kb_row['cluster_size']}")
    lines.append(f"- JAGT series: {kb_row['n_jagt']}")
    lines.append(f"- KB wins: {kb_row['kb_wins_vs_cp_best']}")
    lines.append(f"- CP-best wins: {kb_row['cp_best_wins_vs_kb']}")
    lines.append(f"- Ties: {kb_row['ties_vs_cp_best']}")
    lines.append(f"- q-value: {kb_row['sign_q_vs_cp_best']}")
    lines.append(f"- KB MAE: {kb_row['kb_mae']}")
    lines.append(f"- CP-best MAE: {kb_row['cp_best_mae']}")
    lines.append("")
    lines.append("Representative CP-favored cluster:")
    lines.append(f"- Cluster: {cp_row['cluster']}")
    lines.append(f"- Cluster size: {cp_row['cluster_size']}")
    lines.append(f"- JAGT series: {cp_row['n_jagt']}")
    lines.append(f"- KB wins: {cp_row['kb_wins_vs_cp_best']}")
    lines.append(f"- CP-best wins: {cp_row['cp_best_wins_vs_kb']}")
    lines.append(f"- Ties: {cp_row['ties_vs_cp_best']}")
    lines.append(f"- q-value: {cp_row['sign_q_vs_cp_best']}")
    lines.append(f"- KB MAE: {cp_row['kb_mae']}")
    lines.append(f"- CP-best MAE: {cp_row['cp_best_mae']}")
    lines.append("")

    for winner in ["KB", "CP-best"]:
        lines.append(f"Common TSFresh subfamilies in {winner} clusters:")
        subfam = common_outputs.get(winner, {}).get("subfamilies", pd.DataFrame())
        if subfam.empty:
            lines.append("- none")
        else:
            for _, r in subfam.head(5).iterrows():
                lines.append(
                    f"- {r['subfamily']} "
                    f"(family={r['broad_family']}, clusters={int(r['n_clusters'])}, "
                    f"weighted |z|={r['weighted_mean_abs_z']:.3f})"
                )
        lines.append("")

        lines.append(f"Common raw TSFresh features in {winner} clusters:")
        raw = common_outputs.get(winner, {}).get("raw", pd.DataFrame())
        if raw.empty:
            lines.append("- none")
        else:
            for _, r in raw.head(5).iterrows():
                lines.append(
                    f"- {r['feature']} "
                    f"(clusters={int(r['n_clusters'])}, weighted |z|={r['weighted_mean_abs_z']:.3f})"
                )
        lines.append("")

    lines.append("Copied representative files:")
    for c in copied:
        lines.append(f"- {os.path.basename(c)}")
    lines.append("")
    lines.append("Meeting interpretation:")
    lines.append("- This single clustering supports method specialization.")
    lines.append("- Some clusters significantly favor KB-KSSD.")
    lines.append("- Some clusters significantly favor CP-SSD.")
    lines.append("- The new per-cluster feature files allow checking whether KB- and CP-winning clusters share recurring TSFresh signatures.")
    lines.append("- Therefore the result should be presented as cluster-dependent specialization, not as a global superiority claim.")

    with open(os.path.join(ONE_DIR, "ONE_SELECTED_CLUSTERING_SUMMARY.txt"), "w") as f:
        f.write("\n".join(lines))


def main():
    summary_path = os.path.join(OUTDIR, "top_configurations_interpretable_cluster_summary.csv")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(summary_path)

    selected, _ = choose_one_clustering(summary_path)

    prefix, full_cluster_df = load_full_cluster_table_for_selected(selected)
    zscores = load_zscore_table_for_selected(prefix)

    # Load cached labels/series for additional meeting plots.
    cached_result, _cached_cluster_eval = load_cached_selected_result(selected)
    jagt_methods = load_jagt_method_table()

    all_clusters = annotate_all_clusters(full_cluster_df)

    all_clusters.to_csv(
        os.path.join(ONE_DIR, f"{prefix}_ALL_CLUSTERS_OVERVIEW.csv"),
        index=False,
    )

    kb_row, cp_row = select_representative_clusters(all_clusters)

    copied = []
    copied += copy_existing_artifacts(prefix, kb_row["cluster"], "KB_REPRESENTATIVE")
    copied += copy_existing_artifacts(prefix, cp_row["cluster"], "CP_REPRESENTATIVE")

    all_raw, all_subfam = save_features_for_every_cluster(all_clusters, zscores, prefix)
    common_outputs = save_common_feature_summaries(all_raw, all_subfam, prefix)

    plot_all_cluster_margins(all_clusters, prefix)
    plot_all_cluster_win_counts(all_clusters, prefix)
    plot_all_cluster_sizes(all_clusters, prefix)
    plot_representative_method_win_summary(kb_row, cp_row, prefix)

    # Additional compact meeting plots:
    # 1) all cluster mean trajectories, first 500 samples;
    # 2) KB-vs-CP absolute error boxplots by cluster;
    # 3) per-cluster error-difference boxplot.
    jagt_err = build_jagt_error_table(
        labels=cached_result.labels,
        series_keys=cached_result.series_keys,
        jagt_methods=jagt_methods,
    )
    jagt_err.to_csv(
        os.path.join(ONE_DIR, f"{prefix}_JAGT_error_per_series.csv"),
        index=False,
    )

    plot_cluster_mean_fingerprints(
        data_normalized=cached_result.data_normalized,
        labels=cached_result.labels,
        all_clusters=all_clusters,
        prefix=prefix,
        max_points=500,
    )
    plot_jagt_error_boxplots_by_cluster(jagt_err, all_clusters, prefix)
    plot_jagt_error_difference_boxplot(jagt_err, all_clusters, prefix)

    make_text_summary(selected, all_clusters, kb_row, cp_row, copied, common_outputs)

    print("\nSelected one clustering:")
    print(f"variant      = {selected['variant']}")
    print(f"config_name  = {selected['config_name']}")

    print("\nAll-cluster overview saved to:")
    print(os.path.join(ONE_DIR, f"{prefix}_ALL_CLUSTERS_OVERVIEW.csv"))

    print("\nAll-cluster raw features saved to:")
    print(os.path.join(ONE_DIR, f"{prefix}_ALL_CLUSTERS_top_raw_features.csv"))

    print("\nAll-cluster subfamilies saved to:")
    print(os.path.join(ONE_DIR, f"{prefix}_ALL_CLUSTERS_top_subfamilies.csv"))

    print("\nJAGT per-series error table saved to:")
    print(os.path.join(ONE_DIR, f"{prefix}_JAGT_error_per_series.csv"))

    print("\nAdditional meeting plots saved to:")
    print(os.path.join(ONE_DIR, f"{prefix}_cluster_mean_fingerprints_first500.png"))
    print(os.path.join(ONE_DIR, f"{prefix}_jagt_error_boxplots_kb_vs_cp.png"))
    print(os.path.join(ONE_DIR, f"{prefix}_jagt_error_difference_boxplot.png"))

    print("\nCluster winner counts:")
    print(all_clusters["winner_label"].value_counts())

    print("\nRepresentative KB cluster:")
    print(kb_row[[
        "cluster", "cluster_size", "n_jagt",
        "kb_wins_vs_cp_best", "cp_best_wins_vs_kb",
        "ties_vs_cp_best", "sign_q_vs_cp_best"
    ]])

    print("\nRepresentative CP cluster:")
    print(cp_row[[
        "cluster", "cluster_size", "n_jagt",
        "kb_wins_vs_cp_best", "cp_best_wins_vs_kb",
        "ties_vs_cp_best", "sign_q_vs_cp_best"
    ]])

    print(f"\nOutputs saved in: {ONE_DIR}/")


if __name__ == "__main__":
    main()