#!/usr/bin/env python3

"""
ANOVA / Sobol-like parameter-effect analysis for CP-SSD grid results.

Assumes configurations were generated as:

s1_vals = (0.8, 0.9, 1.0)
curve_vals = ("convex", "concave")
direction_vals = ("decreasing", "increasing")
es_vals = (0.04, 0.05, 0.06)
significance_vals = (0.04, 0.05, 0.06)

config_lst = list(product(*all_param_vals))

Therefore:
classification_0   -> config_lst[0]
classification_1   -> config_lst[1]
...
classification_107 -> config_lst[107]
"""

import os
import re
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================
# CONFIG
# ======================================================

INPUT_CSV = "ssd_best_cpssd_comparison_results/cpssd_config_ranking.csv"
OUTDIR = "cpssd_parameter_effect_analysis"
os.makedirs(OUTDIR, exist_ok=True)

s1_vals = [0.8]
curve_vals = ["convex"]
direction_vals = ["decreasing"]
es_vals = (0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09)
significance_vals = [0.05]
interpolation_methods = [('interp1d', 1), ('polynomial', 2), ('polynomial', 3), ('polynomial', 4), ('polynomial', 5),
                         ('polynomial', 6), ('polynomial', 7)]

PARAMS = ["S1", "curve", "direction", "es", "significance", 'interpolation']

METRICS = [
    "mae_valid_only",
    "false_unsteady",
    "mae_penalized",
]


# ======================================================
# HELPERS
# ======================================================

def prettify(ax):
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def savefig(name):
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/{name}.png", dpi=300)
    plt.close()


def build_config_index_mapping():
    config_lst = list(itertools.product(
        s1_vals,
        curve_vals,
        direction_vals,
        es_vals,
        significance_vals,
        interpolation_methods
    ))

    # assert len(config_lst) == 108, f"Expected 108 configs, got {len(config_lst)}"

    mapping = {}

    for i, (S1, curve, direction, es, significance, interpolation) in enumerate(config_lst):
        mapping[f"classification_{i}"] = {
            "S1": S1,
            "curve": curve,
            "direction": direction,
            "es": es,
            "significance": significance,
            'interpolation': interpolation
        }

    return mapping


def extract_classification_name(config_name):
    """
    Handles either:
        classification_17
    or:
        cpssd-new-results/classification_17
    """

    m = re.search(r"classification_\d+", str(config_name))
    if m is None:
        return None
    return m.group(0)


def load_and_attach_params():
    df = pd.read_csv(INPUT_CSV)
    mapping = build_config_index_mapping()

    rows = []

    for _, r in df.iterrows():
        d = r.to_dict()

        cname = extract_classification_name(d.get("config_name", ""))

        if cname is None:
            cname = extract_classification_name(d.get("config_dir", ""))

        if cname is None:
            print(f"WARNING: cannot identify classification_N for row: {d}")
            rows.append(d)
            continue

        if cname not in mapping:
            print(f"WARNING: no parameter mapping for {cname}")
            rows.append(d)
            continue

        d["classification"] = cname
        d.update(mapping[cname])

        rows.append(d)

    dfp = pd.DataFrame(rows)

    missing = [p for p in PARAMS if p not in dfp.columns]
    if missing:
        raise RuntimeError(
            f"Parameters missing after mapping: {missing}. "
            f"Check config_name/config_dir columns in {INPUT_CSV}."
        )

    dfp.to_csv(f"{OUTDIR}/cpssd_config_ranking_with_params.csv", index=False)

    print("\nLoaded configuration table:")
    print(dfp[["classification"] + PARAMS + METRICS].head())

    return dfp


# ======================================================
# VARIANCE DECOMPOSITION
# ======================================================

def first_order_effects(df, metric):
    """
    Sobol-like first-order effect:

        V_i = Var(E[Y | X_i])

    Contribution:
        S_i = V_i / Var(Y)
    """

    sub_y = df[metric].dropna()
    total_var = np.var(sub_y, ddof=0)

    rows = []

    for p in PARAMS:
        sub = df[[p, metric]].dropna()

        if sub.empty:
            continue

        means = sub.groupby(p)[metric].mean()
        counts = sub.groupby(p)[metric].size()

        weighted_mean = np.average(means.values, weights=counts.values)
        effect_var = np.average(
            (means.values - weighted_mean) ** 2,
            weights=counts.values,
        )

        rows.append({
            "metric": metric,
            "parameter": p,
            "effect_variance": effect_var,
            "first_order_contribution": effect_var / total_var if total_var > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def second_order_effects(df, metric):
    """
    Sobol-like second-order interaction:

        V_ij = Var(E[Y | Xi, Xj]) - V_i - V_j

    Negative finite-grid values are clipped for plotting.
    """

    first = first_order_effects(df, metric)

    if first.empty:
        return pd.DataFrame(columns=[
            "metric",
            "parameter_1",
            "parameter_2",
            "joint_variance",
            "interaction_variance",
            "interaction_variance_clipped",
            "second_order_contribution",
            "second_order_contribution_clipped",
        ])

    first_map = dict(zip(first["parameter"], first["effect_variance"]))

    sub_y = df[metric].dropna()
    total_var = np.var(sub_y, ddof=0)

    rows = []

    for p1, p2 in itertools.combinations(PARAMS, 2):
        sub = df[[p1, p2, metric]].dropna()

        if sub.empty:
            continue

        means = sub.groupby([p1, p2])[metric].mean()
        counts = sub.groupby([p1, p2])[metric].size()

        weighted_mean = np.average(means.values, weights=counts.values)
        joint_var = np.average(
            (means.values - weighted_mean) ** 2,
            weights=counts.values,
        )

        interaction_var = joint_var - first_map.get(p1, 0.0) - first_map.get(p2, 0.0)
        interaction_clipped = max(interaction_var, 0.0)

        rows.append({
            "metric": metric,
            "parameter_1": p1,
            "parameter_2": p2,
            "joint_variance": joint_var,
            "interaction_variance": interaction_var,
            "interaction_variance_clipped": interaction_clipped,
            "second_order_contribution": interaction_var / total_var if total_var > 0 else np.nan,
            "second_order_contribution_clipped": interaction_clipped / total_var if total_var > 0 else np.nan,
        })

    return pd.DataFrame(rows)


# ======================================================
# PLOTS
# ======================================================

def plot_first_order(df_first, metric):
    sub = df_first[df_first["metric"] == metric].copy()
    sub = sub.sort_values("first_order_contribution", ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(sub["parameter"], sub["first_order_contribution"])

    ax.set_ylabel("Variance contribution")
    ax.set_title(f"First-order parameter effects: {metric}")

    prettify(ax)
    savefig(f"first_order_effects_{metric}")


def plot_second_order(df_second, metric):
    sub = df_second[df_second["metric"] == metric].copy()

    if sub.empty:
        return

    sub["pair"] = sub["parameter_1"] + " × " + sub["parameter_2"]
    sub = sub.sort_values("second_order_contribution_clipped", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(sub["pair"], sub["second_order_contribution_clipped"])

    ax.set_ylabel("Variance contribution")
    ax.set_title(f"Second-order interaction effects: {metric}")

    plt.xticks(rotation=45, ha="right")
    prettify(ax)
    savefig(f"second_order_effects_{metric}")


def plot_main_effects(df, metric):
    for p in PARAMS:
        sub = df[[p, metric]].dropna()

        means = sub.groupby(p)[metric].mean()
        stds = sub.groupby(p)[metric].std()

        labels = [str(x) for x in means.index]
        x = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(
            x,
            means.values,
            yerr=stds.values,
            marker="o",
            capsize=4,
            linewidth=2,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_xlabel(p)
        ax.set_ylabel(metric)
        ax.set_title(f"Main effect of {p} on {metric}")

        prettify(ax)
        savefig(f"main_effect_{p}_{metric}")


def plot_boxplots(df, metric):
    for p in PARAMS:
        groups = []
        labels = []

        for val in sorted(df[p].dropna().unique()):
            vals = df.loc[df[p] == val, metric].dropna()
            groups.append(vals)
            labels.append(str(val))

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.boxplot(groups, labels=labels, showfliers=False)

        ax.set_xlabel(p)
        ax.set_ylabel(metric)
        ax.set_title(f"Distribution of {metric} by {p}")

        prettify(ax)
        savefig(f"boxplot_{p}_{metric}")


def plot_interaction_heatmaps(df, metric):
    for p1, p2 in itertools.combinations(PARAMS, 2):
        table = df.pivot_table(
            index=p1,
            columns=p2,
            values=metric,
            aggfunc="mean",
        )

        if table.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))

        im = ax.imshow(table.values, aspect="auto")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f"mean {metric}")

        ax.set_xticks(range(len(table.columns)))
        ax.set_xticklabels([str(x) for x in table.columns], rotation=45, ha="right")

        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels([str(x) for x in table.index])

        ax.set_xlabel(p2)
        ax.set_ylabel(p1)
        ax.set_title(f"{metric}: {p1} × {p2}")

        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/interaction_heatmap_{p1}_x_{p2}_{metric}.png", dpi=300)
        plt.close()


def plot_combined_variance_contributions(df_first, df_second, metric, top_k=15):
    first = df_first[df_first["metric"] == metric].copy()
    first["name"] = first["parameter"]
    first["type"] = "first-order"
    first["value"] = first["first_order_contribution"]

    second = df_second[df_second["metric"] == metric].copy()
    if not second.empty:
        second["name"] = second["parameter_1"] + " × " + second["parameter_2"]
        second["type"] = "second-order"
        second["value"] = second["second_order_contribution_clipped"]
        combined = pd.concat([
            first[["name", "type", "value"]],
            second[["name", "type", "value"]],
        ])
    else:
        combined = first[["name", "type", "value"]]

    combined = combined.sort_values("value", ascending=False).head(top_k)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(combined["name"], combined["value"])

    ax.set_ylabel("Variance contribution")
    ax.set_title(f"Top variance contributions: {metric}")

    plt.xticks(rotation=45, ha="right")
    prettify(ax)
    savefig(f"combined_variance_contributions_{metric}")


# ======================================================
# SUMMARY
# ======================================================

def print_best_regions(df, metric, lower_is_better=True):
    print("\n" + "=" * 80)
    print(f"BEST PARAMETER VALUES BY MEAN {metric}")
    print("=" * 80)

    for p in PARAMS:
        means = df.groupby(p)[metric].mean()

        if lower_is_better:
            means = means.sort_values()
        else:
            means = means.sort_values(ascending=False)

        print(f"\n{p}")
        print(means)


def run_effect_analysis():
    df = load_and_attach_params()

    all_first = []
    all_second = []

    for metric in METRICS:
        if metric not in df.columns:
            print(f"Skipping missing metric: {metric}")
            continue

        print("\n" + "=" * 80)
        print(f"PARAMETER EFFECT ANALYSIS FOR: {metric}")
        print("=" * 80)

        df_first = first_order_effects(df, metric)
        df_second = second_order_effects(df, metric)

        print("\nFirst-order effects:")
        print(df_first.sort_values("first_order_contribution", ascending=False))

        print("\nSecond-order effects:")
        if df_second.empty:
            print("No second-order effects.")
        else:
            print(df_second.sort_values("second_order_contribution_clipped", ascending=False))

        all_first.append(df_first)
        all_second.append(df_second)

        plot_first_order(df_first, metric)
        plot_second_order(df_second, metric)
        plot_main_effects(df, metric)
        plot_boxplots(df, metric)
        plot_interaction_heatmaps(df, metric)
        plot_combined_variance_contributions(df_first, df_second, metric)

        print_best_regions(df, metric, lower_is_better=True)

    df_first_all = pd.concat(all_first, ignore_index=True)
    df_second_all = pd.concat(all_second, ignore_index=True)

    df_first_all.to_csv(f"{OUTDIR}/first_order_effects.csv", index=False)
    df_second_all.to_csv(f"{OUTDIR}/second_order_interaction_effects.csv", index=False)

    return df, df_first_all, df_second_all


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":
    df, first, second = run_effect_analysis()

    print("\nDone.")
    print(f"Effect-analysis results saved in: {OUTDIR}/")