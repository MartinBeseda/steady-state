#!/usr/bin/env python3

import os
import gc
import __main__

import pandas as pd

import targeted_tsfresh_cp_kb_cluster_search_aligned as t


# ============================================================
# PICKLE COMPATIBILITY
# ============================================================
#
# The clustering cache files were created while
# targeted_tsfresh_cp_kb_cluster_search_aligned.py was executed
# directly as a script.
#
# Therefore ClusteringResult was pickled as:
#
#     __main__.ClusteringResult
#
# When this rebuild script reads those pickle files, Python looks
# for ClusteringResult inside *this* script's __main__ module.
#
# Expose the original class under that name so the old caches can
# be unpickled correctly.
#
__main__.ClusteringResult = t.ClusteringResult


# ============================================================
# CONFIG
# ============================================================

TOP_N_PER_VARIANT = 8


def main():

    print("\nRebuilding aligned top-configuration summaries")
    print("=" * 80)

    metrics_path = os.path.join(
        t.TABLES_DIR,
        "all_clustering_configurations.csv",
    )

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(
            f"Cannot find clustering configuration table:\n{metrics_path}"
        )

    print(f"\nReading clustering configuration table:\n{metrics_path}")

    df = pd.read_csv(metrics_path)

    if "variant" not in df.columns:
        raise RuntimeError(
            "Expected column 'variant' in all_clustering_configurations.csv"
        )

    if "config_name" not in df.columns:
        raise RuntimeError(
            "Expected column 'config_name' in all_clustering_configurations.csv"
        )

    if "hypothesis_score" not in df.columns:
        raise RuntimeError(
            "Expected column 'hypothesis_score' in "
            "all_clustering_configurations.csv"
        )

    print("\nAvailable configurations by variant:")
    print(df["variant"].value_counts())

    # ============================================================
    # SELECT TOP CONFIGURATIONS SEPARATELY FOR EACH REGIME
    # ============================================================

    selected_parts = []

    for variant, group in df.groupby("variant"):

        top = (
            group
            .sort_values(
                "hypothesis_score",
                ascending=False,
            )
            .head(TOP_N_PER_VARIANT)
            .copy()
        )

        selected_parts.append(top)

    selected = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    print("\nSelected configurations:")
    print("-" * 80)

    columns_to_show = [
        c for c in [
            "variant",
            "config_name",
            "hypothesis_score",
            "silhouette",
            "coverage",
            "n_clusters",
            "n_outliers",
        ]
        if c in selected.columns
    ]

    print(
        selected[columns_to_show]
        .to_string(index=False)
    )

    # ============================================================
    # REBUILD INTERPRETATION TABLES
    # ============================================================

    summary_parts = []

    n_total = len(selected)

    for config_no, (_, row) in enumerate(
        selected.iterrows(),
        start=1,
    ):

        variant = str(row["variant"])
        config = str(row["config_name"])

        print("\n" + "=" * 80)
        print(
            f"[{config_no}/{n_total}] "
            f"Processing {variant} / {config}"
        )
        print("=" * 80)

        cache_filename = (
            f"{t.safe_name(variant)}__"
            f"{t.safe_name(config)}.pkl"
        )

        cache_path = os.path.join(
            t.CONFIG_CACHE_DIR,
            cache_filename,
        )

        if not os.path.exists(cache_path):
            print(
                "WARNING: cache does not exist, skipping:\n"
                f"{cache_path}"
            )
            continue

        print(f"Loading cache:\n{cache_path}")

        # --------------------------------------------------------
        # Load cached clustering
        # --------------------------------------------------------

        try:
            payload = pd.read_pickle(cache_path)

        except Exception as exc:
            print(
                "\nERROR while loading cached configuration:"
            )
            print(cache_path)
            print(repr(exc))
            print("\nSkipping this configuration.")
            continue

        if not isinstance(payload, dict):
            print(
                "WARNING: unexpected cache object type: "
                f"{type(payload)}"
            )
            del payload
            gc.collect()
            continue

        if "result" not in payload:
            print(
                "WARNING: cache does not contain key 'result'."
            )
            del payload
            gc.collect()
            continue

        if "cluster_eval" not in payload:
            print(
                "WARNING: cache does not contain key 'cluster_eval'."
            )
            del payload
            gc.collect()
            continue

        result = payload["result"]
        cluster_eval = payload["cluster_eval"]

        print(
            f"Loaded clustering with "
            f"{len(result.series_keys)} series."
        )

        # --------------------------------------------------------
        # Recompute cluster feature z-scores
        # --------------------------------------------------------

        print("Computing cluster feature Z-scores...")

        zscores = t.cluster_feature_zscores(
            result.features,
            result.labels,
        )

        # --------------------------------------------------------
        # Save the cluster-level interpretation tables expected by
        # cluster_signature_ssd_recommendation_aligned.py
        # --------------------------------------------------------

        print("Saving cluster interpretation tables...")

        t.save_cluster_interpretation_tables(
            result,
            cluster_eval,
            zscores,
        )

        prefix = t.safe_name(
            f"{variant}_{config}"
        )

        summary_path = os.path.join(
            t.SIG_DIR,
            f"{prefix}_interpretable_cluster_summary.csv",
        )

        if not os.path.exists(summary_path):
            print(
                "WARNING: expected summary was not generated:\n"
                f"{summary_path}"
            )

        else:

            part = pd.read_csv(summary_path)

            if part.empty:
                print(
                    "WARNING: generated summary contains no rows."
                )

            else:
                summary_parts.append(part)

                print(
                    f"Added {len(part)} interpretable "
                    f"cluster rows."
                )

        # --------------------------------------------------------
        # Release memory aggressively
        # --------------------------------------------------------

        del zscores
        del cluster_eval
        del result
        del payload

        gc.collect()

        print("Memory released for this configuration.")

    # ============================================================
    # COMBINE THE REBUILT SUMMARIES
    # ============================================================

    if not summary_parts:
        raise RuntimeError(
            "No interpretable cluster summaries were successfully "
            "reconstructed."
        )

    final = pd.concat(
        summary_parts,
        ignore_index=True,
    )

    output_path = os.path.join(
        t.OUTDIR,
        "top_configurations_interpretable_cluster_summary.csv",
    )

    final.to_csv(
        output_path,
        index=False,
    )

    # ============================================================
    # FINAL CHECKS
    # ============================================================

    print("\n" + "=" * 80)
    print("REBUILD COMPLETE")
    print("=" * 80)

    print(
        f"\nSaved combined summary:\n{output_path}"
    )

    print("\nCluster-summary rows by variant:")

    if "variant" in final.columns:
        print(
            final["variant"].value_counts()
        )

    print("\nUnique configurations represented:")

    if (
        "variant" in final.columns
        and "config_name" in final.columns
    ):
        print(
            final
            .groupby("variant")["config_name"]
            .nunique()
        )

    print("\nConfigurations included:")

    if (
        "variant" in final.columns
        and "config_name" in final.columns
    ):

        configs = (
            final[
                ["variant", "config_name"]
            ]
            .drop_duplicates()
            .sort_values(
                ["variant", "config_name"]
            )
        )

        print(
            configs.to_string(index=False)
        )

    print(
        "\nYou can now run "
        "cluster_signature_ssd_recommendation_aligned.py "
        "with either ANALYSIS_MODE."
    )


if __name__ == "__main__":
    main()

