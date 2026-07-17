#!/usr/bin/env python3

import pandas as pd
import numpy as np

CSV_PATH = "../man_steady_comparison/ssd_best_cpssd_comparison_results/ssd_best_config_comparison_table.csv"

KEY_COL = "key"
ERR_COLS = ["kb_abs_err", "cp_best_abs_err"]
IDX_COLS = ["kb_idx", "cp_best_idx"]
JAGT_COL = "jagt_idx"


def show_raw_problem_values():
    raw = pd.read_csv(
        CSV_PATH,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )

    print("\n=== BASIC INFO ===")
    print("Rows:", len(raw))
    print("Columns:", list(raw.columns))

    for col in ERR_COLS:
        numeric = pd.to_numeric(raw[col], errors="coerce")
        print(f"\n{col} bad numeric values:", int(numeric.isna().sum()))

    kb_abs = pd.to_numeric(raw["kb_abs_err"], errors="coerce")
    cp_abs = pd.to_numeric(raw["cp_best_abs_err"], errors="coerce")

    bad_any = kb_abs.isna() | cp_abs.isna()
    bad = raw.loc[bad_any].copy()

    print("\n=== ROWS BAD IN EITHER KB OR CP ===")
    print("Bad rows:", len(bad))

    for c in ["kb_abs_err", "cp_best_abs_err", "kb_idx", "cp_best_idx", "jagt_idx"]:
        bad[c + "_num"] = pd.to_numeric(bad[c], errors="coerce")

    for _, r in bad.iterrows():
        print("-" * 140)
        print("key:", r[KEY_COL])

        print("jagt_idx raw:", repr(r.get("jagt_idx", "")))
        print("kb_idx raw:", repr(r.get("kb_idx", "")))
        print("cp_best_idx raw:", repr(r.get("cp_best_idx", "")))

        print("kb_abs_err raw:", repr(r.get("kb_abs_err", "")))
        print("cp_best_abs_err raw:", repr(r.get("cp_best_abs_err", "")))

        missing_kb_abs = pd.isna(r["kb_abs_err_num"])
        missing_cp_abs = pd.isna(r["cp_best_abs_err_num"])

        kb_idx = r["kb_idx_num"]
        cp_idx = r["cp_best_idx_num"]
        jagt_idx = r["jagt_idx_num"]

        print("missing kb_abs_err:", missing_kb_abs)
        print("missing cp_best_abs_err:", missing_cp_abs)

        if missing_kb_abs:
            print("KB missing abs err; kb_idx == -1:", kb_idx == -1)
            if not pd.isna(jagt_idx) and not pd.isna(kb_idx):
                print("Potential KB abs err if -1 is penalized:", abs(kb_idx - jagt_idx))

        if missing_cp_abs:
            print("CP missing abs err; cp_best_idx == -1:", cp_idx == -1)
            if not pd.isna(jagt_idx) and not pd.isna(cp_idx):
                print("Potential CP abs err if -1 is penalized:", abs(cp_idx - jagt_idx))

    print("\n=== SUMMARY FOR BAD ROWS ===")

    summary = pd.DataFrame({
        "key": bad[KEY_COL],
        "jagt_idx": bad["jagt_idx_num"],
        "kb_idx": bad["kb_idx_num"],
        "cp_best_idx": bad["cp_best_idx_num"],
        "kb_abs_err": bad["kb_abs_err_num"],
        "cp_best_abs_err": bad["cp_best_abs_err_num"],
        "kb_abs_missing": bad["kb_abs_err_num"].isna(),
        "cp_abs_missing": bad["cp_best_abs_err_num"].isna(),
        "kb_idx_is_minus_1": bad["kb_idx_num"] == -1,
        "cp_best_idx_is_minus_1": bad["cp_best_idx_num"] == -1,
    })

    summary["missing_kb_explained_by_minus_1"] = (
        summary["kb_abs_missing"] & summary["kb_idx_is_minus_1"]
    )
    summary["missing_cp_explained_by_minus_1"] = (
        summary["cp_abs_missing"] & summary["cp_best_idx_is_minus_1"]
    )

    summary["all_missing_explained_by_minus_1"] = (
        (~summary["kb_abs_missing"] | summary["kb_idx_is_minus_1"])
        & (~summary["cp_abs_missing"] | summary["cp_best_idx_is_minus_1"])
    )

    print(summary.to_string(index=False))

    print("\nMissing KB abs err:", int(summary["kb_abs_missing"].sum()))
    print("... explained by kb_idx == -1:", int(summary["missing_kb_explained_by_minus_1"].sum()))

    print("\nMissing CP abs err:", int(summary["cp_abs_missing"].sum()))
    print("... explained by cp_best_idx == -1:", int(summary["missing_cp_explained_by_minus_1"].sum()))

    print("\nAll bad rows explained by -1 index:",
          int(summary["all_missing_explained_by_minus_1"].sum()),
          "/",
          len(summary))

    summary.to_csv("debug_missing_abs_err_minus1_check.csv", index=False)
    print("\nSaved: debug_missing_abs_err_minus1_check.csv")


if __name__ == "__main__":
    show_raw_problem_values()