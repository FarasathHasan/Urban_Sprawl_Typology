#!/usr/bin/env python
"""
compute_gap_sprawl.py

Compute and plot the Gap Statistic for your sliding-window metrics,
to identify an optimal number of clusters (k) before running k-means
in SPrwl_TYPE_CRIT.py.

Usage:
    python compute_gap_sprawl.py --years 2025 2015 2005 --ks 1 10 --B 50
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.ndimage import distance_transform_edt

# Import your main module
import SPrwl_TYPE_CRIT as sprawl

def compute_gap_statistic(X, k_max=10, B=50, random_state=0):
    """
    Compute the Gap Statistic for 1..k_max clusters on data X.
    Returns:
        ks: list of k values
        gaps: array of gap(k)
        sk:   array of standard errors
        dk:   array of Δ criterion values
    """
    rng = np.random.RandomState(random_state)
    n_samples, n_feats = X.shape

    # Uniform reference bounds
    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    # 1) Compute log(W_k) for real data
    Wk = np.zeros(k_max+1)
    for k in range(1, k_max+1):
        km = KMeans(n_clusters=k, random_state=random_state).fit(X)
        Wk[k] = np.log(np.sum(np.min(cdist(X, km.cluster_centers_), axis=1)))

    # 2) Compute log(W_k*) for B reference datasets
    Wkbs = np.zeros((B, k_max+1))
    for b in range(B):
        Xb = rng.uniform(mins, maxs, size=(n_samples, n_feats))
        for k in range(1, k_max+1):
            km = KMeans(n_clusters=k, random_state=random_state).fit(Xb)
            Wkbs[b, k] = np.log(np.sum(np.min(cdist(Xb, km.cluster_centers_), axis=1)))

    # 3) Compute gaps, s_k, and Δ criterion
    ks, gaps, sk = [], [], []
    for k in range(1, k_max+1):
        gap_k = np.mean(Wkbs[:, k]) - Wk[k]
        sk_k  = np.sqrt(np.mean((Wkbs[:, k] - np.mean(Wkbs[:, k]))**2)) * np.sqrt(1 + 1.0/B)
        ks.append(k)
        gaps.append(gap_k)
        sk.append(sk_k)

    gaps = np.array(gaps)
    sk   = np.array(sk)
    dk   = np.zeros_like(gaps)
    # Δ(k) = Gap(k) - [Gap(k+1) - s_{k+1}]
    for i in range(len(ks)-1):
        dk[i] = gaps[i] - (gaps[i+1] - sk[i+1])
    dk[-1] = np.nan

    return ks, gaps, sk, dk

def main(years, k_min, k_max, B):
    # 1. Prepare road distance array once
    road_arr, _   = sprawl.load_and_resample(sprawl.ROAD_FP, sprawl.TR)
    road_mask     = road_arr > 0
    road_dist_arr = distance_transform_edt(~road_mask) * sprawl.TR

    for yr in years:
        print(f"\n=== Year {yr} ===")
        # 2. Extract metrics DataFrame for this year
        df_metrics, centers, shape, valid, meta = sprawl.extract_metrics(
            f"MSPA/{yr}.tif",
            sprawl.TR,
            road_arr,
            road_dist_arr
        )

        # 3. Standardize the 20 metrics
        X = StandardScaler().fit_transform(df_metrics[sprawl.FEATURE_COLS])

        # 4. Compute gap statistic
        ks, gaps, sk, dk = compute_gap_statistic(X, k_max=k_max, B=B)

        # 5. Plot Gap vs. k with error bars
        plt.figure(figsize=(6,4))
        plt.errorbar(ks, gaps, yerr=sk, marker='o', linestyle='-')
        plt.xlabel("Number of clusters k")
        plt.ylabel("Gap(k)")
        plt.title(f"Gap Statistic – Year {yr}")
        plt.xticks(ks)
        plt.grid(True, linewidth=0.5)
        out_png = f"gap_statistic_{yr}.png"
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Saved plot → {out_png}")

        # 6. Print summary table
        tbl = pd.DataFrame({
            'k':                     ks,
            'Gap(k)':                gaps,
            's_k':                   sk,
            'Gap(k) - [Gap(k+1)-s]': dk
        })
        print(tbl.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Gap Statistic on sliding-window metrics"
    )
    parser.add_argument(
        '--years', type=int, nargs='+',
        default=[2025, 2015, 2005],
        help="Years to process in sequence (e.g., 2025 2015 2005)"
    )
    parser.add_argument(
        '--ks', type=int, nargs=2, metavar=('k_min','k_max'),
        default=[1, 10],
        help="Range of k values to test"
    )
    parser.add_argument(
        '--B', type=int, default=50,
        help="Number of reference bootstraps (e.g., 50)"
    )

    args = parser.parse_args()
    k_min, k_max = args.ks
    main(args.years, k_min, k_max, args.B)
