#!/usr/bin/env python
"""
High Resolution Urban Sprawl Classification for 2015, 2020 & 2025
All years classified with the 2020-trained model to ensure consistency.
Sprawl Type Codes (1–5):
    1. Infill Development
    2. Leapfrog Development
    3. Urban Extension
    4. Linear Development
    5. Clustered Development

Note: 2015 & 2020 processing unchanged. 2025 block refined so that
any previously “removed” linear pixels become core (Infill, type 1),
and only true linear growth remains along main arteries outside core.

This version computes CRITIC weights **for the 20 metrics** (not the clusters),
derives a composite score per cluster as the weighted sum of its 20 trimmed‐mean metrics,
and exports three CSVs (one per year) listing:
  • each metric (rows)
  • the five cluster-trimmed-mean values (columns)
  • a “CRITIC_weight” column for each metric’s weight
  • a final row “composite_score” giving each cluster’s composite score.
"""

import os
import math
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage import measure, filters
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.cluster import DBSCAN
from scipy.stats import trim_mean

# — CONFIGURATION —
TR = 30              # target resolution (m)
WS = 30              # window size (pixels)
SS = 5               # sliding step (pixels)

ROAD_DIST_QUANTILE     = 0.25   # bottom 25% = close to roads for 2015/20
ALIGN_THRESH           = 0.8    # cosine similarity ≥ 0.8

# Additional 2025 parameters:
ROAD_DIST_QUANTILE_2025 = 0.20  # bottom 20% = main arteries
CORE_PERC_2025          = 20    # inner 20% = core by centroid distance
ISO_DIST_QUANTILE       = 0.50  # top 50% most isolated clustered patches → leapfrog

ROAD_FP = "Hong Kong DATAnew/distance from main roads_ceaned_HK.tiff"
YEARS   = [2015, 2020, 2025]

FEATURE_COLS = [
    'ED','Mean_NN','LPI','UEII','Fractal_D','MSI',
    'LSI','Entropy','Patch_STD','AWM_expansion',
    'Local_Moran','Ripley_K','Distance_Decay',
    'Elongation','Directional_AC','DBSCAN_Count',
    'Mean_Dist_Roads','Road_Alignment','Patch_Density','Total_Edge'
]

TYPE_NAMES = {
    1: "Infill Development",
    2: "Leapfrog Development",
    3: "Urban Extension",
    4: "Linear Development",
    5: "Clustered Development"
}


# — I/O & RESAMPLING —
def load_and_resample(fp, target_res):
    with rasterio.open(fp) as src:
        data = src.read(1)
        meta = src.meta.copy()
    data[data == meta.get('nodata', 0)] = 0
    orig_res = abs(meta['transform'].a)
    if orig_res != target_res:
        scale = orig_res / target_res
        new_h = int(meta['height'] * scale)
        new_w = int(meta['width'] * scale)
        data2 = np.empty((new_h, new_w), dtype=data.dtype)
        new_tf = rasterio.Affine(target_res, 0, meta['transform'].c,
                                 0, -target_res, meta['transform'].f)
        reproject(
            source=data, destination=data2,
            src_transform=meta['transform'], src_crs=meta['crs'],
            dst_transform=new_tf, dst_crs=meta['crs'],
            resampling=Resampling.nearest
        )
        meta.update(height=new_h, width=new_w, transform=new_tf)
        data = data2
    return data, meta

def save_raster(arr, meta, out_fp):
    m = meta.copy()
    m.update(count=1, dtype=rasterio.uint8, nodata=0)
    with rasterio.open(out_fp, 'w', **m) as dst:
        dst.write(arr.astype('uint8'), 1)


# — UTILITIES —
def fill_unclassified(full, valid):
    mask_zero = (full == 0) & valid
    if mask_zero.any():
        _, inds = ndimage.distance_transform_edt(full != 0, return_indices=True)
        full[mask_zero] = full[inds[0][mask_zero], inds[1][mask_zero]]
    return full

def sliding_windows(arr, ws, ss):
    rows, cols = arr.shape
    for i in range(0, rows - ws + 1, ss):
        for j in range(0, cols - ws + 1, ss):
            yield (i, j), arr[i:i+ws, j:j+ws]


# — METRIC FUNCTIONS — (unchanged)
def boxcount(Z, k):
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
        np.arange(0, Z.shape[1], k), axis=1)
    return np.sum((S > 0) & (S < k*k))

def compute_edge_density(w, ps=1):
    e = filters.sobel(w.astype(float))
    eb = e > e.mean()
    return (eb.sum()*ps) / (w.size*ps**2) * 1e4

def compute_mean_shape_index(w, ps=1):
    lbl, _ = ndimage.label(w)
    regs = measure.regionprops(lbl)
    if not regs:
        return 0
    sis = []
    for r in regs:
        A = r.area * ps**2
        P = r.perimeter * ps
        if A > 0:
            sis.append(P / (2 * math.sqrt(math.pi * A)))
    return np.mean(sis) if sis else 0

def compute_LSI(w, ps=1):
    e = filters.sobel(w.astype(float))
    eb = e > e.mean()
    IE = eb.sum()*ps
    BL = 2*(w.shape[0]+w.shape[1])*ps
    E_star = IE + BL
    A = w.size * ps**2
    return 0.25 * E_star / math.sqrt(A)

def compute_fractal_dimension(w):
    Z = (w>0).astype(np.uint8)
    p = min(Z.shape)
    n = int(math.floor(math.log2(p)))
    sizes = 2**np.arange(n,1,-1)
    counts = np.array([boxcount(Z,s) for s in sizes], float)
    counts[counts==0] = 1e-6
    if len(sizes) < 2: return 0
    m,_ = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -m

def compute_mean_nn_distance(w, ps=1):
    lbl, _ = ndimage.label(w)
    regs = measure.regionprops(lbl)
    if len(regs)<2: return 0
    cents = np.array([r.centroid for r in regs])
    D = np.sqrt(((cents[:,None]-cents[None,:])**2).sum(axis=2))
    np.fill_diagonal(D, np.nan)
    return np.nanmean(np.nanmin(D, axis=1)) * ps

def compute_urban_expansion_intensity_index(w, ps=1):
    return (w.sum()/w.size)*100

def compute_shannon_entropy(w):
    t = w.size
    pu = (w==1).sum()/t
    pn = (w==0).sum()/t
    ent = lambda p: -p*math.log(p) if p>0 else 0
    return ent(pu) + ent(pn)

def compute_patch_std_and_LPI(w, ps=1):
    lbl, _ = ndimage.label(w)
    regs = measure.regionprops(lbl)
    areas = [r.area*ps**2 for r in regs]
    if not areas: return 0, 0
    psd = np.std(areas)
    lpi = max(areas)/(w.size*ps**2)*100
    return psd, lpi

def compute_local_moran(w):
    lm = ndimage.convolve(w.astype(float), np.ones((3,3)), mode='constant')/9
    return np.mean(np.abs(w - lm))

def compute_ripley_k(w):
    lbl, _ = ndimage.label(w)
    regs = measure.regionprops(lbl)
    cents = np.array([r.centroid for r in regs])
    if len(cents)<2: return 0
    ds = [np.linalg.norm(cents[i]-cents[j])
          for i in range(len(cents)) for j in range(i+1,len(cents))]
    return np.std(ds)

def compute_distance_decay(w, pos, gc, ps=1):
    wc = (pos[0]+w.shape[0]/2, pos[1]+w.shape[1]/2)
    d = np.linalg.norm(np.array(wc)-np.array(gc))*ps
    return (w.mean()/d) if d!=0 else w.mean()

def compute_elongation_ratio(w):
    lbl, _ = ndimage.label(w)
    regs = measure.regionprops(lbl)
    rs = [r.major_axis_length/r.minor_axis_length
          for r in regs if r.minor_axis_length>0]
    return np.mean(rs) if rs else 0

def compute_directional_autocorrelation(w):
    lbl, _ = ndimage.label(w)
    os = [r.orientation for r in measure.regionprops(lbl)
          if r.orientation is not None]
    return np.std(os) if os else 0

def compute_dbscan_cluster_count(w):
    y,x = np.where(w==1)
    coords = np.column_stack((y,x))
    if len(coords)==0: return 0
    cl = DBSCAN(eps=3, min_samples=3).fit(coords)
    labs = cl.labels_
    return len(set(labs)) - (1 if -1 in labs else 0)

def compute_patch_density(w, ps=1):
    lbl, _ = ndimage.label(w)
    n_patches = len(np.unique(lbl)) - (1 if 0 in np.unique(lbl) else 0)
    area_ha = (w.size * ps**2) / 10000.0
    return n_patches / area_ha if area_ha>0 else 0

def compute_total_edge(w, ps=1):
    lbl, _ = ndimage.label(w)
    regs = measure.regionprops(lbl)
    total_edge = sum(r.perimeter for r in regs) * ps
    return total_edge


# — FEATURES & CLUSTERING SETUP —
def fit_and_label(df):
    # 1. Standardize & K-means
    X = StandardScaler().fit_transform(df[FEATURE_COLS])
    km = KMeans(n_clusters=5, random_state=0).fit(X)
    df['raw_cluster'] = km.labels_
    df['silhouette'] = silhouette_samples(X, km.labels_)

    # 2. Build cluster-trimmed-mean DataFrame (clusters × metrics)
    clusters = sorted(df.raw_cluster.unique())
    med_dict = {
        c: {
            col: trim_mean(df[df.raw_cluster == c][col], proportiontocut=0.10)
            for col in FEATURE_COLS
        }
        for c in clusters
    }
    med_df = pd.DataFrame(med_dict).T  # index=clusters, columns=metrics

    # 3. CRITIC weighting over metrics:
    metrics_df = med_df.T                              # index=metrics, columns=clusters
    sigma_m    = metrics_df.std(axis=1)                # variability of each metric
    corr_m     = metrics_df.T.corr().fillna(0)         # correlation among metrics
    C_m        = sigma_m * ((1 - corr_m).sum(axis=1))  # raw importance per metric
    w_m        = C_m / C_m.sum()                       # normalized metric weights

    # 4. Composite score per cluster = sum_over_metrics(trimmed_mean_value * metric_weight)
    med_df['composite_score'] = med_df.mul(w_m, axis=1).sum(axis=1)

    # --- export metrics_df + weights + composite_score row for 2020 ---
    export_df       = metrics_df.copy()  # rows=metrics, cols=clusters
    export_df['CRITIC_weight'] = w_m
    comp_row        = med_df['composite_score']      # index=clusters
    export_df.loc['composite_score', export_df.columns] = comp_row.reindex(export_df.columns).values
    os.makedirs('SPrawl outputs_HK', exist_ok=True)
    export_df.to_csv('SPrawl outputs_HK/cluster_metric_weights_2020.csv')
    # ----------------------------------------------------------

    # 5. Rank clusters by composite score
    ranked = med_df['composite_score'].sort_values(ascending=False).index.tolist()

    # 6. Map ranks to sprawl types (1→Infill, 2→Leapfrog, 3→UE, 4→Linear, 5→Clustered)
    order   = [1, 3, 4, 5, 2]
    mapping = {cluster: sp for cluster, sp in zip(ranked, order)}

    # 7. Assign Sprawl_Type & override linear for 2015/2020
    df['Sprawl_Type'] = df['raw_cluster'].map(mapping).astype(int)
    rd = df['Mean_Dist_Roads']
    thr = rd.quantile(ROAD_DIST_QUANTILE)
    mask_lin = (rd <= thr) & (df['Road_Alignment'] >= ALIGN_THRESH)
    df.loc[mask_lin, 'Sprawl_Type'] = 4

    return km, mapping, med_df, w_m


def apply_model(df, km, mapping):
    X = StandardScaler().fit_transform(df[FEATURE_COLS])
    df['raw_cluster'] = km.predict(X)
    df['silhouette']  = silhouette_samples(X, df['raw_cluster'])
    df['Sprawl_Type'] = df['raw_cluster'].map(mapping).astype(int)
    rd = df['Mean_Dist_Roads']
    thr = rd.quantile(ROAD_DIST_QUANTILE)
    mask_lin = (rd <= thr) & (df['Road_Alignment'] >= ALIGN_THRESH)
    df.loc[mask_lin, 'Sprawl_Type'] = 4
    return df

def refine_leapfrog(full, valid):
    lbl, num = ndimage.label((full == 5) & valid)
    props    = measure.regionprops(lbl)
    if len(props) < 2:
        return full
    cents = np.array([r.centroid for r in props])
    dmat  = np.linalg.norm(cents[:,None,:] - cents[None,:,:], axis=2)
    np.fill_diagonal(dmat, np.nan)
    min_dists = np.nanmin(dmat, axis=1)
    thr = np.nanquantile(min_dists, ISO_DIST_QUANTILE)
    for idx, r in enumerate(props):
        if min_dists[idx] > thr:
            full[lbl == r.label] = 2
    return full

def extract_metrics(fp, tr, road_arr, road_dist_arr):
    data, meta = load_and_resample(fp, tr)
    valid = data != 0
    yc, xc = np.where(valid)
    gc     = (yc.mean(), xc.mean()) if yc.size>0 else (data.shape[0]/2, data.shape[1]/2)

    recs, centers = [], []
    for (i,j), w in sliding_windows(data, WS, SS):
        if w.sum() < 1:
            continue
        p  = (w>0).astype(np.uint8)
        dw = road_dist_arr[i:i+WS, j:j+WS]

        rw    = road_arr[i:i+WS, j:j+WS] > 0
        r_lbl, _ = ndimage.label(rw)
        regs_r   = measure.regionprops(r_lbl)
        road_or  = max(regs_r, key=lambda r: r.area).orientation if regs_r else None

        p_lbl, _  = ndimage.label(p)
        regs_p    = measure.regionprops(p_lbl)
        patch_or  = max(regs_p, key=lambda r: r.area).orientation if regs_p else None

        align = abs(math.cos(patch_or - road_or)) if (road_or is not None and patch_or is not None) else 0

        recs.append({
            'ED': compute_edge_density(p, tr),
            'Mean_NN': compute_mean_nn_distance(p, tr),
            'LPI': compute_patch_std_and_LPI(p, tr)[1],
            'UEII': compute_urban_expansion_intensity_index(p, tr),
            'Fractal_D': compute_fractal_dimension(p),
            'MSI': compute_mean_shape_index(p, tr),
            'LSI': compute_LSI(p, tr),
            'Entropy': compute_shannon_entropy(p),
            'Patch_STD': compute_patch_std_and_LPI(p, tr)[0],
            'AWM_expansion': compute_urban_expansion_intensity_index(p, tr),
            'Local_Moran': compute_local_moran(p),
            'Ripley_K': compute_ripley_k(p),
            'Distance_Decay': compute_distance_decay(p, (i,j), gc, tr),
            'Elongation': compute_elongation_ratio(p),
            'Directional_AC': compute_directional_autocorrelation(p),
            'DBSCAN_Count': compute_dbscan_cluster_count(p),
            'Mean_Dist_Roads': dw.mean(),
            'Road_Alignment': align,
            'Patch_Density': compute_patch_density(p, tr),
            'Total_Edge': compute_total_edge(p, tr)
        })
        centers.append((i+WS//2, j+WS//2))

    return pd.DataFrame(recs), np.array(centers), data.shape, valid, meta

def interpolate_to_raster(centers, types, shape):
    gy, gx = np.mgrid[0:shape[0], 0:shape[1]]
    full   = griddata(centers, types, (gy, gx), method='nearest', fill_value=0)
    return full.astype(np.uint8)

def build_and_save(full, valid, meta, out_tif, out_csv):
    full[~valid] = 0
    save_raster(full, meta, out_tif)
    rows, cols = np.nonzero(full>0)
    pd.DataFrame({
        'row': rows,
        'col': cols,
        'Sprawl_Type': full[rows, cols]
    }).to_csv(out_csv, index=False)

    plt.figure(figsize=(8,6))
    plt.imshow(full, cmap='viridis', vmin=1, vmax=5)
    cbar = plt.colorbar(ticks=[1,2,3,4,5])
    cbar.set_label('Sprawl Type')
    plt.title(os.path.basename(out_tif))
    plt.axis('off')
    png_fp = out_tif.replace('.tif', '.png')
    plt.savefig(png_fp, dpi=150)
    plt.close()

def compute_area(full, tr):
    km2 = (tr*tr)/1e6
    return {c: np.sum(full==c)*km2 for c in range(1,6)}


# — MAIN PROCESS —
if __name__ == "__main__":
    os.makedirs('SPrawl outputs_HK', exist_ok=True)

    # Load roads & compute distance-to-road
    road_arr, _      = load_and_resample(ROAD_FP, TR)
    road_mask        = road_arr > 0
    road_dist_arr    = distance_transform_edt(~road_mask) * TR

    summaries = {}

    # 1. Fit on 2020
    df20, ctr20, shp20, valid20, meta20 = extract_metrics(
        "Hong Kong DATAnew/2015hk.tif", TR, road_arr, road_dist_arr
    )
    km20, map20, med_df20, w20 = fit_and_label(df20)
    full20 = interpolate_to_raster(ctr20, df20.Sprawl_Type.values, shp20)
    full20 = fill_unclassified(full20, valid20)
    full20 = refine_leapfrog(full20, valid20)
    build_and_save(full20, valid20, meta20,
                   "SPrawl outputs_HK/sprawl_2020.tif",
                   "SPrawl outputs_HK/sprawl_2020_metrics.csv")
    summaries[2020] = compute_area(full20, TR)

    # Silhouette‐score summary for 2020 clusters
    sil_stats20 = df20.groupby('raw_cluster')['silhouette'] \
                      .agg(['mean', 'std', 'count']) \
                      .rename(columns={'mean': 'sil_mean', 'std': 'sil_std', 'count': 'n_windows'})
    print("\nSilhouette Scores by Cluster (2020 fit):")
    print(sil_stats20)

    # Export raw silhouette scores for 2020
    df20[['raw_cluster','silhouette']].to_csv('SPrawl outputs_HK/2020_silhouette_scores.csv', index=False)

    plt.figure(figsize=(6,4))
    plt.hist(df20['silhouette'], bins=50)
    plt.title('Histogram of Silhouette Scores (2020)')
    plt.xlabel('Silhouette Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('SPrawl outputs_HK/silhouette_histogram_2020.png')
    plt.close()

    plt.figure(figsize=(6,4))
    scores20 = [df20[df20.raw_cluster == i].silhouette for i in sorted(df20.raw_cluster.unique())]
    plt.boxplot(scores20, labels=[f"C{i}" for i in sorted(df20.raw_cluster.unique())])
    plt.xlabel('Cluster')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette‐Score Distribution by Cluster (2020)')
    plt.tight_layout()
    plt.savefig('SPrawl outputs_HK/silhouette_boxplot_2020.png')
    plt.close()

    comp20 = med_df20[['composite_score']].sort_values('composite_score', ascending=False)
    plt.figure(figsize=(5,3))
    plt.plot(range(1, len(comp20)+1), comp20['composite_score'], marker='o', linestyle='-')
    plt.xticks(range(1, len(comp20)+1))
    plt.xlabel('Cluster Rank')
    plt.ylabel('Composite Score')
    plt.title('CRITIC Composite Scores by Cluster Rank (2020)')
    plt.grid(True, linewidth=0.5)
    plt.tight_layout()
    plt.savefig('SPrawl outputs_HK/composite_scree_2020.png')
    plt.close()

    # 2. Apply 2020 model to 2015
    df15, ctr15, shp15, valid15, meta15 = extract_metrics(
        "Hong Kong DATAnew/2005hk.tif", TR, road_arr, road_dist_arr
    )
    df15 = apply_model(df15, km20, map20)
    full15 = interpolate_to_raster(ctr15, df15.Sprawl_Type.values, shp15)
    full15 = fill_unclassified(full15, valid15)
    full15 = refine_leapfrog(full15, valid15)
    build_and_save(full15, valid15, meta15,
                   "SPrawl outputs_HK/sprawl_2015.tif",
                   "SPrawl outputs_HK/sprawl_2015_metrics.csv")
    summaries[2015] = compute_area(full15, TR)

    # Silhouette‐score summary for 2015 clusters
    sil_stats15 = df15.groupby('raw_cluster')['silhouette'] \
                      .agg(['mean', 'std', 'count']) \
                      .rename(columns={'mean': 'sil_mean', 'std': 'sil_std', 'count': 'n_windows'})
    print("\nSilhouette Scores by Cluster (2015 classification):")
    print(sil_stats15)

    df15[['raw_cluster','silhouette']].to_csv('SPrawl outputs_HK/2015_silhouette_scores.csv', index=False)

    plt.figure(figsize=(6,4))
    plt.hist(df15['silhouette'], bins=50)
    plt.title('Histogram of Silhouette Scores (2015)')
    plt.xlabel('Silhouette Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('SPrawl outputs_HK/silhouette_histogram_2015.png')
    plt.close()

    plt.figure(figsize=(6,4))
    scores15 = [df15[df15.raw_cluster == i].silhouette for i in sorted(df15.raw_cluster.unique())]
    plt.boxplot(scores15, labels=[f"C{i}" for i in sorted(df15.raw_cluster.unique())])
    plt.xlabel('Cluster')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette‐Score Distribution by Cluster (2015)')
    plt.tight_layout()
    plt.savefig('SPrawl outputs_HK/silhouette_boxplot_2015.png')
    plt.close()

    # Compute trimmed-mean & CRITIC weights for 2015
    clusters15 = sorted(df15.raw_cluster.unique())
    med_dict15 = {
        c: {
            col: trim_mean(df15[df15.raw_cluster == c][col], proportiontocut=0.10)
            for col in FEATURE_COLS
        }
        for c in clusters15
    }
    med_df15 = pd.DataFrame(med_dict15).T

    metrics_df15 = med_df15.T
    sigma15      = metrics_df15.std(axis=1)
    corr15       = metrics_df15.T.corr().fillna(0)
    C15          = sigma15 * ((1 - corr15).sum(axis=1))
    w15          = C15 / C15.sum()
    med_df15['composite_score'] = med_df15.mul(w15, axis=1).sum(axis=1)

    exp15 = metrics_df15.copy()
    exp15['CRITIC_weight'] = w15
    comp15 = med_df15['composite_score']
    exp15.loc['composite_score', exp15.columns] = comp15.reindex(exp15.columns).values
    exp15.to_csv('SPrawl outputs_HK/cluster_metric_weights_2015.csv')

    # 3. Apply 2020 model to 2025 with refined linear/core & dimension-fix
    df25, ctr25, shp25, valid25, meta25 = extract_metrics(
        "Hong Kong DATAnew/2025hk.tif", TR, road_arr, road_dist_arr
    )
    df25 = apply_model(df25, km20, map20)
    full25 = interpolate_to_raster(ctr25, df25.Sprawl_Type.values, shp25)

    # Fix potential shape mismatch
    rd25 = road_dist_arr
    if rd25.shape != valid25.shape:
        rd25 = rd25[:valid25.shape[0], :valid25.shape[1]]

    # Refine linear vs core for 2025
    yc25, xc25 = np.where(valid25)
    centroid25 = (yc25.mean(), xc25.mean())
    gy, gx     = np.indices(full25.shape)
    dist_center25 = np.sqrt((gy - centroid25[0])**2 + (gx - centroid25[1])**2)
    core_thr25    = np.percentile(dist_center25[valid25], CORE_PERC_2025)
    core_mask25   = dist_center25 <= core_thr25

    artery_thr = np.quantile(rd25[valid25], ROAD_DIST_QUANTILE_2025)
    artery_mask = rd25 <= artery_thr

    lin_mask     = (full25 == 4)
    keep_lin     = lin_mask & artery_mask & ~core_mask25
    removed_lin  = lin_mask & ~keep_lin
    full25[removed_lin] = 1
    full25[keep_lin]    = 4

    full25 = fill_unclassified(full25, valid25)
    full25 = refine_leapfrog(full25, valid25)
    build_and_save(full25, valid25, meta25,
                   "SPrawl outputs_HK/sprawl_2025.tif",
                   "SPrawl outputs_HK/sprawl_2025_metrics.csv")
    summaries[2025] = compute_area(full25, TR)

    # Silhouette‐score summary for 2025 clusters
    sil_stats25 = df25.groupby('raw_cluster')['silhouette'] \
                      .agg(['mean', 'std', 'count']) \
                      .rename(columns={'mean': 'sil_mean', 'std': 'sil_std', 'count': 'n_windows'})
    print("\nSilhouette Scores by Cluster (2025 classification):")
    print(sil_stats25)

    df25[['raw_cluster','silhouette']].to_csv('SPrawl outputs_HK/2025_silhouette_scores.csv', index=False)

    plt.figure(figsize=(6,4))
    plt.hist(df25['silhouette'], bins=50)
    plt.title('Histogram of Silhouette Scores (2025)')
    plt.xlabel('Silhouette Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('SPrawl outputs_HK/silhouette_histogram_2025.png')
    plt.close()

    plt.figure(figsize=(6,4))
    scores25 = [df25[df25.raw_cluster == i].silhouette for i in sorted(df25.raw_cluster.unique())]
    plt.boxplot(scores25, labels=[f"C{i}" for i in sorted(df25.raw_cluster.unique())])
    plt.xlabel('Cluster')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette‐Score Distribution by Cluster (2025)')
    plt.tight_layout()
    plt.savefig('SPrawl outputs_HK/silhouette_boxplot_2025.png')
    plt.close()

    # Compute trimmed-mean & CRITIC weights for 2025
    clusters25 = sorted(df25.raw_cluster.unique())
    med_dict25 = {
        c: {
            col: trim_mean(df25[df25.raw_cluster == c][col], proportiontocut=0.10)
            for col in FEATURE_COLS
        }
        for c in clusters25
    }
    med_df25 = pd.DataFrame(med_dict25).T

    metrics_df25 = med_df25.T
    sigma25      = metrics_df25.std(axis=1)
    corr25       = metrics_df25.T.corr().fillna(0)
    C25          = sigma25 * ((1 - corr25).sum(axis=1))
    w25          = C25 / C25.sum()
    med_df25['composite_score'] = med_df25.mul(w25, axis=1).sum(axis=1)

    exp25 = metrics_df25.copy()
    exp25['CRITIC_weight'] = w25
    comp25 = med_df25['composite_score']
    exp25.loc['composite_score', exp25.columns] = comp25.reindex(exp25.columns).values
    exp25.to_csv('SPrawl outputs_HK/cluster_metric_weights_2025.csv')

    comp25_plot = med_df25[['composite_score']].sort_values('composite_score', ascending=False)
    plt.figure(figsize=(5,3))
    plt.plot(range(1, len(comp25_plot)+1), comp25_plot['composite_score'], marker='o', linestyle='-')
    plt.xticks(range(1, len(comp25_plot)+1))
    plt.xlabel('Cluster Rank')
    plt.ylabel('Composite Score')
    plt.title('CRITIC Composite Scores by Cluster Rank (2025)')
    plt.grid(True, linewidth=0.5)
    plt.tight_layout()
    plt.savefig('SPrawl outputs_HK/composite_scree_2025.png')
    plt.close()

    # Final summary of areas and printing of weights
    print("\nSprawl Type Areas (km²):")
    for yr in YEARS:
        print(f"\nYear {yr}:")
        for t in range(1,6):
            print(f"  Type {t} ({TYPE_NAMES[t]:<18}): {summaries[yr][t]:8.2f}")

    print("\nCRITIC weights for each of the 20 metrics:")
    print("  2015:", w15.to_dict())
    print("  2020:", w20.to_dict())
    print("  2025:", w25.to_dict())

    print("\nDone.")
