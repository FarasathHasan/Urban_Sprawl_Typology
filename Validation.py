import os
import numpy as np
from osgeo import gdal, osr
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.stats import chi2_contingency

# ————— CONFIGURATION —————
MSPA_PATHS = {
    2005: "MSPA_HK/MSPA_classes_2015.tif",
    2015: "MSPA_HK/MSPA_classes_2020.tif",
    2025: "MSPA_HK/MSPA_classes_2025.tif"
}

SPRAWL_PATHS = {
    2005: "SPrawl outputs_HK/sprawl_2015.tif",
    2015: "SPrawl outputs_HK/sprawl_2020.tif",
    2025: "SPrawl outputs_HK/sprawl_2025.tif"
}

SPRAWL_EXPECTATIONS = {1:[1,3], 2:[7], 3:[2], 4:[2,4], 5:[4]}
SPRAWL_RESILIENCE = {
    1:[1,2,3,5], 2:[1,7,2,4,5], 3:[1,2,4,6,7],
    4:[1,2,3,4,7], 5:[1,2,3,4,7]
}

CLASS_NAMES = {
    'sprawl': {1:'Infill',2:'Leapfrog',3:'Urban Extension',4:'Linear',5:'Clustered'},
    'mspa':   {1:'Core',2:'Edge',3:'Perforation',4:'Islet',5:'Bridge',6:'Branch',7:'Loop'}
}

OUTPUT_DIR = "verification"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ————— GLOBAL PLOTTING STYLES —————
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
})

# Ten distinct colors for bins
bin_colors = [
    '#ebec80', '#eac238', '#ed8a1a', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]
cmap_bins = ListedColormap(bin_colors)

def read_raster(path):
    ds = gdal.Open(path)
    if ds is None:
        raise IOError(f"Could not open {path}")
    return ds, ds.GetRasterBand(1).ReadAsArray()

def align_rasters(mspa_ds, sprawl_ds):
    drv = gdal.GetDriverByName('MEM')
    out = drv.Create("", mspa_ds.RasterXSize, mspa_ds.RasterYSize, 1, gdal.GDT_Byte)
    out.SetGeoTransform(mspa_ds.GetGeoTransform())
    out.SetProjection(mspa_ds.GetProjection())
    gdal.ReprojectImage(
        sprawl_ds, out,
        sprawl_ds.GetProjection(), mspa_ds.GetProjection(),
        gdal.GRA_NearestNeighbour
    )
    return out.GetRasterBand(1).ReadAsArray()

def _plot_and_report(year, cm, ua, oa, p_val):
    # Explicit bins capped at 80 000
    thresholds = [0, 2000, 5000, 10000, 20000, 40000, 80000]
    max_count = cm[1:,1:].max()
    # internal upper bound (not shown)
    upper = max(thresholds[-1], max_count) + 1
    bounds = thresholds + [upper]
    norm = BoundaryNorm(boundaries=bounds, ncolors=len(bin_colors), clip=True)

    # Confusion‑matrix heatmap
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(cm[1:,1:], cmap=cmap_bins, norm=norm, aspect='equal')
    ax.set_xticks(np.arange(7))
    ax.set_xticklabels([CLASS_NAMES['mspa'][i] for i in range(1,8)],
                       rotation=45, ha='right')
    ax.set_yticks(np.arange(5))
    ax.set_yticklabels([CLASS_NAMES['sprawl'][i] for i in range(1,6)])
    ax.set_xlabel("MSPA Class")
    ax.set_ylabel("Sprawl Class")
    ax.set_title(f"{year} Class Distribution\nOverall Accuracy: {oa:.2%}")

    # light-ash grid lines
    ax.set_xticks(np.arange(-.5, 7, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
    ax.grid(which='minor', color='#D3D3D3', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    # colorbar: use thresholds for ticks, so no "upper" label
    cbar = fig.colorbar(
        im, ax=ax,
        fraction=0.046, pad=0.04,
        boundaries=bounds
    )
    cbar.set_ticks(thresholds)
    cbar.set_ticklabels([str(t) for t in thresholds])
    cbar.ax.set_ylabel("Pixel Count", rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"class_distribution_{year}.png"))
    plt.close(fig)

    # User‑accuracy bar chart
    fig, ax = plt.subplots(figsize=(6,4))
    labels = [CLASS_NAMES['sprawl'][i] for i in sorted(ua)]
    values = [ua[i] for i in sorted(ua)]
    bars = ax.bar(labels, values, edgecolor='black')
    ax.set_ylim(0,1)
    ax.set_ylabel("User Accuracy")
    ax.set_title(f"{year} User Accuracy by Sprawl")

    # color bars with the first len(bars) colors
    for idx, bar in enumerate(bars):
        bar.set_facecolor(bin_colors[idx])

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"user_accuracy_{year}.png"))
    plt.close(fig)

    # Text report
    rpt = os.path.join(OUTPUT_DIR, f"validation_report_{year}.txt")
    with open(rpt, 'w', encoding='utf-8') as f:
        f.write(f"Urban Sprawl Validation Report — {year}\n")
        f.write("="*50 + "\n")
        f.write(f"Overall Accuracy   : {oa:.4f}\n")
        f.write(f"Chi-square p-value : {p_val:.6f}\n\n")
        f.write("Class-wise User Accuracy:\n")
        for cls, acc in sorted(ua.items()):
            f.write(f"  {CLASS_NAMES['sprawl'][cls]:<18}: {acc:.4f}\n")

def validate_year(year):
    print(f"Processing year {year}...")
    mspa_ds, mspa_arr = read_raster(MSPA_PATHS[year])
    sprawl_ds, sprawl_arr = read_raster(SPRAWL_PATHS[year])

    if (mspa_ds.RasterXSize != sprawl_ds.RasterXSize or
        mspa_ds.RasterYSize != sprawl_ds.RasterYSize or
        not np.allclose(mspa_ds.GetGeoTransform(), sprawl_ds.GetGeoTransform()) or
        osr.SpatialReference(wkt=mspa_ds.GetProjection()).IsSame(
            osr.SpatialReference(wkt=sprawl_ds.GetProjection())
        ) == 0):
        print(" Aligning rasters...")
        sprawl_arr = align_rasters(mspa_ds, sprawl_ds)

    m_flat, s_flat = mspa_arr.flatten(), sprawl_arr.flatten()
    cm = np.zeros((len(SPRAWL_EXPECTATIONS)+1, 8), dtype=int)
    stats = {cls:{'total':0,'correct':0} for cls in SPRAWL_EXPECTATIONS}

    for s_val, m_val in zip(s_flat, m_flat):
        s,m = int(s_val), int(m_val)
        if 0 <= s < cm.shape[0] and 0 <= m < cm.shape[1]:
            cm[s,m] += 1
        if s in stats:
            stats[s]['total'] += 1
            if m in SPRAWL_EXPECTATIONS[s] or m in SPRAWL_RESILIENCE[s]:
                stats[s]['correct'] += 1

    total = sum(v['total'] for v in stats.values())
    overall_acc = sum(v['correct'] for v in stats.values()) / total
    user_acc = {cls:(v['correct']/v['total'] if v['total']>0 else 0) for cls,v in stats.items()}

    try:
        _, p_val, _, _ = chi2_contingency(cm[1:,1:7])
    except ValueError:
        _, p_val, _, _ = chi2_contingency(cm[1:,1:7] + 0.5)

    _plot_and_report(year, cm, user_acc, overall_acc, p_val)
    print(f" → Overall Acc: {overall_acc:.2%}, χ² p-value: {p_val:.4f}")
    return year, overall_acc, p_val

if __name__ == "__main__":
    results = [validate_year(y) for y in (2005, 2015, 2025)]
    with open(os.path.join(OUTPUT_DIR, "summary_report.txt"), 'w', encoding='utf-8') as f:
        f.write("Year,Overall_Accuracy,Chi2_p_value\n")
        for y,oa,p in results:
            f.write(f"{y},{oa:.4f},{p:.6f}\n")
    print("All done. Outputs are in 'verification/'.")
