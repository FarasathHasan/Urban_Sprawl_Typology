import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # For accuracy assessment

# -----------------------------------------------------------------------------
# Matplotlib / Seaborn styling for high-quality figures with small fonts
# -----------------------------------------------------------------------------
plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman'],
    'mathtext.fontset':   'stix',
    'axes.titlesize':     10,
    'axes.labelsize':     8,
    'xtick.labelsize':    6,
    'ytick.labelsize':    6,
    'legend.fontsize':    6,
    'figure.dpi':         600,
    'savefig.dpi':        600,
    'axes.linewidth':     0.8,
    'grid.linewidth':     0.4,
    'grid.color':         '#DDDDDD'
})
sns.set_style("whitegrid")

# -----------------------------------------------------------------------------
# Configuration with all factors
# -----------------------------------------------------------------------------
sprawl_years = [2015, 2020, 2025]
sprawl_type_names = {
    1: 'Infill Development',
    2: 'Leapfrog Development',
    3: 'Urban Extension',
    4: 'Linear Development',
    5: 'Clustered Development'
}

factor_paths = {
    'road_dist':                         'Hong Kong DATAnew/Distance from main roads.tif',
    'population':                        'Hong Kong DATAnew/Hong_Kong_Population_Density.tif',
    'amenities':                         'Hong Kong DATAnew/Amenity density.tif',
    'commercial':                        'Hong Kong DATAnew/commercial density.tif',
    'industrial':                        'Hong Kong DATAnew/industrail density.tif',
    'slope':                             'Hong Kong DATAnew/DEM.tif',
    'town_dist':                         'Hong Kong DATAnew/CBD distance.tif',
    'buil_density':                      'Hong Kong DATAnew/building density.tif',
    'Distance_from_amenity_centers':     'Hong Kong DATAnew/Distance from amenity centers.tif',
    'Distance_from_Commercial_centers':  'Hong Kong DATAnew/Distance from commercial centers.tif',
    'Distance_from_industrail_centers':  'Hong Kong DATAnew/Distance from industrail locations.tif',
    'road_density':                      'Hong Kong DATAnew/Road Density.tif'
}

factor_labels = {
    'road_dist':                         'Road Distance (m)',
    'population':                        'Population Density (people/km²)',
    'amenities':                         'Amenity Density',
    'commercial':                        'Commercial Density',
    'industrial':                        'Industrial Density',
    'slope':                             'Slope (%)',
    'town_dist':                         'CBD Distance (m)',
    'buil_density':                      'Building Density',
    'Distance_from_amenity_centers':     'Distance from Amenity Centers (m)',
    'Distance_from_Commercial_centers':  'Distance from Commercial Centers (m)',
    'Distance_from_industrail_centers':  'Distance from Industrial Centers (m)',
    'road_density':                      'Road Density'
}

# -----------------------------------------------------------------------------
# 1. Data loading with no-data masking (127) & stratified subsampling at 75%
# -----------------------------------------------------------------------------
def load_data(sample_frac=0.75):
    factors = {}
    for name, path in factor_paths.items():
        with rasterio.open(path) as src:
            arr = src.read(1).astype(float)
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            arr[arr == 127] = np.nan
            factors[name] = arr

    df_list = []
    for year in sprawl_years:
        with rasterio.open(f'SPrawl outputs_HK/sprawl_{year}.tif') as src:
            sprawl = src.read(1)
            valid = (sprawl >= 1) & (sprawl <= 5)
            rows, cols = np.where(valid)

        idx = np.random.choice(len(rows),
                               size=int(len(rows) * sample_frac),
                               replace=False)
        data = {
            'year':        year,
            'sprawl_type': sprawl[rows[idx], cols[idx]]
        }
        for name, arr in factors.items():
            vals = arr[rows[idx], cols[idx]]
            rs = RobustScaler()
            data[name] = rs.fit_transform(vals.reshape(-1, 1)).flatten()

        df_list.append(pd.DataFrame(data))
    return pd.concat(df_list, ignore_index=True)

# Load 75% of valid pixels per year
df = load_data(sample_frac=0.75)

# -----------------------------------------------------------------------------
# 2. Simplified statistical analysis (Mann–Whitney U + FDR correction)
# -----------------------------------------------------------------------------
def quick_statistical_analysis():
    records = []
    for st, st_name in sprawl_type_names.items():
        st_df    = df[df['sprawl_type'] == st]
        other_df = df[df['sprawl_type'] != st]
        for factor in factor_paths:
            a = st_df[factor].dropna()
            b = other_df[factor].dropna()
            if len(a) == 0 or len(b) == 0:
                med_diff = np.nan; u_stat = np.nan; p = np.nan
            else:
                med_diff = a.median() - b.median()
                u_stat, p = mannwhitneyu(a, b, alternative='two-sided')
            eff = u_stat / (len(a) * len(b)) if len(a) * len(b) > 0 else np.nan
            records.append({
                'sprawl_type': st_name,
                'factor':      factor,
                'med_diff':    med_diff,
                'effect_size': eff,
                'p_value':     p
            })
    res = pd.DataFrame.from_records(records)
    _, p_corr, _, _ = multipletests(res['p_value'].fillna(1), method='fdr_bh')
    res['p_value_corr'] = p_corr
    stats_df = res.set_index(['factor', 'sprawl_type']).unstack('sprawl_type')
    return stats_df

stats_df = quick_statistical_analysis()

# -----------------------------------------------------------------------------
# 3. Simplified ML analysis & feature importances (with enhanced Random Forest)
# -----------------------------------------------------------------------------
def quick_ml_analysis():
    X = df.drop(columns=['sprawl_type', 'year'])
    y = df['sprawl_type']
    X_imp = SimpleImputer(strategy='median').fit_transform(X)
    X_imp = pd.DataFrame(X_imp, columns=X.columns)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_imp, y, test_size=0.3, stratify=y, random_state=42
    )

    # -------------------------------------------------------------------------
    # Enhanced Random Forest:
    # - More trees (n_estimators=500)
    # - No depth limit (max_depth=None)
    # - Slightly larger leaf/min samples for stability
    # - Use out-of-bag estimate as a sanity check
    # -------------------------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        oob_score=True,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_tr, y_tr)

    # Print OOB accuracy for additional feedback
    print("=== Random Forest OOB Accuracy (for training data) ===")
    print(f"OOB accuracy: {rf.oob_score_:.3f}\n")

    # -------------------------------
    # Held-Out Test Set Accuracy
    # -------------------------------
    y_pred = rf.predict(X_te)
    test_accuracy = accuracy_score(y_te, y_pred)
    print("=== Random Forest Held-Out Test Accuracy ===")
    print(f"Overall accuracy on test set: {test_accuracy:.3f}\n")

    print("Confusion Matrix (rows=true classes, cols=predicted classes):")
    cm = confusion_matrix(y_te, y_pred, labels=[1, 2, 3, 4, 5])
    print(cm, "\n")

    print("Classification Report (precision, recall, f1-score per class):")
    print(classification_report(y_te, y_pred,
                                target_names=[
                                    'Infill Development',
                                    'Leapfrog Development',
                                    'Urban Extension',
                                    'Linear Development',
                                    'Clustered Development'
                                ]))
    print("============================================\n")

    # -------------------------------
    # Confusion Matrix Visualization (with annotated values and stronger color ramp)
    # -------------------------------
    plt.figure(figsize=(6, 5))  # Expanded figure size for better color differentiation
    sns.heatmap(
        cm,
        annot=True,                        # Show numbers inside cells
        fmt='d',
        cmap='viridis',                    # More pronounced color ramp
        xticklabels=[sprawl_type_names[i] for i in [1, 2, 3, 4, 5]],
        yticklabels=[sprawl_type_names[i] for i in [1, 2, 3, 4, 5]],
        cbar=True,                         # Show colorbar as a legend
        cbar_kws={'label': 'Count'},       # Label for colorbar
        linewidths=0.4,
        linecolor='gray',
        annot_kws={'size': 8}              # Small font size for numbers
    )
    plt.xlabel('Predicted Sprawl Type', labelpad=6, fontsize=8)
    plt.ylabel('True Sprawl Type', labelpad=6, fontsize=8)
    plt.title('Confusion Matrix (Color Ramp)', pad=8, fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=600)
    plt.close()

    return pd.Series(rf.feature_importances_, index=X.columns)

fi_series = quick_ml_analysis()

# -----------------------------------------------------------------------------
# 4. Generate high-quality plots (effect-size heatmap + importance bar chart)
# -----------------------------------------------------------------------------
def generate_high_quality_plots(stats_df, fi_series):
    # Effect-size heatmap (without numeric annotations)
    es = stats_df.xs('effect_size', level=0, axis=1)
    plt.figure(figsize=(6, 4))
    cmap = sns.color_palette("RdBu_r", as_cmap=True)
    ax = sns.heatmap(
        es.T.rename(columns=factor_labels),
        cmap=cmap,
        center=0,
        annot=False,          # Remove numbers inside cells
        linewidths=0.4,
        cbar_kws={'label': "Cliff's Δ", 'shrink': 0.75}
    )
    ax.set_xlabel('Spatial Factors', labelpad=6, fontsize=8)
    ax.set_ylabel('Sprawl Types', labelpad=6, fontsize=8)
    ax.tick_params(axis='x', labelsize=6, rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha('right')
    ax.tick_params(axis='y', labelsize=6, rotation=0)
    plt.tight_layout()
    plt.savefig('hq_effect_size_matrix.png', dpi=600)
    plt.close()

    # Feature importance bar chart
    fi = fi_series.rename(index=factor_labels).sort_values()
    plt.figure(figsize=(5, 4))
    bar_color = '#4c72b0'
    ax2 = fi.plot.barh(color=bar_color, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Relative Importance', labelpad=6, fontsize=8)
    ax2.set_title('Random Forest Feature Importance', pad=8, fontsize=10)
    ax2.tick_params(labelsize=6)
    plt.tight_layout()
    plt.savefig('hq_feature_importance.png', dpi=600)
    plt.close()

generate_high_quality_plots(stats_df, fi_series)

# -----------------------------------------------------------------------------
# 5. Quick console report
# -----------------------------------------------------------------------------
print("=== Key Findings ===\n")
print("Statistical Effect Sizes (Cliff's Δ):")
print(stats_df.xs('effect_size', level=0, axis=1)
      .rename(columns=factor_labels).round(2), "\n")
print("Random Forest Feature Importances:")
print(fi_series.rename(index=factor_labels)
      .sort_values(ascending=False).round(3))
