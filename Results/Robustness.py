import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 24})
plt.style.use('tableau-colorblind10')

# ---------------------------------------------------
# 1. Define CSV paths and dataset names.
# ---------------------------------------------------
csv_paths = [
    "parameter_sweep_results_Fasttext_Multiple_methods_with_additional_baselines.csv",
    "parameter_sweep_results_Isolet_Multiple_methods_with_additional_baselines.csv",
    "parameter_sweep_results_Arcene_Multiple_methods_with_additional_baselines.csv",
    "parameter_sweep_results_PBMC3k_Multiple_methods_with_additional_baselines.csv"
]
dataset_names = ["Fasttext", "Isolet", "Arcene", "PBMC3k"]

# ---------------------------------------------------
# 2. Define display methods, colors, etc.
# ---------------------------------------------------
display_methods_config = [
    {'x_label': 'MPAD',  'full_csv_col_name': 'MPAD Accuracy'},
    {'x_label': 'UMAP',  'full_csv_col_name': 'UMAP Accuracy'},
    {'x_label': 'Isomap','full_csv_col_name': 'Isomap Accuracy'},
    {'x_label': 'KPCA',  'full_csv_col_name': 'KernelPCA Accuracy'},
    {'x_label': 'AE',    'full_csv_col_name': 'Autoencoder Accuracy'},
    {'x_label': 'FA',    'full_csv_col_name': 'FeatureAgglomeration Accuracy'},
    {'x_label': 'LLE',   'full_csv_col_name': 'LLE Accuracy'},
    {'x_label': 'NMF',   'full_csv_col_name': 'NMF Accuracy'},
    {'x_label': 'RP',    'full_csv_col_name': 'RandomProjection Accuracy'},
    {'x_label': 'VAE',   'full_csv_col_name': 'VAE Accuracy'},
    {'x_label': 't-SNE', 'full_csv_col_name': 'tSNE Accuracy'},
    {'x_label': 'LSH',   'full_csv_col_name': 'LSH Accuracy'}
]

predefined_method_attributes = {
    'MPAD Accuracy':                    {'color': 'red'},
    'UMAP Accuracy':                    {'color': 'green'},
    'Isomap Accuracy':                  {'color': 'orange'},
    'KernelPCA Accuracy':               {'color': 'plum'},
    'Autoencoder Accuracy':             {'color': 'blue'},
    'Feature Agglomeration Accuracy':   {'color': 'blue'},
    'LLE Accuracy':                     {'color': 'gray'},
    'NMF Accuracy':                     {'color': 'yellowgreen'},
    'RandomProjection Accuracy':        {'color': 'brown'},
    'VAE Accuracy':                     {'color': 'tan'},
    'tSNE Accuracy':                    {'color': 'teal'},
    'LSH Accuracy':                     {'color': 'olive'}
}

# Build a consistent color lookup
default_palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
known_colors = {v['color'] for v in predefined_method_attributes.values()}
extra_colors = [c for c in default_palette if c not in known_colors]

globally_assigned = {}
cycler_idx = 0
styled_display_methods = []
for cfg in display_methods_config:
    full = cfg['full_csv_col_name']
    col  = predefined_method_attributes.get(full, {}).get('color')
    if col is None:
        if full not in globally_assigned:
            globally_assigned[full] = extra_colors[cycler_idx % len(extra_colors)]
            cycler_idx += 1
        col = globally_assigned[full]
    styled_display_methods.append({
        'full':  full,
        'label': cfg['x_label'],
        'color': col
    })

method_display  = {m['full']:(m['label'], m['color']) for m in styled_display_methods}
fixed_fullnames = [m['full'] for m in styled_display_methods]
EXCLUDED        = ["PCA Accuracy", "FastICA Accuracy", "MDS Accuracy"]

# ---------------------------------------------------
# 3. Create a 1×4 grid: 4 datasets × 1 pie each
# ---------------------------------------------------
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(28, 7))
axs = axs.flatten()

# ---------------------------------------------------
# 4. Loop over datasets, tally counts, then plot.
# ---------------------------------------------------
for i, path in enumerate(csv_paths):
    ax = axs[i]

    # -- Load & preprocess
    try:
        df = pd.read_csv(path)
        if 'b' in df.columns:
            df = df[~df['b'].isin([40, 50])]
    except FileNotFoundError:
        ax.text(0.5, 0.5, "CSV not found", ha='center', va='center', fontsize=24)
        ax.axis('off')
        continue
    except Exception:
        ax.text(0.5, 0.5, "Error loading data", ha='center', va='center', fontsize=24)
        ax.axis('off')
        continue

    methods = [c for c in df.columns if c.endswith("Accuracy") and c not in EXCLUDED]

    # Initialize best-method counter
    best_counts = dict.fromkeys(fixed_fullnames, 0)

    # Tally only the best
    for _, row in df.iterrows():
        avail = [m for m in methods if pd.notna(row[m])]
        if not avail:
            continue
        ranks = row[avail].rank(method='dense', ascending=False)
        best_rank = ranks.min()
        for m in avail:
            if ranks[m] == best_rank:
                best_counts[m] += 1

    # Prepare data for the pie
    best_only = [(m, best_counts[m]) for m in fixed_fullnames if best_counts[m] > 0]
    labels    = [method_display[m][0] for m, _ in best_only]
    sizes     = [cnt for _, cnt in best_only]
    colors    = [method_display[m][1] for m, _ in best_only]

    total = sum(sizes)
    labels_with_pct = [f"{lab} {(cnt/total*100):.1f}%" for lab, cnt in zip(labels, sizes)]

    ax.pie(
        sizes,
        labels=labels_with_pct,
        colors=colors,
        startangle=120,
        textprops={'fontsize': 14},
        pctdistance=0.5,
        labeldistance=1.05
    )
    ax.set_title(f"{dataset_names[i]}", fontsize=22, loc='left')
    ax.axis('equal')

plt.tight_layout()
plt.show()
