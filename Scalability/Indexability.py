import pandas as pd
import matplotlib.pyplot as plt

#--------------------------------------------------
# 1. Style configuration (aligned with earlier example)
#--------------------------------------------------
display_methods_config = [
    {'x_label': 'MPAD',                  'full_col': 'dw_pmad_Accuracy'},
    {'x_label': 'UMAP',                  'full_col': 'umap_Accuracy'},
    {'x_label': 'Isomap',                'full_col': 'isomap_Accuracy'},
    {'x_label': 'Kernel PCA',            'full_col': 'kernelpca_Accuracy'},
    {'x_label': 'Random Projection',     'full_col': 'randomprojection_Accuracy'},
    {'x_label': 't-SNE',                 'full_col': 'tsne_Accuracy'},
    {'x_label': 'NMF',                   'full_col': 'nmf_Accuracy'},
    {'x_label': 'LLE',                   'full_col': 'lle_Accuracy'},
    {'x_label': 'Feature Agglomeration', 'full_col': 'featureagglomeration_Accuracy'},
    {'x_label': 'Autoencoder',           'full_col': 'autoencoder_Accuracy'},
    {'x_label': 'VAE',                   'full_col': 'vae_Accuracy'},
]

predefined_colors = {
    'dw_pmad_Accuracy':            'red',
    'umap_Accuracy':               'green',
    'isomap_Accuracy':             'orange',
    'kernelpca_Accuracy':          'plum',
    'randomprojection_Accuracy':   'brown',
    'tsne_Accuracy':               'teal',
    'nmf_Accuracy':                'yellowgreen',
    'lle_Accuracy':                'gray',
    'featureagglomeration_Accuracy':'blue',
    'autoencoder_Accuracy':        'blue',
    'vae_Accuracy':                'tan',
}

manual_styles = {
    'dw_pmad_Accuracy':            {'marker': 'o',  'linestyle': '-'},
    'umap_Accuracy':               {'marker': '^',  'linestyle': '-.'},
    'isomap_Accuracy':             {'marker': 'D',  'linestyle': ':'},
    'kernelpca_Accuracy':          {'marker': 'v',  'linestyle': '-'},
    'randomprojection_Accuracy':   {'marker': '*',  'linestyle': '--'},
    'tsne_Accuracy':               {'marker': 'X',  'linestyle': ':'},
    'nmf_Accuracy':                {'marker': 'h',  'linestyle': '-'},
    'lle_Accuracy':                {'marker': 'p',  'linestyle': ':'},
    'featureagglomeration_Accuracy':{'marker': 's', 'linestyle': '-.'},
    'autoencoder_Accuracy':        {'marker': 'x',  'linestyle': '--'},
    'vae_Accuracy':                {'marker': '+',  'linestyle': '-.'},
}

def normalize(name: str) -> str:
    return name.lower().replace(' ', '').replace('-', '')

# Build method_styles for lookup
method_styles = {}
for conf in display_methods_config:
    key_norm = normalize(conf['full_col'])
    style = manual_styles.get(conf['full_col'], {}).copy()
    style['color'] = predefined_colors.get(conf['full_col'])
    style['short_name'] = conf['x_label']
    method_styles[key_norm] = style

#--------------------------------------------------
# 2. File paths, dataset names, and global style
#--------------------------------------------------
excel_files = [
    "scalability_ANN_SOTA_results_900_PBMC3k_origD200_ratio0p6_alpha6_b90.xlsx"
]
dataset_names = ["PBMC3k"]

plt.rcParams.update({'font.size': 22})
plt.style.use('tableau-colorblind10')

# single row, three columns
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), squeeze=False)

# mapping sheet-name → display title
sheet_name_mapping = {
    "Exact_kNN":      "kNN",
    "HNSWFlat_Faiss": "HNSWFlat",
    "IVFPQ_Faiss":    "IVFPQ"
}

#--------------------------------------------------
# 3. Plotting loop with relative normalization and row label
#--------------------------------------------------
for row_idx, file_path in enumerate(excel_files):
    xls = pd.ExcelFile(file_path)
    original_sheets = xls.sheet_names[:3]

    for col_idx, orig_sheet in enumerate(original_sheets):
        ax = axes[row_idx][col_idx]
        df = xls.parse(orig_sheet).sort_values('k_Neighbors')

        # pick only desired accuracy columns
        acc_cols = [
            c for c in df.columns
            if c.endswith('_Accuracy')
               and c.lower() not in ['pca_accuracy','mds_accuracy','fastica_accuracy']
        ]
        # compute max for relative scaling
        max_acc = df[acc_cols].max().max()

        for col in acc_cols:
            key_norm = normalize(col)
            style = method_styles.get(key_norm, {
                'marker': 'o', 'linestyle': '-', 'color': None, 'short_name': col
            })
            rel_vals = df[col] / max_acc
            ax.plot(
                df['k_Neighbors'], rel_vals,
                label=style['short_name'],
                marker=style['marker'],
                linestyle=style['linestyle'],
                color=style['color']
            )

        # set the renamed title
        display_title = sheet_name_mapping.get(orig_sheet, orig_sheet)
        ax.set_title(display_title, fontsize=22)
        ax.set_xlabel('k', fontsize=22)
        ax.set_ylabel('Relative Accuracy', fontsize=22)

        # set x-ticks to actual k values
        xticks = sorted(df['k_Neighbors'].unique())
        ax.set_xticks(xticks)
        plt.setp(ax.get_xticklabels(), rotation=0, fontsize=22)

        # annotate the dataset name once on the leftmost subplot
        #if col_idx == 0:
            #ax.text(
                #-0.3, 0.5, dataset_names[row_idx],
                #transform=ax.transAxes, fontsize=18,
                #fontweight='bold', va='center', ha='right'
            #)

        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.5)

#--------------------------------------------------
# 4. Legend & layout
#--------------------------------------------------
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=18, frameon=False)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()
