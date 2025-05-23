import os
import pandas as pd
import matplotlib.pyplot as plt

#--------------------------------------------------
# 1. Style configuration (from earlier)
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
    'dw_pmad_Accuracy':             'red',
    'umap_Accuracy':                'green',
    'isomap_Accuracy':              'orange',
    'kernelpca_Accuracy':           'plum',
    'randomprojection_Accuracy':    'brown',
    'tsne_Accuracy':                'teal',
    'nmf_Accuracy':                 'yellowgreen',
    'lle_Accuracy':                 'gray',
    'featureagglomeration_Accuracy':'blue',
    'autoencoder_Accuracy':         'blue',
    'vae_Accuracy':                 'tan',
}

manual_styles = {
    'dw_pmad_Accuracy':             {'marker': 'o',  'linestyle': '-'},
    'umap_Accuracy':                {'marker': '^',  'linestyle': '-.'},
    'isomap_Accuracy':              {'marker': 'D',  'linestyle': ':'},
    'kernelpca_Accuracy':           {'marker': 'v',  'linestyle': '-'},
    'randomprojection_Accuracy':    {'marker': '*',  'linestyle': '--'},
    'tsne_Accuracy':                {'marker': 'X',  'linestyle': ':'},
    'nmf_Accuracy':                 {'marker': 'h',  'linestyle': '-'},
    'lle_Accuracy':                 {'marker': 'p',  'linestyle': ':'},
    'featureagglomeration_Accuracy':{'marker': 's',  'linestyle': '-.'},
    'autoencoder_Accuracy':         {'marker': 'x',  'linestyle': '--'},
    'vae_Accuracy':                 {'marker': '+',  'linestyle': '-.'},
}

def normalize(name: str) -> str:
    # remove spaces, hyphens, underscores for robust matching
    return name.lower().replace(' ', '').replace('-', '').replace('_', '')

method_styles = {}
for conf in display_methods_config:
    key_norm = normalize(conf['full_col'])
    style = manual_styles.get(conf['full_col'], {}).copy()
    style['color'] = predefined_colors.get(conf['full_col'])
    style['short_name'] = conf['x_label']
    method_styles[key_norm] = style

#--------------------------------------------------
# 2. Datasets, sample sizes, metrics, and k-values
#--------------------------------------------------
datasets     = ["Fasttext", "Isolet", "PBMC3k"]
sample_sizes = [300, 600, 900, 1200]
max_sample   = max(sample_sizes)

metrics  = ["Exact_kNN", "HNSWFlat_Faiss", "IVFPQ_Faiss"]
k_values = [1, 3, 6, 10, 15]

# global plot style
plt.rcParams.update({'font.size': 14})
plt.style.use('tableau-colorblind10')

#--------------------------------------------------
# 3. Loop over metrics
#--------------------------------------------------
for metric in metrics:
    fig, axes = plt.subplots(
        nrows=len(datasets), ncols=len(k_values),
        figsize=(20, 12), squeeze=False
    )

    # per-dataset, per-k subplot
    for di, dataset in enumerate(datasets):
        for ki, k in enumerate(k_values):
            ax = axes[di][ki]
            max_acc = 0.0
            records = {}

            # gather data for this dataset/metric/k
            for ss in sample_sizes:
                path = (
                    f"scalability_ANN_SOTA_results_"
                    f"{ss}_{dataset}_origD200_ratio0p6_alpha6_b90.xlsx"
                )
                if not os.path.exists(path):
                    continue
                df = pd.read_excel(path, sheet_name=metric)
                sub = df[df['k_Neighbors'] == k]
                for col in sub.columns:
                    if col.endswith('_Accuracy') and col.lower() not in [
                        'pca_accuracy', 'mds_accuracy', 'fastica_accuracy'
                    ]:
                        acc = sub[col].iat[0]
                        key_norm = normalize(col)
                        records.setdefault(key_norm, {})[ss] = acc
                        if acc > max_acc:
                            max_acc = acc

            # plot each method's relative curve
            for key_norm, ss_dict in records.items():
                style = method_styles.get(key_norm, {})
                xs = [ss / max_sample for ss in sorted(ss_dict)]
                ys = [ss_dict[ss] / max_acc for ss in sorted(ss_dict)]
                ax.plot(
                    xs, ys,
                    label=style.get('short_name', key_norm),
                    marker=style.get('marker', 'o'),
                    linestyle=style.get('linestyle', '-'),
                    color=style.get('color', None)
                )

            # subplot formatting
            ax.set_title(f"k = {k}", fontsize=14)
            if ki == 0:
                ax.text(
                    -0.3, 0.5, dataset,
                    transform=ax.transAxes,
                    fontsize=16, fontweight='bold',
                    va='center', ha='right'
                )
            ax.set_xlabel("Sample Size (%)", fontsize=12)
            ax.set_ylabel("Rel. Accuracy", fontsize=12)
            ticks = [ss / max_sample for ss in sample_sizes]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{int(ss*100/max_sample)}%" for ss in sample_sizes])
            ax.set_ylim(0, 1.05)
            ax.grid(True, linestyle='--', alpha=0.5)

    # legend & layout
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center', ncol=6, fontsize=12, frameon=False
    )
    plt.suptitle(metric, fontsize=18, y=1.02)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.show()
