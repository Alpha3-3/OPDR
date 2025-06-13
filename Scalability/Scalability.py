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
    return name.lower().replace(' ', '').replace('-', '').replace('_', '')

# build lookup styles
method_styles = {}
for conf in display_methods_config:
    key = normalize(conf['full_col'])
    style = manual_styles.get(conf['full_col'], {}).copy()
    style['color']      = predefined_colors.get(conf['full_col'])
    style['short_name'] = conf['x_label']
    method_styles[key]  = style

#--------------------------------------------------
# 2. Only PBMC3k, sample sizes → ratio, metrics, k-values
#--------------------------------------------------
dataset     = "PBMC3k"
sample_sizes = [300, 600, 900, 1200]
max_sample   = max(sample_sizes)
# each step of ratio is sample / (max_sample/4) → 1,2,3,4
metrics     = ["Exact_kNN", "HNSWFlat_Faiss", "IVFPQ_Faiss"]
k_values    = [1, 3, 6, 10, 15]

plt.rcParams.update({'font.size': 22})
plt.style.use('tableau-colorblind10')

#--------------------------------------------------
# 3. Loop over metrics, single-row of subplots
#--------------------------------------------------
for metric in metrics:
    fig, axes = plt.subplots(
        nrows=1, ncols=len(k_values),
        figsize=(20, 5), squeeze=False
    )

    for ki, k in enumerate(k_values):
        ax = axes[0][ki]
        max_acc = 0.0
        records = {}

        # collect accuracy for each sample size
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
                    key = normalize(col)
                    records.setdefault(key, {})[ss] = acc
                    max_acc = max(max_acc, acc)

        # plot each method
        for key, ss_dict in records.items():
            style = method_styles.get(key, {})
            # compute ratio ticks: ss / (max_sample/4)
            xs = [ss / (max_sample/4) for ss in sorted(ss_dict)]
            ys = [ss_dict[ss] / max_acc for ss in sorted(ss_dict)]
            ax.plot(
                xs, ys,
                label=style.get('short_name', key),
                marker=style.get('marker', 'o'),
                linestyle=style.get('linestyle', '-'),
                color=style.get('color', None)
            )

        ax.set_title(f"k = {k}", fontsize=22)
        #if ki == 0:
            #ax.text(-0.3, 0.5, dataset, transform=ax.transAxes, fontsize=16, fontweight='bold', va='center', ha='right')
        ax.set_xlabel("Sample Size Ratio", fontsize=22)
        ax.set_ylabel("Relative Accuracy", fontsize=22)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlim(0.8, 4.2)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.5)

    # legend & layout
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center', ncol=6, fontsize=22, frameon=False
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.show()
