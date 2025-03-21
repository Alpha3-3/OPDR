import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import colorsys
# 如果是逗号分隔，可以用 sep=','；若是制表符(\t)分隔，就用 sep='\t'
df = pd.read_csv('parameter_sweep_results_CIFAR-10.csv')

# 查看前几行，确认数据是否读取正确
print(df.head())
import pandas as pd

melted_df = pd.melt(
    df,
    id_vars=['Dimension','Target Ratio','b','alpha','k'],  # 这些列作为“标识列”
    value_vars=['DW-PMAD Accuracy','PCA Accuracy','UMAP Accuracy',
                'Isomap Accuracy','KernelPCA Accuracy','MDS Accuracy'],
    var_name='Method',        # 新列：方法名
    value_name='Accuracy'     # 新列：准确率数值
)

print(melted_df.head())



accuracy_cols = [
    'DW-PMAD Accuracy','PCA Accuracy','UMAP Accuracy',
    'Isomap Accuracy','KernelPCA Accuracy','MDS Accuracy'
]

# idxmax(axis=1) 会返回每行在这几列中最大值的列名
df['best_method'] = df[accuracy_cols].idxmax(axis=1)

# 看看前几行效果
print(df[['Dimension','Target Ratio','b','alpha','k','best_method']].head())
# 1) 总体统计
print("各方法成为最优的次数：")
print(df['best_method'].value_counts())

# 2) 按 Target Ratio 分组统计
print("\n按 (Target Ratio, best_method) 统计：")
print(df.groupby(['Target Ratio','best_method']).size())

# 3) 按 (Target Ratio, k, best_method) 统计
print("\n按 (Target Ratio, k, best_method) 统计：")
print(df.groupby(['Target Ratio','k','best_method']).size())
import seaborn as sns
import matplotlib.pyplot as plt

# 先过滤一下，若您只想看某个 Dimension=50（若只有50就无需过滤）
subset = df[df['Dimension'] == 50]

# 画散点图：x=b, y=alpha, hue=best_method
# row=k, col=Target Ratio (若某些值太多，可以适当筛选或者拆分)
g = sns.relplot(
    data=subset,
    x='b',
    y='alpha',
    hue='best_method',
    col='Target Ratio',
    row='k',
    kind='scatter',  # 或者 kind='line' 取决于数据分布，散点更直观
    height=4,        # 子图高度
    aspect=1.2       # 子图宽高比
)

# 设置子图标题样式
g.set_titles(row_template="k={row_name}", col_template="Target Ratio={col_name}")

plt.tight_layout()
plt.show()


