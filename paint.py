import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# 读取CSV文件
data1 = pd.read_csv('AUC_result_mean.csv')
data2 = pd.read_csv('AUC2.csv', header=None)
data3 = pd.read_csv('AUC3.csv', header=None)
data4 = pd.read_csv('AUC4.csv', header=None)
data5 = pd.read_csv('AUC5.csv', header=None)
data6 = pd.read_csv('AUC6.csv', header=None)
auc_data1 = data1.iloc[:, 1:].values
row_labels = data1.iloc[:, 0].values
col_labels = data1.columns[1:]
auc_data1 = np.round(auc_data1, 3)
auc_data2 = data2.values
auc_data2 = np.round(auc_data2, 3)
auc_data3 = data3.values
auc_data3 = np.round(auc_data3, 3)
auc_data4 = data4.values
auc_data4 = np.round(auc_data4, 3)
auc_data5 = data5.values
auc_data5 = np.round(auc_data5, 3)
auc_data6 = data6.values
auc_data6 = np.round(auc_data6, 3)

# 使用seaborn绘制热力图
cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#e6e8fa', '#035096'], N=256)
cmap.set_bad(color='#d3d3d3')
fig, axes = plt.subplots(3, 2, figsize=(3.8, 3.8))
heatmap1 = sns.heatmap(auc_data1, vmin=0.5, vmax=1, annot=True, cmap=cmap, yticklabels=row_labels, xticklabels=col_labels,
            fmt='.3f', annot_kws={'size': 3.5, 'color': 'black'}, cbar=False, ax=axes[0, 0], square=False, mask=auc_data1 < 0.5)
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45, horizontalalignment='center', fontsize=3.5, weight='bold')
axes[0, 0].set_yticklabels(axes[0, 0].get_yticklabels(), rotation=0, verticalalignment='center', fontsize=3.5, weight='bold')
axes[0, 0].xaxis.tick_top()  # 将x轴刻度放在上方
axes[0, 0].yaxis.tick_left()  # 将y轴刻度放在左边
axes[0, 0].set_title('TFs+1000', rotation=0, size=5, pad=0, weight='bold')
heatmap1.tick_params(width=0.2, length=0.5)  # 设置刻度线的粗细为0.5
heatmap1.tick_params(axis='x', pad=0.1)  # 设置x轴刻度与标签的距离
heatmap1.tick_params(axis='y', pad=0.1)  # 设置y轴刻度与标签的距离


heatmap2 = sns.heatmap(auc_data2, vmin=0.5, vmax=1, annot=True, cmap=cmap, xticklabels=col_labels,
            fmt='.3f', annot_kws={'size': 3.5, 'color': 'black'}, cbar=False, ax=axes[0, 1], square=False, mask=auc_data2 < 0.5)
axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, horizontalalignment='center', fontsize=3.5, weight='bold')
axes[0, 1].set_yticklabels([])
axes[0, 1].set_yticks([])
axes[0, 1].xaxis.tick_top()  # 将x轴刻度放在上方
axes[0, 1].set_title('TFs+500', rotation=0, size=5, pad=0, weight='bold')
heatmap2.tick_params(width=0.2, length=0.5)
heatmap2.tick_params(axis='x', pad=0.1)  # 设置x轴刻度与标签的距离

heatmap3 = sns.heatmap(auc_data3, vmin=0.5, vmax=1, annot=True, cmap=cmap, yticklabels=row_labels,
            fmt='.3f', annot_kws={'size': 3.5, 'color': 'black'}, cbar=False, ax=axes[1, 0], square=False, mask=auc_data3 < 0.5)
axes[1, 0].set_xticklabels([])
axes[1, 0].set_yticklabels(axes[1, 0].get_yticklabels(), rotation=0, verticalalignment='center', fontsize=3.5, weight='bold')
axes[1, 0].xaxis.tick_top()  # 将x轴刻度放在上方
axes[1, 0].yaxis.tick_left()  # 将y轴刻度放在左边
axes[1, 0].set_xticks([])
heatmap3.tick_params(width=0.2, length=0.5)
heatmap3.tick_params(axis='y', pad=0.1)  # 设置y轴刻度与标签的距离


heatmap4 = sns.heatmap(auc_data4, vmin=0.5, vmax=1, annot=True, cmap=cmap,
            fmt='.3f', annot_kws={'size': 3.5, 'color': 'black'}, cbar=False, ax=axes[1, 1], square=False, mask=auc_data4 < 0.5)
axes[1, 1].set_xticklabels([])
axes[1, 1].set_yticklabels([])
axes[1, 1].xaxis.tick_top()  # 将x轴刻度放在上方
axes[1, 1].set_xticks([])
axes[1, 1].set_yticks([])
heatmap4.tick_params(width=0.2, length=0.5)

heatmap5 = sns.heatmap(auc_data5, vmin=0.5, vmax=1, annot=True, cmap=cmap, yticklabels=row_labels,
            fmt='.3f', annot_kws={'size': 3.5, 'color': 'black'}, cbar=False, ax=axes[2, 0], square=False, mask=auc_data5 < 0.5)
axes[2, 0].set_xticklabels([])
axes[2, 0].set_yticklabels(axes[2, 0].get_yticklabels(), rotation=0, verticalalignment='center', fontsize=3.5, weight='bold')
axes[2, 0].xaxis.tick_top()  # 将x轴刻度放在上方
axes[2, 0].yaxis.tick_left()  # 将y轴刻度放在左边
axes[2, 0].set_xticks([])
heatmap5.tick_params(width=0.2, length=0.5)
heatmap5.tick_params(axis='y', pad=0.1)  # 设置y轴刻度与标签的距离

heatmap6 = sns.heatmap(auc_data6, vmin=0.5, vmax=1, annot=True, cmap=cmap,
            fmt='.3f', annot_kws={'size': 3.5, 'color': 'black'}, cbar=False, ax=axes[2, 1], square=False, mask=auc_data6 < 0.5)
axes[2, 1].set_xticklabels([])
axes[2, 1].set_yticklabels([])
axes[2, 1].xaxis.tick_top()  # 将x轴刻度放在上方
axes[2, 1].set_xticks([])
axes[2, 1].set_yticks([])
heatmap6.tick_params(width=0.2, length=0.5)

# 设置图间距
# plt.tight_layout(pad=0.5)

plt.subplots_adjust(wspace=0.05, hspace=0.01, left=0.23)
fig.text(0.12, 0.304, 'STRING', ha='center', fontsize=5, weight='bold')
fig.text(0.115, 0.533, 'Non-Specific', ha='center', fontsize=5, weight='bold')
fig.text(0.12, 0.761, 'Specific', ha='center', fontsize=5, weight='bold')


# 添加颜色条,调整颜色条粗细和长度
cbar = fig.colorbar(heatmap1.collections[0], ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.1, pad=0.01, aspect=35)
cbar.ax.tick_params(labelsize=3.5, width=0.5, length=1.3)
plt.savefig('Heatmap/heatmap.pdf', dpi=400, bbox_inches='tight')
