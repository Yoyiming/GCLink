import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 读取CSV文件
GENELink_AUROC = pd.read_csv("GENELink_Transfer_result_AUROC.csv", header=0)
GENELink_AUPRC = pd.read_csv("GENELink_Transfer_result_AUROC.csv", header=0)
GCGRN_AUROC = pd.read_csv("GCGRN_Transfer_result_AUROC.csv", header=0)
GCGRN_AUPRC = pd.read_csv("GCGRN_Transfer_result_AUROC.csv", header=0)

# 重塑数据
melted_AUROC = pd.concat([GENELink_AUROC, GCGRN_AUROC], keys=['GENELink', 'GCLink'], names=['Methods'])
melted_AUPRC = pd.concat([GENELink_AUPRC, GCGRN_AUPRC], keys=['GENELink', 'GCLink'], names=['Methods'])
melted_AUROC = melted_AUROC.reset_index().melt(id_vars='Methods', var_name='Dataset', value_name='AUROC')
melted_AUPRC = melted_AUPRC.reset_index().melt(id_vars='Methods', var_name='Dataset', value_name='AUPRC')
# 删除不需要的行
melted_AUROC = melted_AUROC.drop(melted_AUROC[melted_AUROC['Dataset'] == 'level_1'].index)
melted_AUPRC = melted_AUPRC.drop(melted_AUPRC[melted_AUPRC['Dataset'] == 'level_1'].index)
# 绘制分组箱线图
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4.8, 2.8))

sns.boxplot(x='Dataset', y='AUROC', hue='Methods', data=melted_AUROC, palette='Set2', ax=axes[0],
            flierprops={'marker':'D', 'markerfacecolor':'grey', 'markeredgecolor':'grey'}, fliersize=1)
# axes[0].set_title('Source Dataset : mESC', fontsize=8)
axes[0].set_xlabel('', fontsize=4, labelpad=0.5)
axes[0].set_ylabel('AUROC', fontsize=6, labelpad=0.6)
axes[0].legend(title='', loc='upper right', fontsize=5)
axes[0].tick_params(axis='both', which='major', labelsize=4.5, width=0.3, length=1.2)

sns.boxplot(x='Dataset', y='AUPRC', hue='Methods', data=melted_AUPRC, palette='Set2', ax=axes[1],
            flierprops={'marker':'D', 'markerfacecolor':'grey', 'markeredgecolor':'grey'}, fliersize=1)
# axes[1].set_title('Source Dataset : mESC', fontsize=8)
axes[1].set_xlabel('', fontsize=4, labelpad=0.5)
axes[1].set_ylabel('AUPRC', fontsize=6, labelpad=0.6)
axes[1].legend(title='', loc='upper right', fontsize=5)
axes[1].tick_params(axis='both', which='major', labelsize=4.5, width=0.3, length=1.2)

# 调整子图间距
# plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.05, wspace=0.2, hspace=0.3)
# 在两个子图的左上角分别添加一个A和B
axes[0].text(0, 1.03, 'A', fontsize=10, fontweight='bold', transform=axes[0].transAxes)
axes[1].text(0, 1.03, 'B', fontsize=10, fontweight='bold', transform=axes[1].transAxes)


# 保存图像
plt.savefig("boxplot_GENELink_GCLink.pdf", dpi=400)
