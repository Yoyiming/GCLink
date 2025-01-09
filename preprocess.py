import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import itertools
from pathlib import Path

plt.rc('font')
my_colors = ["#1EB2A6", "#ffc4a3", "#e2979c", "#F67575"]
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=400, facecolor='white')  # 设置输出图像格式

# 读取TF list
ActNK_tf_list = pd.read_csv('pbmc_initial_TRN_adj_matrix/ActNK_all_cell_matrix.TF.list.txt', header=None)
ActNK_tf_list = ActNK_tf_list[0].tolist()

CD8_T_tf_list = pd.read_csv('pbmc_initial_TRN_adj_matrix/CD8Tcells_all_cell_matrix.TF.list.txt', header=None)
CD8_T_tf_list = CD8_T_tf_list[0].tolist()

CD4_T_tf_list = pd.read_csv('pbmc_initial_TRN_adj_matrix/NaiveCD4Tcells_all_cell_matrix.TF.list.txt', header=None)
CD4_T_tf_list = CD4_T_tf_list[0].tolist()

CD14_Monocyte_tf_list = pd.read_csv('pbmc_initial_TRN_adj_matrix/Monocytes_all_cell_matrix.TF.list.txt', header=None)
CD14_Monocyte_tf_list = CD14_Monocyte_tf_list[0].tolist()

# 取并集
tf_list = list(set(ActNK_tf_list + CD8_T_tf_list + CD4_T_tf_list + CD14_Monocyte_tf_list))
print(len(tf_list))

adata = sc.read_10x_mtx('pbmc8k_raw_gene_bc_matrices/GRCh38/', var_names='gene_symbols', cache=True)
adata.var_names_make_unique()
adata.obs_names_make_unique()
# 过滤低质量细胞和基因
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# 线粒体基因
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, inplace=True, log1p=False)
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True)
plt.savefig('Preprocess/violin.png')

# 绘制QC图
mito_filter = 8
n_counts_filter = 4000
fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', ax=axs[0], show=False)
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', ax=axs[1], show=False)
axs[0].hlines(y=mito_filter, xmin=0, xmax=max(adata.obs['total_counts']), color='red', ls='dashed')
axs[1].hlines(y=n_counts_filter, xmin=0, xmax=max(adata.obs['total_counts']), color='red', ls='dashed')
fig.tight_layout()
plt.savefig('Preprocess/qc_metrics.png')

adata = adata[adata.obs.n_genes_by_counts < 4000, :]
adata = adata[adata.obs.pct_counts_mt < 8, :]

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

sc.pl.highly_variable_genes(adata)
plt.savefig('Preprocess/hvg.png')
print('HVG numbers:', adata.var.highly_variable.sum())
for tf in tf_list:
    if tf in adata.var_names.values:
        adata.var.highly_variable[adata.var_names == tf] = True
print('HVG numbers:', adata.var.highly_variable.sum())

adata.raw = adata
adata = adata[:, adata.var.highly_variable]
sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack')
sc.pl.pca(adata, color='CST3')
plt.savefig('Preprocess/pca.png')

sc.pl.pca_variance_ratio(adata, log=True)
plt.savefig('Preprocess/pca_variance_ratio.png')

sc.pp.neighbors(adata, n_neighbors=20, n_pcs=30)
sc.tl.tsne(adata)
sc.pl.tsne(adata, color=['CST3', 'NKG7', 'PPBP'])
plt.savefig('Preprocess/tsne.png')
sc.pl.tsne(adata, color=['CST3', 'NKG7', 'PPBP'], use_raw=False)
plt.savefig('Preprocess/tsne_use_raw.png')

sc.pp.neighbors(adata, n_neighbors=20, n_pcs=30)
sc.tl.leiden(adata, resolution=0.5)
sc.pl.tsne(adata, color='leiden', legend_loc='on data', frameon=False, legend_fontsize=10, legend_fontoutline=2)
plt.savefig('Preprocess/leiden.png')

sc.tl.rank_genes_groups(adata, groupby="leiden", use_raw=False, method="wilcoxon")
sc.pl.rank_genes_groups(adata, groupby="leiden", n_genes=25, dendrogram=True, standard_scale='var')
plt.savefig('Preprocess/rank_genes_groups.png')

sc.pl.rank_genes_groups_dotplot(adata, groupby="leiden", n_genes=5, dendrogram=True, standard_scale='var')
plt.savefig('Preprocess/rank_genes_groups_dotplot.png')

result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
marker_genes_df = pd.DataFrame({group + '-' + key[:1]: result[key][group]for group in groups for key in ['names', 'pvals']}).head(10)
marker_genes_df.to_csv("Preprocess/marker_genes.csv")

adata.obs['leiden'].value_counts()
print(adata.obs['leiden'].value_counts())

new_cluster_names = {'0':'CD14 Monocyte',
                     '1':'CD4 T',
                     '2':'B',
                     '3':'ActNK',
                     '4':'CD8+ T',
                     '5':'CD4 T',
                     '6':'CD16 Monocyte',
                     '7':'Dendritic',
                     '8':'CD14 Monocyte',
                     '9':'CD8+ T',
                     '10':'Other',
                     '11':'Other',
                     '12':'Other',
                     }
adata.obs['cell type'] = adata.obs['leiden'].map(new_cluster_names).astype('category')
sc.pl.tsne(adata, color='cell type', legend_loc='on data', frameon=False, legend_fontsize=10, legend_fontoutline=2)
plt.savefig('Preprocess/cell_annotation.png')
print(adata)
# 保存数据
adata.write(Path('Preprocess/pbmc8k.h5ad'))

clus0 = adata[adata.obs['leiden'].isin(['1', '5']), :]
clus0.copy().T.to_df().to_csv("Preprocess/CD4 T.csv")
print(clus0.X.shape)

clus1 = adata[adata.obs['leiden'].isin(['0', '8']), :]
clus1.copy().T.to_df().to_csv("Preprocess/CD14 Monocyte.csv")
print(clus1.X.shape)

clus3 = adata[adata.obs['leiden'].isin(['4', '9']), :]
clus3.copy().T.to_df().to_csv("Preprocess/CD8+ T.csv")
print(clus3.X.shape)

clus4 = adata[adata.obs['leiden'].isin(['3']), :]
clus4.copy().T.to_df().to_csv("Preprocess/ActNK.csv")
print(clus4.X.shape)

clus5 = adata[adata.obs['leiden'].isin(['2']), :]
clus5.copy().T.to_df().to_csv("Preprocess/B.csv")
print(clus5.X.shape)


