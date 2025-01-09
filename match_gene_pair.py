import pandas as pd

# 读取TF-Target pair文件
tf_target_file = 'human TF-Target-information.txt'
tf_target_pairs = pd.read_csv(tf_target_file, sep='\t', header=None, names=['TF', 'target', 'tissue'])

# 读取基因表达矩阵文件
gene_expr_file = '10k_PBMCs Dataset/HVG_matrix.csv'
gene_expr_matrix = pd.read_csv(gene_expr_file, index_col=0)

# 获取基因表达矩阵中的基因名
gene_names = gene_expr_matrix.index

# 筛选TF-Target pair
filtered_pairs = tf_target_pairs.loc[(tf_target_pairs['TF'].isin(gene_names)) & (tf_target_pairs['target'].isin(gene_names)), ['TF', 'target']]
filtered_pairs.columns = ['TF', 'Target']

# 将所有TF保存为一个文件叫做TF_list.csv， 将所有Target保存为一个文件叫做Target_list.csv
tf_list = filtered_pairs['TF'].unique()
target_list = filtered_pairs['Target'].unique()

# 将每个TF与gene_expr_matrix中的基因名匹配，并记录TF在gene_expr_matrix中的索引，保存成'index'列
tf_list = pd.DataFrame(tf_list, columns=['TF'])
tf_list['index'] = tf_list['TF'].apply(lambda x: gene_expr_matrix.index.get_loc(x))
tf_list = tf_list.sort_values(by='index')
tf_list = tf_list.reset_index(drop=True)

# 将每个Target与gene_expr_matrix中的基因名匹配，并记录Target在gene_expr_matrix中的索引，保存成'index'列
target_list = pd.DataFrame(target_list, columns=['Target'])
target_list['index'] = target_list['Target'].apply(lambda x: gene_expr_matrix.index.get_loc(x))
target_list = target_list.sort_values(by='index')
target_list.columns = ['Gene', 'index']
target_list = target_list.reset_index(drop=True)

pd.DataFrame(tf_list).to_csv('10k_PBMCs Dataset/TF_list.csv', index=True)
pd.DataFrame(target_list).to_csv('10k_PBMCs Dataset/Target_list.csv', index=True)

# 将筛选后的TF-Target pair保存为CSV文件, 重排索引
filtered_pairs = filtered_pairs.reset_index(drop=True)
filtered_pairs.to_csv('10k_PBMCs Dataset/HVG_refnetwork.csv', index=True)

# filtered_pairs中的TF列和Target列替换成tf_list和target_list中的'index'列的值
filtered_pairs['TF'] = filtered_pairs['TF'].map(tf_list.set_index('TF')['index'])
filtered_pairs['Target'] = filtered_pairs['Target'].map(target_list.set_index('Gene')['index'])
filtered_pairs.to_csv('10k_PBMCs Dataset/Label.csv', index=True)

