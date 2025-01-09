import pandas as pd
import numpy as np

train_file = 'Data/STRING/mDC 1000/sample1/Train_set.csv'
test_file = 'Data/STRING/mDC 1000/sample1/Test_set.csv'
val_file = 'Data/STRING/mDC 1000/sample1/Validation_set.csv'

train_data = pd.read_csv(train_file, index_col=0)
test_data = pd.read_csv(test_file, index_col=0)
validation_data = pd.read_csv(val_file, index_col=0)

train_edges_num = train_data['Label'].sum()
test_edges_num = test_data['Label'].sum()
validation_edges_num = validation_data['Label'].sum()

print('Train edges:', train_edges_num)
print('Validation edges:', validation_edges_num)
print('Test edges:', test_edges_num)
