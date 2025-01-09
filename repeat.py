import os

file_list = ['sample1', 'sample2', 'sample3', 'sample4', 'sample5']
# file_list = ['sample1']
cell_list = ['mHSC-GM', 'hESC', 'mHSC-L', 'mDC', 'mHSC-E', 'hHEP']
# cell_list = ['mHSC-GM', 'mHSC-L', 'mHSC-E']

for sample in file_list:
    for cell_type in cell_list:
        os.system('python main_transfer_shared.py -sample %s -cell_type %s' % (sample, cell_type))
