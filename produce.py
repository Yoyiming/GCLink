import csv

file_list = ['sample1', 'sample2', 'sample3', 'sample4', 'sample5']
cell_list = ['mHSC-E', 'mHSC-GM', 'mHSC-L', 'hESC', 'hHEP', 'mDC']
# cell_list = ['mHSC-GM', 'mHSC-L', 'mHSC-E']


# For transfer result
# with open('GRNResult/Specific/mHSC-E1000_shared_result.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['AUC', 'AUPR'])
#     for sample in file_list:
#         for cell_type in cell_list:
#             txt_filename = f'AUC_AUPR_Result/Specific/TF+1000/hHEP500_{sample}_{cell_type}.txt'
#             with open(txt_filename, 'r') as f:
#                 auc = float(f.readline().strip())
#                 aupr = float(f.readline().strip())
#
#             writer.writerow([auc, aupr])

with open('GRNResult/mESC1000_finetune_shared_i+e_False_0.05_nommd_result.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['AUC', 'AUPR'])
    for cell_type in cell_list:
        auc_list = []
        aupr_list = []
        for sample in file_list:
            txt_filename = f'AUC_AUPR_Result/Specific/TF+1000/mESC1000_{sample}_{cell_type}_i+e_False_transfer_finetune_nommd.txt'
            with open(txt_filename, 'r') as f:
                auc = float(f.readline().strip())
                aupr = float(f.readline().strip())
                auc_list.append(auc)
                aupr_list.append(aupr)

        # Calculate the average AUC and AUPR
        avg_auc = sum(auc_list) / len(auc_list)
        avg_aupr = sum(aupr_list) / len(aupr_list)

        # Write the average values in the last row
        writer.writerow(['Average', ''])
        writer.writerow([avg_auc, avg_aupr])

# For main result
# for cell_type in cell_list:
#     auc_list = []
#     aupr_list = []
#     with open('Result/Specific/TF+1000/' + cell_type + '_i+e_shared_False_contrast_result.csv', 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['AUC', 'AUPR'])
#         for sample in file_list:
#             txt_filename = f'Result/Specific/TF+1000/{sample}_{cell_type}_i+e_shared_False_contrast.txt'
#             with open(txt_filename, 'r') as f:
#                 auc = float(f.readline().strip())
#                 aupr = float(f.readline().strip())
#                 auc_list.append(auc)
#                 aupr_list.append(aupr)
# 
#                 writer.writerow([auc, aupr])
#         # Calculate the average AUC and AUPR
#         avg_auc = sum(auc_list) / len(auc_list)
#         avg_aupr = sum(aupr_list) / len(aupr_list)
# 
#         # Write the average values in the last row
#         writer.writerow(['Average', ''])
#         writer.writerow([avg_auc, avg_aupr])
