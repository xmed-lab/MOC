#%%
import pandas as pd
import json
import os

result_path = "/home/txiang/pathology/CLAM/main_zs_policy_analysis_results_old.json"
output_path = "/home/txiang/pathology/CLAM/main_zs_policy_analysis_results_old.csv"

with open(result_path, 'r') as f:
    results = json.load(f)


#%%
'''
result 
  nsclc/c16/rcc/eb12/eb30
       topk/delta-softmax/delta-diff/topk*delta-softmax/topk*delta-diff/bottomk-irrel/bottomk-irrel*delta-softmax/bottomk-irrel*delta-diff/bottomk-irrel*delta-softmax then topk/bottomk-irrel*delta-diff then topk

  auc/bacc

'''

datasets = ['nsclc', 'c16', 'rcc', 'ebrains12', 'ebrains30']
methods = ['topk', 'delta-softmax', 'delta-diff', 'topk * delta-softmax', 'topk * delta-diff', 'bottomk-irrel', 'bottomk-irrel * delta-softmax', 'bottomk-irrel * delta-diff', 'bottomk-irrel * delta-softmax then topk', 'bottomk-irrel * delta-diff then topk']
metrics = ['roc_auc', 'bacc']

data = []

for method in methods:
    row = [method]
    for dataset in datasets:
        roc_auc = results[dataset][method]['roc_auc']['10']
        roc_auc = round(roc_auc, 3)
        bacc = results[dataset][method]['bacc']['10']
        bacc = round(bacc, 3)
        row.extend([roc_auc, bacc])
    data.append(row)

columns = ['Method']
for dataset in datasets:
    columns.extend([f'{dataset}_roc_auc', f'{dataset}_bacc'])

df = pd.DataFrame(data, columns=columns)
df.to_csv(output_path, index=False)
