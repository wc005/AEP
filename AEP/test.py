import torch

import pandas as pd
import torch.nn.functional as F


# aa = pd.DataFrame([[1,2],[4,5],[3,2],[2,4]], columns=list('ab'))
#
#
# aa['retweeted_users'] = aa.apply(
#         lambda row: aa[aa['b'] == row['a']].to_dict('records'),
#         axis=1
#     )
# aa = aa[aa['retweeted_users'].apply(lambda x: len(x) > 1)]
# print(aa['retweeted_users'][3])
# print(len(aa['retweeted_users'][3]))
#
# s = pd.Series([1, 2, 3, 4, 5])
# print(type(s))
# # 获取Series中的元素数量
# number_of_elements = len(s)
# print(number_of_elements)

import os
import numpy as np
# import pandas as pd
#
# file_path = '../casflow-data/weibo/dataset.txt'
# data_pkl = 'data.pkl'
# data1pkl_file = 'preprocess/data1pkl_file.pkl'
#
#
# data_df = pd.read_pickle(data1pkl_file)
# print('aaaa')
# data_df = pd.read_pickle('preprocess/AERpkl_file.pkl')
# print(data_df)


import json

# 示例字典
# data_dict = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
#
# # 指定文件路径
# file_path = 'data.json'
#
# file_path = 'preprocess/users_id_emd.pkl'
# with open(file_path, 'r') as file:
#     nodes_dic = json.load(file)
# for key in nodes_dic.keys():
#     if key == '997145':
#         print('jjjjjjjjjjjjjjjfind')
#         break

import torch

# 示例张量
tensor = torch.tensor([3, 56, 74, 2])

# Step 1: 对张量进行排序，获得排序索引
sorted_tensor, sorted_indices = torch.sort(tensor)

# Step 2: 创建一个与张量相同形状的结果张量，用于存放排序替换后的数值
result = torch.zeros_like(tensor)

# Step 3: 用排序顺序替换原始值
# 比如 sorted_tensor 中最小值的排序替换为 1，次小值为 2，依次类推
for i, idx in enumerate(sorted_indices):
    result[idx] = i + 1  # 这里从 1 开始排序

print("原始张量:", tensor)
print("排序替换后的张量:", result)

