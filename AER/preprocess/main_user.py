import os
import pandas as pd
from save import process_and_save
import  query_user
import pickle
import torch
import json

if __name__ == "__main__":
    file_path = '../../casflow-data/weibo/dataset.txt'
    data_pkl = 'data.pkl'
    user1pkl_file = 'user_file.pkl'
    user2pkl_file = 'user2_file.pkl'

    # 检查是否已经存在处理后的 CSV 文件
    # 预处理
    if not (os.path.exists(data_pkl)):
        process_and_save(file_path, data_pkl)
    data_df = pd.read_pickle(data_pkl)
    data_df = data_df.reset_index(drop=False)
    data_df.set_index(['user_id'], inplace=True, drop=True)
    num_participants = 26

    # 检查是否已经存在处理后的 pkl 文件user_id
    if not (os.path.exists(user1pkl_file)):
        # 获取每个用户的冲激序列
        query_user.cascade_sequence(data_df, user1pkl_file)
    # 读取处理时间因子序列
    data_df = pd.read_pickle(user1pkl_file)
    if not (os.path.exists(user2pkl_file)):
        query_user.impulse_time(data_df, user2pkl_file)
    data_df = pd.read_pickle(user2pkl_file)

    data_t = []
    for group_name, group_value in data_df.groupby('user_id'):
        temp = pd.DataFrame()
        temp['time_since_start'] = group_value['retweet_time']
        temp['time_since_last_event'] = group_value['time_since_last_event']
        temp['type_event'] = group_value['cascade_id']
        temp['user_id'] = group_name
        temp['num_participants'] = group_value['num_participants']
        cascade = temp.to_dict('records')
        data_t.append(cascade)

    print(len(data_t))

    # 保存到 PKL 文件
    pkl_file = 'AER_user_pkl_file.pkl'
    # 保存到 Pickle 文件
    with open(pkl_file, 'wb') as f:
        pickle.dump(data_t, f)

    with open(pkl_file, 'rb') as pkl_file:
        aa = pickle.load(pkl_file)
    # print(aa)
    # 查询用户在时间段内的转发行为
    # result2 = query.query_user_in_timeframe(data_df, user_id=1239, start_time=1464710400, end_time=1464712619)
    # print("Columns:", result2.columns)
    # print(result2)