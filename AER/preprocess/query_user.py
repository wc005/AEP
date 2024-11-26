import pandas as pd
import math
import numpy as np

def query_user_in_timeframe(data_df, start_time, end_time):
    """
    查询指定级联在指定时间段内的转发行为。
    """
    filtered_data = data_df.loc[start_time:end_time]
    data = []
    for group_name, group_value in filtered_data.groupby('user_id'):
        try:
            user_id = group_name
            filtered_data1 = filtered_data.xs(user_id).reset_index()

            for index, value in filtered_data1.iterrows():
                one = {
                    'user_id': user_id,
                    'cascade_id': int(value['cascade_id']),
                    'retweet_time': int(value['retweet_time']),
                    'num_participants': value['num_participants']
                }
                data.append(one)
        except KeyError:
            print('键错误！')
            continue

    data_df = pd.DataFrame(data)
    return data_df

def cascade_sequence(data_df, data1pkl_file):
    df_select = data_df.drop(['retweeted_from', 'original_user', 'path', 'original_user', 'relative_time'], axis=1)
    df_select = df_select.groupby('user_id').filter(lambda x: len(x) > 3 and len(x) < 6)
    df_select = df_select.groupby('user_id').apply(lambda x: x.sort_values('retweet_time'))
    df_select_head = df_select.groupby('user_id').head(4)
    df_select = query_user_in_timeframe(df_select_head, start_time=0, end_time=146471261900)

    df_select.to_pickle(data1pkl_file)

def impulse_time(df_sorted, data2pkl_file):
    '''
    计算间隔时间
    '''
    # 绝对时间
    df_sorted['time_since_last_event'] = df_sorted.groupby('user_id')['retweet_time'].diff()
    df_sorted['time_since_last_event'] = df_sorted['time_since_last_event'].fillna(0)
    df_sorted.to_pickle(data2pkl_file)


