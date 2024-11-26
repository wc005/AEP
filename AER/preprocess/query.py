import pandas as pd
import math
import numpy as np

def query_cascade_in_timeframe(data_df, start_time, end_time, num_participants):
    """
    查询指定级联在指定时间段内的转发行为。
    """
    filtered_data = data_df.loc[start_time:end_time]  # 基于时间范围过滤
    # filtered_data_select = filtered_data[filtered_data['num_participants'] == num_participants]
    data = []
    numi = 0
    for group_name, group_value in filtered_data.groupby('cascade_id'):
        try:
            # print(f'group_name: {group_name}, group_value: {group_value}')
            cascade_id = group_name
            # cascade_id = 4
            filtered_data1 = filtered_data.xs(cascade_id).reset_index()
            filtered_data1['retweeted_users'] = filtered_data1.apply(
                lambda row: filtered_data1[filtered_data1['retweeted_from'] == row['user_id']].to_dict('records'),
                axis=1
            )
            filtered_data1['time_factor'] = 0.0
            # 针对每一个级联中的节点计算时间影响因子
            for index, value in filtered_data1.iterrows():
                # print(f'Index: {index}, Value: {value}')
                num = len(value['retweeted_users'])
                # 如果转发节点大于0，则重新计算节点的时间影响因子
                if num > 0:
                    retweeted_users = value['retweeted_users']
                    time_factor = []
                    # 计算事件时间因子
                    for item in retweeted_users:
                        # 被转发用户转发微博的时间
                        retweet_time_for_user = value['retweet_time']
                        relative_time_for_user = value['relative_time']
                        # 转发时间
                        retweet_time = item['retweet_time']
                        # 相对最初的时间
                        relative_time = item['relative_time']
                        # 相对时间距离因子
                        t_kn = 1 / (1 + math.exp(
                            math.log(int(relative_time_for_user), 16) - math.log(int(relative_time), 16)))
                        # 绝对时间距离因子
                        t_k1 = 1 / (1 + math.exp(math.log(int(relative_time_for_user), 16)))
                        fa = t_k1 * (1 - t_kn)
                        time_factor.append(fa)
                    aaa = np.array(time_factor)
                    filtered_data1.at[index, 'time_factor'] = np.mean(aaa)
                    # print(aaa)
                    # 最初发布时间
                    # pub_timestamp = item['pub_timestamp']
                one = {
                    'cascade_id': cascade_id,
                    'user_id': int(value['user_id']),
                    'relative_time': int(value['relative_time']),
                    'retweet_time': int(value['retweet_time']),
                    'pub_timestamp': int(value['pub_timestamp']),
                    'time_factor': filtered_data1.at[index, 'time_factor'],
                    'num_participants': value['num_participants']
                }
                data.append(one)
            numi = numi + 1
            if numi % 100 == 0:
                print('cascade: {}'.format(cascade_id))
        # filtered_data = filtered_data[filtered_data['retweeted_users'].apply(lambda x: len(x) > 4)]
        except KeyError:
            print('键错误！')
            continue

    data_df = pd.DataFrame(data)
    return data_df

def user_sequence(data_df, num_participants, data1pkl_file):
    df_select = data_df.groupby('cascade_id').filter(lambda x: len(x) > 15 and len(x) < 35)
    df_select = df_select.groupby('cascade_id').apply(lambda x: x.sort_values('relative_time'))
    df_select_head = df_select.groupby('cascade_id').head(10)
    df_select = query_cascade_in_timeframe(df_select_head, start_time=0, end_time=146471261900,
                                               num_participants=num_participants)
    # 分组并在组内按转发时间排序
    # df_select = result1.groupby('cascade_id').filter(lambda x: len(x) > 4)
    df_sorted = df_select.groupby('cascade_id').apply(lambda x: x.sort_values('relative_time')).reset_index(drop=True)
    df_sorted['time_factor'] = df_sorted['time_factor'] + 1

    df_sorted.to_pickle(data1pkl_file)

def retweet_time(df_sorted, data2pkl_file):
    '''
    计算间隔时间
    '''
    df_sorted['time_since_last_event'] = df_sorted.groupby('cascade_id')['relative_time'].diff()
    df_sorted['time_since_last_event'] = df_sorted.groupby('cascade_id')['time_since_last_event'].fillna(0)
    df_sorted.to_pickle(data2pkl_file)
