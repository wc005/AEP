import pandas as pd
import os


def read_data_in_batches(file_path, batch_size=300):
    """
    逐批次读取 data.txt 文件数据
    """
    with open(file_path, 'r') as file:
        while True:
            lines = [file.readline().strip() for _ in range(batch_size)]
            if not lines[0]:
                break
            yield lines


def process_data(lines):
    data = []

    for line in lines:
        if not line:
            continue  # 跳过空行

        parts = line.split('\t')
        cascade_id = int(parts[0])  # 级联ID
        original_user = int(parts[1])  # 原始用户ID
        pub_timestamp = int(parts[2])  # 发布时间戳
        num_participants = int(parts[3])  # 参与者数量

        participants_info = parts[4].split(' ')
        for info in participants_info:
            path, relative_time = info.split(':')
            retweet_time = pub_timestamp + int(relative_time)
            retweeted_from = path.split('/')[-2] if '/' in path else original_user
            user_id = path.split('/')[-1]

            if int(user_id) != original_user:
                data.append({
                    'cascade_id': cascade_id,
                    'user_id': int(user_id),
                    'retweeted_from': int(retweeted_from),
                    'relative_time': relative_time,
                    'retweet_time': retweet_time,
                    'original_user': original_user,
                    'pub_timestamp': pub_timestamp,
                    'num_participants': num_participants,
                    'path': path
                })

    data_df = pd.DataFrame(data)

    return data_df


def process_and_save(file_path, pkl_file, batch_size=300):
    """
    处理数据并保存为 PKL 文件
    """
    all_data = pd.DataFrame()

    # 逐批读取和处理数据
    for batch in read_data_in_batches(file_path, batch_size):
        # 处理数据并生成DataFrame
        data_df = process_data(batch)

        # 将每批数据合并到 all_data
        all_data = pd.concat([all_data, data_df], ignore_index=True)

    # 设置索引
    all_data.set_index(['cascade_id'], inplace=True, drop=True)
    # all_data.set_index(['cascade_id'], inplace=True, drop=True)
    all_data.sort_index(inplace=True)

    # 保存到 PKL 文件
    all_data.to_pickle(pkl_file)

    print("数据已成功批量处理并保存为 PKL 文件！")

