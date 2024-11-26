import numpy as np
import torch
import torch.utils.data

from transformer import Constants

def flatten_and_rank(lst):
    # 展平二维列表并记录原始索引
    flat_list = [(value, (i, j)) for i, sublist in enumerate(lst) for j, value in enumerate(sublist)]

    # 对展平后的列表进行排序
    sorted_flat_list = sorted(flat_list, key=lambda x: x[0])

    # 创建一个字典来映射每个值到其排序后的序号（从1开始）
    rank_dict = {value: rank + 1 for rank, (value, _) in enumerate(sorted_flat_list)}

    # 构建一个新的二维列表，用排序后的序号替换原始值
    ranked_2d_list = []
    for i, sublist in enumerate(lst):
        ranked_sublist = []
        for j, value in enumerate(sublist):
            # 使用原始索引来找到正确的排序序号
            original_index = (i, j)
            # 注意：这里我们其实不需要找到原始索引，因为rank_dict已经通过值映射了序号
            # 但为了展示如何关联原始位置和排序后的序号，我保留了这一步
            ranked_value = rank_dict[value]
            ranked_sublist.append(ranked_value)
        ranked_2d_list.append(ranked_sublist)

    return ranked_2d_list

class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self, data):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """
        # 绝对时间
        self.time = [[elem['time_since_start'] for elem in inst] for inst in data]
        # 间隔时间
        self.time_gap = [[elem['time_since_last_event'] for elem in inst] for inst in data]
        # plus 1 since there could be event type 0, but we use 0 as padding
        event_type = [[elem['type_event'] for elem in inst] for inst in data]
        self.event_type = flatten_and_rank(event_type)
        a = max(max(self.event_type))
        print(a)
        self.time_factor = [[elem['time_factor'] for elem in inst] for inst in data]

        self.cascade_id = [[elem['cascade_id'] for elem in inst] for inst in data]
        self.num_participants = [[elem['num_participants'] for elem in inst] for inst in data]

        self.length = len(data)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        return self.time[idx], self.time_gap[idx], self.event_type[idx], self.cascade_id[idx], self.time_factor[idx], self.num_participants[idx]

def pad_time(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.float32)


def pad_type(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.long)


def pad_time_factor(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.long)


def pad_cascade_id(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)
    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])
    return torch.tensor(batch_seq, dtype=torch.long)


def pad_num_participants(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)
    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])
    return torch.tensor(batch_seq, dtype=torch.long)


def collate_fn(insts):
    """ Collate function, as required by PyTorch. """

    time, time_gap, event_type, time_factor, cascade_id, num_participants = list(zip(*insts))
    time = pad_time(time)
    time_gap = pad_time(time_gap)
    event_type = pad_type(event_type)
    time_factor = pad_time_factor(time_factor)
    cascade_id = pad_cascade_id(cascade_id)
    num_participants = pad_num_participants(num_participants)
    return time, time_gap, event_type, time_factor, cascade_id, num_participants


def user_get_dataloader(data, batch_size, shuffle=True):
    """ Prepare dataloader. """

    ds = EventData(data)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
    )
    return dl
