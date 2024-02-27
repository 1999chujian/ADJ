#!pip install dgl -i https://mirrors.aliyun.com/pypi/simple/ --target /dfs/data
!export DGLBACKEND=pytorch
import argparse
import copy
import logging
import math
import multiprocessing
import os
import sys
from datetime import datetime, timedelta
import random
import inspect
import numpy as np
import torch.optim as optim
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from scipy import stats
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from multiprocessing import Pool
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import warnings
import yaml
import time
from multiprocessing.pool import ThreadPool

warnings.filterwarnings("ignore")


class Utils():
    def __init__(self, configs=None, n_thres=100):
        self.n_thres = n_thres # 计算指标时非nan样本数要求
        self.configs = configs

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_filename(self, configs):
        name = '-'.join([configs['model']['typ'], configs['model']['model_name']])
        name = name + configs['others']['suffix']
        return name

    def get_device(self):
        if torch.cuda.is_available():
            n_gpu = torch.cuda.device_count()
            print(f'Number of gpu: {n_gpu}')
            print(f'GPU name: {torch.cuda.get_device_name(0)}')
            print('GPU is on')
        else:
            print('GPU is off, using CPU instead.')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device

    def to_device(self, model, device, local_rank):
        print("Device index:", local_rank)  # 添加打印语句，输出设备索引
        if self.configs['others']['DDP']: # DDP
            model = model.to(local_rank)
            model = MyDistributedDataParallel(model,
                                              device_ids=[local_rank],
                                              output_device=local_rank,
                                              find_unused_parameters=True)
        else: # DP
            model.to(device)  # to cuda
            model = MyDataParallel(model)
        return model

    def model_configuration(self, model):
        print(f'Model Architecture: {model}')
        num_params = sum(p.numel() for p in model.parameters())
        print(f'Number of Model Parameters: {num_params}')

    def disable_dropout(self, model):
        for layer in model.children():
            if isinstance(layer, nn.Dropout):
                layer.p = 0.0 # 将dropout层的概率设置为0
            elif isinstance(layer, nn.Module):
                self.disable_dropout(layer) # 递归调用子模块

    def gen_timestamp(self, start_time_hour, start_time_min, interval, step):
        # 固定日期
        base_date = datetime(2000, 1, 1, start_time_hour, start_time_min, 0)
        # 时间间隔
        interval = timedelta(minutes=interval)
        # 生成6位数字表示的时间
        timestamps = [(base_date + i * interval).strftime('%H%M%S') for i in range(step)]
        return timestamps

    def get_calendar(self, timestamp):
        timestamp = datetime.strptime(timestamp, '%Y-%m-%d')
        month = timestamp.month
        week = timestamp.isocalendar()[1]
        day_of_week = timestamp.weekday() + 1
        return (month, week, day_of_week)
        
    def get_date_range(self, data_info, mode):
        date_train_start = self.configs['data']['date_train_start']
        date_val_start = self.configs['data']['date_val_start']
        date_test_start = self.configs['data']['date_test_start']

        if mode == 'train':
            print(f'Loading training data...')
            idx = [datetime.strptime(_, '%Y%m%d') >= datetime.strptime(date_train_start, '%Y%m%d') and
                   datetime.strptime(_, '%Y%m%d') < datetime.strptime(date_val_start, '%Y%m%d')
                   for _ in data_info['dates_trade']]
        elif mode == 'val':
            print(f'Loading validation data...')
            idx = [datetime.strptime(_, '%Y%m%d') >= datetime.strptime(date_val_start, '%Y%m%d') and
                   datetime.strptime(_, '%Y%m%d') < datetime.strptime(date_test_start, '%Y%m%d')
                   for _ in data_info['dates_trade']]
        elif mode == 'test':
            print(f'Loading testing data...')
            idx = [datetime.strptime(_, '%Y%m%d') >= datetime.strptime(date_test_start, '%Y%m%d')
                   for _ in data_info['dates_trade']]
        else:
            raise NotImplementedError
        return idx

    def rm_nan(self, v1, v2):
        assert v1.ndim == 1 and v2.ndim == 1
        idx = ~np.isnan(v1) & ~np.isnan(v2)
        v1 = v1[idx]
        v2 = v2[idx]
        return v1, v2

    def zscore(self, y, eps=1e-6):
        y = (y - torch.mean(y)) / (torch.std(y) + eps)
        return y

    def IC(self, y_pred, y_true):
        y_pred, y_true = self.rm_nan(y_pred, y_true)
        if len(y_pred) >= self.n_thres:
            return np.corrcoef(y_pred, y_true)[0, 1]
        else:
            return np.nan

    def RankIC(self, y_pred, y_true):
        y_pred, y_true = self.rm_nan(y_pred, y_true)
        if len(y_pred) >= self.n_thres:
            return stats.spearmanr(y_pred, y_true)[0]
        else:
            return np.nan

    def LongShort(self, y_pred, y_true):
        y_pred, y_true = self.rm_nan(y_pred, y_true)
        if len(y_pred) >= self.n_thres:
            idx_short = np.argsort(y_pred)[:int(0.2 * len(y_pred))] # argsort 升序
            idx_long = np.argsort(y_pred)[-int(0.2 * len(y_pred)):]
            return np.mean(y_true[idx_long]) - np.mean(y_true[idx_short])
        else:
            return np.nan

    def GroupReturn(self, y_pred, y_true, quantile_l, quantile_r):
        y_pred, y_true = self.rm_nan(y_pred, y_true)
        if len(y_pred) >= self.n_thres:
            idx_group = np.argsort(y_pred)[int(quantile_l * len(y_pred)):
                                           int(quantile_r * len(y_pred))]
            return np.mean(y_true[idx_group])
        else:
            return np.nan

    def Backtest(self, y_pred, y_true):
        data_info = np.load(self.configs['others']['info_path'], allow_pickle=True)
        # 现在每个batch的维度是(stock, seq, feature), 然后在date * timestamps的维度上迭代取batch
        # 所以y应该reshape成(date, timestamp, stock)的三维张量
        y_pred = y_pred.reshape(-1, len(data_info['timestamps'])-3, len(data_info['codes']))
        y_true = y_true.reshape(-1, len(data_info['timestamps'])-3, len(data_info['codes']))

        # 向量化自定义函数
        vectorized_func_IC = np.vectorize(self.IC, signature='(n),(n)->()')
        vectorized_func_RankIC = np.vectorize(self.RankIC, signature='(n),(n)->()')
        vectorized_func_LongShort = np.vectorize(self.LongShort, signature='(n),(n)->()')
        vectorized_func_GroupReturn = np.vectorize(self.GroupReturn, signature='(n),(n),(),()->()')

        # 应用函数到两个矩阵上, 得到(date, timestamp)维度的回测表现matrix, np.vectorize在最后一维进行
        result_matrix = {}
        result_matrix['Alpha'] = y_pred
        result_matrix['GroundTruth'] = y_true
        result_matrix['IC'] = vectorized_func_IC(y_pred, y_true)
        result_matrix['RankIC'] = vectorized_func_RankIC(y_pred, y_true)
        result_matrix['LongShort'] = vectorized_func_LongShort(y_pred, y_true)
        for i, _ in enumerate(np.arange(0, 1.0, 0.2)):
            result_matrix[f'GroupReturn{str(i+1)}'] = vectorized_func_GroupReturn(y_pred, y_true, _, (_+0.2))

        return result_matrix

utils = Utils()

timestamps = []
timestamps += utils.gen_timestamp(9, 30, interval=30, step=4+1)
timestamps += utils.gen_timestamp(13, 30, interval=30, step=4)
df_timestamps = pd.DataFrame(range(9), columns=['minuteCode']).astype('float')

# training: 2018.12.31-2020.06.30
# validation: 2020.07.01-2020.08.15
# testing: 2020.08.16-2020.09.30
dates = []
start_date = datetime(2018, 12, 31)
end_date = datetime(2020, 9,30)
current_date = start_date
while current_date <= end_date:
    year = current_date.year
    month = current_date.month
    day = current_date.day
    dates.append(str(year).zfill(4) + str(month).zfill(2) + str(day).zfill(2))
    current_date += timedelta(days=1)

count = 0
dates_trade = []
for date in tqdm(dates):
    try:
        df = pd.read_feather(f'/dfs/dataset/10-1704454471157/data/data_norm/tc1/{date[:4]}/{date}/e100000.feather')
        if count == 0:
            codes = set(df['symbol'])
        else:
            codes = codes.union(set(df['symbol']))
        count += 1
        dates_trade.append(date)
    except Exception as error:
        #print(error)
        pass
        continue

del dates
codes = list(codes)
df_codes = pd.DataFrame(codes, columns=['code'])  #4063*1

print(len(dates_trade))

x_columns = [f'f{i+1}' for i in range(170)] # 170因子
y_columns = ['yhat_raw_ret_v2v_1d']


datetimestamps = []
for date in dates_trade:
    for timestamp in timestamps:
        datetimestamp = date + timestamp
        datetimestamps.append(datetimestamp)


# def generator(datetimestamps):
#     df_list = []
#     for dt in tqdm(datetimestamps):
#         df_x = pd.read_feather(f'/dfs/dataset/10-1704454471157/data/data_norm/tc1/{dt[:4]}/{dt[:8]}/e{dt[-6:]}.feather')
#     # y file (30min label, 复权后)
#         df_y = pd.read_feather(f'/dfs/dataset/10-1704454471157/data/data_norm/label/{dt[:4]}/{dt[:8]}/e{dt[-6:]}.feather')
#         df_y = df_y[['skey','yhat_raw_ret_v2v_1d','date']]

#         # x
#         df_x.rename(columns={'symbol': 'code'}, inplace=True)
#         df_x = df_codes.merge(df_x, how='left', on='code')

#         # y
#         df_y.rename(columns={'skey': 'code'}, inplace=True)
#         df_y = df_codes.merge(df_y, how='left', on='code')
        

#         df_temp = pd.merge(df_x, df_y, on='code', how='left')
#         df_temp =df_temp.drop('code',axis=1)
#         df_temp['datetimestamp'] = dt
#         df_list.append(df_temp)
#     df_merged = pd.concat(df_list, ignore_index=True)
#     return df_merged

import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

# 假设 df_codes 已经在外部定义



def process_datetimestamp(dt):
    df_x = pd.read_feather(f'/dfs/dataset/10-1704454471157/data/data_norm/tc1/{dt[:4]}/{dt[:8]}/e{dt[-6:]}.feather')
    df_y = pd.read_feather(f'/dfs/dataset/10-1704454471157/data/data_norm/label/{dt[:4]}/{dt[:8]}/e{dt[-6:]}.feather')
    df_y = df_y[['skey','yhat_raw_ret_v2v_1d','date','isZT','isDT']]
    
    df_x.rename(columns={'symbol': 'code'}, inplace=True)
    df_x = df_codes.merge(df_x, how='left', on='code')

    df_y.rename(columns={'skey': 'code'}, inplace=True)
    df_y = df_codes.merge(df_y, how='left', on='code')

    df_temp = pd.merge(df_x, df_y, on='code', how='left')
    df_temp.loc[~((df_temp['isZT'] == 0.0) & (df_temp['isDT'] == 0.0)), 'yhat_raw_ret_v2v_1d'] = np.nan
    df_temp= df_temp.dropna(subset = 'yhat_raw_ret_v2v_1d')
    df_temp['datetimestamp'] = dt
    
    
    
    return df_temp

def generator(datetimestamps):
    with Pool(processes=152) as pool:  # 选择一个适当的进程数量
        df_list = list(tqdm(pool.imap(process_datetimestamp, datetimestamps), total=len(datetimestamps)))
    df_merged = pd.concat(df_list, ignore_index=True)
    return df_merged

RAW_DATA = generator(datetimestamps)
RAW_DATA['datetimestamp'] = RAW_DATA['datetimestamp'].astype(int)
size_in_bytes = sys.getsizeof(RAW_DATA)
size_in_mb = size_in_bytes / (1024**2)
print(f"RAW_DATA size: {size_in_mb:.2f} MB")

RAW_DATA.to_parquet('/dfs/dataset/1704939107951/data/rawdata/RAW_DATA_ztdttonan.parquet')

# RAW_DATAv = RAW_DATA.values
# f= open(f'/dfs/dataset/1704939107951/data/rawdata/RAW_DATA.dat', 'wb')
# f.write(RAW_DATAv.astype(np.float32).tobytes())
# f.flush()


# x = np.memmap('/dfs/dataset/1704939107951/data/rawdata/RAW_DATA.dat', dtype=np.float32, mode='r',
#                            shape=(len(datetimestamp)*len(df_codes), 173)) 


def train_test_oos_cut(RAW_DATA, date1, date2):
    RAW_DATA["date"] = pd.to_datetime(RAW_DATA["date"], format="%Y%m%d")
    dt =RAW_DATA["date"]
    train_data = RAW_DATA[dt < date1]
    test_data = RAW_DATA[(dt < date2 )& (dt >= date1)]
    predict_data = RAW_DATA[dt >= date2]
    return train_data,test_data,predict_data


class myDataSet(Dataset):
    def __init__(self, x, y, times, relations):
        super(myDataSet, self).__init__()
        self.feature = x
        self.label = y
        self.relation = relations
        self.t = times.unique()
        
    def __getitem__(self, index):
        feature = self.feature[index]
        label = self.label[index]
        relation = self.relation[index]
        return feature, label, relation
    
    def __len__(self):
        return len(self.t)
    

#codes_list = df_codes['code'].tolist()

def load_adj(args):
    time = args
    adj = pd.read_csv(f'/dfs/dataset/190-1705298898613/data/StockIndustry/{str(time)[:4]}/{str(time)[:8]}.csv', index_col=0)
    codes_list = df1[df1['datetimestamp']== time]['code'].unique().tolist()
    new_adj = adj.reindex(index=codes_list, columns=codes_list, fill_value=0)
    print(time)
    new_adj =new_adj.values
    f= open(f'/dfs/dataset/1704939107951/data/adj/Industry_{time}_ynonan.dat', 'wb')
    f.write(new_adj.astype(np.float32).tobytes())
    f.flush()
    del adj, new_adj

def load_all_adjs(unique_times):
    with Pool(processes=152) as pool:
        time_codes_pairs = [time for (i,time) in enumerate(unique_times)]
        pool.map(load_adj, time_codes_pairs)

    

other_cols = ['datetimestamp', 'date','yhat_raw_ret_v2v_1d']

train_data,test_data,predict_data = train_test_oos_cut(RAW_DATA, '20200701', '20200816')

unique_times = np.concatenate((RAW_DATA['datetimestamp'].unique()))

print('开始取图')
relations = load_all_adjs(RAW_DATA['datetimestamp'].unique())
print('结束取图')
