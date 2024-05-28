"""
Dataset classes for ease of training. For readability, we have created a class per dataset type.

These classes have been simplified from other repositories (e.g., including PatchTST [TODO])
"""
    
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os


class DatasetETTs():
    def __init__(self, 
                 path,
                 freq='h',
                 split='train', 
                 context_length=336, 
                 horizon=336, 
                 ):
        assert freq in ['h', 'm']
        assert split in ['train', 'test', 'val']
        
        self.path = path
        self.context_length = context_length
        self.horizon = horizon
        self.split = split
        
        if freq=='h':
            self.border1s = [0, 12 * 30 * 24 - self.context_length, 12 * 30 * 24 + 4 * 30 * 24 - self.context_length]
            self.border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        else:
            self.border1s = [0, 12 * 30 * 24 * 4 - self.context_length, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.context_length]
            self.border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.path)       
        
        idx = 0
        if self.split == 'val': idx = 1
        elif self.split == 'test': idx = 2
        border1 = self.border1s[idx]
        border2 = self.border2s[idx]

        data_columns = df_raw.columns[1:]
        df_data = df_raw[data_columns]

        train_data = df_data[self.border1s[0]:self.border2s[0]]
        self.scaler.fit(train_data.values)
        data = self.scaler.transform(df_data.values)


        self.data = data[border1:border2]

    def __getitem__(self, index):
        context = self.data[index:index+self.context_length]
        targets = self.data[index+self.context_length:index+self.context_length+self.horizon]
        return context, targets

    def __len__(self):
        return len(self.data) - self.context_length - self.horizon + 1

class DatasetCustom():
    def __init__(self, 
                 path, 
                 split='train', 
                 context_length=336, 
                 horizon=336,  
                 ):
        assert split in ['train', 'test', 'val']
        
        self.path = path
        self.context_length = context_length
        self.horizon = horizon
        self.split = split

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.path)
        
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.context_length, len(df_raw) - num_test - self.context_length]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        idx = 0
        if self.split == 'val': idx = 1
        elif self.split == 'test': idx = 2
        border1 = border1s[idx]
        border2 = border2s[idx]

        data_columns = df_raw.columns[1:]
        df_data = df_raw[data_columns]

        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data.values)
        data = self.scaler.transform(df_data.values)

        self.data = data[border1:border2]

    def __getitem__(self, index):
        context = self.data[index:index+self.context_length]
        targets = self.data[index+self.context_length:index+self.context_length+self.horizon]
        return context, targets

    def __len__(self):
        return len(self.data) - self.context_length - self.horizon + 1

def dataset_selector(key, context_length, horizon, root='data/'):
    if key=='ETTh1':
        dataset_train = DatasetETTs(path=os.path.join(root, 'ETTh1.csv'), freq='h', split='train', context_length=context_length, horizon=horizon)
        dataset_val = DatasetETTs(path=os.path.join(root, 'ETTh1.csv'), freq='h', split='val', context_length=context_length, horizon=horizon)
        dataset_test = DatasetETTs(path=os.path.join(root, 'ETTh1.csv'), freq='h', split='test', context_length=context_length, horizon=horizon)
    elif key=='ETTh2':
        dataset_train = DatasetETTs(path=os.path.join(root, 'ETTh2.csv'), freq='h', split='train', context_length=context_length, horizon=horizon)
        dataset_val = DatasetETTs(path=os.path.join(root, 'ETTh2.csv'), freq='h', split='val', context_length=context_length, horizon=horizon)
        dataset_test = DatasetETTs(path=os.path.join(root, 'ETTh2.csv'), freq='h', split='test', context_length=context_length, horizon=horizon)
    elif key=='ETTm1':
        dataset_train = DatasetETTs(path=os.path.join(root, 'ETTm1.csv'), freq='m', split='train', context_length=context_length, horizon=horizon)
        dataset_val = DatasetETTs(path=os.path.join(root, 'ETTm1.csv'), freq='m', split='val', context_length=context_length, horizon=horizon)
        dataset_test = DatasetETTs(path=os.path.join(root, 'ETTm1.csv'), freq='m', split='test', context_length=context_length, horizon=horizon)
    elif key=='ETTm2':
        dataset_train = DatasetETTs(path=os.path.join(root, 'ETTm2.csv'), freq='m', split='train', context_length=context_length, horizon=horizon)
        dataset_val = DatasetETTs(path=os.path.join(root, 'ETTm2.csv'), freq='m', split='val', context_length=context_length, horizon=horizon)
        dataset_test = DatasetETTs(path=os.path.join(root, 'ETTm2.csv'), freq='m', split='test', context_length=context_length, horizon=horizon)
    elif key=='electricity':
        dataset_train = DatasetCustom(path=os.path.join(root, 'electricity.csv'), split='train', context_length=context_length, horizon=horizon)
        dataset_val = DatasetCustom(path=os.path.join(root, 'electricity.csv'), split='val', context_length=context_length, horizon=horizon)
        dataset_test = DatasetCustom(path=os.path.join(root, 'electricity.csv'), split='test', context_length=context_length, horizon=horizon)
    elif key=='weather':
        dataset_train = DatasetCustom(path=os.path.join(root, 'weather.csv'), split='train', context_length=context_length, horizon=horizon)
        dataset_val = DatasetCustom(path=os.path.join(root, 'weather.csv'), split='val', context_length=context_length, horizon=horizon)
        dataset_test = DatasetCustom(path=os.path.join(root, 'weather.csv'), split='test', context_length=context_length, horizon=horizon)
    elif key=='traffic':
        dataset_train = DatasetCustom(path=os.path.join(root, 'traffic.csv'), split='train', context_length=context_length, horizon=horizon)
        dataset_val = DatasetCustom(path=os.path.join(root, 'traffic.csv'), split='val', context_length=context_length, horizon=horizon)
        dataset_test = DatasetCustom(path=os.path.join(root, 'traffic.csv'), split='test', context_length=720, horizon=720)
    elif key=='exchange':
        dataset_train = DatasetCustom(path=os.path.join(root, 'exchange_rate.csv'), split='train', context_length=context_length, horizon=horizon)
        dataset_val = DatasetCustom(path=os.path.join(root, 'exchange_rate.csv'), split='val', context_length=context_length, horizon=horizon)
        dataset_test = DatasetCustom(path=os.path.join(root, 'exchange_rate.csv'), split='test', context_length=context_length, horizon=horizon)
    else:
        raise NotImplementedError
    return dataset_train, dataset_val, dataset_test


if __name__=='__main__':
    """
    Simple tests.
    """
    ETTh1_train = DatasetETTs(path='data/ETTh1.csv', freq='h', split='train', context_length=720, horizon=720)
    ETTh1_val = DatasetETTs(path='data/ETTh1.csv', freq='h', split='val', context_length=720, horizon=720)
    ETTh1_test = DatasetETTs(path='data/ETTh1.csv', freq='h', split='test', context_length=720, horizon=720)
    print(f'ETTh1 train shape={ETTh1_train.data.shape}, val shape={ETTh1_val.data.shape}, test shape={ETTh1_test.data.shape}')

    ETTh2_train = DatasetETTs(path='data/ETTh2.csv', freq='h', split='train', context_length=720, horizon=720)
    ETTh2_val = DatasetETTs(path='data/ETTh2.csv', freq='h', split='val', context_length=720, horizon=720)
    ETTh2_test = DatasetETTs(path='data/ETTh2.csv', freq='h', split='test', context_length=720, horizon=720)
    print(f'ETTh2 train shape={ETTh2_train.data.shape}, val shape={ETTh2_val.data.shape}, test shape={ETTh2_test.data.shape}')

    ETTm1_train = DatasetETTs(path='data/ETTm1.csv', freq='m', split='train', context_length=720, horizon=720)
    ETTm1_val = DatasetETTs(path='data/ETTm1.csv', freq='m', split='val', context_length=720, horizon=720)
    ETTm1_test = DatasetETTs(path='data/ETTm1.csv', freq='m', split='test', context_length=720, horizon=720)
    print(f'ETTm1 train shape={ETTm1_train.data.shape}, val shape={ETTm1_val.data.shape}, test shape={ETTm1_test.data.shape}')

    ETTm2_train = DatasetETTs(path='data/ETTm2.csv', freq='m', split='train', context_length=720, horizon=720)
    ETTm2_val = DatasetETTs(path='data/ETTm2.csv', freq='m', split='val', context_length=720, horizon=720)
    ETTm2_test = DatasetETTs(path='data/ETTm2.csv', freq='m', split='test', context_length=720, horizon=720)
    print(f'ETTm2 train shape={ETTm2_train.data.shape}, val shape={ETTm2_val.data.shape}, test shape={ETTm2_test.data.shape}')

    electricity_train = DatasetCustom(path='data/electricity.csv', split='train', context_length=720, horizon=720)
    electricity_val = DatasetCustom(path='data/electricity.csv', split='val', context_length=720, horizon=720)
    electricity_test = DatasetCustom(path='data/electricity.csv', split='test', context_length=720, horizon=720)
    print(f'Electricity train shape={electricity_train.data.shape}, val shape={electricity_val.data.shape}, test shape={electricity_test.data.shape}')

    weather_train = DatasetCustom(path='data/weather.csv', split='train', context_length=720, horizon=720)
    weather_val = DatasetCustom(path='data/weather.csv', split='val', context_length=720, horizon=720)
    weather_test = DatasetCustom(path='data/weather.csv', split='test', context_length=720, horizon=720)
    print(f'Weather train shape={weather_train.data.shape}, val shape={weather_val.data.shape}, test shape={weather_test.data.shape}')

    traffic_train = DatasetCustom(path='data/traffic.csv', split='train', context_length=720, horizon=720)
    traffic_val = DatasetCustom(path='data/traffic.csv', split='val', context_length=720, horizon=720)
    traffic_test = DatasetCustom(path='data/traffic.csv', split='test', context_length=720, horizon=720)
    print(f'Traffic train shape={traffic_train.data.shape}, val shape={traffic_val.data.shape}, test shape={traffic_test.data.shape}')

    exchange_train = DatasetCustom(path='data/exchange_rate.csv', split='train', context_length=720, horizon=720)
    exchange_val = DatasetCustom(path='data/exchange_rate.csv', split='val', context_length=720, horizon=720)
    exchange_test = DatasetCustom(path='data/exchange_rate.csv', split='test', context_length=720, horizon=720)
    print(f'Traffic train shape={exchange_train.data.shape}, val shape={exchange_val.data.shape}, test shape={exchange_test.data.shape}')