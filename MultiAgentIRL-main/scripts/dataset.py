from torch.utils.data import Dataset
import pickle
from termcolor import cprint
import numpy as np

class BCDataset(Dataset):
    def __init__(self, pickle_file='data/BC_data.pkl', sequence_len=1):
        with open(pickle_file, 'rb') as f:
            self.data = pickle.load(f)
            cprint('Loaded data from {}'.format(pickle_file), 'green')
        data = []
        for key in self.data.keys():
            for samples in self.data[key]:
                for sample in samples:
                    data.append(sample)
        data = np.asarray(data)
        self.data = data
        self.sequence_len = sequence_len
        
        # weight assigned to each data point
        self.data_weights = self.__get_weights__()
        
    def __get_weights__(self):
        return pickle.load(open('data/sp_weights.pkl', 'rb'))
    
    def __len__(self):
        return max(0, self.data.shape[0] - self.sequence_len + 1)
    
    def __getitem__(self, idx):
        # sample = self.data[idx]
        # v = np.asarray([sample[2], sample[3]]).astype(np.float32)
        # x_y_theta_onehot = np.asarray([sample[0], sample[1], sample[4], 
        #                                sample[5], sample[6], sample[7], sample[8]]).astype(np.float32)
        # return x_y_theta_onehot, v, self.data_weights[idx]
        
        sample = self.data[idx:idx+self.sequence_len]
        v = sample[-1, 2:4].astype(np.float32)
        x_y_theta_onehot = sample[:, [0, 1, 4, 5, 6, 7, 8]].astype(np.float32)
        return x_y_theta_onehot, v, self.data_weights[idx+self.sequence_len-1]
  