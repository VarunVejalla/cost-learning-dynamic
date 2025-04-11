import torch
import torch.nn as nn
import math

import pickle

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
from termcolor import cprint
# to spliit the data into train and test
from torch.utils.data import random_split
import tensorboard as tb
import os
import matplotlib.pyplot as plt

# def dyna_func(vec,vel, theta):
def dyna_func(vec,vel):
    x = vec[0]
    y = vec[1]
    angle = vec[2]
    x_new = x + vel * math.cos(angle) * 0.1
    y_new = y + vel * math.sin(angle) * 0.1
    return x_new, y_new

class BCDataset(Dataset):
    def __init__(self, pickle_file='data/BC_data.pkl'):
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
        
        # weight assigned to each data point
        self.data_weights = self.__get_weights__()
    
    def __get_weights__(self):
        return pickle.load(open('data/sp_weights.pkl', 'rb'))
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        # v = np.asarray([sample[2], sample[3]]).astype(np.float32)
        # x_y_theta_onehot = np.asarray([sample[0], sample[1], sample[4], 
        #                                sample[5], sample[6], sample[7], sample[8]]).astype(np.float32)
        v = np.asarray(sample[2]).astype(np.float32)
        x_y_theta_onehot = np.asarray([sample[0], sample[1], sample[3], sample[4], 
                                       sample[5], sample[6], sample[7], sample[8]]).astype(np.float32)
        return x_y_theta_onehot, v, self.data_weights[idx]
    
class BCDataModule(pl.LightningDataModule):
    def __init__(self, pickle_file='data/BC_data.pkl', batch_size=32, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        dataset = BCDataset(pickle_file)
        print('Dataset length: {}'.format(len(dataset)))
        self.train_set, self.val_set = random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])
        
        # train_dl_tmp = DataLoader(self.train_set, batch_size=1, num_workers=1, shuffle=False)
        train_dl_tmp = DataLoader(self.train_set, batch_size=1, num_workers=0, shuffle=False)
        self.train_weights = []
        for _, _, w in train_dl_tmp:
            self.train_weights.append(w)
        self.train_weights = np.asarray(self.train_weights[0]).flatten().astype(np.float32)
        
        print('Train set length: {}'.format(len(self.train_set)))
        print('Val set length: {}'.format(len(self.val_set)))
        print('Train set length: {}'.format(len(self.train_set)))
        print('Val set length: {}'.format(len(self.val_set)))
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=4, pin_memory=True, 
                          drop_last=True if len(self.train_set)%self.batch_size==0 else False, shuffle=True)
                        #   sampler=torch.utils.data.WeightedRandomSampler(self.train_weights, len(self.train_weights)))
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=4, shuffle=False, pin_memory=True, 
                          drop_last=False)
        
class Policy(nn.Module):
    def __init__(self, mean_std_pkl='data/sp_mean_std.pkl') -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            # nn.Linear(7, 64), nn.Tanh(), # x, y, one_hot1, one_hot2, one_hot3, ref_coord1, ref_coord2
            nn.Linear(8, 64), nn.Tanh(), # x, y, one_hot1, one_hot2, one_hot3, ref_coord1, ref_coord2
            nn.Dropout(0.2),
            nn.Linear(64, 64), nn.Tanh(),
            # nn.Linear(64, 2) # velocity, theta
            nn.Linear(64, 1) # velocity
        )
        
        # load mean and std from pickle file for normalizing the data
        tmp = pickle.load(open(mean_std_pkl, 'rb'))
        self.vel_mean, self.vel_std = tmp['vel_mean'], tmp['vel_std']
        self.feat_mean, self.feat_std = tmp['feat_mean'], tmp['feat_std']
        self.vel_mean, self.vel_std = np.asarray(self.vel_mean).astype(np.float32), np.asarray(self.vel_std).astype(np.float32)
        
        # convert to tensors
        self.vel_mean, self.vel_std = torch.from_numpy(self.vel_mean).float().cuda(), torch.from_numpy(self.vel_std).float().cuda()
        self.feat_mean, self.feat_std = torch.from_numpy(self.feat_mean).float().cuda(), torch.from_numpy(self.feat_std).float().cuda()
        
        # donot train the mean and std
        self.vel_mean.requires_grad = False
        self.vel_std.requires_grad = False
        self.feat_mean.requires_grad = False
        self.feat_std.requires_grad = False
    
    def forward(self, x):
        x = (x - self.feat_mean) / (self.feat_std + 1e-6)
        v = self.model(x)
        v = v * self.vel_std + self.vel_mean
        return v
        # return v[0], v[1]
        
class BCModel(pl.LightningModule):
    def __init__(self, lr=3e-4, weight_decay=1e-5, batch_size=32, expt_log_path=None):
        super().__init__()
        assert expt_log_path is not None, 'Please provide a valid path to save the model'
        self.expt_log_path = expt_log_path
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        
        self.model = Policy(mean_std_pkl='data/sp_mean_std.pkl')
        
        # self.loss = nn.SmoothL1Loss()
        self.loss = nn.MSELoss()
        
        self.best_val_loss = 1e10
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x_y_theta_onehot, v, _ = batch
        v_hat = self.forward(x_y_theta_onehot)
        loss = self.loss(v_hat, v)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x_y_theta_onehot, v, _ = batch
        v_hat = self.forward(x_y_theta_onehot)
        loss = self.loss(v_hat, v)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)
    
    def on_validation_end(self):
        # perform this only on GPU 0
        if torch.cuda.current_device() == 0:
            # get the validation loss across all batches and GPUs from the trainer
            val_loss = self.trainer.callback_metrics['val_loss']
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                # save this best model as a .pt file
                file_name = os.path.join(self.expt_log_path, 'data/bc_policy.pt')
                torch.save(self.model.state_dict(), file_name)
                print('Saved best model to {}'.format(file_name))
    
if __name__ == "__main__":

    import datetime # get current time as expt_log_path
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    expt_log_path = os.path.join('logs', current_time)
    if not os.path.exists(expt_log_path): os.makedirs(expt_log_path)
    
    dm = BCDataModule()
    model = BCModel(expt_log_path=expt_log_path)
    
    # trainer = pl.Trainer(devices=1, max_epochs=1000,
    #                      gradient_clip_val=1.0, gradient_clip_algorithm='norm',
    #                     logger=pl.loggers.TensorBoardLogger(save_dir = expt_log_path, name=""),
    #                      accelerator='gpu', use_distributed_sampler=False)
    
    # trainer.fit(model, dm)


    # model_path = 'data/bc_policy_theta.pt'
    model_path = 'data/bc_policy.pt'
    # Load the pretrained model
    device = torch.device("cuda:0")
    model.load_state_dict(torch.load(model_path), strict=False)   
    model = model.to(device)
    model.eval()

    pickle_file = 'data/BC_data.pkl'
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
        cprint('Loaded data from {}'.format(pickle_file), 'green')
    # data_traj = []
    # for key in data.keys():
    #     for samples in data[key]:
    #         for sample in samples:
    #             data_traj.append(sample)
    # # data_traj = np.asarray(data_traj)
    BC_learned = {}
    plt.figure()
    for agent in ["agent1","agent2","agent3"]:
        data_traj = data[agent]
        sample = data_traj[0]

        x_next = sample[0][0]
        y_next = sample[0][1]
        tr_1_x = []
        tr_1_y = [] 
        tr_2_x = []
        tr_2_y = []
        for t in range(60):
            tr_1_x.append(x_next)
            tr_1_y.append(y_next)
            tr_2_x.append(sample[t][7])
            tr_2_y.append(sample[t][8])
            try:
                x_y_theta_onehot = np.asarray([x_next, y_next, sample[t][3], sample[t][4], sample[t][5], sample[t][6], sample[t][7], sample[t][8]]).astype(np.float32)
            except:
                pass
            # x_y_theta_onehot = np.asarray([x_next, y_next, sample[t][4], sample[t][5], sample[t][6], sample[t][7], sample[t][8]]).astype(np.float32)

            # v, theta = model(torch.Tensor(x_y_theta_onehot).cuda())
            v = model(torch.Tensor(x_y_theta_onehot).cuda())
            # print(v, sample[t][2])
            # x_next, y_next = dyna_func(x_y_theta_onehot, v.detach().cpu().numpy(), theta.detach().cpu().numpy())
            x_next, y_next = dyna_func(x_y_theta_onehot, v.detach().cpu().numpy())
            x_next = x_next[0]
            y_next = y_next[0]
        # Create a new figure
            # Plot the first trajectory
        plt.plot(tr_1_x, tr_1_y, label='BC', marker='o')

    # Plot the second trajectory
        plt.plot(tr_2_x, tr_2_y, label='Reference')
        BC_learned[agent] = [tr_1_x, tr_1_y]
    datapath = os.path.expanduser("~/Research/bluecity_example/data/")
    fname = os.path.join(datapath, "learned_BC_speedway.pkl")
    with open(fname, "wb") as file:
        pickle.dump(BC_learned, file)


    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.xlim([-4,4])
    # plt.ylim([-4,4])
    plt.xlim([-15, 0])
    plt.ylim([-2, 12])
    plt.title('Comparison of Trajectories')

    # Add legend
    plt.legend()

    # Display the plot
    plt.show()