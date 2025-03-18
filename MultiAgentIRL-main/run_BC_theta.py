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
from scripts.models import BC_MLP, BC_LSTM, BC_Transformer, GMM_MLP, GMM_LSTM, GMM_Transformer, IBC, PECNet, MLP
from scripts.dataset import BCDataset
from scripts.loss import GMMLoss

# below are the algorithms that is supported by this training script. 
algorithms = ['bc_mlp', 'bc_lstm', 'bc_transformer', 'implicit_clone', 'gmm_mlp', 'gmm_lstm', 'gmm_transformer', 'pecnet']

def dyna_func(vec,vel, theta):
# def dyna_func(vec,vel):
    x = vec[0]
    y = vec[1]
    angle = vec[2] if np.mean(vel)==np.mean(theta) else theta
    x_new = x + vel * math.cos(angle) * 0.1 
    y_new = y + vel * math.sin(angle) * 0.1
    return x_new, y_new

class DataModule(pl.LightningDataModule):
    def __init__(self, pickle_file='data/BC_data.pkl', batch_size=32, num_workers=4, sequence_length=1):
        super().__init__()
        self.batch_size = batch_size
        dataset = BCDataset(pickle_file, sequence_len=sequence_length)
        self.save_hyperparameters()
        
        print('Dataset length: {}'.format(len(dataset)))
        torch.manual_seed(42) # for reproducibility
        self.train_set, self.val_set = random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])
        
        train_dl_tmp = DataLoader(self.train_set, batch_size=1, num_workers=0, shuffle=False)
        self.train_weights = []
        for _, _, w in train_dl_tmp:
            self.train_weights.append(w)
        self.train_weights = np.asarray(self.train_weights).flatten().astype(np.float32)
        
        print('Train set length: {}'.format(len(self.train_set)))
        print('Val set length: {}'.format(len(self.val_set)))
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=4, pin_memory=True, 
                          drop_last=True if len(self.train_set)%self.batch_size==0 else False, shuffle=True)
                        #   sampler=torch.utils.data.WeightedRandomSampler(self.train_weights, len(self.train_weights)))
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=4, shuffle=False, pin_memory=True, 
                          drop_last=False)
        
class BCModel(pl.LightningModule):
    def __init__(self, lr=3e-4, weight_decay=1e-5, batch_size=32, 
                 expt_log_path=None, algorithm='bc_mlp', sequence_length=1, n_gaussians=3):
        super().__init__()
        assert expt_log_path is not None, 'Please provide a valid path to save the model'
        self.expt_log_path = expt_log_path
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.algorithm = algorithm
        self.sequence_length = sequence_length
        if 'gmm' in self.algorithm:
            self.n_gaussians = n_gaussians
        else:
            self.n_gaussians = None
        self.save_hyperparameters()

        """DEFINE THE MODEL AND LOSS HERE"""
        if self.algorithm == 'bc_mlp':
            self.model = BC_MLP(mean_std_pkl='data/sp_mean_std_theta.pkl', sequence_length=self.sequence_length)
            self.loss = nn.MSELoss()
        elif self.algorithm == 'bc_lstm':
            self.model = BC_LSTM(mean_std_pkl='data/sp_mean_std_theta.pkl', sequence_length=self.sequence_length)
            self.loss = nn.MSELoss()
        elif self.algorithm == 'bc_transformer':
            self.model = BC_Transformer(mean_std_pkl='data/sp_mean_std_theta.pkl', sequence_length=self.sequence_length)
            self.loss = nn.MSELoss()
        elif self.algorithm == 'gmm_mlp':
            self.model = GMM_MLP(mean_std_pkl='data/sp_mean_std_theta.pkl', sequence_length=self.sequence_length,
                                 n_gaussians=self.n_gaussians)
            self.loss = GMMLoss(n_gaussians=self.n_gaussians)
        elif self.algorithm == 'gmm_lstm':
            self.model = GMM_LSTM(mean_std_pkl='data/sp_mean_std_theta.pkl', sequence_length=self.sequence_length,
                                  n_gaussians=self.n_gaussians)
            self.loss = GMMLoss(n_gaussians=self.n_gaussians)
        elif self.algorithm == 'gmm_transformer':
            self.model = GMM_Transformer(mean_std_pkl='data/sp_mean_std_theta.pkl', sequence_length=self.sequence_length)
            self.loss = GMMLoss(n_gaussians=self.n_gaussians)
        elif self.algorithm == 'implicit_clone':
            self.model = IBC(mean_std_pkl='data/sp_mean_std_theta.pkl', sequence_length=self.sequence_length)
            self.loss = nn.BCEWithLogitsLoss()
        elif self.algorithm == 'pecnet':   
            checkpoint = torch.load('data/pecnet.pt', map_location=torch.device("cuda:0"))
            hyper_params = checkpoint["hyper_params"]   
            self.model = PECNet(hyper_params["enc_past_size"], hyper_params["enc_dest_size"], hyper_params["enc_latent_size"], hyper_params["dec_size"], hyper_params["predictor_hidden_size"], hyper_params['non_local_theta_size'], hyper_params['non_local_phi_size'], hyper_params['non_local_g_size'], hyper_params["fdim"], hyper_params["zdim"], hyper_params["nonlocal_pools"], hyper_params['non_local_dim'], hyper_params["sigma"], 3, 1)
            self.model = self.model.float()
        else:
            raise NotImplementedError
        
        # self.loss = nn.SmoothL1Loss()
        
        self.best_val_loss = 1e10
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x_y_theta_onehot, v, _ = batch
        if 'bc' in self.algorithm:
            v_hat = self.forward(x_y_theta_onehot)
            loss = self.loss(v_hat, v)
        elif 'gmm' in self.algorithm:
            v_means, v_covs, v_weights = self.forward(x_y_theta_onehot) 
            loss = self.loss(v_means, v_covs, v_weights, v).mean()
        elif 'implicit_clone' in self.algorithm:
            # implement implicit behavior cloning
            # x_y_theta_onehot shape is [B, sequence_length, 7]
            # v shape is [B, 2]
            loss = 0
            positive_logits = self.model(x_y_theta_onehot, v)
            loss += self.loss(positive_logits, torch.ones_like(positive_logits))
            
            # create negative samples by shuffling the v values
            for _ in range(10): # create 10 negatives
                # shuffle the v such that it is not the same as the original v
                v_neg = v[torch.randperm(v.shape[0])]
                # use v_neg as mean and 0.1 as std, sample from a Gaussian distribution
                v_neg = torch.normal(v_neg, 0.1)
                
                negative_logits = self.model(x_y_theta_onehot, v_neg)
                loss += self.loss(negative_logits, torch.zeros_like(negative_logits)) / 10.0
            
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x_y_theta_onehot, v, _ = batch
        if 'bc' in self.algorithm:
            v_hat = self.forward(x_y_theta_onehot)
            loss = self.loss(v_hat, v)
        elif 'gmm' in self.algorithm:
            v_means, v_covs, v_weights = self.forward(x_y_theta_onehot) 
            loss = self.loss(v_means, v_covs, v_weights, v).mean()
        elif 'implicit_clone' in self.algorithm:
            # implicit behavior cloning
            # x_y_theta_onehot shape is [B, sequence_length, 7]
            # v shape is [B, 2]
            loss = 0
            positive_logits = self.model(x_y_theta_onehot, v)
            loss += self.loss(positive_logits, torch.ones_like(positive_logits))
            
            # create negative samples by shuffling the v values
            for _ in range(10): # create 10 negatives
                # shuffle the v such that it is not the same as the original v
                v_neg = v[torch.randperm(v.shape[0])]
                negative_logits = self.model(x_y_theta_onehot, v_neg)
                loss += self.loss(negative_logits, torch.zeros_like(negative_logits)) / 10.0
                
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)
    
    def on_validation_end(self):
        # perform this only on GPU 0
        if self.trainer.global_rank == 0:
            # get the validation loss across all batches and GPUs from the trainer
            val_loss = self.trainer.callback_metrics['val_loss']
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                # save this best model as a .pt file
                file_name = os.path.join(self.expt_log_path, 'bc_policy.pt')
                torch.save(self.model.state_dict(), file_name)
                if not 'implicit_clone' in self.algorithm:
                    # also save the model as a .jit file
                    sample_input = torch.randn(self.batch_size, self.sequence_length, 7).to(self.device)
                    traced_script_module = torch.jit.trace(self.model, sample_input)
                    file_name = os.path.join(self.expt_log_path, 'bc_policy.jit')
                    traced_script_module.save(file_name)
                    print('Saved best model at {}'.format(file_name))                
                
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--algorithm', type=str, choices=algorithms, required=False, default='pecnet')
    parser.add_argument('--sequence_length', type=int, required=False, default=3)
    parser.add_argument('--num_gaussians', type=int, default=3)
    args = parser.parse_args()

    import datetime # get current time as expt_log_path
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_folder_name = current_time + '_alg_' + args.algorithm + '_seqlen_' + str(args.sequence_length)
  
    if 'gmm' in args.algorithm: log_folder_name += '_numgauss_' + str(args.num_gaussians)
    expt_log_path = os.path.join('logs', log_folder_name)
    if not os.path.exists(expt_log_path): os.makedirs(expt_log_path)
    
    dm = DataModule(sequence_length=args.sequence_length, batch_size=args.batch_size)
    model = BCModel(expt_log_path=expt_log_path, algorithm=args.algorithm, 
                    batch_size=args.batch_size, lr=args.lr, sequence_length=args.sequence_length,
                    n_gaussians=args.num_gaussians)
    
    # trainer = pl.Trainer(devices=1, max_epochs=1000,
    #                      gradient_clip_val=5.0, gradient_clip_algorithm='norm',
    #                      logger=pl.loggers.TensorBoardLogger(save_dir = expt_log_path, name=""),
    #                      accelerator='gpu', deterministic=True,
    #                      use_distributed_sampler=False)
    
    # trainer.fit(model, dm)


    model_path = 'data/pecnet.pt'
    # model_path = 'data/bc_policy.pt'
    # Load the pretrained model
    device = torch.device("cuda:0")
    model.load_state_dict(torch.load(model_path), strict=False)   
    model = model.to(device)
    model.eval()
    model_IBC = IBC(mean_std_pkl='data/sp_mean_std_theta.pkl', sequence_length=args.sequence_length)
    model_IBC.load_state_dict(torch.load(model_path), strict=False)   
    model_IBC = model_IBC.to(device)
    model_IBC.eval()
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
        for t in range(3,60):
            tr_1_x.append(x_next)
            tr_1_y.append(y_next)
            tr_2_x.append(sample[t][7])
            tr_2_y.append(sample[t][8])
            x_y_theta_onehot_full = sample[t:t+args.sequence_length]

            x_y_theta_onehot_curr = np.asarray([x_next, y_next, sample[t][4], sample[t][5], sample[t][6], sample[t][7], sample[t][8]]).astype(np.float32)
            x_y_theta_onehot = x_y_theta_onehot_full[:, [0, 1, 4, 5, 6, 7, 8]].astype(np.float32)
            v_implicit = np.array([sample[t][2],sample[t][3]]).astype(np.float32)
            v_implicit = np.expand_dims(v_implicit, axis=0)
            x_y_theta_onehot = np.expand_dims(x_y_theta_onehot, axis=0)
            if args.algorithm == "implicit_clone":
                grid = model_IBC.grid()
                out = [model_IBC.forward(torch.Tensor(x_y_theta_onehot).cuda(), torch.Tensor(np.array([grid[j]])).cuda()) for j in range(len(grid))]
                max_value = max([t.item() for t in out])
                max_index = [i for i, t in enumerate(out) if t.item() == max_value][0]
            elif 'gmm' in args.algorithm:
                mu,cov,_ = model.forward(torch.Tensor(x_y_theta_onehot).cuda())
                mu = mu.detach().cpu().numpy()
                cov = cov.detach().cpu().numpy()
                k = np.random.choice(len(mu[0]))
                gmm_sample = np.random.normal(mu[0][k], cov[0][k])
            elif 'pecnet' in args.algorithm:
                dest, _ = model(torch.Tensor(x_y_theta_onehot).cuda())   
                v, theta = model([torch.Tensor(x_y_theta_onehot).cuda(), dest])
                print(theta.detach().cpu().numpy())
            else:
                v, theta = model(torch.Tensor(x_y_theta_onehot).cuda()) 

            # v = model(torch.Tensor(x_y_theta_onehot).cuda())
            # print(v, sample[t][2])
            if 'gmm' in args.algorithm:
                x_next, y_next = dyna_func(x_y_theta_onehot_curr, gmm_sample[0], gmm_sample[1])
            elif args.algorithm == "implicit_clone":
                x_next, y_next = dyna_func(x_y_theta_onehot_curr, grid[max_index][0], grid[max_index][1])
            elif args.algorithm == "pecnet":
                x_next, y_next = dyna_func(x_y_theta_onehot_curr, v.detach().cpu().numpy(), theta.detach().cpu().numpy())
                x_next = x_next[0]
                y_next = y_next[0]
            else:
                x_next, y_next = dyna_func(x_y_theta_onehot_curr, v.detach().cpu().numpy(), theta.detach().cpu().numpy())
            # x_next, y_next = dyna_func(x_y_theta_onehot, v.detach().cpu().numpy())
            # x_next = x_next.detach().cpu().numpy()[0]
            # y_next = y_next.detach().cpu().numpy()[0]
        # Create a new figure
            # Plot the first trajectory
        plt.plot(tr_1_x, tr_1_y, label='BC', marker='o')

    # Plot the second trajectory
        plt.plot(tr_2_x, tr_2_y, label='Reference')
        BC_learned[agent] = [tr_1_x, tr_1_y]
    with open("/home/rchandra/Research/bluecity_example/data/learned_BC_speedway.pkl", "wb") as file:
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
    # plt.show()