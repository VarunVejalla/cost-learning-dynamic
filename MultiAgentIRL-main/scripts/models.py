"""
Implements different models for the different behavior cloning algorithms
"""

import torch
import torch.nn as nn
import pickle
import numpy as np
import math
from torch.nn.utils import spectral_norm, weight_norm

from torch.autograd import Variable
import torch.nn.functional as F

# base model that contains the mean and std of the data, common to all models
class BaseModel(nn.Module):
    def __init__(self, mean_std_pkl='data/sp_mean_std_theta.pkl', sequence_length=1) -> None:
        super().__init__()
        
        self.sequence_length = sequence_length
        
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

class BC_MLP(BaseModel):
    def __init__(self, mean_std_pkl='data/sp_mean_std_theta.pkl', sequence_length=1) -> None:
        super().__init__(mean_std_pkl, sequence_length)
        self.model_type = 'MLP'
        self.model = torch.nn.Sequential(
            nn.Linear(7*self.sequence_length, 64), nn.Tanh(), # x, y, one_hot1, one_hot2, one_hot3, ref_coord1, ref_coord2
            nn.Dropout(0.2),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2) # velocity, theta
        )
    
    def forward(self, x): # x is of shape [B, seq_len, 7]
        x = (x - self.feat_mean) / (self.feat_std + 1e-6)
        x = x.view(x.shape[0], -1) # [B, seq_len*7]
        v = self.model(x) # [B, 2]
        v = v * self.vel_std + self.vel_mean
        return v
    
class BC_LSTM(BaseModel):
    def __init__(self, mean_std_pkl='data/sp_mean_std_theta.pkl', sequence_length=1) -> None:
        super().__init__(mean_std_pkl, sequence_length)
        self.model_type = 'LSTM'
        self.lstm = nn.LSTM(7, 64, batch_first=True, num_layers=2, dropout=0.2)
        self.linear = nn.Linear(64, 2)
        
    def forward(self, x):
        x = (x - self.feat_mean) / (self.feat_std + 1e-6) # [B, seq_len, 7]
        x, _ = self.lstm(x) # [B, seq_len, 64]
        x = self.linear(x) # [B, seq_len, 2]
        v = x[:, -1, :] # [B, 2]
        v = v * self.vel_std + self.vel_mean
        return v[0][0], v[0][1]
        
class BC_Transformer(BaseModel):
    def __init__(self, mean_std_pkl='data/sp_mean_std_theta.pkl', sequence_length=1) -> None:
        super().__init__(mean_std_pkl, sequence_length)
        self.model_type = 'Transformer'
        self.src_mask = None
        
        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, dropout=0.1, max_len=5000):
                super(PositionalEncoding, self).__init__()
                self.dropout = nn.Dropout(p=dropout)

                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0).transpose(0, 1)
                self.register_buffer('pe', pe)

            def forward(self, x):
                x = x + self.pe[:x.size(0), :]
                return self.dropout(x)

        self.tokenizer = nn.Sequential(
            nn.Linear(7, 16), nn.Tanh(),
        )
            
        self.pos_encoder = PositionalEncoding(16, dropout=0.1)
        self.transformer = nn.Transformer(d_model=16, nhead=2, num_encoder_layers=2, 
                                          num_decoder_layers=2, dropout=0.2)
        self.decoder = nn.Linear(16, 2)
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, x):
        x = (x - self.feat_mean) / (self.feat_std + 1e-6) # [B, seq_len, 7]
        
        x = self.tokenizer(x) # [B, seq_len, 64]
        
        # transform x to [seq_len, B, 7]
        x = x.permute(1, 0, 2)
        
        if self.src_mask is None or self.src_mask.size(0) != len(x):
            device = x.device
            mask = self._generate_square_subsequent_mask(len(x)).to(device)
            self.src_mask = mask
        
        x = self.pos_encoder(x)
        output = self.transformer(x, x, self.src_mask)
        output = self.decoder(output)
        v = output[-1, :, :]
        v = v * self.vel_std + self.vel_mean
        return v[0][0], v[0][1]
        
class GMM_MLP(BaseModel):
    def __init__(self, mean_std_pkl='data/sp_mean_std_theta.pkl', sequence_length=1, n_gaussians=3) -> None:
        super().__init__(mean_std_pkl, sequence_length)
        self.model_type = 'GMM_MLP'
        self.n_gaussians = n_gaussians
        self.model = nn.Sequential(
            nn.Linear(7*self.sequence_length, 64), nn.Tanh(), # x, y, one_hot1, one_hot2, one_hot3, ref_coord1, ref_coord2
            nn.Dropout(0.2),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2 * 2 * n_gaussians + n_gaussians) # (velocity, theta) - mean, std, weight
        )
    
    def forward(self, x): # x is of shape [B, seq_len, 7]
        x = (x - self.feat_mean) / (self.feat_std + 1e-6)
        x = x.reshape(x.shape[0], -1) # [B, seq_len*7]
        params = self.model(x) # [B, 2 * 2 * n_gaussians + n_gaussians]
        means, covs, weights = params[:, :2*self.n_gaussians], params[:, 2*self.n_gaussians:4*self.n_gaussians], params[:, 4*self.n_gaussians:]
        means = means.view(-1, self.n_gaussians, 2) # [B, n_gaussians, 2]
        covs = covs.view(-1, self.n_gaussians, 2) # [B, n_gaussians, 2]
        covs = torch.exp(covs) # [B, n_gaussians, 2]
        covs = torch.clamp(covs, min=1e-6)
        
        weights = weights.view(-1, self.n_gaussians) # [B, n_gaussians]
        weights = torch.softmax(weights, dim=-1) # [B, n_gaussians]
        weights = torch.clamp(weights, min=1e-8, max=1.0-1e-8)
        
        # unnormalize mean and std
        means = means * self.vel_std + self.vel_mean
        covs = covs * self.vel_std
        
        return means, covs, weights
        
class GMM_LSTM(BaseModel):
    def __init__(self, mean_std_pkl='data/sp_mean_std_theta.pkl', sequence_length=1, n_gaussians=3) -> None:
        super().__init__(mean_std_pkl, sequence_length)
        self.model_type = 'GMM_LSTM'
        self.n_gaussians = n_gaussians
        self.lstm = nn.LSTM(7, 64, batch_first=True, num_layers=2, dropout=0.2)
        self.linear = nn.Linear(64, 2 * 2 * n_gaussians + n_gaussians)
    
    def forward(self, x): 
        x = (x - self.feat_mean) / (self.feat_std + 1e-6)
        x, _ = self.lstm(x) # [B, seq_len, 64]
        x = self.linear(x) # [B, seq_len, 2 * 2 * n_gaussians + n_gaussians]
        x = x[:, -1, :] # [B, 2 * 2 * n_gaussians + n_gaussians]
        means, covs, weights = x[:, :2*self.n_gaussians], x[:, 2*self.n_gaussians:4*self.n_gaussians], x[:, 4*self.n_gaussians:]
        means = means.view(-1, self.n_gaussians, 2) # [B, n_gaussians, 2]
        covs = covs.view(-1, self.n_gaussians, 2) # [B, n_gaussians, 2]
        covs = torch.exp(covs) # [B, n_gaussians, 2]
        covs = torch.clamp(covs, min=1e-6)
        
        weights = weights.view(-1, self.n_gaussians) # [B, n_gaussians]
        weights = torch.softmax(weights, dim=-1) # [B, n_gaussians]
        weights = torch.clamp(weights, min=1e-8, max=1.0-1e-8)
        
        # unnormalize mean and std
        means = means * self.vel_std + self.vel_mean
        covs = covs * self.vel_std
        
        return means, covs, weights
    
class GMM_Transformer(BaseModel):
    def __init__(self, mean_std_pkl='data/sp_mean_std_theta.pkl', sequence_length=1, n_gaussians=3) -> None:
        super().__init__(mean_std_pkl, sequence_length)
        self.model_type = 'GMM_Transformer'
        self.n_gaussians = n_gaussians
        self.src_mask = None
        
        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, dropout=0.1, max_len=5000):
                super(PositionalEncoding, self).__init__()
                self.dropout = nn.Dropout(p=dropout)

                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0).transpose(0, 1)
                self.register_buffer('pe', pe)

            def forward(self, x):
                x = x + self.pe[:x.size(0), :]
                return self.dropout(x)

        self.tokenizer = nn.Sequential(
            nn.Linear(7, 16), nn.Tanh(),
        )
            
        self.pos_encoder = PositionalEncoding(16, dropout=0.1)
        self.transformer = nn.Transformer(d_model=16, nhead=2, num_encoder_layers=2, 
                                          num_decoder_layers=2, dropout=0.2)
        self.decoder = nn.Linear(16, 2 * 2 * n_gaussians + n_gaussians)
        
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        x = (x - self.feat_mean) / (self.feat_std + 1e-6) # [B, seq_len, 7]
        
        x = self.tokenizer(x) # [B, seq_len, 64]
        
        # transform x to [seq_len, B, 7]
        x = x.permute(1, 0, 2)
        
        if self.src_mask is None or self.src_mask.size(0) != len(x):
            device = x.device
            mask = self._generate_square_subsequent_mask(len(x)).to(device)
            self.src_mask = mask
        
        x = self.pos_encoder(x)
        output = self.transformer(x, x, self.src_mask)
        output = self.decoder(output)
        v = output[-1, :, :]
        means, covs, weights = v[:, :2*self.n_gaussians], v[:, 2*self.n_gaussians:4*self.n_gaussians], v[:, 4*self.n_gaussians:]
        means = means.view(-1, self.n_gaussians, 2) # [B, n_gaussians, 2]
        covs = covs.view(-1, self.n_gaussians, 2) # [B, n_gaussians, 2]
        covs = torch.exp(covs) # [B, n_gaussians, 2]
        covs = torch.clamp(covs, min=1e-6)
        
        weights = weights.view(-1, self.n_gaussians) # [B, n_gaussians]
        weights = torch.softmax(weights, dim=-1) # [B, n_gaussians]
        weights = torch.clamp(weights, min=1e-8, max=1.0-1e-8)
        
        # unnormalize mean and std
        means = means * self.vel_std + self.vel_mean
        covs = covs * self.vel_std

        return means, covs, weights
    
# implement implicit behavior cloning
class IBC(BaseModel):
    def __init__(self, mean_std_pkl='data/sp_mean_std_theta.pkl', sequence_length=1) -> None:
        super().__init__(mean_std_pkl, sequence_length)
        self.model_type = 'IBC'
        # using tanh activation since its Lipschitz constant is 1 
        self.model = torch.nn.Sequential(
            spectral_norm(nn.Linear(7*self.sequence_length + 2, 64)), nn.Tanh(), # x, y, one_hot1, one_hot2, one_hot3, ref_coord1, ref_coord2 + velocity
            spectral_norm(nn.Linear(64, 64)), nn.Tanh(),
            spectral_norm(nn.Linear(64, 1))
        )
    
    def forward(self, x, v): # x is of shape [B, seq_len, 7]
        x = (x - self.feat_mean) / (self.feat_std + 1e-6)
        v = (v - self.vel_mean) / (self.vel_std + 1e-6)
        x = x.reshape(x.shape[0], -1) # [B, seq_len*7]
        # v = v.reshape(v.shape[0], -1) # [B, seq_len*7]
        x = torch.cat([x, v], dim=-1) # [B, seq_len*7 + 2]
        y = self.model(x) # [B, 1]
        return y
    def grid(self):
        # Conversion factor from mph to m/s
        mph_to_m_s = 0.44704

        # Range of walking speeds in mph
        speeds_mph = [2, 3, 4, 5]

        # Convert speeds to m/s
        speeds_m_s = [speed * mph_to_m_s for speed in speeds_mph]

        # Degrees of a circle with a resolution of 30 degrees
        degrees = np.arange(0, 361, 30)

        # Initialize list to store results
        grid_search_results = []

        # Perform grid search
        for speed in speeds_m_s:
            for degree in degrees:
                # Do something with the combination of speed and degree
                # For demonstration, we'll just append them to our results list
                grid_search_results.append((speed, degree))
        return grid_search_results

'''MLP model'''
class MLP(BaseModel):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        # x = x.T.squeeze(2)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers)-1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x

class PECNet(BaseModel):

    def __init__(self, enc_past_size, enc_dest_size, enc_latent_size, dec_size, predictor_size, non_local_theta_size, non_local_phi_size, non_local_g_size, fdim, zdim, nonlocal_pools, non_local_dim, sigma, past_length, future_length):
        '''
        Args:
            size parameters: Dimension sizes
            nonlocal_pools: Number of nonlocal pooling operations to be performed
            sigma: Standard deviation used for sampling N(0, sigma)
            past_length: Length of past history (number of timesteps)
            future_length: Length of future trajectory to be predicted
        '''
        super(PECNet, self).__init__()

        self.zdim = zdim
        self.nonlocal_pools = nonlocal_pools
        self.sigma = sigma

        # takes in the past
        self.encoder_past = MLP(input_dim = past_length*1, output_dim = fdim, hidden_size=enc_past_size)

        self.encoder_dest = MLP(input_dim = 2, output_dim = fdim, hidden_size=enc_dest_size)

        self.encoder_latent = MLP(input_dim = 2*fdim, output_dim = 2*zdim, hidden_size=enc_latent_size)

        self.decoder = MLP(input_dim = fdim + zdim, output_dim = 2, hidden_size=dec_size)

        self.non_local_theta = MLP(input_dim = 2*fdim + 2, output_dim = non_local_dim, hidden_size=non_local_theta_size)
        self.non_local_phi = MLP(input_dim = 2*fdim + 2, output_dim = non_local_dim, hidden_size=non_local_phi_size)
        self.non_local_g = MLP(input_dim = 2*fdim + 2, output_dim = 2*fdim + 2, hidden_size=non_local_g_size)

        self.predictor = MLP(input_dim = 2*fdim + 1, output_dim = 1*(future_length), hidden_size=predictor_size)

        architecture = lambda net: [l.in_features for l in net.layers] + [net.layers[-1].out_features]

        # if verbose:
        #     print("Past Encoder architecture : {}".format(architecture(self.encoder_past)))
        #     print("Dest Encoder architecture : {}".format(architecture(self.encoder_dest)))
        #     print("Latent Encoder architecture : {}".format(architecture(self.encoder_latent)))
        #     print("Decoder architecture : {}".format(architecture(self.decoder)))
        #     print("Predictor architecture : {}".format(architecture(self.predictor)))

        #     print("Non Local Theta architecture : {}".format(architecture(self.non_local_theta)))
        #     print("Non Local Phi architecture : {}".format(architecture(self.non_local_phi)))
        #     print("Non Local g architecture : {}".format(architecture(self.non_local_g)))

    def non_local_social_pooling(self, feat, mask):

        # N,C
        theta_x = self.non_local_theta(feat)

        # C,N
        phi_x = self.non_local_phi(feat).transpose(1,0)

        # f_ij = (theta_i)^T(phi_j), (N,N)
        f = torch.matmul(theta_x, phi_x)

        # f_weights_i =  exp(f_ij)/(\sum_{j=1}^N exp(f_ij))
        f_weights = F.softmax(f, dim = -1)

        # setting weights of non neighbours to zero
        f_weights = f_weights * mask

        # rescaling row weights to 1
        f_weights = F.normalize(f_weights, p=1, dim=1)

        # ith row of all_pooled_f = \sum_{j=1}^N f_weights_i_j * g_row_j
        pooled_f = torch.matmul(f_weights, self.non_local_g(feat))

        return pooled_f + feat

    def forward(self, x, dest = None, mask = None, device=torch.device('cuda:0')):
        if not isinstance(x, list):
            # x = x[0]
            initial_pos = x[0][0][0:2]
            x = x.T.squeeze(2)
            # provide destination iff training
            # assert model.training
            assert self.training ^ (dest is None)
            assert self.training ^ (mask is None)

            # encode
            ftraj = self.encoder_past(x)

            if not self.training:
                z = torch.Tensor(x.size(0), self.zdim)
                z.normal_(0, self.sigma)

            else:
                # during training, use the destination to produce generated_dest and use it again to predict final future points

                # CVAE code
                dest_features = self.encoder_dest(dest)
                features = torch.cat((ftraj, dest_features), dim = 1)
                latent =  self.encoder_latent(features)

                mu = latent[:, 0:self.zdim] # 2-d array
                logvar = latent[:, self.zdim:] # 2-d array

                var = logvar.mul(0.5).exp_()
                eps = torch.DoubleTensor(var.size()).normal_()
                eps = eps.to(device)
                z = eps.mul(var).add_(mu)

            # z = z.double().to(device)
            z = z.to(device)
            decoder_input = torch.cat((ftraj, z), dim = 1)
            generated_dest = self.decoder(decoder_input)

            if self.training:
                # prediction in training, no best selection
                generated_dest_features = self.encoder_dest(generated_dest)

                prediction_features = torch.cat((ftraj, generated_dest_features, initial_pos), dim = 1)

                for i in range(self.nonlocal_pools):
                    # non local social pooling
                    prediction_features = self.non_local_social_pooling(prediction_features, mask)

                pred_future = self.predictor(prediction_features)
                return generated_dest, mu, logvar, pred_future

            return generated_dest, generated_dest
        else:
            past = x[0]
            dest = x[1]
            # initial_pos = past[0][0][0:2]
            past = past.T.squeeze(2)
            initial_pos = torch.unsqueeze(past[:,0], axis=1)
            ftraj = self.encoder_past(past)
            generated_dest_features = self.encoder_dest(dest)
            prediction_features = torch.cat((ftraj, generated_dest_features, initial_pos), dim = 1)

            # for i in range(self.nonlocal_pools):
            #     # non local social pooling
            #     prediction_features = self.non_local_social_pooling(prediction_features, mask)

            interpolated_future = self.predictor(prediction_features)
            return interpolated_future[0], interpolated_future[1]

    # # separated for forward to let choose the best destination
    # def predict(self, past, generated_dest):
    #     initial_pos = x[0][0][0:2]
    #     x = x.T.squeeze(2)
    #     ftraj = self.encoder_past(past)
    #     generated_dest_features = self.encoder_dest(generated_dest)
    #     prediction_features = torch.cat((ftraj, generated_dest_features, initial_pos), dim = 1)

    #     # for i in range(self.nonlocal_pools):
    #     #     # non local social pooling
    #     #     prediction_features = self.non_local_social_pooling(prediction_features, mask)

    #     interpolated_future = self.predictor(prediction_features)
    #     return interpolated_future