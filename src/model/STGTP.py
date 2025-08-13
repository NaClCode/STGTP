import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings 
import torchvision.models as models
from torchvision.models.vision_transformer import VisionTransformer

warnings.filterwarnings('ignore')

class LearnableMask(nn.Module):
    def __init__(self, init_steepness=1000.0):
        super().__init__()

        self.steepness = nn.Parameter(torch.tensor(init_steepness))
        
        self.offset = nn.Parameter(torch.tensor(1.0)) 

    def forward(self, x):

        shifted_x = x - self.offset
        positive_part = torch.relu(shifted_x)
        return -positive_part * torch.exp(self.steepness * positive_part)

def gaussian_2d_torch(X, Y, mu_x, mu_y, sigma_x, sigma_y):
    mu_x = mu_x.unsqueeze(-1).unsqueeze(-1)
    mu_y = mu_y.unsqueeze(-1).unsqueeze(-1)  
    Z = torch.exp(- (((X - mu_x)**2) / (2 * sigma_x**2) + ((Y - mu_y)**2) / (2 * sigma_y**2)))
    return Z


def get_noise(shape, noise_type, device):
    if noise_type == "gaussian":
        return torch.randn(shape, device=device)
    elif noise_type == "uniform":
        return torch.rand(*shape, device=device).sub_(0.5).mul_(2.0)
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class TransformerModel(nn.Module):

    def __init__(self, encoder_layer, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.src_mask = None
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers)

    def forward(self, src, n_mask):

        # mask = n_mask.float().masked_fill(n_mask == 1., float(-1e20)).masked_fill(n_mask == 1., float(0.0))
        output = self.transformer_encoder(src, mask=n_mask)
        return output

class STGTP(torch.nn.Module):

    def __init__(self, args, dropout_prob=0.1):
        super(STGTP, self).__init__()

        self.embedding_size = [32]
        self.output_size = 2
        self.dropout_prob = dropout_prob
        self.args = args

        
        self.learn_mask = LearnableMask()
        self.fusion_num = 0
        self.input_embedding_layer_temporal = nn.Linear(2, 32)
        self.tenc = nn.TransformerEncoderLayer(d_model=32, nhead=8, batch_first=True, dim_feedforward=self.args.dim)
        self.temporal_encoder_1 = nn.TransformerEncoder(self.tenc, 2)
        self.temporal_encoder_2 = nn.TransformerEncoder(self.tenc, 2)
        self.fusion_num += 32

        if self.args.S:
            self.senc = nn.TransformerEncoderLayer(d_model=32, nhead=8, batch_first=True, dim_feedforward=self.args.dim)
            self.input_embedding_layer_spatial = nn.Linear(2, 32)
            self.spatial_encoder_1 = TransformerModel(self.senc, 2)
            self.spatial_encoder_2 = TransformerModel(self.senc, 2)
            self.encoder2_layer = nn.Linear(64, 32)
            self.fusion_num += 32
        
        if self.args.P:
            self.encode_vit = VisionTransformer(image_size=self.args.image_size, patch_size=self.args.patch_size, hidden_dim=self.args.vit_dim, num_layers=2, num_heads=8, mlp_dim=self.args.vit_dim)
            self.encode_vit.conv_proj = nn.Conv2d(
                    in_channels=1, out_channels=self.args.vit_dim, kernel_size=self.args.patch_size, stride=self.args.patch_size
                )

            self.encode_vit.heads = nn.Sequential(nn.Linear(in_features=512, out_features=32))
            self.fusion_num += 32

        self.fusion_layer = nn.Linear(self.fusion_num, 32)
     
        self.output_layer = nn.Linear(36, 2)
        self.output = nn.Linear(self.args.obs_length, self.args.pred_length)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_prob)

    def get_node_index(self, seq_list):
        """

        :param seq_list: mask indicates whether pedestrain exists
        :type seq_list: numpy array [F, N], F: number of frames. N: Number of pedestrians (a mask to indicate whether
                                                                                            the pedestrian exists)
        :return: All the pedestrians who exist from the beginning to current frame
        :rtype: numpy array
        """
        for idx, framenum in enumerate(seq_list):

            if idx == 0:
                node_indices = framenum > 0
            else:
                node_indices *= (framenum > 0)

        return node_indices
    
    def forward(self, inputs, iftest=False):

        nodes_abs, seq_list, nei_lists, nei_num, batch_pednum = inputs
        
        device = nodes_abs.device
        outputs = torch.zeros((self.args.pred_length, nodes_abs.shape[1], 2), device=device)
            
        node_index = self.get_node_index(seq_list)
        
        num_ped = sum(node_index)
        noise = get_noise((self.args.obs_length, num_ped, 4), 'gaussian', device=device)
        
        nodes_current = nodes_abs[:, node_index, :]
        
        nei_list = nei_lists[:, node_index, :]
        nei_list = nei_list[:, :, node_index]

        fusion_features = []

        temporal_input_embedded = self.dropout(
            self.relu(
                self.input_embedding_layer_temporal(nodes_current)
            )
        )
        temporal_input_embedded = self.temporal_encoder_1(temporal_input_embedded.permute(1, 0, 2))
        fusion_features.append(temporal_input_embedded.permute(1, 0, 2))

        if self.args.S:
            spatial_input_embedded = self.dropout(
                self.relu(
                    self.input_embedding_layer_spatial(nodes_current)
                )
            )
       
            attn_mask = self.learn_mask(nei_list).unsqueeze(1).repeat(1, 8, 1, 1)
            if not iftest:
                attn_mask = attn_mask.view(self.args.obs_length * 8, num_ped, num_ped)
                
            spatial_input_embedded = self.spatial_encoder_1(spatial_input_embedded, attn_mask)
            fusion_features.append(spatial_input_embedded)
        
        if self.args.P:
            sigma_x, sigma_y = self.args.sigma, self.args.sigma
            grid_size = self.args.image_size
            mu_x = nodes_current[:, :, 0] * self.args.image_size
            mu_y = nodes_current[:, :, 1] * self.args.image_size

            x = torch.linspace(0, self.args.image_size, grid_size).cuda()
            y = torch.linspace(0, self.args.image_size, grid_size).cuda()
            X, Y = torch.meshgrid(x, y)
            
            total_gaussian = torch.zeros((sum(node_index), grid_size, grid_size), device=device)
            for j in range(self.args.obs_length):
                Z = gaussian_2d_torch(X, Y, mu_x[j], mu_y[j], sigma_x, sigma_y)
                total_gaussian = total_gaussian + Z  
        
            heatmap = self.encode_vit(total_gaussian.unsqueeze(1))
            fusion_features.append(heatmap.unsqueeze(0).expand(8, -1, -1))
        
        fusion_feat = torch.cat(fusion_features, dim=2)
        
        fusion_feat = self.fusion_layer(fusion_feat)

        if self.args.S:
            spatial_input_embedded = self.spatial_encoder_2(fusion_feat, attn_mask)
            temporal_input_embedded = torch.cat((temporal_input_embedded, spatial_input_embedded.permute(1, 0, 2)), dim=2)
            temporal_input_embedded = self.encoder2_layer(temporal_input_embedded).permute(1, 0, 2)
        else:
            temporal_input_embedded = temporal_input_embedded.permute(1, 0, 2)
        temporal_input_embedded = self.temporal_encoder_2(temporal_input_embedded)
        
        temporal_input_embedded_wnoise = torch.cat((temporal_input_embedded, noise), dim=2)
        
        
        outputs_current = self.output_layer(
            self.dropout(
                self.relu(
                    self.output(temporal_input_embedded_wnoise.permute(1, 2, 0)).permute(2, 0, 1)
                )
            )
        )
        
        outputs[:, node_index, :] = outputs_current

        return outputs

