import torch.optim as optim
import util
import numpy as np
import configparser
import argparse
import models
import torch
import pandas as pd

def load_weighted_adjacency_matrix(file_path):
    df = pd.read_csv(file_path, header=None)
    return df.to_numpy()

class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid, dropout, normalization, lrate, wdecay, device, days=288,
                 dims=40, order=2, resolution=12, zero_=-3, predict_point=0):

        dataset = "metr-la"
        self.predict_point = predict_point
        print("predicting the traffic speed in " + str(5+5*self.predict_point) + ' min.')
        time_intvl = 5
        n_his = 12
        Kt = 3
        stblock_num = 2
        if ((Kt - 1) * 2 * stblock_num > n_his) or ((Kt - 1) * 2 * stblock_num <= 0):
            raise ValueError(f'ERROR: {Kt} and {stblock_num} are unacceptable.')
        Ko = n_his - (Kt - 1) * 2 * stblock_num
        drop_rate = 0.5
        batch_size = 32
        learning_rate = lrate
        weight_decay_rate = wdecay
        step_size = 10
        gamma = 0.999
        self.zero_ = zero_

        # gated_act_func = ConfigSectionMap('casualconv')['gated_act_func']
        gated_act_func = "glu"
        
        # graph_conv_type = "grdcnconv"
        # graph_conv_type = "reactiondiffusionconv"
        graph_conv_type = "gcnconv"
        # graph_conv_type = "chebconv"
        
        if (graph_conv_type != "chebconv") and (graph_conv_type != "gcnconv") and (graph_conv_type != "grdcnconv") and (graph_conv_type != "reactiondiffusionconv"):
            raise NotImplementedError(f'ERROR: {graph_conv_type} is not implemented.')
        else:
            graph_conv_type = graph_conv_type
        
        Ks = 2
        if (graph_conv_type == 'gcnconv') and (Ks != 2):
            Ks = 2

        # blocks: settings of channel size in st_conv_blocks and output layer,
        # using the bottleneck design in st_conv_blocks
        blocks = []
        blocks.append([1])
        for l in range(stblock_num):
            blocks.append([64, 16, 64])
        if Ko == 0:
            blocks.append([128])
        elif Ko > 0:
            blocks.append([128, 128])
        blocks.append([1])
        

        day_slot = int(24 * 60 / time_intvl)
        n_pred = 1

        time_pred = n_pred * time_intvl
        time_pred_str = str(time_pred) + '_mins'
        model_name = "stgcn_gcnconv_sym_gtu"
        # model_save_path = model_save_path + model_name + '_' + dataset + '_' + time_pred_str + "mins" + '_' + model_name + "_" + str(learning_rate) +'.pth'

        # wam_path = "/localproject/users/yus516/data/code/explainable-graph/Physics-Informed-STGCN/original/orig/STGCN/data/train/road_traffic/metr-la/metr-la-road-graph-stgcn-input.csv"
        # wam_path = "/localproject/users/yus516/data/code/explainable-graph/Physics-Informed-STGCN/original/orig/STGCN/data/train/road_traffic/metr-la/adj_mat.csv"
        # wam_path = "data/seattle-loop/adj_seattle.csv"
        # wam_path = "data/pems-bay/adj_pems_bay.csv"
        wam_path = "data/metr-la/adj_mat.csv"
        adj_mat = load_weighted_adjacency_matrix(wam_path)

        n_vertex_vel = num_nodes
        n_vertex_adj = num_nodes
        if n_vertex_vel != n_vertex_adj:
            raise ValueError(f'ERROR: number of vertices in dataset is not equal to number of vertices in weighted adjacency matrix.')
        else:
            n_vertex = n_vertex_vel
        
        mat_type = "hat_sym_normd_lap_mat"
        # mat_type = "wid_sym_normd_lap_mat"
        

        if graph_conv_type == "chebconv":
            if (mat_type != "wid_sym_normd_lap_mat") and (mat_type != "wid_rw_normd_lap_mat"):
                raise ValueError(f'ERROR: {mat_type} is wrong.')
            mat = util.calculate_laplacian_matrix(adj_mat, mat_type)
            chebconv_matrix = torch.from_numpy(mat).float().to(device)
            stgcn_chebconv = models.STGCN_ChebConv(Kt, Ks, blocks, n_his, n_vertex, gated_act_func, graph_conv_type, chebconv_matrix, drop_rate).to(device)
            model = stgcn_chebconv

        elif graph_conv_type == "gcnconv":
            if (mat_type != "hat_sym_normd_lap_mat") and (mat_type != "hat_rw_normd_lap_mat"):
                raise ValueError(f'ERROR: {mat_type} is wrong.')
            mat = util.calculate_laplacian_matrix(adj_mat, mat_type)
            gcnconv_matrix = torch.from_numpy(mat).float().to(device)
            stgcn_gcnconv = models.STGCN_GCNConv(Kt, Ks, blocks, n_his, n_vertex, gated_act_func, graph_conv_type, gcnconv_matrix, drop_rate).to(device)
            model = stgcn_gcnconv

        elif graph_conv_type == "grdcnconv":
            if (mat_type != "hat_sym_normd_lap_mat") and (mat_type != "hat_rw_normd_lap_mat"):
                raise ValueError(f'ERROR: {mat_type} is wrong.')
            mat = util.calculate_laplacian_matrix(adj_mat, mat_type)
            mat_plus = util.calculate_laplacian_matrix_plus(adj_mat, mat_type)
            gcnconv_matrix = torch.from_numpy(mat).float().to(device)
            gcnconv_matrixplus = torch.from_numpy(mat_plus).float().to(device)
            stgcn_gcnconv = models.STGRDCN_GCNConv(Kt, Ks, blocks, n_his, n_vertex, gated_act_func, graph_conv_type, [gcnconv_matrix, gcnconv_matrixplus], drop_rate).to(device)
            model = stgcn_gcnconv

        elif graph_conv_type == "reactiondiffusionconv":
            # stgcn_gcnconv = models.STGCN_GRDCNConv(Kt, Ks, blocks, n_his, n_vertex, gated_act_func, graph_conv_type, adj_mat, drop_rate).to(device)
            stgcn_gcnconv = models.STGCN_GRDCNConv(Kt, Ks, blocks, n_his, n_vertex, gated_act_func, graph_conv_type, adj_mat, drop_rate).to(device)
            model = stgcn_gcnconv

        self.model = model

        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scaeduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=.3, patience=10, threshold=1e-3,
                                                              min_lr=1e-5, verbose=True)
        self.loss = util.masked_mae
        print("the loss function is masked mae")
        # self.loss = util.masked_mse
        # print("the loss function is masked mse")
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val, ind):

        self.model.train()
        self.optimizer.zero_grad()
        # input = nn.functional.pad(input, (1, 0, 0, 0))
        # input = input[:, :, :, -1][:, :, :, None]
        input = input.transpose(2, 3)
        output = self.model(input)
        output = output.transpose(2, 3)
        real = torch.unsqueeze(real_val, dim=1)[:, :, :, self.predict_point][:, :, :, None]
        predict = self.scaler.inverse_transform(output)

        x_mask = (input[:, :, -1, :] >= self.zero_)
        x_mask = x_mask.float()[:, :, :, None]
        x_mask /= torch.mean((x_mask))
        x_mask = torch.where(torch.isnan(x_mask), torch.zeros_like(x_mask), x_mask) 

        loss = self.loss(predict, real, 0.0, x_mask)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict, real, 0.0, x_mask).item()
        mape = util.masked_mape(predict, real, 0.0, x_mask).item()
        rmse = util.masked_rmse(predict, real, 0.0, x_mask).item()
        return mae, mape, rmse

    def eval(self, input, real_val, ind):
        self.model.eval()
        input = input.transpose(2, 3)
        output = self.model(input)
        output = output.transpose(2, 3)
        outputs = output
        
    
        # generate training mask
        x_mask = (input[:, :, -1, :] >= self.zero_)
        x_mask = x_mask.float()[:, :, :, None]
        x_mask /= torch.mean((x_mask))
        x_mask = torch.where(torch.isnan(x_mask), torch.zeros_like(x_mask), x_mask) 


        predict = self.scaler.inverse_transform(outputs)
        real = torch.unsqueeze(real_val, dim=1)
        real = real[:, :, :, self.predict_point][:, :, :, None]
        mae = util.masked_mae(predict, real, 0.0, x_mask).item()
        mape = util.masked_mape(predict, real, 0.0, x_mask).item()
        rmse = util.masked_rmse(predict, real, 0.0, x_mask).item()
        return mae, mape, rmse

    # let the model diff 12 times
    # def eval(self, input, real_val, ind):
    #     self.model.eval()
    #     input = input.transpose(2, 3)
    #     output = self.model(input)
    #     output = output.transpose(2, 3)
    #     outputs = output
    #     for i in range(1, 12):
    #         ind = [(ii + i) % 288 for ii in ind]
    #         output = output.transpose(2, 3)
    #         input_loop = input[:, :, 1:, :]
    #         input_loop = torch.cat((input_loop, output), 2)
    #         output = self.model(input)
    #         output = output.transpose(2, 3)
    #         outputs = torch.cat((outputs, output), 3)

    
    #     # generate training mask
    #     x_mask = (input[:, :, -1, :] >= self.zero_)
    #     x_mask = x_mask.float()[:, :, :, None]
    #     x_mask /= torch.mean((x_mask))
    #     x_mask = torch.where(torch.isnan(x_mask), torch.zeros_like(x_mask), x_mask) 


    #     predict = self.scaler.inverse_transform(outputs)
    #     real = torch.unsqueeze(real_val, dim=1)
    #     mae = util.masked_mae(predict, real, 0.0, x_mask).item()
    #     mape = util.masked_mape(predict, real, 0.0, x_mask).item()
    #     rmse = util.masked_rmse(predict, real, 0.0, x_mask).item()
    #     return mae, mape, rmse
