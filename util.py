import pickle
import numpy as np
import os
import torch
from scipy.linalg import eigvalsh
from scipy.linalg import fractional_matrix_power


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, begin=0, days=288, pad_with_last_sample=True, add_ind=True, ind=0):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        if add_ind:
            self.ind = np.arange(begin, begin + self.size)
        else:
            self.ind = ind
        self.days = days
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            ind_padding = np.repeat(self.ind[-1:], num_padding, axis=0)
            self.xs = np.concatenate([xs, x_padding], axis=0)
            self.ys = np.concatenate([ys, y_padding], axis=0)
            self.ind = np.concatenate([self.ind, ind_padding], axis=0)
        self.size = len(self.xs)

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.ind = self.ind[permutation]
        self.xs = xs
        self.ys = ys

    def filter_by_slice(self, start_point, end_point):
        from_point = start_point * 12
        to_point = end_point * 12
        print("filtering samples from " + str(from_point) + "to " + str(to_point))
        mid = (from_point + to_point) / 2
        width = np.abs((from_point - to_point) / 2)
        good_index = np.where((np.abs(self.ind % 288 - mid) <= width))
        self.xs = self.xs[good_index[0]]
        self.ys = self.ys[good_index[0]]
        self.ind = self.ind[good_index[0]]
        self.size = len(self.ind)



    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                i_i = self.ind[start_ind: end_ind, ...] % self.days
                # xi_i = np.tile(np.arange(x_i.shape[1]), [x_i.shape[0], x_i.shape[2], 1, 1]).transpose(
                #     [0, 3, 1, 2]) + self.ind[start_ind: end_ind, ...].reshape([-1, 1, 1, 1])
                # x_i = np.concatenate([x_i, xi_i % self.days / self.days, np.eye(7)[xi_i // self.days % 7].squeeze(-2)],
                #                      axis=-1)
                yield (x_i, y_i, i_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_dataset_weekday_weekend(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None, days=288, sequence=12,
                 in_seq=12, keep_order=True, filter=0, start_point=0, lastinghours=24):
    data = {}

    weekday_data = np.load(os.path.join(dataset_dir, 'train_weekday.npz'))
    weekend_data = np.load(os.path.join(dataset_dir, 'test_weekend.npz'))


    train_val_data_x = weekday_data['x'][:, -in_seq:, :, 0:2]
    train_val_data_y = weekday_data['y'][:, -in_seq:, :, 0:1]
    test_data_x = weekend_data['x'][:, -in_seq:, :, 0:2]
    test_data_y = weekend_data['y'][:, -in_seq:, :, 0:1]

    train_val_data = DataLoader(train_val_data_x, train_val_data_y, batch_size, days=days, begin=0, add_ind=True, ind=0)
    test_data = DataLoader(train_val_data_x, train_val_data_y, batch_size, days=days, begin=0, add_ind=True, ind=0)

    train_val_data.size = 288*12
    train_val_data.xs, train_val_data.ys = train_val_data.xs[-train_val_data.size:, ...], train_val_data.ys[-train_val_data.size:, ...]
    train_val_data.ind = train_val_data.ind[0:train_val_data.size]

    train_val_data.filter_by_slice(start_point, start_point + lastinghours)

    train_val_data_x, train_val_data_y = train_val_data.xs, train_val_data.ys
    test_data_x, test_data_y = test_data.xs, test_data.ys

    train_set_size = int(train_val_data.size / 4)

    permutation = np.random.RandomState(seed=42).permutation(train_val_data.size)
    train_val_data_x, train_val_data_y = train_val_data_x[permutation], train_val_data_y[permutation]

    train_x = train_val_data_x[:3*train_set_size, ...]
    train_y = train_val_data_y[:3*train_set_size, ...]
    val_x = train_val_data_x[-train_set_size:, ...]
    val_y = train_val_data_y[-train_set_size:, ...]

    data['scaler'] = StandardScaler(mean=train_x[..., 0].mean(), std=train_x[..., 0].std())
    train_x = data['scaler'].transform(train_x[..., 0])
    val_x = data['scaler'].transform(val_x[..., 0])
    test_x = data['scaler'].transform(test_data_x[..., 0])

    train_x, val_x, test_x = train_x[:, :, :, None], val_x[:, :, :, None], test_x[:, :, :, None]

    data['train_loader'] = DataLoader(train_x, train_y, batch_size, days=days, begin=0, add_ind=True, ind=0)
    data['val_loader'] = DataLoader(val_x, val_y, valid_batch_size, days=days,
                                    begin=0, add_ind=True, ind=0)
    data['test_loader'] = DataLoader(test_x, test_data_y, test_batch_size, days=days,
                                    begin=0, add_ind=True, ind=0)


    return data

def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None, days=288, sequence=12,
                 in_seq=12, keep_order=True, filter=0, start_point=0, lastinghours=24, missingnode=-1, norm_y=False):
    
    if (missingnode >= 0):
        global missing_node
        missing_node = missingnode
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x'][:, -in_seq:, :, 0:2]  # B T N F speed flow
        data['y_' + category] = cat_data['y'][:, :sequence, :, 0:1]

        if category == "train":
            data['scaler'] = StandardScaler(mean=cat_data['x'][..., 0].mean(), std=cat_data['x'][..., 0].std())
    for si in range(0, data['x_' + category].shape[-1]):
        scaler_tmp = StandardScaler(mean=data['x_train'][..., si].mean(), std=data['x_train'][..., si].std())
        for category in ['train', 'val', 'test']:
            data['x_' + category][..., si] = scaler_tmp.transform(data['x_' + category][..., si])
    if (norm_y):
        for category in ['train', 'val', 'test']:
            data['y_' + category] = data['scaler'].transform(data['y_' + category])

    if (keep_order):
        data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, days=days, begin=0, add_ind=True, ind=0)
        data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size, days=days,
                                        begin=data['x_train'].shape[0], add_ind=True, ind=0)
        data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size, days=days,
                                        begin=data['x_train'].shape[0] + data['x_val'].shape[0], add_ind=True, ind=0)    
    else:
        all_data_x = np.concatenate([data['x_train'], data['x_val'], data['x_test']], 0)
        all_data_y = np.concatenate([data['y_train'], data['y_val'], data['y_test']], 0)
        all_data = DataLoader(all_data_x, all_data_y, batch_size, days=days, begin=0, add_ind=True, ind=0)

        # do the filter
        if (filter > 0):
            all_data.filter_by_slice(start_point, start_point + lastinghours)
        
        all_data.shuffle()
        test_size = int(all_data.size * 0.2)
        break1 = all_data.size - 2 * test_size
        break2 = all_data.size - test_size
        data['train_loader'] = DataLoader(all_data.xs[:break1, ...], all_data.ys[:break1, ...], batch_size, days=days, begin=0, add_ind=False, ind=all_data.ind[:break1, ...])
        data['val_loader'] = DataLoader(all_data.xs[break1:break2, ...], all_data.ys[break1:break2, ...], batch_size, days=days, begin=0, add_ind=False, ind=all_data.ind[break1:break2, ...])
        data['test_loader'] = DataLoader(all_data.xs[:test_size, ...], all_data.ys[:test_size, ...], batch_size, days=days, begin=0, add_ind=False, ind=all_data.ind[:test_size, ...])
    return data

def masked_mse(preds, labels, null_val=np.nan, x_mask=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    # mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    # strength the mask by remove the prediction from 0  => real speed
    if (len(x_mask.shape)==4):
        mask = mask * x_mask[:, :, :, 0][:, :, :, None]
    else:
        if (len(mask.shape)==3):
            mask = mask.reshape(mask.shape[0], mask.shape[1])
        mask = mask * x_mask
    mask /= torch.mean((mask))
    loss = (preds - labels) ** 2
    if (len(loss.shape)==3):
        loss = mask.reshape(loss.shape[0], loss.shape[1])
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan, x_mask=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val, x_mask=x_mask))


def masked_mae(preds, labels, null_val=np.nan, x_mask=np.nan):

    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)


    if (len(x_mask.shape)==4):
        mask = mask * x_mask[:, :, :, 0][:, :, :, None]
    else:
        if (len(mask.shape)==3):
            mask = mask.reshape(mask.shape[0], mask.shape[1])
        mask = mask * x_mask

    mask = mask.float()
    # mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    if (len(loss.shape) == 3):
        loss = loss.reshape(loss.shape[0], loss.shape[1])
    loss = loss * mask
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan, x_mask=np.nan):

    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    # if not np.isnan(x_mask):
    #     mask = mask * x_mask

    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    # return torch.mean(loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse

def metric_unmasked(pred, real):
    mae = masked_mae(pred, real, 1000).item()
    mape = masked_mape(pred, real, 1000).item()
    rmse = masked_rmse(pred, real, 1000).item()
    return mae, mape, rmse

def metric_strength(pred, real, x_mask):
    mae = masked_mae(pred, real, 0.0, x_mask).item()
    mape = masked_mape(pred, real, 0.0, x_mask).item()
    rmse = masked_rmse(pred, real, 0.0, x_mask).item()
    return mae, mape, rmse

def calculate_laplacian_matrix(adj_mat, mat_type):
    n_vertex = adj_mat.shape[0]
    id_mat = np.identity(n_vertex)

    # D_row
    deg_mat_row = np.diag(np.sum(adj_mat, axis=1))
    # D_com
    #deg_mat_col = np.diag(np.sum(adj_mat, axis=0))

    # D = D_row as default
    deg_mat = deg_mat_row

    # wid_A = A + I
    wid_adj_mat = adj_mat + id_mat
    # wid_D = D + I
    wid_deg_mat = deg_mat + id_mat

    # Combinatorial Laplacian
    # L_com = D - A
    com_lap_mat = deg_mat - adj_mat

    if mat_type == 'id_mat':
        return id_mat
    elif mat_type == 'com_lap_mat':
        return com_lap_mat

    if (mat_type == 'sym_normd_lap_mat') or (mat_type == 'wid_sym_normd_lap_mat') or (mat_type == 'hat_sym_normd_lap_mat'):
        deg_mat_inv_sqrt = fractional_matrix_power(deg_mat, -0.5)
        wid_deg_mat_inv_sqrt = fractional_matrix_power(wid_deg_mat, -0.5)

        # Symmetric normalized Laplacian
        # For SpectraConv
        # L_sym = D^{-0.5} * L_com * D^{-0.5} = I - D^{-0.5} * A * D^{-0.5}
        sym_normd_lap_mat = np.matmul(np.matmul(deg_mat_inv_sqrt, com_lap_mat), deg_mat_inv_sqrt)

        # For ChebConv
        # wid_L_sym = 2 * L_sym / lambda_max_sym - I
        ev_max_sym = max(eigvalsh(sym_normd_lap_mat))
        wid_sym_normd_lap_mat = 2 * sym_normd_lap_mat / ev_max_sym - id_mat

        # For GCNConv
        # hat_L_sym = wid_D^{-0.5} * wid_A * wid_D^{-0.5}
        hat_sym_normd_lap_mat = np.matmul(np.matmul(wid_deg_mat_inv_sqrt, wid_adj_mat), wid_deg_mat_inv_sqrt)

        if mat_type == 'sym_normd_lap_mat':
            return sym_normd_lap_mat
        elif mat_type == 'wid_sym_normd_lap_mat':
            return wid_sym_normd_lap_mat
        elif mat_type == 'hat_sym_normd_lap_mat':
            return hat_sym_normd_lap_mat

    elif (mat_type == 'rw_normd_lap_mat') or (mat_type == 'wid_rw_normd_lap_mat') or (mat_type == 'hat_rw_normd_lap_mat'):

        deg_mat_inv = fractional_matrix_power(deg_mat, -1)
        wid_deg_mat_inv = fractional_matrix_power(wid_deg_mat, -1)

        # Random Walk normalized Laplacian
        # For SpectraConv
        # L_rw = D^{-1} * L_com = I - D^{-1} * A
        rw_normd_lap_mat = np.matmul(deg_mat_inv, com_lap_mat)

        # For ChebConv
        # wid_L_rw = 2 * L_rw / lambda_max_rw - I
        ev_max_rw = max(eigvalsh(rw_normd_lap_mat))
        wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / ev_max_rw - id_mat

        # For GCNConv
        # hat_L_rw = wid_D^{-1} * wid_A
        hat_rw_normd_lap_mat = np.matmul(wid_deg_mat_inv, wid_adj_mat)

        if mat_type == 'rw_normd_lap_mat':
            return rw_normd_lap_mat
        elif mat_type == 'wid_rw_normd_lap_mat':
            return wid_rw_normd_lap_mat
        elif mat_type == 'hat_rw_normd_lap_mat':
            return hat_rw_normd_lap_mat

def evaluate_model(model, loss, data_iter, zscore):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x).view(len(x), -1)
            l = loss(y_pred, y, zscore)
            # l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        mse = l_sum / n
        
        return mse

def evaluate_metric(model, data_iter, scaler):
    model.eval()
    with torch.no_grad():
        mae, sum_y, mape, mse = [], [], [], []
        for x, y in data_iter:
            y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred = scaler.inverse_transform(model(x).view(len(x), -1).cpu().numpy()).reshape(-1)

            # add mask
            mask = np.where(y >= 0.3)
            y = y[mask]
            y_pred = y_pred[mask]

            d = np.abs(y - y_pred)
            mae += d.tolist()
            sum_y += y.tolist()
            mape += (d / y).tolist()
            mse += (d ** 2).tolist()
        MAE = np.array(mae).mean()
        MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        WMAPE = np.sum(np.array(mae)) / np.sum(np.array(sum_y))

        #return MAE, MAPE, RMSE
        return MAE, RMSE, WMAPE


def data_transform(data_raw, data_masked, n_his, n_pred, day_slot):
    # produce data slices for x_data and y_data

    n_vertex = data_raw.shape[1]
    len_record = len(data_raw)
    num = len_record - n_his - n_his
    
    x = np.zeros([num, n_his, n_vertex, 1])
    y = np.zeros([num, n_his, n_vertex, 1])
    
    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :, :] = data_masked[head: tail].reshape(n_his, n_vertex, 1)
        y[i, :, :, :] = data_raw[tail: tail + 12].reshape(n_his, n_vertex, 1)

    return x, y