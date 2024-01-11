import dgl
import numpy as np
import torch
import os
import random
import scipy.io as sio
import scipy.sparse as sp
from ano_inject import make_anomalies


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
    return labels_one_hot

def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1)).astype(np.float64)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def load_mat(dataset, train_rate=0.3, val_rate=0.1):
    data = sio.loadmat("./dataset/{}.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']

    adj = sp.csr_matrix(network) # csc(Network) -> csr 輸出dense matrix格式: (1, 2)	1 -> (列，行) 值
    feat = sp.lil_matrix(attr) # 使用两列表存非0元素。data保存每列中的非零元素值,rows保存每列中的非零元素所在的行。

    '''
    labels = np.squeeze(np.array(data['Class'],dtype=np.int64) - 1) # 從0開始
    num_classes = np.max(labels) + 1
    labels = dense_to_one_hot(labels,num_classes)
    '''

    # Label = str+attr_anomaly_label (沒有用到類別)
    ano_labels = np.squeeze(np.array(label))
    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    # ACM沒有anomaly_label
    else:
        str_ano_labels = None
        attr_ano_labels = None

    # train-test split
    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[ : num_train]
    idx_val = all_idx[num_train : num_train + num_val]
    idx_test = all_idx[num_train + num_val : ]

    #return adj, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels
    return adj, feat, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels

def load_dataset(args, dataset):
    print('Dataset: {}'.format(dataset), flush=True)

    if args.data_type == 'prem':
        adj, features, _, _, _, ano_label, str_ano_label, attr_ano_label = load_mat(dataset)
    elif args.data_type == 'inject':
        adj, features, ano_label, str_ano_label, attr_ano_label = make_anomalies(dataset, rate=0.05, clique_size=10, degree=30, surround=25, scale_factor=5)

    features, _ = preprocess_features(features)
    src, dst = np.nonzero(adj)
    g = dgl.graph((src, dst))
    g = dgl.add_self_loop(g)
    return g, features, ano_label, str_ano_label, attr_ano_label


def set_random_seeds(seed):
    dgl.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def rescale(x):
    return (x + 1) / 2