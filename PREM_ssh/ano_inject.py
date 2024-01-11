import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import argparse

# - Load dataset 
def load_ano_injection_mat(dataset):

    #data_mat = sio.loadmat("./ano_data/{0}/{0}.mat".format(dataset))
    data_mat = sio.loadmat("./dataset/{}.mat".format(dataset))

    try:
        adj = data_mat['A']
        feat = data_mat['X']
        label = data_mat['gnd']
    except Exception:
        adj = data_mat['Network']
        feat = data_mat['Attributes']
        label = data_mat['Label']

    label = label.flatten()
    #?  flatten sparse matrix
    if not isinstance(adj, np.ndarray):
        adj = adj.toarray()
    if not isinstance(feat, np.ndarray):
        feat = feat.toarray()

    return data_mat, adj, feat, label

# - Anomaly injection
def make_anomalies(dataset, rate=0.05,clique_size=5, degree=30, surround=25, scale_factor=5):

    '''
    Input:
        adj: adjacency matrix (Network)
        feat: feature matrix (Attributes)
        rate: anomaly rate -> prob低於rate就執行anomaly injection
        clique_size: size of clique -> Structure: add clique (High-degree)
        surround: candidate node set -> Attribute (Deviated: change feature to the furthest of n nodes) 
        scale_factor: scale factor -> Attribute (Disproportionate:scale attr )
    Output:
        adj_aug: augmented adjacency matrix
        feat_aug: augmented feature matrix
        label_aug: augmented label
        str_label_aug: augmented structural anomaly label
        atr_label_aug: augmented attribute anomaly label
    '''
    _, adj, feat, _ = load_ano_injection_mat(dataset)
    adj_aug, feat_aug = adj.copy(), feat.copy()
    label_aug = np.zeros(adj.shape[0]).astype(np.uint8)
    str_label_aug = np.zeros(adj.shape[0]).astype(np.uint8)
    atr_label_aug = np.zeros(adj.shape[0]).astype(np.uint8)
    
    assert(adj_aug.shape[0]==feat_aug.shape[0]) #? node數要相同，否則會出錯
    num_nodes = adj_aug.shape[0]

    #* Anomaly injection (node) 
    for i in range(num_nodes):
        #! 設置機率，若小於rate，該node就執行anomaly injection
        prob = np.random.uniform() #? numpy.random.uniform(low=,high=1) [0.1)
        if prob > rate: 
            continue #? 跳過此次迴圈，繼續下一次迴圈
        label_aug[i] = 1 #? 1:anomaly, 0:normal
        
        #! 決定是何種anomaly
        #? numpy.random.randint(low,high) [0,5) : 傳回隨機int
        #? Original fixed pattern
        #one_fifth = np.random.choice(np.arange(5), size=1, p=[0.1,0,0,0.9,0])
        #? Modified pattern
        one_fifth = np.random.choice(np.arange(5), size=1, p=[0.025,0.225,0.25,0.25,0.25])

        #! Structure 
        if one_fifth == 0:
            str_label_aug[i] = 1 #? 1:structural anomaly, 0:normal
            # add clique 
            new_neighbors = np.random.choice(np.arange(num_nodes), clique_size, replace=False)
            new_neighbors = np.append(new_neighbors, i)
            new_neighbors = np.unique(new_neighbors)
            for idx_i in new_neighbors:
                for idx_j in new_neighbors:
                    adj_aug[idx_i][idx_j] = 1
                    adj_aug[idx_j][idx_i] = 1
                
                label_aug[idx_i] = 1
                str_label_aug[idx_i] = 1

        if one_fifth == 1:
            str_label_aug[i] = 1 #? 1:structural anomaly, 0:normal
            # High-degree
            new_neighbors = np.random.choice(np.arange(num_nodes), degree, replace=False)
            for n in new_neighbors:
                adj_aug[n][i] = 1
                adj_aug[i][n] = 1               

        elif one_fifth == 2:
            str_label_aug[i] = 1 #? 1:structural anomaly, 0:normal
            # drop all connection (Outlying)
            neighbors = np.nonzero(adj[i]) #? 取得非0元素的index

            #? 作用與 if 語句相反:測試條件是否為假
            #? 如果無鄰居，則跳過此次迴圈，繼續下一次迴圈
            if not neighbors[0].any():
                    continue
            else: 
                neighbors = neighbors[0]

            for n in neighbors:
                adj_aug[i][n] = 0
                adj_aug[n][i] = 0

        #! Attribute
        elif one_fifth == 3:
            atr_label_aug[i] = 1 #? 1:attribute anomaly, 0:normal
            # Deviated(change feature to the furthest of n nodes)
            candidates = np.random.choice(np.arange(num_nodes), surround, replace=False)
            max_dev, max_idx = 0, i

            for c in candidates:
                dev = np.square(feat[i]-feat[c]).sum() #? 計算距離
                if dev > max_dev:
                    max_dev = dev
                    max_idx = c
            feat_aug[i] = feat[max_idx] #? 將feature改成最遠的node的feature

        else:
            atr_label_aug[i] = 1 #? 1:attribute anomaly, 0:normal
            # scale attr (Disproportionate)
            prob = np.random.uniform(0, 1)
            if prob > 0.5:
                feat_aug[i] = feat_aug[i]*scale_factor #? 將feature放大
            else:
                feat_aug[i] = feat_aug[i]/scale_factor #? 將feature縮小

    #* Modify data type
    adj_aug = sp.csr_matrix(adj_aug)    
    feat_aug = sp.lil_matrix(feat_aug)

    print("Total anomaly: {}".format(np.nonzero(label_aug)[0].shape[0]))
    print("Total  structural anomaly: {}".format(np.nonzero(str_label_aug)[0].shape[0]))
    print("Total attribute anomaly: {}".format(np.nonzero(atr_label_aug)[0].shape[0]))


    return adj_aug, feat_aug, label_aug, str_label_aug, atr_label_aug

'''
parser = argparse.ArgumentParser(description='Anomaly injection dataset')
parser.add_argument('--dataset', type=str, default='Flickr')

args = parser.parse_args()
network, attribute, label, str_label, atr_label = make_anomalies(args.dataset, rate=0.2, clique_size=20, surround=50, scale_factor=10)
#network, attribute, label = load_ano_injection_mat(args.dataset)
print(len(np.nonzero(label)[0]))
#print(np.nonzero(str_label))
#print(np.nonzero(atr_label))
print(type(label[0]))
'''