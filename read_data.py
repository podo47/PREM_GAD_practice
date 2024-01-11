import scipy.io as sio
import scipy.sparse as sp
import numpy as np
import os

'''
if not os.path.exists('./ano_data/inject'):
            os.makedirs('./ano_data/inject')
'''

data = sio.loadmat("ano_data/{0}/{0}.mat".format('Amazon'))
#data = sio.loadmat("dataset/{0}.mat".format('cora'))
#print(data.keys())
#print(type(data['Network']))
#print(data['gnd'])
#print(type(data['Label']))

label = data['gnd']
label = label.flatten()
#print(label)
#ano_labels = np.array(label)
#ano_labels = np.squeeze(np.array(label))
#print(ano_labels)

network = data['A']
print(network.shape[0])
feat = data['X']
print(feat.shape[0])

#network = sp.csr_matrix(data['A'])
#print(network)
#network = network.toarray()
#print(network)

#feat = sp.lil_matrix(data['Attributes'])
#print(type(network))
#print(network)
#print(feat)
#print("prem")
#adj = sp.csr_matrix(network)
#print(adj)
#print(len(np.squeeze(np.array(data['attr_anomaly_label']))))
#print(data['str_anomaly_label'])
'''

arr = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 2]])
print("arr")
print(arr)
csr = sp.csr_matrix(arr)
print("CSR")
print(csr)
csc = sp.csc_matrix(arr)
print("CSC")
print(csc)
trans = sp.csr_matrix(csc)
print("trans")
print(trans)
print(type(trans))

all_ted = [4,2,4,6]
ted = "all"
print(eval(ted+"_ted"))

'''

