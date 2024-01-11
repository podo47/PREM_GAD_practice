import scipy.io as sio
import scipy.sparse as sp
import numpy as np
from  modules.utils import rescale
import torch


#data = sio.loadmat("dataset/{}.mat".format('citeseer'))
#print(data.keys())
#print(data['Network'])
#network = data['Network']
#print(type(network))
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

sed = 0.5
#print(rescale(sed))

print(torch.__version__)
