import random

import numpy as np
from cffi.backend_ctypes import xrange

dataset_source_folder_path = './data/' + 'cora' + '/original/'
dataset_save_folder_path = './data/' + 'cora' + '/'

idx_features_labels = np.genfromtxt("{}/node".format(dataset_source_folder_path), dtype=np.dtype(str))
idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
edges_unordered = np.genfromtxt("{}/link".format(dataset_source_folder_path), dtype=np.int32)

temp_idx = idx[1:500]
edges_temp = edges_unordered
index_list = []
for id, idx in enumerate(edges_temp):
    if (idx[0] in temp_idx) & (idx[1] in temp_idx):
        index_list.append(id)

k = len(index_list) * 20 // 100
indicies = random.sample(xrange(len(index_list)), k)
#delte_index = [index_list[i] for i in indicies]
print(indicies)
edges_unordered_del = np.delete(edges_unordered,indicies,0)
print(edges_unordered_del)
np.savetxt(dataset_save_folder_path + 'link', edges_unordered_del, fmt="%s")
