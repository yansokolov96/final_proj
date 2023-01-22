import random

import numpy as np
from cffi.backend_ctypes import xrange

def runScript(precentage = 10):

    nodesToDelete = precentage

    if nodesToDelete > 90:
        nodesToDelete = 90

    if nodesToDelete < 0:
        nodesToDelete = 0

    dataset_source_folder_path = './data/' + 'cora' + '/original/'
    dataset_save_folder_path = './data/' + 'cora' + '/'

    #contains the information about the nodes in the dataset, such as their index and any associated features or labels.
    idx_features_labels = np.genfromtxt("{}/node".format(dataset_source_folder_path), dtype=np.dtype(str))

    # creates a new numpy array, idx, which contains the first column of the idx_features_labels array, which is the index of each node in the dataset. The np.array() function is used to convert the selected column into a numpy array. The dtype of this new array is set to np.int32, which stands for 32-bit integer. This means that the index of each node is stored as a 32-bit integer in the idx array.
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)

    # uses the numpy function genfromtxt to load the file "link" from the dataset_source_folder_path, which is the "cora/original" folder. This file contains the information about the edges in the dataset, such as the indices of the nodes that the edges connect. The function reads the file, which is a text file, and converts its contents into a numpy array with dtype of 'int32'. This array, edges_unordered, contains the information about the edges in the dataset, such as the indices of the nodes that the edges connect, in an unordered manner.
    edges_unordered = np.genfromtxt("{}/link".format(dataset_source_folder_path), dtype=np.int32)

    # creates a new numpy array called temp_idx, which contains a subset of the elements from the idx array, specifically the elements from index 1 to index 499 (not 500) . This means that temp_idx will contain the indices of the first 500 nodes in the dataset. This is used to filter the edges array to include only those that have both nodes within this subset of 500 nodes.
    temp_idx = idx[1:500]
    edges_temp = edges_unordered

    # The variable index_list is a list that is created by iterating through the edges_temp array, which is the edges array filtered to include only those that have both nodes within the subset of 500 nodes. For each edge represented by a pair of indices in the edges_temp array, the code checks if both of the indices are present in the temp_idx array. If both of the indices are present in temp_idx array, the index of the edge in the edges_temp array is appended to the index_list list. Therefore, index_list will contain a list of indices of the edges in the edges_temp array that have both nodes within the subset of 500 nodes.
    index_list = []
    for id, idx in enumerate(edges_temp):
        if (idx[0] in temp_idx) & (idx[1] in temp_idx):
            index_list.append(id)

    k = len(index_list) * nodesToDelete // 100
    indicies = random.sample(xrange(len(index_list)), k)
    #delte_index = [index_list[i] for i in indicies]
    edges_unordered_del = np.delete(edges_unordered,indicies,0)

    # Uses the numpy function savetxt() to save the modified edges array, edges_unordered_del, to a file named "link" in the dataset_save_folder_path, which is the "cora" folder. This function writes the array to a text file, each row of the array represents a line in the text file. The fmt parameter is set to '%s', which means that the elements in the array will be written to the text file as strings.
    # This operation will save the modified edges array in the "cora" folder, which will be used later on.
    np.savetxt(dataset_save_folder_path + 'link', edges_unordered_del, fmt="%s")
