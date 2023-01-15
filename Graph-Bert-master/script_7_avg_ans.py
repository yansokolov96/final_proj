import os
from builtins import enumerate

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from code.DatasetLoader import DatasetLoader
from code.MethodBertComp import GraphBertConfig
from code.MethodGraphBertNodeClassification import MethodGraphBertNodeClassification
from code.ResultSaving import ResultSaving
from code.Settings import Settings
import numpy as np
import torch
import matplotlib.lines as mlines
import hungarian_algorithm
import sys
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import confusion_matrix
import seaborn as sns
# sns.set_theme()
from matplotlib import pyplot as plt
from munkres import Munkres

number_of_cluster = 7

dataset_source_folder_temp = "./result/temp20p/"

dataset_source_folder_path = "./result/taged_nodes/1000"


cluster_list = []
number_of_files = 10

cluster_orig = np.genfromtxt(dataset_source_folder_path, dtype=np.int32)

for i in range(1,number_of_files + 1):
    cluster_list.append(np.genfromtxt(dataset_source_folder_temp + "file" + str(i), dtype=np.int32))
'''
res = np.zeros(7)

for i in cluster_list:
    for j in i:
        res[j] = res[j] + 1

res = res / number_of_files
'''
new_cm_temp = []
for i in range(number_of_files):
    new_cm_temp.append(confusion_matrix(cluster_orig, cluster_list[i], labels=range(7)))

new_cm_res = new_cm_temp[0]
for i in range(1,number_of_files):
    new_cm_res = new_cm_res + new_cm_temp[i]

new_cm_res = new_cm_res/number_of_files
new_cm = (np.rint(new_cm_res)).astype(int)

#new_cm = new_cm_res

print(type(new_cm))
plt.figure()
l = [f"c{i}" for i in range(1, number_of_cluster + 1)]
ax = sns.heatmap(new_cm, cmap="YlGn", annot=True, xticklabels=l, yticklabels=l, cbar=True, fmt=',')
ax.set(title="Confusion Matrix", xlabel="Predict", ylabel="True")
ax.get_figure().show()
plt.show()





