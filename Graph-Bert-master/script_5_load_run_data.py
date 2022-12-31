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
number_of_cluster = 7

def load_and_change_cluster2(clusters_net_t):
    clusters_Orig = np.genfromtxt(dataset_source_folder_path + str(len(clusters_net_t)), dtype=np.int32)
    clust_orig_group = dict()

    for i, c in enumerate(clusters_Orig):
        if  c not in clust_orig_group:
            clust_orig_group[c] = {i}
        else:
            clust_orig_group[c].add(i)

    clust_net_group = dict()

    for i, c in enumerate(clusters_net_t):
        if c not in clust_net_group:
            clust_net_group[c] = {i}
        else:
            clust_net_group[c].add(i)

    def myIOU(x,y):
        o = clust_orig_group[x]
        n = clust_net_group[y]
        i = len(o.intersection(n))
        u = len(o.union(n))
        return i/u

    dist = np.fromfunction(np.vectorize(myIOU),(number_of_cluster,number_of_cluster))
    m = np.argmax(dist,axis=0)

    res = np.zeros([len(clusters_net_t)])
    for i,c in enumerate(m):
        res[list(clust_net_group[c])] = i

    return res


def load_and_change_cluster(clusters_net_t):
    clusters_Orig = np.genfromtxt(dataset_source_folder_path + str(len(clusters_net_t)), dtype=np.int32)

    for i in range(number_of_cluster):
        cluster_i = []
        error = 0
        for index, value in enumerate(clusters_Orig):
            if value == i:
                cluster_i.append([index, value])
        val2 = -1

        for j in cluster_i:
            if clusters_net_t[j[0]] != j[1]:
                error = error + 1
                val2 = clusters_net_t[j[0]]
        if error / len(cluster_i) > 0.9:
            val1 = i
            if val2 != -1:
                for k_t in range(clusters_net_t):
                    if clusters_net_t[k_t] == val1:
                        clusters_net_t[k_t] = val2
                    if clusters_net_t[k_t] == val2:
                        clusters_net_t[k_t] = val1

    return clusters_net_t


# ---- 'cora' , 'citeseer', 'pubmed' ----
from code.base_class.method import method

dataset_source_folder_path = './result/taged_nodes/'
dataset_name = 'cora'

np.random.seed(1)
torch.manual_seed(1)

if dataset_name == 'cora-small':
    nclass = 7
    nfeature = 1433
    ngraph = 10
elif dataset_name == 'cora':
    nclass = 7
    nfeature = 1433
    ngraph = 2708
elif dataset_name == 'citeseer':
    nclass = 6
    nfeature = 3703
    ngraph = 3312
elif dataset_name == 'pubmed':
    nclass = 3
    nfeature = 500
    ngraph = 19717

if dataset_name == 'pubmed':
    lr = 0.001
    k = 30
    max_epoch = 1000  # 500 ---- do an early stop when necessary ----
elif dataset_name == 'cora':
    lr = 0.01
    k = 7
    max_epoch = 150  # 150 ---- do an early stop when necessary ----
elif dataset_name == 'citeseer':
    k = 5
    lr = 0.001
    max_epoch = 2000  # 2000 # it takes a long epochs to get good results, sometimes can be more than 2000

x_size = nfeature
hidden_size = intermediate_size = 32
num_attention_heads = 2
num_hidden_layers = 2
y_size = nclass
graph_size = ngraph
residual_type = 'graph_raw'
# --------------------------

data_obj = DatasetLoader()
data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
data_obj.dataset_name = dataset_name
data_obj.k = k
data_obj.load_all_tag = True

bert_config = GraphBertConfig(residual_type=residual_type, k=k, x_size=nfeature, y_size=y_size,
                              hidden_size=hidden_size, intermediate_size=intermediate_size,
                              num_attention_heads=num_attention_heads, num_hidden_layers=num_hidden_layers)

method_obj = MethodGraphBertNodeClassification(bert_config)
method_obj.spy_tag = True
method_obj.max_epoch = max_epoch
method_obj.lr = lr

# add load net
loaded_data = data_obj.load()
model = method_obj.from_pretrained(
    './result/PreTrained_GraphBert/' + dataset_name + '/node_classification_complete_model/', config=bert_config)
model.data = loaded_data

output = model.forward(loaded_data['raw_embeddings'], loaded_data['wl_embedding'], loaded_data['int_embeddings'],
                       loaded_data['hop_embeddings'], loaded_data['idx_test'])
clusters_net = output.max(1)[1]
tensor_features = loaded_data['X']
array_features = np.array(np.array(tensor_features, dtype=int))
'''
reducer = umap.UMAP()
embedding = reducer.fit_transform(array_features)
temp = np.array(embedding[500:1500], dtype=np.float32)
'''
pca = PCA(n_components=2)
# prepare transform on dataset
pca.fit(array_features)
# apply transform to dataset
transformed = pca.transform(array_features)
# tsne = TSNE(n_components=2, verbose=1, random_state=123)
# transformed = tsne.fit_transform(array_features)

x = []
y = []
for i in transformed[500:1500]:
    x.append(i[0])
    y.append(i[1])
# colorsa = np.array(["r" for _ in range(7)])
colorsa =dict()
# ["#0000FF", "#00FF00", "#FF0066"]
colorsa[0] = "b"
colorsa[1] = "c"
colorsa[2] = "g"
colorsa[3] = "k"
colorsa[4] = "m"
colorsa[5] = "r"
colorsa[6] = "y"

# 'b'- blue.
# 'c' - cyan.
# 'g' - green.
# 'k' - black.
# 'm' - magenta.
# 'r' - red.
# 'y' - yellow

# for i in b:
#    temp = i.reshape([1, 1433])
#    print(np.squeeze(temp, axis=0))

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

ans = os.path.exists(dataset_source_folder_path + str(len(clusters_net)))
if not ans:
    np.savetxt(dataset_source_folder_path + str(len(clusters_net)), np.array(clusters_net, dtype=int), fmt="%s")
else:
    clusters_net = load_and_change_cluster2(np.array(clusters_net, dtype=int))

fig = plt.figure()
ax = fig.add_subplot(111)
# for i in range(len(x)):
#     scatter = ax.scatter(x[i], y[i], color=colorsa[clusters_net[i]])
color = np.vectorize(lambda c:colorsa[c])
scatter = ax.scatter(x, y, color=color(clusters_net))
# plt.legend(["c1 - blue", "c2 - cyan", "c3 - green", "c4 - black", "c5 - magenta", "c6 - red", "c7 - yellow"])
# legend1 = ax.legend(*scatter.legend_elements())

c1 = mlines.Line2D([], [], color='b', marker=".", linestyle='None', markersize=10, label='c1')
c2 = mlines.Line2D([], [], color='c', marker=".", linestyle='None', markersize=10, label='c2')
c3 = mlines.Line2D([], [], color='g', marker=".", linestyle='None', markersize=10, label='c3')
c4 = mlines.Line2D([], [], color='k', marker=".", linestyle='None', markersize=10, label='c4')
c5 = mlines.Line2D([], [], color='m', marker=".", linestyle='None', markersize=10, label='c5')
c6 = mlines.Line2D([], [], color='r', marker=".", linestyle='None', markersize=10, label='c6')
c7 = mlines.Line2D([], [], color='y', marker=".", linestyle='None', markersize=10, label='c7')

plt.legend(handles=[c1, c2, c3, c5, c6, c7])

# plt.legend(c=colorsa,label=["c1 - blue", "c2 - cyan", "c3 - green", "c4 - black", "c5 - magenta", "c6 - red", "c7 - yellow"])
plt.show()
