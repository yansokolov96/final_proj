from code.DatasetLoader import DatasetLoader
from code.MethodBertComp import GraphBertConfig
from code.MethodGraphBertNodeClassification import MethodGraphBertNodeClassification
from code.ResultSaving import ResultSaving
from code.Settings import Settings
import numpy as np
import torch


#---- 'cora' , 'citeseer', 'pubmed' ----
from code.base_class.method import method

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
    max_epoch = 1000 # 500 ---- do an early stop when necessary ----
elif dataset_name == 'cora':
    lr = 0.01
    k = 7
    max_epoch = 150 # 150 ---- do an early stop when necessary ----
elif dataset_name == 'citeseer':
    k = 5
    lr = 0.001
    max_epoch = 2000 #2000 # it takes a long epochs to get good results, sometimes can be more than 2000

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

#add load net
loaded_data = data_obj.load()
model = method_obj.from_pretrained('./result/PreTrained_GraphBert/' + dataset_name + '/node_classification_complete_model/',config=bert_config)
model.data = loaded_data

output = model.forward(loaded_data['raw_embeddings'], loaded_data['wl_embedding'], loaded_data['int_embeddings'],
                       loaded_data['hop_embeddings'], loaded_data['idx_test'])
clusters_net = output.max(1)[1]
print("#####################################################")
a = loaded_data['X'][loaded_data['idx_test']].numpy()
b = np.array(a)
print(np.squeeze(b))
#print(loaded_data['y'][loaded_data['idx_train']])