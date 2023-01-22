import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tk

from transformers.models.bert.modeling_bert import BertPreTrainedModel
from code.MethodGraphBert import MethodGraphBert

import time

BertLayerNorm = torch.nn.LayerNorm

def updateGUI(Progress_Bar_Precentage,GUI_Text,GUI,update_progress=None):
    global GUI_Total_Text,progressPercentage
    GUI_Total_Text = "\n\n" + GUI_Text
    GUI.delete(1.0, tk.END)
    GUI.insert(tk.END, GUI_Total_Text)
    progressPercentage += Progress_Bar_Precentage
    GUI.see(tk.END)

    if progressPercentage > 100: progressPercentage = 100

    if update_progress:
        update_progress(progressPercentage, 100)  # Update progress to 1%

class MethodGraphBertNodeConstruct(BertPreTrainedModel):
    learning_record_dict = {}
    lr = 0.001
    weight_decay = 5e-4
    max_epoch = 500
    load_pretrained_path = ''
    save_pretrained_path = ''

    def __init__(self, config,GUI=None, update_progress=None, *args, **kwargs):
        super(MethodGraphBertNodeConstruct, self).__init__(config,*args, **kwargs)
        self.GUI = GUI
        self.update_progress = update_progress
        self.config = config
        self.bert = MethodGraphBert(config)
        self.cls_y = torch.nn.Linear(config.hidden_size, config.x_size)
        self.init_weights()

    def forward(self, raw_features, wl_role_ids, init_pos_ids, hop_dis_ids, idx=None):

        outputs = self.bert(raw_features, wl_role_ids, init_pos_ids, hop_dis_ids)

        sequence_output = 0
        for i in range(self.config.k+1):
            sequence_output += outputs[0][:,i,:]
        sequence_output /= float(self.config.k+1)

        x_hat = self.cls_y(sequence_output)

        return x_hat


    def train_model(self, max_epoch,GUI = None,update_progress=None):
        t_begin = time.time()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for epoch in range(max_epoch):
            t_epoch_begin = time.time()

            # -------------------------

            self.train()
            optimizer.zero_grad()

            output = self.forward(self.data['raw_embeddings'], self.data['wl_embedding'], self.data['int_embeddings'], self.data['hop_embeddings'])

            loss_train = F.mse_loss(output, self.data['X'])

            loss_train.backward()
            optimizer.step()

            self.learning_record_dict[epoch] = {'loss_train': loss_train.item(), 'time': time.time() - t_epoch_begin}

            # -------------------------
            if epoch % 50 == 0:
                toPrint = 'Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.4f}'.format(loss_train.item()), 'time: {:.4f}s'.format(time.time() - t_epoch_begin)
                updateGUI(1,toPrint,GUI, update_progress)
                print(toPrint)

        updateGUI(1, "Optimization Finished!", GUI, update_progress)
        updateGUI(1, "Total time elapsed: {:.4f}s".format(time.time() - t_begin), GUI, update_progress)
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_begin))
        return time.time() - t_begin

    def run(self):

        self.train_model(self.max_epoch,GUI = None,update_progress = None)

        return self.learning_record_dict