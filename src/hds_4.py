# import os

# os.environ["HF_HOME"] = "/projectnb/ds598/students/jun/misc"
# os.environ["TORCH_HOME"] = "/projectnb/ds598/students/jun/misc"

import os
import csv
import json
import pathlib
import pandas as pd
from simpletransformers.ner import NERModel

import torch

cuda_available = torch.cuda.is_available()
print("cuda status", cuda_available)


data_path = os.path.expanduser("~/Sync/cdata")
run_path = os.getcwd()

lec1 = "dl4ds_graph_nn"
lec2 = "dl4ds_diffusion_models"


st_trn_doc_id = 1
end_trn_doc_id = 220068
tst_doc_id = 220069
end_tst_doc_id = 247576 ## end_tst_doc_id = end_trn_doc_id + 27508
st_val_doc_id = 247577
end_val_doc_id = 275085 # total doc_doc_id’s


ps_udpipe_hds_4 = "hds_4"
ps_bert_hds_4 = "bert_token_hds_4"

ps_udpipe_pobs_hds_4 = "pobs_hds_4"
lbls = ['eob', 'eol', 'S1', 'S2']

manual_seed = 42
n_gpu = 1

num_train_epochs = 20
max_seq_length = 512

# don’t save steps for every 2000 steps
save_steps = -1
overwrite_output_dir = True,
best_model_dir = "outputs/best_model"

weight = [1/0.113, 1/0.0442, 1/0.636, 1/0.207]

learning_rate = 1e-5
df = (pd.
      read_csv(data_path + "/MustJ/exp/" + ps + ".csv", na_filter=False).
      rename(columns = {'doc_id': 'sentence_id', 'obs': 'words', 'hds': 'labels'}))

model = NERModel("bert", "bert-large-uncased",
                 args={
                     "manual_seed": manual_seed,

                     "n_gpu": n_gpu,

                     "num_train_epochs": num_train_epochs,
                     "max_seq_length": max_seq_length,
                     "learning_rate": learning_rate,

                     "save_steps": save_steps,
                     "overwrite_output_dir": overwrite_output_dir,
                     "best_model_dir": best_model_dir

                        # True only when file path is given
                     # "lazy_loading": True
                       },
                 use_cuda=cuda_available,
                 labels=lbls,
                 weight = weight)

# # Train the model
model.train_model(df)
