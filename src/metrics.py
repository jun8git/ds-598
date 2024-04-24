import numpy as np
import pandas as pd

data = pd.read_csv(data_path + "/MustJ/exp/bert_token_hds_4_pred_val.csv")

def get_f1_score(pred_hds, hds):
    PP = np.sum(pred_hds.isin(["eob", "eol"]))
    AP = np.sum(hds.isin(["eob", "eol"]))
    TP = np.sum(pred_hds[pred_hds == hds].isin(["eob", "eol"]))
    p = TP / PP
    r = TP / AP
    return (2 * p * r)/(p + r)

# Calculate F1 score
f1_score = get_f1_score(data['pred_hds'], data['hds'])
print(f1_score)
