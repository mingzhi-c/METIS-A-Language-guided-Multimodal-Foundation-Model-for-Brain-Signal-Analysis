from Metis import Metis
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from pyhealth.metrics.binary import binary_metrics_fn
from pyhealth.metrics import multiclass_metrics_fn

from  dataset import ISRUC, Dreams, IEDS, Mayo, SanDiego, MDD, ADFSU, ADHD_Adult, ADHD_Children, Schizophrenia28


def get_metric(num_classes, y_true, y_prob):
    y_true = y_true.cpu().numpy()
    if num_classes == 2:
        all_metrics = ["roc_auc"]
        y_prob = F.softmax(y_prob, dim=-1)[:, 1].cpu().numpy()
        metrics = binary_metrics_fn(y_true, y_prob, metrics=all_metrics)
        return metrics['roc_auc']
    else:
        all_metrics = ["roc_auc_macro_ovo"]
        y_prob = F.softmax(y_prob, dim=-1).cpu().numpy()
        metrics = multiclass_metrics_fn(y_true, y_prob, metrics=all_metrics)
        return metrics['roc_auc_macro_ovo']

model = Metis(n_layers=8, dim=512)
device = 'cuda:0'
model_path = '/Metis_1_0.pt'
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model = model.to(device).eval()
test_dataset = ISRUC(folder_path='', n=1, type='test')
#test_dataset = Dreams(folder_path='', n=1, type='test')
#test_dataset = IEDS(folder_path='', n=1, type='test')
#test_dataset = Mayo(folder_path='', n=1, type='test')
#test_dataset = MDD(folder_path='', n=1, type='test')
#test_dataset = ADFSU(folder_path='', n=1, type='test')
#test_dataset = ADHD_Adult(folder_path='', n=1, type='test')
#test_dataset = ADHD_Children(folder_path='', n=1, type='test')
#test_dataset = Schizophrenia28(folder_path='', n=1, type='test')
#test_dataset = AdoSchizophrenia(folder_path='', n=1, type='test')
#test_dataset = RatEpilepsy(folder_path='', n=1, type='test')
#test_dataset = Siena(folder_path='', n=1, type='test')
#test_dataset = APAVA(folder_path='', n=1, type='test')
#test_dataset = NMTAB(type='test')
#test_dataset = SEE(n=1, type='test')
num_classes = len(test_dataset.options)
dataloader = DataLoader(test_dataset, batch_size=128, drop_last=False)
all_probs = []
all_labels = []

with torch.no_grad():
    for signal, input_ids, option_token_ids, Y in tqdm(dataloader):
        signal = signal.to(device)
        input_ids = input_ids.to(device)
        option_token_ids = option_token_ids.to(device)
        Y = Y.to(device)
        outputs = model(signal, input_ids)
        last_token_logits = outputs[:, -1, :]
        option_logits = last_token_logits.gather(1, option_token_ids)
        all_probs.append(option_logits)
        all_labels.append(Y)

all_probs = torch.cat(all_probs)
all_labels = torch.cat(all_labels)
roc_auc = get_metric(num_classes=num_classes, y_true=all_labels, y_prob=all_probs)
print('roc_auc:', roc_auc)