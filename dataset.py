import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from tqdm import tqdm
import scipy.io as scio
import torch
import re
import pickle
from transformers import AutoTokenizer
from scipy.signal import resample
import scipy.signal as signal
from scipy.signal import filtfilt
import os
import math
from torch.utils.data.sampler import RandomSampler
os.environ["TOKENIZERS_PARALLELISM"] = "false"

##-------------------------------------------Pretrain-------------------------------------------------------------------
def prepare_qwen_inputs(question: str, answer: str, tokenizer, max_length=64):
    text = (
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n{answer}<|im_end|>"
    )
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length" if max_length else False,
        add_special_tokens=True
    )
    prefix = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    prefix_ids = tokenizer(
        prefix,
        return_tensors="pt",
        add_special_tokens=True
    ).input_ids
    answer_start_idx = prefix_ids.shape[1]
    labels = inputs.input_ids.clone()
    labels[:, :answer_start_idx] = -100
    if max_length:
        padding_mask = (inputs.input_ids == tokenizer.pad_token_id)
        labels[padding_mask] = -100
    return inputs.input_ids[0], labels[0]

##----------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------用于测试-------------------------------------------------------------
def prepare_qwen_test_inputs(question: str, options: list, tokenizer):
    text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    option_token_ids = [tokenizer.convert_tokens_to_ids(opt) for opt in options]
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        add_special_tokens=True
    )
    prefix_ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True
    ).input_ids
    return inputs.input_ids[0], torch.tensor(option_token_ids, dtype=torch.long)
##----------------------------------------------------------------------------------------------------------------------
class SHHSPretrain(Dataset):
    def __init__(self, root, sampling_rate=200, seed=1, max_length=48):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        self.root = root
        self.max_length = max_length
        self.files = os.listdir(root)
        np.random.shuffle(self.files)
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.scale = 37.9747
        self.tokenizer = AutoTokenizer.from_pretrained('')
        # 'Wake, N1, N2, N3, REM'
        self.question_prompt = 'Which sleep stage does this signal belong to?'
        self.label_0 = 'Wake'
        self.label_1 = 'N1'
        self.label_2 = 'N1'
        self.label_3 = 'N3'
        self.label_4 = 'REM'
        self.Option_question_prompt = 'Which sleep stage does this signal belong to? Options:(A)Wake (B)N1 (C)N2 (D)N3 (E)REM'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'
        self.Option_label_2 = 'C'
        self.Option_label_3 = 'D'
        self.Option_label_4 = 'E'

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        signal = sample["X"]
        Y = sample["y"]
        if np.random.rand() < 0.5:
            question = self.question_prompt
            if Y == 0:
                answer = self.label_0
            elif Y == 1:
                answer = self.label_1
            elif Y == 2:
                answer = self.label_2
            elif Y == 3:
                answer = self.label_3
            else:
                answer = self.label_4
        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            elif Y == 1:
                answer = self.Option_label_1
            elif Y == 2:
                answer = self.Option_label_2
            elif Y == 3:
                answer = self.Option_label_3
            else:
                answer = self.Option_label_4
        ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer, max_length=self.max_length)

        return signal, ids, label, Y

    def __len__(self):
        return len(self.files)


class SleepEDF(Dataset):
    def __init__(self, folder_path, max_length=48):
        file_names = os.listdir(folder_path)
        self.max_length = max_length
        self.origin_sample_rate = 100
        self.scale = 42.14
        self.b_bandpass, self.a_bandpass = signal.butter(4, [0.001, 0.75], btype='bandpass')
        self.b_notch, self.a_notch = signal.iirnotch(0.5, 0.016666666666666666)
        self.data_x = []
        self.data_label = []
        'Wake, N1, N2, N3, REM'
        for file_name in tqdm(file_names, desc='loading data:'):
            if file_name.endswith(".npz"):
                file_path = os.path.join(folder_path, file_name)
                npz_data = np.load(file_path)
                x = np.swapaxes(npz_data['x'], 2, 1)
                num = int(x.shape[-1] * 200 / self.origin_sample_rate)
                x = resample(x, axis=-1, num=num)
                x = signal.filtfilt(self.b_bandpass, self.a_bandpass, x, axis=-1)
                x = signal.filtfilt(self.b_notch, self.a_notch, x, axis=-1)
                y = torch.tensor(npz_data['y'])
                self.data_x.append(torch.FloatTensor(x.copy()))
                self.data_label.append(y)
        self.data_x = torch.cat(self.data_x, dim=0)
        self.data_x = self.data_x / self.scale
        self.data_label = torch.cat(self.data_label, dim=0)
        self.data_label = self.data_label.to(torch.long)

        self.tokenizer = AutoTokenizer.from_pretrained('')
        # 'Wake, N1, N2, N3, REM'
        self.question_prompt = 'Which sleep stage does this signal belong to?'
        self.label_0 = 'Wake'
        self.label_1 = 'N1'
        self.label_2 = 'N1'
        self.label_3 = 'N3'
        self.label_4 = 'REM'
        self.Option_question_prompt = 'Which sleep stage does this signal belong to? Options:(A)Wake (B)N1 (C)N2 (D)N3 (E)REM'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'
        self.Option_label_2 = 'C'
        self.Option_label_3 = 'D'
        self.Option_label_4 = 'E'

    def __getitem__(self, index):
        signal = self.data_x[index]
        Y = self.data_label[index].item()
        # simple
        if np.random.rand() < 0.5:
            question = self.question_prompt
            if Y == 0:
                answer = self.label_0
            elif Y == 1:
                answer = self.label_1
            elif Y == 2:
                answer = self.label_2
            elif Y == 3:
                answer = self.label_3
            else:
                answer = self.label_4
        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            elif Y == 1:
                answer = self.Option_label_1
            elif Y == 2:
                answer = self.Option_label_2
            elif Y == 3:
                answer = self.Option_label_3
            else:
                answer = self.Option_label_4
        ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                         max_length=self.max_length)

        return signal, ids, label, Y

    def __len__(self):
        return len(self.data_x)


class HMC(Dataset):
    def __init__(self, folder_path, max_length=48):
        file_names = os.listdir(folder_path)
        self.max_length = max_length
        file_names = os.listdir(folder_path)
        self.origin_sample_rate = 256
        self.scale = 0.0002
        self.data_x = []
        self.data_label = []
        'Wake, N1, N2, N3, REM'
        for file_name in tqdm(file_names, desc='loading data:'):
            if file_name.endswith(".npz"):
                file_path = os.path.join(folder_path, file_name)
                npz_data = np.load(file_path, allow_pickle=True)
                x = npz_data['x']
                y = npz_data['y']
                self.data_x.append(torch.FloatTensor(x))
                self.data_label.append(torch.tensor(y, dtype=torch.long))
        self.data_x = torch.cat(self.data_x, dim=0)
        self.data_x = self.data_x / self.scale
        self.data_label = torch.cat(self.data_label, dim=0)
        self.data_label = self.data_label.to(torch.long)

        self.tokenizer = AutoTokenizer.from_pretrained('')
        # 'Wake, N1, N2, N3, REM'
        self.question_prompt = 'Which sleep stage does this signal belong to?'
        self.label_0 = 'Wake'
        self.label_1 = 'N1'
        self.label_2 = 'N1'
        self.label_3 = 'N3'
        self.label_4 = 'REM'
        self.Option_question_prompt = 'Which sleep stage does this signal belong to? Options:(A)Wake (B)N1 (C)N2 (D)N3 (E)REM'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'
        self.Option_label_2 = 'C'
        self.Option_label_3 = 'D'
        self.Option_label_4 = 'E'

    def __getitem__(self, index):
        signal = self.data_x[index]
        Y = self.data_label[index].item()
        # simple
        if np.random.rand() < 0.5:
            question = self.question_prompt
            if Y == 0:
                answer = self.label_0
            elif Y == 1:
                answer = self.label_1
            elif Y == 2:
                answer = self.label_2
            elif Y == 3:
                answer = self.label_3
            else:
                answer = self.label_4
        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            elif Y == 1:
                answer = self.Option_label_1
            elif Y == 2:
                answer = self.Option_label_2
            elif Y == 3:
                answer = self.Option_label_3
            else:
                answer = self.Option_label_4
        ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                         max_length=self.max_length)

        return signal, ids, label, Y

    def __len__(self):
        return len(self.data_x)

#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
class SeizelT2Pretrain(Dataset):
    def __init__(self, root, sampling_rate=200, seed=1, max_length=48):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        self.root = root
        self.max_length = max_length
        self.files = []
        for sub_folder in os.listdir(root):
            sub_folder_path = os.path.join(root, sub_folder)
            if os.path.isdir(sub_folder_path):
                files = [os.path.join(sub_folder_path, f) for f in os.listdir(sub_folder_path) if
                         os.path.isfile(os.path.join(sub_folder_path, f))]
                self.files.extend(files)

        np.random.shuffle(self.files)
        self.sampling_rate = sampling_rate
        self.scale = 0.000141025

        self.tokenizer = AutoTokenizer.from_pretrained('')
        # 'interictal, ictal, preictal, postictal'
        self.question_prompt = 'Which epilepsy state does this signal belong to?'
        self.label_0 = 'interictal'
        self.label_1 = 'ictal'
        self.label_2 = 'preictal'
        self.label_3 = 'postictal'
        self.Option_question_prompt = 'Which epilepsy state does this signal belong to? Options:(A)interictal (B)ictal (C)preictal (D)postictal'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'
        self.Option_label_2 = 'C'
        self.Option_label_3 = 'D'

    def __getitem__(self, index):
        file_path = self.files[index]
        sample = pickle.load(open(file_path, "rb"))
        signal = sample["X"]
        Y = sample["y"]
        # simple
        if np.random.rand() < 0.5:
            question = self.question_prompt
            if Y == 0:
                answer = self.label_0
            elif Y == 1:
                answer = self.label_1
            elif Y == 2:
                answer = self.label_2
            else:
                answer = self.label_3

        else:
            question= self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            elif Y == 1:
                answer = self.Option_label_1
            elif Y == 2:
                answer = self.Option_label_2
            else:
                answer = self.Option_label_3

        ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                         max_length=self.max_length)

        return signal, ids, label, Y

    def __len__(self):
        return len(self.files)


class CHBMITPretrain(Dataset):
    def __init__(self, root, sampling_rate=200, seed=1, max_length=48):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        self.root = root
        self.max_length = max_length
        self.files = []
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(root, split)
            split_files = [os.path.join(split_dir, f) for f in os.listdir(split_dir)]
            self.files.extend(split_files)
        np.random.shuffle(self.files)
        self.sampling_rate = sampling_rate
        self.scale = 242.88

        self.tokenizer = AutoTokenizer.from_pretrained('')
        # 'interictal, ictal, preictal, postictal'
        self.question_prompt = 'Which epilepsy state does this signal belong to?'
        self.label_0 = 'interictal'
        self.label_1 = 'ictal'
        self.label_2 = 'preictal'
        self.label_3 = 'postictal'
        self.Option_question_prompt = 'Which epilepsy state does this signal belong to? Options:(A)interictal (B)ictal (C)preictal (D)postictal'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'
        self.Option_label_2 = 'C'
        self.Option_label_3 = 'D'

    def __getitem__(self, index):
        file_path = self.files[index]
        sample = pickle.load(open(file_path, "rb"))
        signal = sample["X"]
        Y = sample["y"]
        # simple
        if np.random.rand() < 0.5:
            question = self.question_prompt
            if Y == 0:
                answer = self.label_0
            elif Y == 1:
                answer = self.label_1
            elif Y == 2:
                answer = self.label_2
            else:
                answer = self.label_3

        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            elif Y == 1:
                answer = self.Option_label_1
            elif Y == 2:
                answer = self.Option_label_2
            else:
                answer = self.Option_label_3

        ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                         max_length=self.max_length)

        return signal, ids, label, Y

    def __len__(self):
        return len(self.files)

class FNUSA(Dataset):
    def __init__(self, folder_path, max_length=48):
        self.max_length = max_length
        file_names = os.listdir(folder_path)
        self.origin_sample_rate = 200
        self.scale = 1.603
        self.data_x = []
        self.data_label = []
        'intracranial, intracranial, pathological activity'
        for file_name in tqdm(file_names, desc='loading data:'):
            if file_name.endswith(".npz"):
                file_path = os.path.join(folder_path, file_name)
                npz_data = np.load(file_path, allow_pickle=True)
                x = npz_data['x']
                y = npz_data['y']
                self.data_x.append(torch.FloatTensor(x))
                self.data_label.append(torch.tensor(y, dtype=torch.long))
        self.data_x = torch.cat(self.data_x, dim=0)
        self.data_x = self.data_x / self.scale
        self.data_label = torch.cat(self.data_label, dim=0)
        self.data_label = self.data_label.to(torch.long)

        self.tokenizer = AutoTokenizer.from_pretrained('')
        # 'intracranial, intracranial, pathological activity'
        self.question_prompt = 'Which epilepsy state does this signal belong to?'
        self.label_0 = 'intracranial'
        self.label_1 = 'intracranial, pathological activity'
        self.Option_question_prompt = 'Which epilepsy state does this signal belong to? Options:(A)intracranial (B)intracranial, pathological activity'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'

    def __getitem__(self, index):
        signal = self.data_x[index]
        Y = self.data_label[index][0].item()
        # simple
        if np.random.rand() < 0.5:
            question = self.question_prompt
            if Y == 0:
                answer = self.label_0
            else:
                answer = self.label_1

        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            else:
                answer = self.Option_label_1

        ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                         max_length=self.max_length)

        return signal, ids, label, Y

    def __len__(self):
        return len(self.data_x)


class SWEC_ETHZ(Dataset):
    def __init__(self, folder_path, max_length=48):
        file_names = os.listdir(folder_path)
        self.max_length = max_length
        file_names = os.listdir(folder_path)
        self.origin_sample_rate = 512
        self.scale = 216.5097504
        self.data_x = []
        self.data_label = []

        for file_name in tqdm(file_names, desc='loading data:'):
            if file_name.endswith(".npz"):
                file_path = os.path.join(folder_path, file_name)
                npz_data = np.load(file_path, allow_pickle=True)
                x = npz_data['x']
                y = npz_data['y']
                self.data_x.append(torch.FloatTensor(x))
                self.data_label.append(torch.tensor(y, dtype=torch.long))
        self.data_x = torch.cat(self.data_x, dim=0)
        self.data_x = self.data_x / self.scale
        self.data_label = torch.cat(self.data_label, dim=0)
        self.data_label = self.data_label.to(torch.long)

        self.tokenizer = AutoTokenizer.from_pretrained('')
        # 'Wake, N1, N2, N3, REM'
        self.question_prompt = 'Which epilepsy state does this signal belong to?'
        self.label_0 = 'preictal'
        self.label_1 = 'ictal'
        self.label_2 = 'postictal'

        self.Option_question_prompt = 'Which epilepsy state does this signal belong to? Options:(A)preictal (B)ictal (C)postictal'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'
        self.Option_label_2 = 'C'

    def __getitem__(self, index):
        signal = self.data_x[index]
        Y = self.data_label[index].item()
        # simple
        if np.random.rand() < 0.5:
            question = self.question_prompt
            if Y == 0:
                answer = self.label_0
            elif Y == 1:
                answer = self.label_1
            else:
                answer = self.label_2


        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            elif Y == 1:
                answer = self.Option_label_1
            else:
                answer = self.Option_label_2

        ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                         max_length=self.max_length)

        return signal, ids, label, Y

    def __len__(self):
        return len(self.data_x)

class HUP(Dataset):
    def __init__(self, max_length=48):
        folder_path = ''
        file_names = os.listdir(folder_path)
        self.max_length = max_length
        self.origin_sample_rate = [1024,512]
        self.scale = 0.00022688
        self.data_x = []
        self.data_label = []

        for file_name in tqdm(file_names, desc='loading data:'):
            if file_name.endswith(".npz"):
                file_path = os.path.join(folder_path, file_name)
                npz_data = np.load(file_path, allow_pickle=True)
                x = npz_data['x']
                y = npz_data['y']
                self.data_x.append(torch.FloatTensor(x))
                self.data_label.append(torch.tensor(y, dtype=torch.long))
        self.data_x = torch.cat(self.data_x, dim=0)
        self.data_x = self.data_x / self.scale
        self.data_label = torch.cat(self.data_label, dim=0)
        self.data_label = self.data_label.to(torch.long)

        self.tokenizer = AutoTokenizer.from_pretrained('')
        # 'Wake, N1, N2, N3, REM'
        self.question_prompt = 'Which epilepsy state does this signal belong to?'
        self.label_0 = 'interictal'
        self.label_1 = 'ictal'
        self.label_2 = 'preictal'
        self.label_3 = 'postictal'
        self.Option_question_prompt = 'Which epilepsy state does this signal belong to? Options:(A)interictal (B)ictal (C)preictal (D)postictal'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'
        self.Option_label_2 = 'C'
        self.Option_label_3 = 'D'

    def __getitem__(self, index):
        signal = self.data_x[index]
        Y = self.data_label[index][0].item()
        # simple
        if np.random.rand() < 0.5:
            question = self.question_prompt
            if Y == 0:
                answer = self.label_0
            elif Y == 1:
                answer = self.label_1
            elif Y == 2:
                answer = self.label_2
            else:
                answer = self.label_3

        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            elif Y == 1:
                answer = self.Option_label_1
            elif Y == 2:
                answer = self.Option_label_2
            else:
                answer = self.Option_label_3
        ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                         max_length=self.max_length)

        return signal, ids, label, Y

    def __len__(self):
        return len(self.data_x)


class KaggleIEEGEpilepsy(Dataset):
    def __init__(self, max_length=48):
        folder_path = ''
        file_names = os.listdir(folder_path)
        self.max_length = max_length
        self.origin_sample_rate = 500
        self.scale = 199.61695000
        self.data_x = []
        self.data_label = []

        for file_name in tqdm(file_names, desc='loading data:'):
            if file_name.endswith(".npz"):
                file_path = os.path.join(folder_path, file_name)
                npz_data = np.load(file_path, allow_pickle=True)
                x = npz_data['x']
                y = npz_data['y']
                self.data_x.append(torch.FloatTensor(x))
                self.data_label.append(torch.tensor(y, dtype=torch.long))
        self.data_x = torch.cat(self.data_x, dim=0)
        self.data_x = self.data_x / self.scale
        self.data_label = torch.cat(self.data_label, dim=0)
        self.data_label = self.data_label.to(torch.long)

        self.tokenizer = AutoTokenizer.from_pretrained('')
        # 'Wake, N1, N2, N3, REM'
        self.question_prompt = 'Which epilepsy state does this signal belong to?'
        self.label_0 = 'interictal'
        self.label_1 = 'ictal'
        self.Option_question_prompt = 'Which epilepsy state does this signal belong to? Options:(A)interictal (B)ictal'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'

    def __getitem__(self, index):
        signal = self.data_x[index]
        Y = self.data_label[index].item()
        # simple
        if np.random.rand() < 0.5:
            question = self.question_prompt
            if Y == 0:
                answer = self.label_0
            else:
                answer = self.label_1
        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            else:
                answer = self.Option_label_1

        ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                         max_length=self.max_length)

        return signal, ids, label, Y

    def __len__(self):
        return len(self.data_x)

#-----------------------------------------------------------------------------------------------------------------------
class AlzheimerDataset(Dataset):
    def __init__(self, max_length=48):
        folder_path = ''
        self.max_length = max_length
        self.origin_sample_rate = 500
        self.b_bandpass, self.a_bandpass = signal.butter(4, [0.001, 0.75], btype='bandpass')
        self.b_notch, self.a_notch = signal.iirnotch(0.5, 0.016666666666666666)
        self.scale = 0.000049175
        self.data_x = []
        self.data_label = []
        for i in tqdm(range(1, 89), desc='loading data:'):
            sub_folder = f"sub-{i:03d}"
            sub_folder_path = os.path.join(folder_path, sub_folder)
            file_path_X = os.path.join(sub_folder_path, 'X.pt')
            file_path_Y = os.path.join(sub_folder_path, 'Y.pt')
            data_X = torch.load(file_path_X)
            data_Y = torch.load(file_path_Y)
            self.data_x.append(data_X)
            self.data_label.append(data_Y)

        self.data_x = torch.cat(self.data_x, dim=0)
        self.data_x = self.data_x / self.scale
        self.data_label = torch.cat(self.data_label, dim=0)
        self.data_label = self.data_label.to(torch.long)
        num = int(self.data_x.shape[-1] * 200 / self.origin_sample_rate)
        self.data_x = torch.FloatTensor(resample(self.data_x.numpy(), axis=-1, num=num))
        for i in tqdm(range(len(self.data_x)), desc='band filtering'):
            x = signal.filtfilt(self.b_bandpass, self.a_bandpass, self.data_x[i].numpy(), axis=-1)
            x = signal.filtfilt(self.b_notch, self.a_notch, x, axis=-1)
            self.data_x[i] = torch.FloatTensor(x.copy())

        self.tokenizer = AutoTokenizer.from_pretrained('')
        # 'Wake, N1, N2, N3, REM'
        self.question_prompt = 'Which disease does this signal belong to?'
        self.label_0 = 'normal'
        self.label_1 = 'alzheimer\'s disease'
        self.label_2 = 'frontotemporal dementia'
        self.Option_question_prompt = 'Which disease does this signal belong to? Options:(A)normal (B)alzheimer\'s disease (C)frontotemporal dementia'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'
        self.Option_label_2 = 'C'

    def __getitem__(self, index):
        signal = self.data_x[index]
        Y = self.data_label[index].item()
        # simple
        if np.random.rand() < 0.5:
            question = self.question_prompt
            if Y == 0:
                answer = self.label_0
            elif Y == 1:
                answer = self.label_1
            else:
                answer = self.label_2

        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            elif Y == 1:
                answer = self.Option_label_1
            else:
                answer = self.Option_label_2

        ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                         max_length=self.max_length)

        return signal, ids, label, Y

    def __len__(self):
        return len(self.data_x)


class RestEyesOpen(Dataset):
    def __init__(self, max_length=48):
        folder_path = ''
        file_names = os.listdir(folder_path)
        self.max_length = max_length
        self.origin_sample_rate = 500
        self.scale = 0.000088685
        self.data_x = []
        self.data_label = []
        for file_name in tqdm(file_names, desc='loading data:'):
            file_path = os.path.join(folder_path, file_name)  # 构建文件路径
            data = np.load(file_path, allow_pickle=True)
            data_x = data['x']
            data_label = torch.tensor(data['y'], dtype=torch.long)
            self.data_x.append(torch.FloatTensor(data_x))
            self.data_label.append(data_label)
        self.data_x = torch.cat(self.data_x, dim=0)
        self.data_x = self.data_x / self.scale
        self.data_label = torch.cat(self.data_label, dim=0)

        self.tokenizer = AutoTokenizer.from_pretrained('')
        # 'Wake, N1, N2, N3, REM'
        self.question_prompt = 'Which disease does this signal belong to?'
        self.label_0 = 'normal'
        self.label_1 = 'parkinson\'s disease'
        self.Option_question_prompt = 'Which disease does this signal belong to? Options:(A)normal (B)parkinson\'s disease'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'

    def __getitem__(self, index):
        signal = self.data_x[index]
        Y = self.data_label[index].item()
        # simple
        if np.random.rand() < 0.5:
            question = self.question_prompt
            if Y == 0:
                answer = self.label_0
            else:
                answer = self.label_1
        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            else:
                answer = self.Option_label_1

        ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                         max_length=self.max_length)

        return signal, ids, label, Y

    def __len__(self):
        return len(self.data_x)


class ADauditory(Dataset):
    def __init__(self, folder_path, max_length=48):
        file_names = os.listdir(folder_path)
        self.scale = 16.72125
        self.max_length = max_length
        self.data_x = []
        self.data_label = []
        for file_name in tqdm(file_names, desc='loading data:'):
            if file_name.endswith(".npz"):
                file_path = os.path.join(folder_path, file_name)  # 构建文件路径
                data = np.load(file_path, allow_pickle=True)
                x = data['x']
                y = data['y']
                self.data_x.append(torch.FloatTensor(x))
                self.data_label.append(torch.tensor(y, dtype=torch.long))
        self.data_x = torch.cat(self.data_x, dim=0)
        self.data_label = torch.cat(self.data_label, dim=0)
        self.data_x = self.data_x / self.scale

        self.tokenizer = AutoTokenizer.from_pretrained('')
        self.question_prompt = 'Which disease does this signal belong to?'
        self.label_0 = 'normal'
        self.label_1 = 'alzheimer\'s disease'
        self.label_2 = 'mild cognitive impairment'
        self.Option_question_prompt = 'Which disease does this signal belong to? Options:(A)normal (B)alzheimer\'s disease (C)mild cognitive impairment'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'
        self.Option_label_2 = 'C'

    def __getitem__(self, index):
        signal = self.data_x[index]
        Y = self.data_label[index].item()
        # simple
        if np.random.rand() < 0.5:
            question = self.question_prompt
            if Y == 0:
                answer = self.label_0
            elif Y == 1:
                answer = self.label_1
            else:
                answer = self.label_2

        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            elif Y == 1:
                answer = self.Option_label_1
            else:
                answer = self.Option_label_2

        ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                         max_length=self.max_length)

        return signal, ids, label, Y

    def __len__(self):
        return len(self.data_x)


class BrainLat(Dataset):
    def __init__(self, folder_path, max_length=64):
        self.max_length = max_length
        self.origin_sample_rate = 256
        self.scale = 0.00002537
        self.data_x = []
        self.data_label = []
        for sub_folder in os.listdir(folder_path):
            sub_folder_path = os.path.join(folder_path, sub_folder)
            if not os.path.isdir(sub_folder_path):
                continue
            file_names = os.listdir(sub_folder_path)
            for file_name in tqdm(file_names, desc=f'Loading data from {sub_folder}:'):
                if file_name.endswith(".npz"):
                    file_path = os.path.join(sub_folder_path, file_name)
                    data = np.load(file_path, allow_pickle=True)
                    x = data['x'][:, 0:128, :]
                    y = data['y']
                    self.data_x.append(torch.FloatTensor(x))
                    self.data_label.append(torch.tensor(y, dtype=torch.long))
        self.data_x = torch.cat(self.data_x, dim=0)
        self.data_label = torch.cat(self.data_label, dim=0)
        self.data_x = self.data_x / self.scale

        self.tokenizer = AutoTokenizer.from_pretrained('')
        # 'Wake, N1, N2, N3, REM'
        self.question_prompt = 'Which disease does this signal belong to?'
        self.label_0 = 'normal'
        self.label_1 = 'alzheimer\'s disease'
        self.label_2 = 'frontotemporal dementia'
        self.label_3 = 'parkinson\'s disease'
        self.label_4 = 'multiple sclerosis'
        self.Option_question_prompt = 'Which disease does this signal belong to? Options:(A)normal (B)alzheimer\'s disease (C)frontotemporal dementia (D)parkinson\'s disease (E)multiple sclerosis'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'
        self.Option_label_2 = 'C'
        self.Option_label_3 = 'D'
        self.Option_label_4 = 'E'

    def __getitem__(self, index):
        signal = self.data_x[index]
        Y = self.data_label[index].item()
        # simple
        if np.random.rand() < 0.5:
            question = self.question_prompt
            if Y == 0:
                answer = self.label_0
            elif Y == 1:
                answer = self.label_1
            elif Y == 2:
                answer = self.label_2
            elif Y == 3:
                answer = self.label_3
            else:
                answer = self.label_4
        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            elif Y == 1:
                answer = self.Option_label_1
            elif Y == 2:
                answer = self.Option_label_2
            elif Y == 3:
                answer = self.Option_label_3
            else:
                answer = self.Option_label_4
        ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                         max_length=self.max_length)

        return signal, ids, label, Y

    def __len__(self):
        return len(self.data_x)


class TDBrain(Dataset):
    def __init__(self, folder_path, label_dict_path, max_length=64):
        file_names = os.listdir(folder_path)
        self.origin_sample_rate = 500
        self.scale = 140.31875
        self.label_dict = np.load(label_dict_path, allow_pickle=True).item()
        self.data_x = []
        self.data_label = []
        b_notch, a_notch = signal.iirnotch(0.6, 30)
        self.max_length = max_length
        for file_name in tqdm(file_names, desc='loading data:'):
            if file_name.endswith(".npz"):
                file_path = os.path.join(folder_path, file_name)
                npz_data = np.load(file_path)
                x = npz_data['x']
                x = signal.filtfilt(b_notch, a_notch, x)
                x = torch.FloatTensor(x.copy())
                y = torch.tensor(npz_data['y'])
                self.data_x.append(x)
                self.data_label.append(y)
        self.data_x = torch.cat(self.data_x, dim=0)
        self.data_label = torch.cat(self.data_label, dim=0)
        self.data_label = self.data_label.to(torch.long)
        self.data_x = self.data_x / self.scale
        self.tokenizer = AutoTokenizer.from_pretrained('')
        self.question_prompt = 'Which disease does this signal belong to?'

    def __getitem__(self, index):
        signal = self.data_x[index]
        Y = self.data_label[index].item()
        question = self.question_prompt
        answer = self.label_dict[Y]
        ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                         max_length=self.max_length)

        return signal, ids, label, Y

    def __len__(self):
        return len(self.data_x)

#-----------------------------------------------------------------------------------------------------------------------
class TUABPretrain(Dataset):
    def __init__(self, root, sampling_rate=200, seed=1, max_length=48):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        self.b_pass, self.a_pass = signal.butter(4, 0.005, btype='high')
        self.b_notch, self.a_notch = signal.iirnotch(0.6, 30)
        self.max_length = max_length
        self.files = []
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(root, split)
            split_files = [os.path.join(split_dir, f) for f in os.listdir(split_dir)]
            self.files.extend(split_files)
        np.random.shuffle(self.files)
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.scale = 0.000128425

        self.tokenizer = AutoTokenizer.from_pretrained('')
        self.question_prompt = 'Is the signal normal or abnormal?'
        self.label_0 = 'normal'
        self.label_1 = 'abnormal'
        self.Option_question_prompt = 'Is the signal normal or abnormal? Options:(A)normal (B)abnormal'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'

    def __getitem__(self, index):
        file_path = self.files[index]
        sample = pickle.load(open(file_path, "rb"))
        signal = sample["X"]
        Y = sample["y"]
        # 重采样（如果需要）
        if self.sampling_rate != self.default_rate:
            signal = resample(signal, 10 * self.sampling_rate, axis=-1)
        signal = filtfilt(self.b_pass, self.a_pass, signal, axis=-1)
        signal = filtfilt(self.b_notch, self.a_notch, signal, axis=-1)
        signal = torch.FloatTensor(signal.copy())
        # simple
        if np.random.rand() < 0.5:
            question = self.question_prompt
            if Y == 0:
                answer = self.label_0
            else:
                answer = self.label_1

        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            else:
                answer = self.Option_label_1

        ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                         max_length=self.max_length)

        return signal, ids, label, Y

    def __len__(self):
        return len(self.files)

class TUEVPretrain(Dataset):
    def __init__(self, root, sampling_rate=200, max_length=64):
        self.files = []
        self.max_length = max_length
        train_files = os.listdir(os.path.join(root, "processed_train"))
        train_files = [os.path.join(root, "processed_train", f) for f in train_files]
        self.files.extend(train_files)
        test_files = os.listdir(os.path.join(root, "processed_eval"))
        test_files = [os.path.join(root, "processed_eval", f) for f in test_files]
        self.files.extend(test_files)

        np.random.shuffle(self.files)
        self.default_rate = 256
        self.sampling_rate = sampling_rate
        self.scale = 0.000111025

        self.tokenizer = AutoTokenizer.from_pretrained('')
        self.question_prompt = 'Which type does this signal belong to?'
        self.label_0 = 'spike and slow wave'
        self.label_1 = 'generalized periodic epileptiform discharge'
        self.label_2 = 'periodic lateralized epileptiform dischage'
        self.label_3 = 'eye movement'
        self.label_4 = 'artifact'
        self.label_5 = 'background'
        self.Option_question_prompt = 'Which type does this signal belong to? Options:(A)spike and slow wave (B)generalized periodic epileptiform discharge (C)periodic lateralized epileptiform dischage (D)eye movement (E)artifact (F)background'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'
        self.Option_label_2 = 'C'
        self.Option_label_3 = 'D'
        self.Option_label_4 = 'E'
        self.Option_label_5 = 'F'

    def __getitem__(self, index):
        file_path = self.files[index]
        sample = pickle.load(open(file_path, "rb"))
        signal = sample["signal"]
        Y = int(sample["label"][0] - 1)
        if self.sampling_rate != self.default_rate:
            signal = resample(signal, 5 * self.sampling_rate, axis=-1)

        signal = signal / self.scale
        signal = torch.FloatTensor(signal)
        # simple
        if np.random.rand() < 0.5:
            question = self.question_prompt
            if Y == 0:
                answer = self.label_0
            elif Y == 1:
                answer = self.label_1
            elif Y == 2:
                answer = self.label_2
            elif Y == 3:
                answer = self.label_3
            elif Y == 4:
                answer = self.label_4
            else:
                answer = self.label_5
        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            elif Y == 1:
                answer = self.Option_label_1
            elif Y == 2:
                answer = self.Option_label_2
            elif Y == 3:
                answer = self.Option_label_3
            elif Y == 4:
                answer = self.Option_label_4
            else:
                answer = self.Option_label_5

        ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                         max_length=self.max_length)

        return signal, ids, label, Y

    def __len__(self):
        return len(self.files)

class TUEPPretrain(Dataset):
    def __init__(self, root, sampling_rate=200, max_length=48):
        self.max_length = max_length
        self.files = []
        self.files.extend([os.path.join(root, f) for f in os.listdir(root)])
        np.random.shuffle(self.files)
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.scale = 0.00044910

        self.tokenizer = AutoTokenizer.from_pretrained('')
        self.question_prompt = 'Does this signal belong to epilepsy?'
        self.label_0 = 'No'
        self.label_1 = 'Yes'
        self.Option_question_prompt = 'Does this signal belong to epilepsy? Options:(A)No (B)Yes'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'

    def __getitem__(self, index):
        file_path = self.files[index]
        sample = pickle.load(open(file_path, "rb"))
        signal = sample["X"]
        Y = sample["y"]
        signal = torch.FloatTensor(signal)
        # simple
        if np.random.rand() < 0.5:
            question = self.question_prompt
            if Y == 0:
                answer = self.label_0
            else:
                answer = self.label_1

        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            else:
                answer = self.Option_label_1

        ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                         max_length=self.max_length)

        return signal, ids, label, Y

    def __len__(self):
        return len(self.files)


class NMTEVPretrain(Dataset):
    def __init__(self, root, sampling_rate=200, max_length=48):
        self.files = []
        self.files.extend([os.path.join(root, f) for f in os.listdir(root)])
        np.random.shuffle(self.files)
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.scale = 0.00000172
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained('')
        self.question_prompt = 'Is the signal normal or abnormal?'
        self.label_0 = 'normal'
        self.label_1 = 'abnormal'
        self.Option_question_prompt = 'Is the signal normal or abnormal? Options:(A)normal (B)abnormal'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'

    def __getitem__(self, index):
        file_path = self.files[index]
        sample = pickle.load(open(file_path, "rb"))
        signal = sample["X"]
        Y = sample["y"]
        signal = torch.FloatTensor(signal)
        if np.random.rand() < 0.5:
            question = self.question_prompt
            if Y == 0:
                answer = self.label_0
            else:
                answer = self.label_1

        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            else:
                answer = self.Option_label_1

        ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                         max_length=self.max_length)

        return signal, ids, label, Y

    def __len__(self):
        return len(self.files)

#-----------------------------------------------------------------------------------------------------------------------
class TUSZPretrain(Dataset):
    def __init__(self, folder_path, label_dict_path, max_length=20):
        super(TUSZPretrain, self).__init__()
        self.max_length = max_length
        file_names = os.listdir(folder_path)
        self.origin_sample_rate = [250, 256, 200]
        self.scale = 77.09871979
        self.label_dict = np.load(label_dict_path, allow_pickle=True).item()
        self.data_x = []
        self.data_label = []
        for file_name in tqdm(file_names, desc='loading data:'):
            if file_name.endswith(".npz"):
                file_path = os.path.join(folder_path, file_name)
                npz_data = np.load(file_path)
                x = npz_data['x']
                x = torch.FloatTensor(x)
                y = torch.tensor(npz_data['y'])
                self.data_x.append(x)
                self.data_label.append(y)
        self.data_x = torch.cat(self.data_x, dim=0)
        self.data_label = torch.cat(self.data_label, dim=0)
        self.data_label = self.data_label.to(torch.long)
        self.data_x = self.data_x / self.scale
        self.tokenizer = AutoTokenizer.from_pretrained('')
        self.question_prompt = 'Which type does this signal belong to?'

    def __getitem__(self, index):
        question = self.question_prompt
        Y = self.data_label[index]
        answer = self.label_dict[Y.item()]
        signal = self.data_x[index]
        ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                         max_length=self.max_length)

        return signal, ids, label, Y

    def __len__(self):
        return len(self.data_x)

class ShuDatasetPretrain(Dataset):
    def __init__(self, folder_path, max_length=32):
        file_names = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
        self.max_length = max_length
        self.data_x = []
        self.data_label = []
        self.scale = 152.633786
        self.origin_sample_rate = 250
        self.b_bandpass, self.a_bandpass = signal.butter(4, [0.001, 0.75], btype='bandpass')
        self.b_notch, self.a_notch = signal.iirnotch(0.5, 0.016666666666666666)
        for file_name in tqdm(file_names, desc='loading data:'):
            file_path = os.path.join(folder_path, file_name)
            data = scio.loadmat(file_path)
            x = data['data']
            num = int(x.shape[-1] * 200 / self.origin_sample_rate)
            x = resample(x, axis=-1, num=num)
            x = signal.filtfilt(self.b_bandpass, self.a_bandpass, x, axis=-1)
            x = signal.filtfilt(self.b_notch, self.a_notch, x, axis=-1)
            self.data_x.append(torch.tensor(x.copy(), dtype=torch.float32))
            self.data_label.append(torch.tensor(data['labels'], dtype=torch.long).view(-1))
        self.data_x = torch.cat(self.data_x, dim=0)
        self.data_x = self.data_x / self.scale
        self.data_label = torch.cat(self.data_label, dim=0)
        self.data_label = self.data_label - 1

        self.tokenizer = AutoTokenizer.from_pretrained('')
        self.question_prompt = 'Which movement is imagined in this signal?'
        self.label_0 = 'left hand'
        self.label_1 = 'right hand'
        self.Option_question_prompt = 'Which movement is imagined in this signal? Options:(A)left hand (B)right hand'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'

    def __getitem__(self, index):
        signal = self.data_x[index]
        Y = self.data_label[index].item()
        # simple
        if np.random.rand() < 0.5:
            question = self.question_prompt
            if Y == 0:
                answer = self.label_0
            else:
                answer = self.label_1
        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            else:
                answer = self.Option_label_1

        ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                         max_length=self.max_length)

        return signal, ids, label, Y

    def __len__(self):
        return len(self.data_x)

class PhysionetMI(Dataset):
    def __init__(self, max_length=48):
        super(PhysionetMI, self).__init__()
        self.max_length = max_length
        self.origin_sample_rate = 160
        self.scale = 0.00022956
        folder_path = ''
        file_names = os.listdir(folder_path)
        self.data_x = []
        self.data_label = []
        for file_name in tqdm(file_names, desc='loading data:'):
            file_path = os.path.join(folder_path, file_name)  # 构建文件路径
            data = np.load(file_path, allow_pickle=True)
            data_x = data['x']
            data_label = torch.tensor(data['y'], dtype=torch.long)
            self.data_x.append(torch.FloatTensor(data_x))
            self.data_label.append(data_label)
        self.data_x = torch.cat(self.data_x, dim=0)
        self.data_x = self.data_x / self.scale
        self.data_label = torch.cat(self.data_label, dim=0)

        self.tokenizer = AutoTokenizer.from_pretrained('')
        self.question_prompt = 'Which movement is imagined in this signal?'
        self.label_0 = 'left fist'
        self.label_1 = 'right fist'
        self.label_2 = 'both fists'
        self.label_3 = 'both feet'
        self.Option_question_prompt = 'Which movement is imagined in this signal? Options:(A)left fist (B)right fist (C)both fists (D)both feet'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'
        self.Option_label_2 = 'C'
        self.Option_label_3 = 'D'

    def __getitem__(self, index):
        signal = self.data_x[index]
        Y = self.data_label[index].item()
        # simple
        if np.random.rand() < 0.5:
            question = self.question_prompt
            if Y == 0:
                answer = self.label_0
            elif Y == 1:
                answer = self.label_1
            elif Y == 2:
                answer = self.label_2
            else:
                answer = self.label_3
        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            elif Y == 1:
                answer = self.Option_label_1
            elif Y == 2:
                answer = self.Option_label_2
            else:
                answer = self.Option_label_3
        ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                         max_length=self.max_length)

        return signal, ids, label, Y

    def __len__(self):
        return len(self.data_x)

#-----------------------------------------------------------------------------------------------------------------------
class SchedulerSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, batch_size, largest_dataset_size, sequential_datasets_idx=None):
        super(SchedulerSampler, self).__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = largest_dataset_size
        # sequential_datasets_idx is a list of indices for datasets that should be read sequentially
        self.sequential_datasets_idx = sequential_datasets_idx if sequential_datasets_idx is not None else []

    def __len__(self):
        return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []

        # Initialize samplers for each dataset
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]

            # If this dataset needs sequential access, use a SequentialSampler
            if dataset_idx in self.sequential_datasets_idx:
                sampler = torch.utils.data.SequentialSampler(cur_dataset)
            else:
                sampler = RandomSampler(cur_dataset)

            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        # Offset values to ensure correct indexing across datasets
        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        samples_to_grab = self.batch_size
        epoch_samples = self.largest_dataset_size * self.number_of_datasets

        final_samples_list = []  # This will hold the final list of indices from the combined dataset

        # Generate samples for the epoch
        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []

                # For each dataset, grab a batch of samples
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # If we reach the end of the iterator, restart it for that dataset
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)

                final_samples_list.extend(cur_samples)

        # Return an iterator over the final list of samples
        return iter(final_samples_list)

#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
class Dreams(Dataset):
    def __init__(self, folder_path, n, K=5, max_length=64, type='train'):
        super(Dreams, self).__init__()
        self.max_length = max_length
        file_names = os.listdir(folder_path)
        npz_files = [f for f in file_names if f.endswith(".npz")]
        npz_files.sort(key=lambda x: int(x.split('.')[0]))  # 按文件名数字排序

        total = len(npz_files)
        assert K <= total, f"Cannot split {total} files into {K} folds"

        div, mod = divmod(total, K)
        folds = []
        start = 0
        for i in range(K):
            fold_size = div + (1 if i < mod else 0)
            folds.append(npz_files[start:start + fold_size])
            start += fold_size

        if n < 1 or n > K:
            raise ValueError(f"n should be between 1 and {K}")

        test_files = folds[n - 1]
        train_files = []
        for idx in range(K):
            if idx != n - 1:
                train_files.extend(folds[idx])

        if type == 'train':
            selected_files = train_files
        elif type == 'test':
            selected_files = test_files
        elif type == 'all':
            selected_files = train_files + test_files
        else:
            raise ValueError("type should be 'train' or 'test'")

        self.origin_sample_rate = 256
        self.scale = 0.00007163
        self.data_x = []
        self.data_label = []
        print(selected_files)
        for file_name in tqdm(selected_files, desc='loading data:'):
            file_path = os.path.join(folder_path, file_name)
            npz_data = np.load(file_path, allow_pickle=True)
            x = npz_data['x']
            y = npz_data['y']
            x = torch.FloatTensor(x)
            y = torch.tensor(y, dtype=torch.long)
            self.data_x.append(x)
            self.data_label.append(y)

        self.data_x = torch.cat(self.data_x, dim=0)
        self.data_x = self.data_x / self.scale
        self.data_label = torch.cat(self.data_label, dim=0)
        self.data_label = self.data_label.to(torch.long)

        self.tokenizer = AutoTokenizer.from_pretrained('')
        self.Option_question_prompt = 'Which sleep stage does this signal belong to? Options:(A)Wake (B)N1 (C)N2 (D)N3 (E)REM'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'
        self.Option_label_2 = 'C'
        self.Option_label_3 = 'D'
        self.Option_label_4 = 'E'
        self.options = [self.Option_label_0, self.Option_label_1, self.Option_label_2, self.Option_label_3,
                        self.Option_label_4]
        self.type = type

    def __getitem__(self, index):
        signal = self.data_x[index]
        Y = self.data_label[index]
        if self.type == 'test':
            input_ids, option_token_ids = prepare_qwen_test_inputs(question=self.Option_question_prompt,
                                                                   options=self.options, tokenizer=self.tokenizer)

            return signal, input_ids, option_token_ids, Y
        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            elif Y == 1:
                answer = self.Option_label_1
            elif Y == 2:
                answer = self.Option_label_2
            elif Y == 3:
                answer = self.Option_label_3
            else:
                answer = self.Option_label_4

            ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                         max_length=self.max_length)

            return signal, ids, label, Y

    def __len__(self):
        return len(self.data_x)

class ISRUC(Dataset):
    def __init__(self, folder_path, n, K=5, max_length=64, type='train'):
        super(ISRUC, self).__init__()
        self.max_length = max_length
        file_names = os.listdir(folder_path)
        npz_files = [f for f in file_names if f.endswith(".npz")]
        npz_files.sort(key=lambda x: int(x.split('.')[0]))

        total = len(npz_files)
        assert K <= total, f"Cannot split {total} files into {K} folds"

        div, mod = divmod(total, K)
        folds = []
        start = 0
        for i in range(K):
            fold_size = div + (1 if i < mod else 0)
            folds.append(npz_files[start:start + fold_size])
            start += fold_size
        if n < 1 or n > K:
            raise ValueError(f"n should be between 1 and {K}")
        test_files = folds[n - 1]
        train_files = []
        for idx in range(K):
            if idx != n - 1:
                train_files.extend(folds[idx])
        if type == 'train':
            selected_files = train_files
        elif type == 'test':
            selected_files = test_files
        elif type == 'all':
            selected_files = train_files + test_files
        else:
            raise ValueError("type should be 'train' or 'test'")

        self.origin_sample_rate = 256
        self.scale = 0.00007163
        self.data_x = []
        self.data_label = []
        print(selected_files)
        for file_name in tqdm(selected_files, desc='loading data:'):
            file_path = os.path.join(folder_path, file_name)
            npz_data = np.load(file_path, allow_pickle=True)
            x = npz_data['x']
            y = npz_data['y']
            x = torch.FloatTensor(x)
            y = torch.tensor(y, dtype=torch.long)
            self.data_x.append(x)
            self.data_label.append(y)

        self.data_x = torch.cat(self.data_x, dim=0)
        self.data_x = self.data_x / self.scale
        self.data_label = torch.cat(self.data_label, dim=0)
        self.data_label = self.data_label.to(torch.long)

        self.tokenizer = AutoTokenizer.from_pretrained('')
        self.Option_question_prompt = 'Which sleep stage does this signal belong to? Options:(A)Wake (B)N1 (C)N2 (D)N3 (E)REM'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'
        self.Option_label_2 = 'C'
        self.Option_label_3 = 'D'
        self.Option_label_4 = 'E'
        self.options = [self.Option_label_0, self.Option_label_1, self.Option_label_2, self.Option_label_3, self.Option_label_4]
        self.type = type

    def __getitem__(self, index):
        signal = self.data_x[index]
        Y = self.data_label[index]
        if self.type == 'test':
            input_ids, option_token_ids = prepare_qwen_test_inputs(question=self.Option_question_prompt,
                                                                   options=self.options, tokenizer=self.tokenizer)

            return signal, input_ids, option_token_ids, Y
        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            elif Y == 1:
                answer = self.Option_label_1
            elif Y == 2:
                answer = self.Option_label_2
            elif Y == 3:
                answer = self.Option_label_3
            else:
                answer = self.Option_label_4

            ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                             max_length=self.max_length)

            return signal, ids, label, Y

    def __len__(self):
        return len(self.data_x)

class IEDS(Dataset):
    def __init__(self, folder_path, n, K=5, max_length=64, type='train'):
        super(IEDS, self).__init__()
        self.max_length = max_length
        file_names = os.listdir(folder_path)
        npz_files = [f for f in file_names if f.endswith(".npz")]
        npz_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
        total = len(npz_files)
        assert K <= total, f"Cannot split {total} files into {K} folds"
        div, mod = divmod(total, K)
        folds = []
        start = 0
        for i in range(K):
            fold_size = div + (1 if i < mod else 0)
            folds.append(npz_files[start:start + fold_size])
            start += fold_size
        if n < 1 or n > K:
            raise ValueError(f"n should be between 1 and {K}")
        test_files = folds[n - 1]
        train_files = []
        for idx in range(K):
            if idx != n - 1:
                train_files.extend(folds[idx])
        if type == 'train':
            selected_files = train_files
        elif type == 'test':
            selected_files = test_files
        elif type == 'all':
            selected_files = train_files + test_files
        else:
            raise ValueError("type should be 'train' or 'test'")
        self.origin_sample_rate = 256
        self.scale = 128.763525
        self.data_x = []
        self.data_label = []
        print(selected_files)
        for file_name in tqdm(selected_files, desc='loading data:'):
            file_path = os.path.join(folder_path, file_name)
            npz_data = np.load(file_path, allow_pickle=True)
            x = npz_data['x']
            y = npz_data['y']
            x = torch.FloatTensor(x)
            y = torch.tensor(y, dtype=torch.long)
            self.data_x.append(x)
            self.data_label.append(y)
        self.data_x = torch.cat(self.data_x, dim=0)
        self.data_x = self.data_x / self.scale
        self.data_label = torch.cat(self.data_label, dim=0)
        self.data_label = self.data_label.to(torch.long)
        self.tokenizer = AutoTokenizer.from_pretrained('')
        self.Option_question_prompt = 'Which epilepsy state does this signal belong to? Options:(A)intracranial (B)intracranial, pathological activity'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'
        self.options = [self.Option_label_0, self.Option_label_1]
        self.type = type

    def __getitem__(self, index):
        signal = self.data_x[index]
        Y = self.data_label[index][0]
        if self.type == 'test':
            input_ids, option_token_ids = prepare_qwen_test_inputs(question=self.Option_question_prompt, options=self.options, tokenizer=self.tokenizer)

            return signal, input_ids, option_token_ids, Y
        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            else:
                answer = self.Option_label_1

            ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                             max_length=self.max_length)
            return signal, ids, label, Y

    def __len__(self):
        return len(self.data_x)


class Mayo(Dataset):
    def __init__(self, folder_path, n, K=5, max_length=64, type='train'):
        super(Mayo, self).__init__()
        self.max_length = max_length
        file_names = os.listdir(folder_path)
        npz_files = [f for f in file_names if f.endswith(".npz")]
        npz_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
        total = len(npz_files)
        assert K <= total, f"Cannot split {total} files into {K} folds"
        div, mod = divmod(total, K)
        folds = []
        start = 0
        for i in range(K):
            fold_size = div + (1 if i < mod else 0)
            folds.append(npz_files[start:start + fold_size])
            start += fold_size
        if n < 1 or n > K:
            raise ValueError(f"n should be between 1 and {K}")
        test_files = folds[n - 1]
        train_files = []
        for idx in range(K):
            if idx != n - 1:
                train_files.extend(folds[idx])
        if type == 'train':
            selected_files = train_files
        elif type == 'test':
            selected_files = test_files
        elif type == 'all':
            selected_files = train_files + test_files
        else:
            raise ValueError("type should be 'train' or 'test'")
        self.origin_sample_rate = 256
        self.scale = 128.763525
        self.data_x = []
        self.data_label = []
        print(selected_files)
        for file_name in tqdm(selected_files, desc='loading data:'):
            file_path = os.path.join(folder_path, file_name)
            npz_data = np.load(file_path, allow_pickle=True)
            x = npz_data['x']
            y = npz_data['y']
            x = torch.FloatTensor(x)
            y = torch.tensor(y, dtype=torch.long)
            self.data_x.append(x)
            self.data_label.append(y)

        self.data_x = torch.cat(self.data_x, dim=0)
        self.data_x = self.data_x / self.scale
        self.data_label = torch.cat(self.data_label, dim=0)
        self.data_label = self.data_label.to(torch.long)

        self.tokenizer = AutoTokenizer.from_pretrained('')
        self.Option_question_prompt = 'Which epilepsy state does this signal belong to? Options:(A)intracranial (B)intracranial, pathological activity'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'
        self.options = [self.Option_label_0, self.Option_label_1]
        self.type = type

    def __getitem__(self, index):
        signal = self.data_x[index]
        Y = self.data_label[index][0]
        if self.type == 'test':
            input_ids, option_token_ids = prepare_qwen_test_inputs(question=self.Option_question_prompt,
                                                                   options=self.options, tokenizer=self.tokenizer)

            return signal, input_ids, option_token_ids, Y
        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            else:
                answer = self.Option_label_1

            ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                             max_length=self.max_length)
            return signal, ids, label, Y

    def __len__(self):
        return len(self.data_x)

class SanDiego(Dataset):
    def __init__(self, folder_path, n, K=5, max_length=64, type='train',
                 healthy_prefix='sub-hc',
                 patient_prefix='sub-pd'):
        super(SanDiego, self).__init__()
        self.max_length = max_length
        file_names = os.listdir(folder_path)
        npz_files = [f for f in file_names if f.endswith(".npz")]

        healthy_files = [f for f in npz_files if f.startswith(healthy_prefix)]
        patient_files = [f for f in npz_files if f.startswith(patient_prefix)]
        healthy_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
        patient_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))

        def kfold_split(files, K, n):
            total = len(files)
            assert K <= total, f"Cannot split {total} files into {K} folds"

            div, mod = divmod(total, K)
            folds = []
            start = 0
            for i in range(K):
                fold_size = div + (1 if i < mod else 0)
                folds.append(files[start:start + fold_size])
                start += fold_size

            test_files = folds[n - 1]
            train_files = []
            for idx in range(K):
                if idx != n - 1:
                    train_files.extend(folds[idx])

            return train_files, test_files

        healthy_train, healthy_test = kfold_split(healthy_files, K, n)
        patient_train, patient_test = kfold_split(patient_files, K, n)

        if type == 'train':
            selected_files = healthy_train + patient_train
        elif type == 'test':
            selected_files = healthy_test + patient_test
        elif type == 'all':
            selected_files = healthy_train + patient_train + healthy_test + patient_test
        else:
            raise ValueError("type should be 'train' or 'test'")

        self.scale = 7.78637E-05
        self.data_x = []
        self.data_label = []
        print(selected_files)
        for file_name in tqdm(selected_files, desc='loading data:'):
            file_path = os.path.join(folder_path, file_name)
            npz_data = np.load(file_path, allow_pickle=True)
            x = npz_data['x']
            y = npz_data['y']
            x = torch.FloatTensor(x)
            y = torch.tensor(y, dtype=torch.long)
            self.data_x.append(x)
            self.data_label.append(y)

        self.data_x = torch.cat(self.data_x, dim=0)
        self.data_x = self.data_x / self.scale
        self.data_label = torch.cat(self.data_label, dim=0)
        self.data_label = self.data_label.to(torch.long)

        self.tokenizer = AutoTokenizer.from_pretrained('')
        self.Option_question_prompt = 'Which disease does this signal belong to? Options:(A)normal (B)parkinson\'s disease'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'
        self.options = [self.Option_label_0, self.Option_label_1]
        self.type = type

    def __getitem__(self, index):
        signal = self.data_x[index]
        Y = self.data_label[index][0]
        if self.type == 'test':
            input_ids, option_token_ids = prepare_qwen_test_inputs(question=self.Option_question_prompt,
                                                                   options=self.options, tokenizer=self.tokenizer)

            return signal, input_ids, option_token_ids, Y
        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            else:
                answer = self.Option_label_1

            ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                             max_length=self.max_length)
            return signal, ids, label, Y

    def __len__(self):
        return len(self.data_x)


class MDD(Dataset):
    def __init__(self, folder_path, n, K=5, max_length=64, type='train',
                 healthy_prefix='H_S',
                 patient_prefix='MDD_S'):
        super(MDD, self).__init__()
        self.max_length = max_length
        self.origin_sample_rate = 256
        self.b_bandpass, self.a_bandpass = signal.butter(4, [0.001, 0.75], btype='bandpass')
        self.b_notch, self.a_notch = signal.iirnotch(0.6, 0.016666666666666666)
        file_names = os.listdir(folder_path)
        npz_files = [f for f in file_names if f.endswith(".npz")]

        healthy_files = [f for f in npz_files if f.startswith(healthy_prefix)]
        patient_files = [f for f in npz_files if f.startswith(patient_prefix)]
        healthy_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
        patient_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))

        def kfold_split(files, K, n):
            total = len(files)
            assert K <= total, f"Cannot split {total} files into {K} folds"

            div, mod = divmod(total, K)
            folds = []
            start = 0
            for i in range(K):
                fold_size = div + (1 if i < mod else 0)
                folds.append(files[start:start + fold_size])
                start += fold_size

            test_files = folds[n - 1]
            train_files = []
            for idx in range(K):
                if idx != n - 1:
                    train_files.extend(folds[idx])

            return train_files, test_files

        healthy_train, healthy_test = kfold_split(healthy_files, K, n)
        patient_train, patient_test = kfold_split(patient_files, K, n)
        if type == 'train':
            selected_files = healthy_train + patient_train
        elif type == 'test':
            selected_files = healthy_test + patient_test
        elif type == 'all':
            selected_files = healthy_train + patient_train + healthy_test + patient_test
        else:
            raise ValueError("type should be 'train' or 'test'")
        self.scale = 7.72176E-05
        self.data_x = []
        self.data_label = []
        print(selected_files)
        for file_name in tqdm(selected_files, desc='loading data:'):
            file_path = os.path.join(folder_path, file_name)
            npz_data = np.load(file_path, allow_pickle=True)
            x = npz_data['x']
            y = npz_data['y']
            num = int(x.shape[-1] * 200 / self.origin_sample_rate)
            x = resample(x, axis=-1, num=num)
            x = signal.filtfilt(self.b_bandpass, self.a_bandpass, x, axis=-1)
            x = signal.filtfilt(self.b_notch, self.a_notch, x, axis=-1)
            x = torch.FloatTensor(x.copy())
            y = torch.tensor(y, dtype=torch.long)
            self.data_x.append(x)
            self.data_label.append(y)
        self.data_x = torch.cat(self.data_x, dim=0)
        self.data_x = self.data_x / self.scale
        self.data_label = torch.cat(self.data_label, dim=0)
        self.data_label = self.data_label.to(torch.long)

        self.tokenizer = AutoTokenizer.from_pretrained('')
        self.Option_question_prompt = 'Which disease does this signal belong to? Options:(A)normal (B)major depressive disorder'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'
        self.options = [self.Option_label_0, self.Option_label_1]
        self.type = type

    def __getitem__(self, index):
        signal = self.data_x[index]
        Y = self.data_label[index]
        if self.type == 'test':
            input_ids, option_token_ids = prepare_qwen_test_inputs(question=self.Option_question_prompt,
                                                                   options=self.options, tokenizer=self.tokenizer)

            return signal, input_ids, option_token_ids, Y
        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            else:
                answer = self.Option_label_1

            ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                             max_length=self.max_length)
            return signal, ids, label, Y

    def __len__(self):
        return len(self.data_x)

class ADFSU(Dataset):
    def __init__(self, folder_path, n, K=5, max_length=64, type='train',
                 healthy_prefix='Healthy_Paciente',
                 patient_prefix='AD_Paciente'):
        super(ADFSU, self).__init__()
        self.max_length = max_length

        file_names = os.listdir(folder_path)
        npz_files = [f for f in file_names if f.endswith(".npz")]

        healthy_files = [f for f in npz_files if f.startswith(healthy_prefix)]
        patient_files = [f for f in npz_files if f.startswith(patient_prefix)]
        healthy_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
        patient_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
        def kfold_split(files, K, n):
            total = len(files)
            assert K <= total, f"Cannot split {total} files into {K} folds"

            div, mod = divmod(total, K)
            folds = []
            start = 0
            for i in range(K):
                fold_size = div + (1 if i < mod else 0)
                folds.append(files[start:start + fold_size])
                start += fold_size

            test_files = folds[n - 1]
            train_files = []
            for idx in range(K):
                if idx != n - 1:
                    train_files.extend(folds[idx])

            return train_files, test_files

        healthy_train, healthy_test = kfold_split(healthy_files, K, n)
        patient_train, patient_test = kfold_split(patient_files, K, n)
        if type == 'train':
            selected_files = healthy_train + patient_train
        elif type == 'test':
            selected_files = healthy_test + patient_test
        elif type == 'all':
            selected_files = healthy_train + patient_train + healthy_test + patient_test
        else:
            raise ValueError("type should be 'train' or 'test'")
        self.scale = 1
        self.data_x = []
        self.data_label = []
        print(selected_files)
        for file_name in tqdm(selected_files, desc='loading data:'):
            file_path = os.path.join(folder_path, file_name)
            npz_data = np.load(file_path, allow_pickle=True)
            x = npz_data['x']
            y = npz_data['y']
            x = torch.FloatTensor(x)
            y = torch.tensor(y, dtype=torch.long)
            self.data_x.append(x)
            self.data_label.append(y)
        self.data_x = torch.cat(self.data_x, dim=0)
        self.data_x = self.data_x / self.scale
        self.data_label = torch.cat(self.data_label, dim=0)
        self.data_label = self.data_label.to(torch.long)

        self.tokenizer = AutoTokenizer.from_pretrained('')
        self.Option_question_prompt = 'Which disease does this signal belong to? Options:(A)normal (B)alzheimer\'s disease'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'
        self.options = [self.Option_label_0, self.Option_label_1]
        self.type = type

    def __getitem__(self, index):
        signal = self.data_x[index]
        Y = self.data_label[index]
        if self.type == 'test':
            input_ids, option_token_ids = prepare_qwen_test_inputs(question=self.Option_question_prompt,
                                                                   options=self.options, tokenizer=self.tokenizer)

            return signal, input_ids, option_token_ids, Y
        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            else:
                answer = self.Option_label_1

            ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                             max_length=self.max_length)
            return signal, ids, label, Y

    def __len__(self):
        return len(self.data_x)

class ADHD_Adult(Dataset):
    def __init__(self, folder_path, n, K=5, max_length=64, type='train',
                 healthy_male_prefix='FC_',
                 healthy_female_prefix='MC_',
                 patient_male_prefix='FADHD_',
                 patient_female_prefix='MADHD_'):
        super(ADHD_Adult, self).__init__()
        self.max_length = max_length
        file_names = os.listdir(folder_path)
        npz_files = [f for f in file_names if f.endswith(".npz")]
        healthy_male_files = [f for f in npz_files if f.startswith(healthy_male_prefix)]
        healthy_female_files = [f for f in npz_files if f.startswith(healthy_female_prefix)]
        patient_male_files = [f for f in npz_files if f.startswith(patient_male_prefix)]
        patient_female_files = [f for f in npz_files if f.startswith(patient_female_prefix)]
        print(f"Healthy male files ({len(healthy_male_files)}): {healthy_male_files}")
        print(f"Healthy female files ({len(healthy_female_files)}): {healthy_female_files}")
        print(f"Patient male files ({len(patient_male_files)}): {patient_male_files}")
        print(f"Patient female files ({len(patient_female_files)}): {patient_female_files}")
        for files in [healthy_male_files, healthy_female_files, patient_male_files, patient_female_files]:
            files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
        def kfold_split(files, K, n):
            total = len(files)
            if K > total:
                raise ValueError(f"Cannot split {total} files into {K} folds")

            div, mod = divmod(total, K)
            folds = []
            start = 0
            for i in range(K):
                fold_size = div + (1 if i < mod else 0)
                folds.append(files[start:start + fold_size])
                start += fold_size

            test_files = folds[n - 1]
            train_files = [f for idx in range(K) if idx != n - 1 for f in folds[idx]]
            return train_files, test_files

        hm_train, hm_test = kfold_split(healthy_male_files, K, n)
        hf_train, hf_test = kfold_split(healthy_female_files, K, n)
        pm_train, pm_test = kfold_split(patient_male_files, K, n)
        pf_train, pf_test = kfold_split(patient_female_files, K, n)

        if type == 'train':
            selected_files = hm_train + hf_train + pm_train + pf_train
        elif type == 'test':
            selected_files = hm_test + hf_test + pm_test + pf_test
        elif type == 'all':
            selected_files = hm_train + hf_train + pm_train + pf_train + hm_test + hf_test + pm_test + pf_test
        else:
            raise ValueError("type should be 'train' or 'test'")
        self.scale = 1
        self.data_x = []
        self.data_label = []
        print(selected_files)
        for file_name in tqdm(selected_files, desc='loading data:'):
            file_path = os.path.join(folder_path, file_name)
            npz_data = np.load(file_path, allow_pickle=True)
            x = npz_data['x']
            y = npz_data['y']
            x = torch.FloatTensor(x)
            y = torch.tensor(y, dtype=torch.long)
            self.data_x.append(x)
            self.data_label.append(y)
        self.data_x = torch.cat(self.data_x, dim=0)
        self.data_x = self.data_x / self.scale
        self.data_label = torch.cat(self.data_label, dim=0)
        self.data_label = self.data_label.to(torch.long)

        self.tokenizer = AutoTokenizer.from_pretrained('')
        self.Option_question_prompt = 'Which disease does this signal belong to? Options:(A)normal (B)attention deficit hyperactivity disorder'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'
        self.options = [self.Option_label_0, self.Option_label_1]
        self.type = type

    def __getitem__(self, index):
        signal = self.data_x[index]
        Y = self.data_label[index]
        if self.type == 'test':
            input_ids, option_token_ids = prepare_qwen_test_inputs(question=self.Option_question_prompt,
                                                                   options=self.options, tokenizer=self.tokenizer)

            return signal, input_ids, option_token_ids, Y
        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            else:
                answer = self.Option_label_1

            ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                             max_length=self.max_length)
            return signal, ids, label, Y

    def __len__(self):
        return len(self.data_x)

class ADHD_Children(Dataset):
    def __init__(self, folder_path, n, K=5, max_length=64, type='train',
                 healthy_1_prefix='Control_part1_',
                 healthy_2_prefix='Control_part2_',
                 patient_1_prefix='ADHD_part1_',
                 patient_2_prefix='ADHD_part2_'):
        super(ADHD_Children, self).__init__()
        self.max_length = max_length

        file_names = os.listdir(folder_path)
        npz_files = [f for f in file_names if f.endswith(".npz")]

        healthy_1_files = [f for f in npz_files if f.startswith(healthy_1_prefix)]
        healthy_2_files = [f for f in npz_files if f.startswith(healthy_2_prefix)]
        patient_1_files = [f for f in npz_files if f.startswith(patient_1_prefix)]
        patient_2_files = [f for f in npz_files if f.startswith(patient_2_prefix)]
        print(f"Healthy 1 files ({len(healthy_1_files)}): {healthy_1_files}")
        print(f"Healthy 2 files ({len(healthy_2_files)}): {healthy_2_files}")
        print(f"Patient 1 files ({len(patient_1_files)}): {patient_1_files}")
        print(f"Patient 2 files ({len(patient_2_files)}): {patient_2_files}")
        for files in [healthy_1_files, healthy_2_files, patient_1_files, patient_2_files]:
            files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
        def kfold_split(files, K, n):
            total = len(files)
            if K > total:
                raise ValueError(f"Cannot split {total} files into {K} folds")

            div, mod = divmod(total, K)
            folds = []
            start = 0
            for i in range(K):
                fold_size = div + (1 if i < mod else 0)
                folds.append(files[start:start + fold_size])
                start += fold_size

            test_files = folds[n - 1]
            train_files = [f for idx in range(K) if idx != n - 1 for f in folds[idx]]
            return train_files, test_files
        h1_train, h1_test = kfold_split(healthy_1_files, K, n)
        h2_train, h2_test = kfold_split(healthy_2_files, K, n)
        p1_train, p1_test = kfold_split(patient_1_files, K, n)
        p2_train, p2_test = kfold_split(patient_2_files, K, n)
        if type == 'train':
            selected_files = h1_train + h2_train + p1_train + p2_train
        elif type == 'test':
            selected_files = h1_test + h2_test + p1_test + p2_test
        elif type == 'all':
            selected_files = h1_train + h2_train + p1_train + p2_train + h1_test + h2_test + p1_test + p2_test
        else:
            raise ValueError("type should be 'train' or 'test'")
        self.scale = 1
        self.data_x = []
        self.data_label = []
        print(selected_files)
        for file_name in tqdm(selected_files, desc='loading data:'):
            file_path = os.path.join(folder_path, file_name)
            npz_data = np.load(file_path, allow_pickle=True)
            x = npz_data['x']
            y = npz_data['y']
            x = torch.FloatTensor(x)
            y = torch.tensor(y, dtype=torch.long)
            self.data_x.append(x)
            self.data_label.append(y)
        self.data_x = torch.cat(self.data_x, dim=0)
        self.data_x = self.data_x / self.scale
        self.data_label = torch.cat(self.data_label, dim=0)
        self.data_label = self.data_label.to(torch.long)

        self.tokenizer = AutoTokenizer.from_pretrained('')
        self.Option_question_prompt = 'Which disease does this signal belong to? Options:(A)normal (B)attention deficit hyperactivity disorder'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'
        self.options = [self.Option_label_0, self.Option_label_1]
        self.type = type

    def __getitem__(self, index):
        signal = self.data_x[index]
        Y = self.data_label[index]
        if self.type == 'test':
            input_ids, option_token_ids = prepare_qwen_test_inputs(question=self.Option_question_prompt,
                                                                   options=self.options, tokenizer=self.tokenizer)

            return signal, input_ids, option_token_ids, Y
        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            else:
                answer = self.Option_label_1

            ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                             max_length=self.max_length)
            return signal, ids, label, Y

    def __len__(self):
        return len(self.data_x)

class Schizophrenia28(Dataset):
    def __init__(self, folder_path, n, K=5, max_length=64, type='train',
                 healthy_prefix='h',
                 patient_prefix='s'):
        super(Schizophrenia28, self).__init__()
        self.max_length = max_length
        file_names = os.listdir(folder_path)
        npz_files = [f for f in file_names if f.endswith(".npz")]
        healthy_files = [f for f in npz_files if f.startswith(healthy_prefix)]
        patient_files = [f for f in npz_files if f.startswith(patient_prefix)]
        healthy_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
        patient_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
        def kfold_split(files, K, n):
            total = len(files)
            assert K <= total, f"Cannot split {total} files into {K} folds"

            div, mod = divmod(total, K)
            folds = []
            start = 0
            for i in range(K):
                fold_size = div + (1 if i < mod else 0)
                folds.append(files[start:start + fold_size])
                start += fold_size

            test_files = folds[n - 1]
            train_files = []
            for idx in range(K):
                if idx != n - 1:
                    train_files.extend(folds[idx])

            return train_files, test_files

        healthy_train, healthy_test = kfold_split(healthy_files, K, n)
        patient_train, patient_test = kfold_split(patient_files, K, n)

        if type == 'train':
            selected_files = healthy_train + patient_train
        elif type == 'test':
            selected_files = healthy_test + patient_test
        elif type == 'all':
            selected_files = healthy_train + patient_train + healthy_test + patient_test
        else:
            raise ValueError("type should be 'train' or 'test'")
        self.scale = 6.66556E-05
        self.data_x = []
        self.data_label = []
        print(selected_files)
        for file_name in tqdm(selected_files, desc='loading data:'):
            file_path = os.path.join(folder_path, file_name)
            npz_data = np.load(file_path, allow_pickle=True)
            x = npz_data['x']
            y = npz_data['y']
            x = torch.FloatTensor(x)
            y = torch.tensor(y, dtype=torch.long)
            self.data_x.append(x)
            self.data_label.append(y)

        self.data_x = torch.cat(self.data_x, dim=0)
        self.data_x = self.data_x / self.scale
        self.data_label = torch.cat(self.data_label, dim=0)
        self.data_label = self.data_label.to(torch.long)

        self.tokenizer = AutoTokenizer.from_pretrained('')
        self.Option_question_prompt = 'Which disease does this signal belong to? Options:(A)normal (B)schizophrenia'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'
        self.options = [self.Option_label_0, self.Option_label_1]
        self.type = type

    def __getitem__(self, index):
        signal = self.data_x[index]
        Y = self.data_label[index]
        if self.type == 'test':
            input_ids, option_token_ids = prepare_qwen_test_inputs(question=self.Option_question_prompt,
                                                                   options=self.options, tokenizer=self.tokenizer)

            return signal, input_ids, option_token_ids, Y
        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            else:
                answer = self.Option_label_1

            ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                             max_length=self.max_length)
            return signal, ids, label, Y

    def __len__(self):
        return len(self.data_x)

class AdoSchizophrenia(Dataset):
    def __init__(self, folder_path, n, K=5, max_length=64, type='train',
                 healthy_prefix='N',
                 patient_prefix='S'):
        super(AdoSchizophrenia, self).__init__()
        self.max_length = max_length
        file_names = os.listdir(folder_path)
        npz_files = [f for f in file_names if f.endswith(".npz")]
        healthy_files = [f for f in npz_files if f.startswith(healthy_prefix)]
        patient_files = [f for f in npz_files if f.startswith(patient_prefix)]
        healthy_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
        patient_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))

        def kfold_split(files, K, n):
            total = len(files)
            assert K <= total, f"Cannot split {total} files into {K} folds"

            div, mod = divmod(total, K)
            folds = []
            start = 0
            for i in range(K):
                fold_size = div + (1 if i < mod else 0)
                folds.append(files[start:start + fold_size])
                start += fold_size

            test_files = folds[n - 1]
            train_files = []
            for idx in range(K):
                if idx != n - 1:
                    train_files.extend(folds[idx])

            return train_files, test_files

        healthy_train, healthy_test = kfold_split(healthy_files, K, n)
        patient_train, patient_test = kfold_split(patient_files, K, n)

        if type == 'train':
            selected_files = healthy_train + patient_train
        elif type == 'test':
            selected_files = healthy_test + patient_test
        elif type == 'all':
            selected_files = healthy_train + patient_train + healthy_test + patient_test
        else:
            raise ValueError("type should be 'train' or 'test'")

        self.scale = 1
        self.data_x = []
        self.data_label = []
        print(selected_files)
        for file_name in tqdm(selected_files, desc='loading data:'):
            file_path = os.path.join(folder_path, file_name)
            npz_data = np.load(file_path, allow_pickle=True)
            x = npz_data['x']
            y = npz_data['y']
            x = torch.FloatTensor(x)
            y = torch.tensor(y, dtype=torch.long)
            self.data_x.append(x)
            self.data_label.append(y)

        self.data_x = torch.cat(self.data_x, dim=0)
        self.data_x = self.data_x / self.scale
        self.data_label = torch.cat(self.data_label, dim=0)
        self.data_label = self.data_label.to(torch.long)

        self.tokenizer = AutoTokenizer.from_pretrained('')
        self.Option_question_prompt = 'Which disease does this signal belong to? Options:(A)normal (B)schizophrenia'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'
        self.options = [self.Option_label_0, self.Option_label_1]
        self.type = type

    def __getitem__(self, index):
        signal = self.data_x[index]
        Y = self.data_label[index]
        if self.type == 'test':
            input_ids, option_token_ids = prepare_qwen_test_inputs(question=self.Option_question_prompt,
                                                                   options=self.options, tokenizer=self.tokenizer)

            return signal, input_ids, option_token_ids, Y
        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            else:
                answer = self.Option_label_1

            ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                             max_length=self.max_length)
            return signal, ids, label, Y

    def __len__(self):
        return len(self.data_x)

class RatEpilepsy(Dataset):
    def __init__(self, folder_path, n, K=5, max_length=64, type='train'):
        super(RatEpilepsy, self).__init__()
        self.max_length = max_length
        file_names = os.listdir(folder_path)
        npz_files = [f for f in file_names if f.endswith(".npz")]
        npz_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
        total = len(npz_files)
        assert K <= total, f"Cannot split {total} files into {K} folds"

        div, mod = divmod(total, K)
        folds = []
        start = 0
        for i in range(K):
            fold_size = div + (1 if i < mod else 0)
            folds.append(npz_files[start:start + fold_size])
            start += fold_size

        if n < 1 or n > K:
            raise ValueError(f"n should be between 1 and {K}")

        test_files = folds[n - 1]
        train_files = []
        for idx in range(K):
            if idx != n - 1:
                train_files.extend(folds[idx])

        if type == 'train':
            selected_files = train_files
        elif type == 'test':
            selected_files = test_files
        elif type == 'all':
            selected_files = train_files + test_files
        else:
            raise ValueError("type should be 'train' or 'test'")

        self.origin_sample_rate = 6000
        self.scale = 3008.25
        self.data_x = []
        self.data_label = []
        print(selected_files)
        for file_name in tqdm(selected_files, desc='loading data:'):
            file_path = os.path.join(folder_path, file_name)
            npz_data = np.load(file_path, allow_pickle=True)
            x = npz_data['x']
            y = npz_data['y']
            x = torch.FloatTensor(x)
            y = torch.tensor(y, dtype=torch.long)
            self.data_x.append(x)
            self.data_label.append(y)


        self.data_x = torch.cat(self.data_x, dim=0)
        self.data_x = self.data_x / self.scale
        self.data_label = torch.cat(self.data_label, dim=0)
        self.data_label = self.data_label.to(torch.long)

        self.tokenizer = AutoTokenizer.from_pretrained('')
        self.Option_question_prompt = 'Which epilepsy state does this signal belong to? Options:(A)interictal (B)ictal'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'
        self.options = [self.Option_label_0, self.Option_label_1]
        self.type = type

    def __getitem__(self, index):
        signal = self.data_x[index]
        Y = self.data_label[index]
        if self.type == 'test':
            input_ids, option_token_ids = prepare_qwen_test_inputs(question=self.Option_question_prompt,
                                                                   options=self.options, tokenizer=self.tokenizer)

            return signal, input_ids, option_token_ids, Y
        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            else:
                answer = self.Option_label_1

            ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                             max_length=self.max_length)
            return signal, ids, label, Y

    def __len__(self):
        return len(self.data_x)

class Siena(Dataset):
    def __init__(self, folder_path, n, K=5, max_length=64, type='train'):
        super(Siena, self).__init__()
        self.max_length = max_length
        file_names = os.listdir(folder_path)
        npz_files = [f for f in file_names if f.endswith(".npz")]
        npz_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))

        total = len(npz_files)
        assert K <= total, f"Cannot split {total} files into {K} folds"

        div, mod = divmod(total, K)
        folds = []
        start = 0
        for i in range(K):
            fold_size = div + (1 if i < mod else 0)
            folds.append(npz_files[start:start + fold_size])
            start += fold_size

        if n < 1 or n > K:
            raise ValueError(f"n should be between 1 and {K}")

        test_files = folds[n - 1]
        train_files = []
        for idx in range(K):
            if idx != n - 1:
                train_files.extend(folds[idx])

        if type == 'train':
            selected_files = train_files
        elif type == 'test':
            selected_files = test_files
        elif type == 'all':
            selected_files = train_files + test_files
        else:
            raise ValueError("type should be 'train' or 'test'")

        self.origin_sample_rate = 256
        self.scale = 0.00040748
        self.data_x = []
        self.data_label = []
        print(selected_files)
        for file_name in tqdm(selected_files, desc='loading data:'):
            file_path = os.path.join(folder_path, file_name)
            npz_data = np.load(file_path, allow_pickle=True)
            x = npz_data['x']
            y = npz_data['y']
            x = torch.FloatTensor(x)
            y = torch.tensor(y, dtype=torch.long)
            self.data_x.append(x)
            self.data_label.append(y)

        self.data_x = torch.cat(self.data_x, dim=0)
        self.data_x = self.data_x / self.scale
        self.data_label = torch.cat(self.data_label, dim=0)
        self.data_label = self.data_label.to(torch.long)

        self.tokenizer = AutoTokenizer.from_pretrained('')
        self.Option_question_prompt = 'Which epilepsy state does this signal belong to? Options:(A)interictal (B)ictal (C)preictal (D)postictal'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'
        self.Option_label_2 = 'C'
        self.Option_label_3 = 'D'
        self.options = [self.Option_label_0, self.Option_label_1, self.Option_label_2, self.Option_label_3]
        self.type = type

    def __getitem__(self, index):
        signal = self.data_x[index]
        Y = self.data_label[index]
        if self.type == 'test':
            input_ids, option_token_ids = prepare_qwen_test_inputs(question=self.Option_question_prompt,
                                                                   options=self.options, tokenizer=self.tokenizer)

            return signal, input_ids, option_token_ids, Y
        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            elif Y==1:
                answer = self.Option_label_1
            elif Y==2:
                answer = self.Option_label_2
            else:
                answer = self.Option_label_3
            ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                             max_length=self.max_length)
            return signal, ids, label, Y

    def __len__(self):
        return len(self.data_x)

class APAVA(Dataset):
    def __init__(self, folder_path, n, K=5, max_length=64, type='train',
                 healthy_prefix='H',
                 patient_prefix='A'):
        super(APAVA, self).__init__()
        self.max_length = max_length
        file_names = os.listdir(folder_path)
        npz_files = [f for f in file_names if f.endswith(".npz")]

        healthy_files = [f for f in npz_files if f.startswith(healthy_prefix)]
        patient_files = [f for f in npz_files if f.startswith(patient_prefix)]

        healthy_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
        patient_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))

        def kfold_split(files, K, n):
            total = len(files)
            assert K <= total, f"Cannot split {total} files into {K} folds"

            div, mod = divmod(total, K)
            folds = []
            start = 0
            for i in range(K):
                fold_size = div + (1 if i < mod else 0)
                folds.append(files[start:start + fold_size])
                start += fold_size

            test_files = folds[n - 1]
            train_files = []
            for idx in range(K):
                if idx != n - 1:
                    train_files.extend(folds[idx])

            return train_files, test_files

        healthy_train, healthy_test = kfold_split(healthy_files, K, n)
        patient_train, patient_test = kfold_split(patient_files, K, n)

        if type == 'train':
            selected_files = healthy_train + patient_train
        elif type == 'test':
            selected_files = healthy_test + patient_test
        elif type == 'all':
            selected_files = healthy_train + patient_train + healthy_test + patient_test
        else:
            raise ValueError("type should be 'train' or 'test'")
        self.scale = 21.92150000
        self.data_x = []
        self.data_label = []
        print(selected_files)
        for file_name in tqdm(selected_files, desc='loading data:'):
            file_path = os.path.join(folder_path, file_name)
            npz_data = np.load(file_path, allow_pickle=True)
            x = npz_data['x']
            y = npz_data['y']
            x = torch.FloatTensor(x)
            y = torch.tensor(y, dtype=torch.long)
            self.data_x.append(x)
            self.data_label.append(y)

        self.data_x = torch.cat(self.data_x, dim=0)
        self.data_x = self.data_x / self.scale
        self.data_label = torch.cat(self.data_label, dim=0)
        self.data_label = self.data_label.to(torch.long)

        self.tokenizer = AutoTokenizer.from_pretrained('')
        self.Option_question_prompt = 'Which epilepsy state does this signal belong to? Options:(A)normal (B)alzheimer\'s disease'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'
        self.options = [self.Option_label_0, self.Option_label_1]
        self.type = type

    def __getitem__(self, index):
        signal = self.data_x[index]
        Y = self.data_label[index]
        if self.type == 'test':
            input_ids, option_token_ids = prepare_qwen_test_inputs(question=self.Option_question_prompt,
                                                                   options=self.options, tokenizer=self.tokenizer)

            return signal, input_ids, option_token_ids, Y
        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            else:
                answer = self.Option_label_1

            ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                             max_length=self.max_length)
            return signal, ids, label, Y

    def __len__(self):
        return len(self.data_x)

class NMTAB(Dataset):
    def __init__(self, n=1, max_length=64, type='test'):
        super(NMTAB, self).__init__()
        self.max_length = max_length
        self.type = type
        if self.type == 'train':
            folder_path = ''
        else:
            folder_path = ''
        file_names = os.listdir(folder_path)
        file_names = [file for file in file_names]
        file_names = sorted([file for file in file_names if file.endswith(".npz")])  # Sort and filter .npz files
        # Only split if it's training data
        if self.type == 'train':
            # Split file_names into 5 parts
            num_files = len(file_names)
            part_size = num_files // 5
            parts = []
            for i in range(5):
                start = i * part_size
                end = (i + 1) * part_size if i < 4 else num_files
                parts.append(file_names[start:end])

            # Select 4 parts based on n (1-5)
            selected_parts = []
            for i in range(5):
                if i != (n - 1):  # Skip the nth part (0-based index)
                    selected_parts.extend(parts[i])
            file_names = selected_parts
        self.scale = 1.81309E-06
        self.data_x = []
        self.data_label = []
        for file_name in tqdm(file_names, desc='loading data:'):
            if file_name.endswith(".npz"):
                file_path = os.path.join(folder_path, file_name)
                npz_data = np.load(file_path, allow_pickle=True)
                x = npz_data['x']
                y = npz_data['y']
                x = torch.FloatTensor(x)
                y = torch.tensor(y, dtype=torch.long)
                self.data_x.append(x)
                self.data_label.append(y)
        self.data_x = torch.cat(self.data_x, dim=0)
        self.data_x = self.data_x / self.scale

        self.data_label = torch.cat(self.data_label, dim=0)
        self.data_label = self.data_label.to(torch.long)

        self.tokenizer = AutoTokenizer.from_pretrained('')
        self.question_prompt = 'Is the signal normal or abnormal?'
        self.label_0 = 'normal'
        self.label_1 = 'abnormal'
        self.Option_question_prompt = 'Is the signal normal or abnormal? Options:(A)normal (B)abnormal'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'
        self.options = [self.Option_label_0, self.Option_label_1]
        self.type = type

    def __getitem__(self, index):
        signal = self.data_x[index]
        Y = self.data_label[index]
        if self.type == 'test':
            input_ids, option_token_ids = prepare_qwen_test_inputs(question=self.Option_question_prompt,
                                                                   options=self.options, tokenizer=self.tokenizer)

            return signal, input_ids, option_token_ids, Y
        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            else:
                answer = self.Option_label_1

            ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                             max_length=self.max_length)
            return signal, ids, label, Y

    def __len__(self):
        return len(self.data_x)

class SEE(Dataset):
    def __init__(self, n=1, max_length=64, type='train'):
        super(SEE, self).__init__()
        self.max_length = max_length
        self.origin_sample_rate = 173.61
        self.scale = 46.93925
        if type == 'train':
            data = np.load('',
                           allow_pickle=True)
            self.data_x = torch.FloatTensor(data['x'])
            self.data_label = torch.tensor(data['y'], dtype=torch.long)
            # Only split if it's training data
            num_samples = len(self.data_x)
            part_size = num_samples // 5
            self.indices = []

            # Create 5 parts
            parts = []
            for i in range(5):
                start = i * part_size
                end = (i + 1) * part_size if i < 4 else num_samples
                parts.append(list(range(start, end)))

            # Select 4 parts based on n (1-5)
            selected_indices = []
            for i in range(5):
                if i != (n - 1):  # Skip the nth part (0-based index)
                    selected_indices.extend(parts[i])
            self.data_x = self.data_x[selected_indices]
            self.data_label = self.data_label[selected_indices]
        else:
            data = np.load('',
                           allow_pickle=True)
            self.data_x = torch.FloatTensor(data['x'])
            self.data_label = torch.tensor(data['y'], dtype=torch.long)
        self.data_x = self.data_x / self.scale

        self.tokenizer = AutoTokenizer.from_pretrained('')
        self.question_prompt = 'Does this signal belong to epilepsy?'
        self.label_0 = 'No'
        self.label_1 = 'Yes'
        self.Option_question_prompt = 'Does this signal belong to epilepsy? Options:(A)No (B)Yes'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'
        self.options = [self.Option_label_0, self.Option_label_1]
        self.type = type

    def __getitem__(self, index):
        signal = self.data_x[index]
        Y = self.data_label[index]
        if self.type == 'test':
            input_ids, option_token_ids = prepare_qwen_test_inputs(question=self.Option_question_prompt,
                                                                   options=self.options, tokenizer=self.tokenizer)

            return signal, input_ids, option_token_ids, Y
        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            else:
                answer = self.Option_label_1

            ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                             max_length=self.max_length)
            return signal, ids, label, Y

    def __len__(self):
        return len(self.data_x)


class NTUBIS(Dataset):
    def __init__(self, folder_path, n, K=5, max_length=64, type='train'):
        super(NTUBIS, self).__init__()
        self.max_length = max_length
        self.b_bandpass, self.a_bandpass = signal.butter(4, [0.001, 0.75], btype='bandpass')
        self.b_notch, self.a_notch = signal.iirnotch(0.5, 0.016666666666666666)
        file_names = os.listdir(folder_path)
        npz_files = [f for f in file_names if f.endswith(".npz")]
        npz_files.sort(key=lambda x: int(x.split('.')[0]))

        total = len(npz_files)
        assert K <= total, f"Cannot split {total} files into {K} folds"
        div, mod = divmod(total, K)
        folds = []
        start = 0
        for i in range(K):
            fold_size = div + (1 if i < mod else 0)
            folds.append(npz_files[start:start + fold_size])
            start += fold_size

        if n < 1 or n > K:
            raise ValueError(f"n should be between 1 and {K}")
        test_files = folds[n - 1]
        train_files = []
        for idx in range(K):
            if idx != n - 1:
                train_files.extend(folds[idx])

        if type == 'train':
            selected_files = train_files
        elif type == 'test':
            selected_files = test_files
        else:
            raise ValueError("type should be 'train' or 'test'")
        self.origin_sample_rate = 128
        self.scale = 1
        self.data_x = []
        self.data_label = []
        print(selected_files)
        for file_name in tqdm(selected_files, desc='loading data:'):
            file_path = os.path.join(folder_path, file_name)
            npz_data = np.load(file_path, allow_pickle=True)
            x = npz_data['x']
            num = int(x.shape[-1] * 200 / self.origin_sample_rate)
            x = resample(x, axis=-1, num=num)
            x = signal.filtfilt(self.b_bandpass, self.a_bandpass, x, axis=-1)
            x = signal.filtfilt(self.b_notch, self.a_notch, x, axis=-1)
            y = npz_data['y']
            x = torch.FloatTensor(x.copy())
            y = torch.tensor(y, dtype=torch.long)
            self.data_x.append(x)
            self.data_label.append(y)
        self.data_x = torch.cat(self.data_x, dim=0)
        self.data_x = self.data_x / self.scale
        self.data_label = torch.cat(self.data_label, dim=0)
        self.data_label = self.data_label.to(torch.long)

        self.tokenizer = AutoTokenizer.from_pretrained('')
        self.Option_question_prompt = 'Which anesthesia depth does this signal belong to? Options:(A)Deep Hypnotic State (B)General Anesthesia (C)Moderate sedation (D)Awake/Light sedation'
        self.Option_label_0 = 'A'
        self.Option_label_1 = 'B'
        self.Option_label_2 = 'C'
        self.Option_label_3 = 'D'
        self.options = [self.Option_label_0, self.Option_label_1, self.Option_label_2, self.Option_label_3]
        self.type = type

    def __getitem__(self, index):
        signal = self.data_x[index]
        Y = self.data_label[index] - 1
        if self.type == 'test':
            input_ids, option_token_ids = prepare_qwen_test_inputs(question=self.Option_question_prompt,
                                                                   options=self.options, tokenizer=self.tokenizer)

            return signal, input_ids, option_token_ids, Y
        else:
            question = self.Option_question_prompt
            if Y == 0:
                answer = self.Option_label_0
            elif Y == 1:
                answer = self.Option_label_1
            elif Y == 2:
                answer = self.Option_label_2
            else:
                answer = self.Option_label_3
            ids, label = prepare_qwen_inputs(question=question, answer=answer, tokenizer=self.tokenizer,
                                             max_length=self.max_length)
            return signal, ids, label, Y

    def __len__(self):
        return len(self.data_x)

if __name__ == "__main__":
    pass
