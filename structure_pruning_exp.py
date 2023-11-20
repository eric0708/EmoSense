import os
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW

dataset_dir = 'Dataset/IEMOCAP'
# load the list from the JSON file and separate text and label data
with open(os.path.join(dataset_dir, 'text_and_label.json'), 'r') as json_file:
    text_and_label = json.load(json_file)

text_data = []
label_data = []

for text, label in text_and_label:
    text_data.append(text)
    label_data.append(label)

# separate data into train and valid datasets
train_texts, valid_texts, train_labels, valid_labels = train_test_split(text_data, label_data, test_size=0.12, random_state=42, stratify=label_data)

# define dataset for IEMOCAP text emotion classification
class IEMOCAP_Text_Dataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

# load idx to label dictionary
with open(os.path.join(dataset_dir, 'idx_2_label.json'), 'r') as json_file:
    idx_2_label = json.load(json_file)

# define tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("eric0708/finetuned_roberta_text_emotion_recognition")
model = RobertaForSequenceClassification.from_pretrained("eric0708/finetuned_roberta_text_emotion_recognition", num_labels=len(idx_2_label))

# define dataset and dataloader
batch_size = 4
valid_dataset = IEMOCAP_Text_Dataset(valid_texts, valid_labels, tokenizer)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

import os
import time
import numpy as np

def print_size_of_model(model, tag=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",tag,' \t','Size (MB):', size/1e6)
    os.remove('temp.p')
    return size

def sparcity_model_overall(model, tag=""):
    num_zero = 0
    num_para = 0
    for name, module in model.named_modules():
        B = list(module.named_buffers())
        for name, b in B:
            num_zero += torch.sum(b == 0).item()
        P = list(module.named_parameters())
        for name, p in P:
            num_para += p.numel()
    print(f'{tag} sparcity: {num_zero / num_para * 100}')

def release_memory():
    torch.cuda.empty_cache()

def val(model, tag=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Start Validation")
    model.eval()
    valid_loss = 0
    valid_correct_preds = 0
    inference_latency_list = []
    with torch.no_grad():
        for batch in tqdm(valid_dataloader):
            i_start_time = time.time()
            inputs = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**inputs)
            i_end_time = time.time()
            logits = outputs.logits
            labels = inputs['labels'].to(device)
            inference_latency_list += [(i_end_time-i_start_time)/labels.size(0)] * labels.size(0)

            # calculate validation accuracy
            _, valid_predictions = torch.max(logits, dim=1)
            valid_correct_preds += torch.sum(valid_predictions == labels).item()

    valid_accuracy = valid_correct_preds / len(valid_dataset)
    print(f"{tag} Valid Accuracy: {valid_accuracy:.4f}\n")
    inference_latency_list = inference_latency_list[4:]
    print(f'{tag} Average Inference Latency: {np.mean(inference_latency_list)/4}')
    print_size_of_model(model, tag)
    sparcity_model_overall(model, tag)

import torch.nn.utils.prune as prune
import torch.nn as nn
def apply_structure_pruning(pruning_percentage, label = ""):
    model = RobertaForSequenceClassification.from_pretrained("eric0708/finetuned_roberta_text_emotion_recognition", num_labels=len(idx_2_label))
    release_memory()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
              prune.l1_unstructured(module, name='weight', amount=pruning_percentage)
              prune.remove(module, name='weight')
    val(model)

for i in [10, 20, 30, 40 ,50 ,60, 70, 80, 90]:
  apply_structure_pruning(i * 0.01, f'[s-{i}]')
