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
batch_size = 128
valid_dataset = IEMOCAP_Text_Dataset(valid_texts, valid_labels, tokenizer)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# validation loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Start Validation")
model.eval()
valid_loss = 0
valid_correct_preds = 0
with torch.no_grad():
    for batch, i in enumerate(valid_dataloader):
        print(f'[LOG] iteration-{i} start')
        inputs = {key: val.to(device) for key, val in batch.items()}
        print(f'[LOG] iteration-{i} input formated')
        outputs = model(**inputs)
        print(f'[LOG] iteration-{i} output inferenced')
        logits = outputs.logits
        print(f'[LOG] iteration-{i} output logited')
        labels = inputs['labels'].to(device)
        print(f'[LOG] iteration-{i} label to device')

        # calculate validation accuracy
        _, valid_predictions = torch.max(logits, dim=1)
        print(f'[LOG] iteration-{i} pred maxed')
        valid_correct_preds += torch.sum(valid_predictions == labels).item()
        print(f'[LOG] iteration-{i} preds accumulated')

valid_accuracy = valid_correct_preds / len(valid_dataset)
print(f"Valid Accuracy: {valid_accuracy:.4f}\n")
