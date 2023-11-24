# Import Libraries
import os
import wget
import json
import numpy as np
from tqdm import tqdm

import torch
import torchaudio
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, AdamW

# load label to idx dictionary
dataset_dir = 'Dataset/IEMOCAP'

with open(os.path.join(dataset_dir, 'label_2_idx.json'), 'r') as json_file:
    label_2_idx = json.load(json_file)

# Generate Train and Validation Set 

# load the list from the JSON file and separate text and label data
with open(os.path.join(dataset_dir, 'new_audio_and_label.json'), 'r') as json_file:
    audio_and_label = json.load(json_file)

valid_audios = []
valid_labels = []

for audio, label in audio_and_label:
    valid_audios.append(audio)
    valid_labels.append(label)

# Define Dataset

class IEMOCAP_Audio_Dataset(Dataset):
    def __init__(self, audios, labels, processor, max_length=160000):
        self.audios = audios
        self.labels = torch.tensor(labels)
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio_path = self.audios[idx]
        waveform, _ = torchaudio.load(audio_path)
        waveform = waveform / torch.max(torch.abs(waveform))
        inputs = {"input_values": self.processor(waveform.squeeze().numpy(), return_tensors="pt", max_length=self.max_length, sampling_rate=16000).input_values[0]}
        inputs["label"] = self.labels[idx]

        return inputs

# load idx to label dictionary
with open(os.path.join(dataset_dir, 'idx_2_label.json'), 'r') as json_file:
    idx_2_label = json.load(json_file)

# load processor and model
processor = Wav2Vec2Processor.from_pretrained("eric0708/finetuned_wav2vec_audio_emotion_recognition")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model_path_url = "https://huggingface.co/eric0708/finetuned_wav2vec_audio_emotion_recognition/resolve/main/model.pth"
model_destination = "Models/model.pth"
wget.download(model_path_url, model_destination)

# create new model to add classification head to original model 
class Wav2VecClassifier(nn.Module):
    def __init__(self, wav2vec_model, num_classes):
        super(Wav2VecClassifier, self).__init__()
        self.wav2vec_model = wav2vec_model
        self.classification_head = nn.Sequential(
            nn.Linear(32, num_classes)
        )

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.wav2vec_model(input_ids).logits
        pooled_output = torch.mean(hidden_states, dim=1)
        logits = self.classification_head(pooled_output)

        return logits

num_classes = len(idx_2_label)
model =  Wav2VecClassifier(model, num_classes)
model.load_state_dict(torch.load(model_destination))

# define validation dataset
valid_dataset = IEMOCAP_Audio_Dataset(valid_audios, valid_labels, processor)

# Evaluate Model
batch_size = 1
loss_fn = torch.nn.CrossEntropyLoss()

# define collate_fn to pad sequences to same length
def collate_fn(batch):
    input_values = [item["input_values"] for item in batch]
    labels = [item["label"] for item in batch]
    input_values_padded = pad_sequence(input_values, batch_first=True, padding_value=0)

    return {"input_values": input_values_padded, "label": torch.stack(labels)}

# create validation data loader
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Evaluation loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))
# device = torch.device("mps")
model.to(device)

# validation
print("Start Validation")
model.eval()
valid_loss = 0
valid_correct_preds = 0
with torch.no_grad():
    for batch in tqdm(valid_dataloader):
        inputs = batch['input_values'].to(device)
        labels = batch['label'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        valid_loss += loss.item()

        # calculate validation accuracy
        _, valid_predictions = torch.max(outputs, dim=1)
        valid_correct_preds += torch.sum(valid_predictions == labels).item()

avg_valid_loss = valid_loss / len(valid_dataloader)
valid_accuracy = valid_correct_preds / len(valid_dataset)

print(f"Valid Loss: {avg_valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}\n")
