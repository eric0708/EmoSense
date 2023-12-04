# Import Libraries
import os
import wget
import json
from tqdm import tqdm

import torch
import torchaudio
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from codecarbon import track_emissions

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

if not os.path.exists(model_destination):
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

if torch.cuda.is_available():
    model.load_state_dict(torch.load(model_destination))
else: 
    model.load_state_dict(torch.load(model_destination, map_location=torch.device('cpu')))

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

# @track_emissions()
def val(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # device = torch.device("mps")
    print("Using device: {}".format(device))
    model.to(device)

    # validation
    print("Start Validation")
    model.eval()
    valid_loss = 0
    valid_correct_preds = 0
    valid_labels_list = []
    valid_predictions_list = []
    with torch.no_grad():
        for batch in tqdm(valid_dataloader):
            inputs = batch['input_values'].to(device)
            labels = batch['label'].to(device)
            # with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            valid_loss += loss.item()

            # calculate validation accuracy
            _, valid_predictions = torch.max(outputs, dim=1)
            valid_correct_preds += torch.sum(valid_predictions == labels).item()

            valid_labels_list += labels.to("cpu").tolist()
            valid_predictions_list += valid_predictions.to("cpu").tolist()

    avg_valid_loss = valid_loss / len(valid_dataloader)
    valid_accuracy = valid_correct_preds / len(valid_dataset)

    # Print loss and accuracy
    print(f"Valid Loss: {avg_valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}\n")

    def validation_metrics(valid_labels_list, valid_predictions_list):
        # Print accuracy
        accuracy = accuracy_score(valid_labels_list, valid_predictions_list)
        print(f"Accuracy: {accuracy:.4f}")

        # Print precision, recall, and F1 score for each class
        precision = precision_score(valid_labels_list, valid_predictions_list, average=None)
        recall = recall_score(valid_labels_list, valid_predictions_list, average=None)
        f1 = f1_score(valid_labels_list, valid_predictions_list, average=None)

        # Print precision, recall, and F1 score averaged across classes
        precision_macro = precision_score(valid_labels_list, valid_predictions_list, average='macro')
        recall_macro = recall_score(valid_labels_list, valid_predictions_list, average='macro')
        f1_macro = f1_score(valid_labels_list, valid_predictions_list, average='macro')

        print(f"Precision (Per Class): {precision}")
        print(f"Recall (Per Class): {recall}")
        print(f"F1 Score (Per Class): {f1}")

        print(f"Macro-Averaged Precision: {precision_macro:.4f}")
        print(f"Macro-Averaged Recall: {recall_macro:.4f}")
        print(f"Macro-Averaged F1 Score: {f1_macro:.4f}")

        # Print confusion matrix
        conf_matrix = confusion_matrix(valid_labels_list, valid_predictions_list)
        print("Confusion Matrix:")
        print(conf_matrix)

        # Print classification report
        class_names = [idx_2_label[str(i)] for i in range(8)]
        class_report = classification_report(valid_labels_list, valid_predictions_list, target_names=class_names)
        print("Classification Report:")
        print(class_report)

    validation_metrics(valid_labels_list, valid_predictions_list)
    
val(model)