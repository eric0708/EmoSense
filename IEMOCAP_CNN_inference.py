# Import Libraries
import os
import wget
import json
import librosa
import numpy as np
from tqdm import tqdm

import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

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
    def __init__(self, audios, labels, spectrogram_length=3408):
        self.audios = audios
        self.labels = torch.tensor(labels)
        self.spectrogram_length = spectrogram_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio_path = self.audios[idx]
        label = self.labels[idx]

        # load the speech file and calculate spectrogram
        speech_audio, _ = librosa.load(audio_path, sr = 16000)
        spectrogram = librosa.stft(speech_audio, n_fft=1024, hop_length=160, center=False, win_length=1024)
        spectrogram = abs(spectrogram)
        
        feature_size, length = spectrogram.shape

        # modify the length of the spectrogram to be the same as the specified length
        if length > self.spectrogram_length:
            spectrogram = spectrogram[:, :self.spectrogram_length]
        else:
            cols_needed = self.spectrogram_length - length
            spectrogram = np.concatenate((spectrogram, np.zeros((feature_size, cols_needed))), axis=1)

        return np.expand_dims(spectrogram.astype(np.float32), axis=0) , label

# load idx to label dictionary
with open(os.path.join(dataset_dir, 'idx_2_label.json'), 'r') as json_file:
    idx_2_label = json.load(json_file)

# define model
class Four_Layer_SpectrogramCNN(nn.Module):
    def __init__(self, num_classes):
        super(Four_Layer_SpectrogramCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 32 * 213, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 32 * 32 * 213)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

num_classes = len(idx_2_label)
model_destination = 'Models/IEMOCAP_spectrogram_cnn_model.pth'
model = Four_Layer_SpectrogramCNN(num_classes)

if torch.cuda.is_available():
    model.load_state_dict(torch.load(model_destination))
else: 
    model.load_state_dict(torch.load(model_destination, map_location=torch.device('cpu')))

# define validation dataset
valid_dataset = IEMOCAP_Audio_Dataset(valid_audios, valid_labels)
valid_dataloader = DataLoader(valid_dataset, batch_size=1)

# Evaluate Model
criterion = torch.nn.CrossEntropyLoss()

# Evaluation loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))
# device = torch.device("mps")
model.to(device)

# validation
print("Start Validation")

# validation
model.eval()
total_loss = 0
correct_predictions = 0
total_samples = 0
valid_labels_list = []
valid_predictions_list = []

with torch.no_grad():
    for i, (inputs, labels) in tqdm(enumerate(valid_dataloader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        valid_labels_list += labels.to("cpu").tolist()
        valid_predictions_list += predicted.to("cpu").tolist()

    average_loss = total_loss / len(valid_dataloader)
    accuracy = correct_predictions / total_samples * 100

# Print loss and accuracy
print(f"Valid Loss: {average_loss:.4f}, Valid Accuracy: {accuracy:.4f}\n")

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