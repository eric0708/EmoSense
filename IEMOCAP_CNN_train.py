# Import Libraries
import os
import json
import numpy as np
import librosa
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Preprocess Dataset 

# function to help construct dictionary of sentence id to audio array using 
def id_2_audio_construct(id_2_audio, audio_dir):
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                file_path = os.path.join(root, file)
                id_2_audio[file.split(".")[0]] = file_path
    
    return id_2_audio

# function to help construct dictionary of sentence id to label using evaluation files
def id_2_label_construct(id_2_label, evaluation_dir):
    for filename in os.listdir(evaluation_dir):
        if filename.split('.')[-1] != 'txt':
            continue

        with open(os.path.join(evaluation_dir, filename), 'r') as file:
            for line in file:
                line_split = line.split()

                if len(line_split) >= 4 and line_split[3].startswith('Ses'):
                    sentence_id = line_split[3]
                    label = line_split[4]
                    if label != 'xxx' and label != 'oth':
                        id_2_label[sentence_id] = label
                        
    return id_2_label

# initialize dictionaries
id_2_audio = {}
id_2_label = {}

# iterate through all the session directories
for session_num in range(1, 6):
    session_dir = 'Dataset/IEMOCAP/Session{}'.format(session_num)
    audio_dir = os.path.join(session_dir, 'sentences/wav')
    evaluation_dir = os.path.join(session_dir, 'dialog/EmoEvaluation')

    id_2_audio = id_2_audio_construct(id_2_audio, audio_dir)
    id_2_label = id_2_label_construct(id_2_label, evaluation_dir)

# load label to idx dictionary
dataset_dir = 'Dataset/IEMOCAP'

with open(os.path.join(dataset_dir, 'label_2_idx.json'), 'r') as json_file:
    label_2_idx = json.load(json_file)

# iterate through all id_2_label sentence ids and locate corresponding audio
audio_and_label = []

for sentence_id in id_2_label:
    label = id_2_label[sentence_id]
    audio = id_2_audio[sentence_id]
    
    audio_and_label.append((audio, label_2_idx[label]))

# save the list to dataset directory
with open(os.path.join(dataset_dir, 'audio_and_label.json'), 'w') as json_file:
    json.dump(audio_and_label, json_file)

# Generate Train and Validation Set 

# load the list from the JSON file and separate text and label data
with open(os.path.join(dataset_dir, 'audio_and_label.json'), 'r') as json_file:
    audio_and_label = json.load(json_file)

audio_data = []
label_data = []

for audio, label in audio_and_label:
    audio_data.append(audio)
    label_data.append(label)

# separate data into train and valid datasets
train_audios, valid_audios, train_labels, valid_labels = train_test_split(audio_data, label_data, test_size=0.12, random_state=42, stratify=label_data)

# Define Dataset

class IEMOCAP_Audio_Dataset(Dataset):
    def __init__(self, audios, labels, spectrogram_length):
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

# find the longest spectrogram length in the training dataset
spectrogram_length = 0
feature_size = 0

for filename in train_audios:
    speech_audio, _ = librosa.load(filename, sr = 16000)
    spectrogram = librosa.stft(speech_audio, n_fft=1024, hop_length=160, center=False, win_length=1024)
    spectrogram = abs(spectrogram)
    feature_size, length = spectrogram.shape

    if length > spectrogram_length:
        spectrogram_length = length

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
        # print(x.shape)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        # print(x.shape)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        # print(x.shape)
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        # print(x.shape)
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        # print(x.shape)
        x = x.view(-1, 32 * 32 * 213)
        # print(x.shape)
        x = self.relu(self.fc1(x))
        # print(x.shape)
        x = self.fc2(x)
        # print(x.shape)
        return x

num_classes = len(idx_2_label)
model = Four_Layer_SpectrogramCNN(num_classes)

# define dataset
# create training and validation datasets and dataloaders
train_dataset = IEMOCAP_Audio_Dataset(train_audios, train_labels, spectrogram_length)
valid_dataset = IEMOCAP_Audio_Dataset(valid_audios, valid_labels, spectrogram_length)

# create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=32)

# Finetune Model

# define training parameters
learning_rate = 1e-3
num_epochs = 15

# set up optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps")
model.to(device)
print("Using device: {}".format(device))

for epoch in range(num_epochs):
    # train
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    for i, (inputs, labels) in tqdm(enumerate(train_dataloader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        # print(inputs.shape)
        # print(labels.shape)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
    
    average_loss = total_loss / len(train_dataloader)
    accuracy = correct_predictions / total_samples * 100
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}, Training Accuracy: {accuracy:.2f}%')


    # validation
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    for i, (inputs, labels) in tqdm(enumerate(valid_dataloader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
    
    average_loss = total_loss / len(valid_dataloader)
    accuracy = correct_predictions / total_samples * 100
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')

# Save Model

# specify the directory to save the models
model_dir = 'Models'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

output_filename = os.path.join(model_dir, 'IEMOCAP_spectrogram_cnn_model.pth')

torch.save(model.state_dict(), output_filename)