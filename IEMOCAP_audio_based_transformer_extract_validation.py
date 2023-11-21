# Import Libraries
import os
import json
import shutil

from sklearn.model_selection import train_test_split

# load label to idx dictionary
dataset_dir = 'Dataset/IEMOCAP'

with open(os.path.join(dataset_dir, 'label_2_idx.json'), 'r') as json_file:
    label_2_idx = json.load(json_file)

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

# create validation audio directory if it does not exist
validation_audio_dir = os.path.join(dataset_dir, 'validation_audio')

if not os.path.exists(validation_audio_dir):
    # Create the directory
    os.makedirs(validation_audio_dir)

# copy validation audios into validation audio directory
new_valid_audios = []

for source_file_path in valid_audios:
    filename = source_file_path.split('/')[-1]
    destination_file_path = os.path.join(validation_audio_dir, filename)
    new_valid_audios.append(destination_file_path)
    shutil.copy(source_file_path, destination_file_path)

# save the new valid audio paths and labels into a json file
new_audio_and_label = []
for (new_valid_audio, valid_label) in zip(new_valid_audios, valid_labels):
    new_audio_and_label.append((new_valid_audio, valid_label))

# save the list to dataset directory
with open(os.path.join(dataset_dir, 'new_audio_and_label.json'), 'w') as json_file:
    json.dump(new_audio_and_label, json_file)