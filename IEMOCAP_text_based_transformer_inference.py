import os
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

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
tokenizer = RobertaTokenizer.from_pretrained("Models/finetuned_roberta_IEMOCAP")
model = RobertaForSequenceClassification.from_pretrained("Models/finetuned_roberta_IEMOCAP", num_labels=len(idx_2_label))

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
valid_labels_list = []
valid_predictions_list = []
with torch.no_grad():
    for batch in tqdm(valid_dataloader):
        inputs = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs['labels'].to(device)

        # calculate validation accuracy
        _, valid_predictions = torch.max(logits, dim=1)
        valid_correct_preds += torch.sum(valid_predictions == labels).item()

        valid_labels_list += labels.to("cpu").tolist()
        valid_predictions_list += valid_predictions.to("cpu").tolist()

valid_accuracy = valid_correct_preds / len(valid_dataset)
print(f"Valid Accuracy: {valid_accuracy:.4f}\n")

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