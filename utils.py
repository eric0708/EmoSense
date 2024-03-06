import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.utils.prune as prune
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# function to help construct dictionary of sentence id to text using transcriptions files
def id_2_text_construct(id_2_text, transcriptions_dir):
    for filename in os.listdir(transcriptions_dir):
        if filename.split('.')[-1] != 'txt':
            continue

        with open(os.path.join(transcriptions_dir, filename), 'r') as file:
            for line in file:
                line_split = line.split()

                if line_split[0].startswith('Ses'):
                    id_2_text[line_split[0]] = ' '.join(line_split[2:])
                    
    return id_2_text

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

# define collate_fn to pad sequences to same length
def collate_fn(batch):
    input_values = [item["input_values"] for item in batch]
    labels = [item["label"] for item in batch]
    input_values_padded = pad_sequence(input_values, batch_first=True, padding_value=0)

    return {"input_values": input_values_padded, "label": torch.stack(labels)}

# define function for showing evaluation metrics
def validation_metrics(valid_labels_list, valid_predictions_list, idx_2_label):
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

# define function for applying unstructured pruning
def apply_unstructured_pruning(model, pruning_percentage):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
              prune.l1_unstructured(module, name='weight', amount=pruning_percentage)
              prune.remove(module, name='weight')

def apply_structured_pruning(model, pruning_percentage):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
              prune.ln_structured(module, name='weight', amount=pruning_percentage, n=2, dim=0)
              prune.remove(module, name='weight')

def apply_structured_pruning_to_encoder(model, pruning_percentage):
    for layer in model.roberta.encoder.layer:
        for name, submodule in layer.named_modules():
            if isinstance(submodule, nn.Linear) and "self.query" in name:
                prune.ln_structured(submodule, name='weight', amount=pruning_percentage, n=2, dim=0)
                prune.remove(submodule, name='weight')
            elif isinstance(submodule, nn.Linear) and "self.key" in name:
                prune.ln_structured(submodule, name='weight', amount=pruning_percentage, n=2, dim=0)
                prune.remove(submodule, name='weight')
            elif isinstance(submodule, nn.Linear) and "self.value" in name:
                prune.ln_structured(submodule, name='weight', amount=pruning_percentage, n=2, dim=0)
                prune.remove(submodule, name='weight')