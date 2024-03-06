# Import Libraries
import os
import wget
import json
import argparse
from tqdm import tqdm

import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import RobertaTokenizer, RobertaForSequenceClassification

from dataset import IEMOCAP_Audio_Dataset, IEMOCAP_Text_Dataset
from model import Wav2VecClassifier
from utils import collate_fn, validation_metrics
from utils import apply_unstructured_pruning, apply_structured_pruning, apply_structured_pruning_to_encoder

def eval_model(args):
    dataset_dir = args.dataset_dir

    # Generate Train and Validation Set 

    # load the list from the JSON file and separate input and label data
    with open(os.path.join(dataset_dir, args.input_and_label_file), 'r') as json_file:
        input_and_label = json.load(json_file)

    valid_inputs = []
    valid_labels = []

    for input, label in input_and_label:
        valid_inputs.append(input)
        valid_labels.append(label)

    # separate data into train and valid datasets
    _, valid_inputs, _, valid_labels = train_test_split(valid_inputs, valid_labels, test_size=0.12, random_state=42, stratify=valid_labels)

    # load idx to label dictionary
    with open(os.path.join(dataset_dir, args.idx_to_label_file), 'r') as json_file:
        idx_2_label = json.load(json_file)

    batch_size = args.batch_size
    loss_fn = torch.nn.CrossEntropyLoss()
    device = args.device
    print("Using device: {}".format(device))

    if args.model_input == 'audio':
        # load processor and model
        processor = Wav2Vec2Processor.from_pretrained(args.pretrained_processor_path)
        model = Wav2Vec2ForCTC.from_pretrained(args.pretrained_model_path)
        model_path_url = args.model_path_url
        model_destination = args.model_destination

        if not os.path.exists(model_destination):
            wget.download(model_path_url, model_destination)

        num_classes = len(idx_2_label)
        model =  Wav2VecClassifier(model, num_classes)

        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_destination))
        else: 
            model.load_state_dict(torch.load(model_destination, map_location=torch.device('cpu')))

        # define validation dataset
        valid_dataset = IEMOCAP_Audio_Dataset(valid_inputs, valid_labels, processor)
        # create validation data loader
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    elif args.model_input == 'text':
        # define tokenizer and model
        tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_tokenizer_path)
        model = RobertaForSequenceClassification.from_pretrained(args.pretrained_model_path, num_labels=len(idx_2_label))
        
        valid_dataset = IEMOCAP_Text_Dataset(valid_inputs, valid_labels, tokenizer)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    elif args.model_input == 'audio_to_text':
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        audio_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        audio_model.to(device)

        tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_tokenizer_path)
        model = RobertaForSequenceClassification.from_pretrained(args.pretrained_model_path, num_labels=len(idx_2_label))

        valid_dataset = IEMOCAP_Audio_Dataset(valid_inputs, valid_labels, processor)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # apply pruning if specified
    if args.pruning_method == 'unstructured':
        apply_unstructured_pruning(model, args.pruning_percentage)
    elif args.pruning_method == 'structured':
        apply_structured_pruning(model, args.pruning_percentage)
    elif args.pruning_method == 'structured_to_encoder':
        apply_structured_pruning_to_encoder(model, args.pruning_percentage)

    # Evaluation loop
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
            if args.model_input == 'audio':
                inputs = batch['input_values'].to(device)
                labels = batch['label'].to(device)
                outputs = model(inputs)
            elif args.model_input == 'text':
                inputs = {key: val.to(device) for key, val in batch.items()}
                outputs = model(**inputs).logits
                labels = inputs['labels'].to(device)
            elif args.model_input == 'audio_to_text':
                inputs = batch['input_values'].to(device)
                labels = batch['label'].to(device)
                logits = audio_model(inputs).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.batch_decode(predicted_ids)
                encoding = tokenizer(transcription, truncation=True, padding=True, max_length=128)
                encoding['input_ids'] = torch.tensor(encoding['input_ids'])
                encoding['attention_mask'] = torch.tensor(encoding['attention_mask'])
                encoding['labels'] = torch.tensor(labels)
                outputs = model(**encoding).logits

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

    validation_metrics(valid_labels_list, valid_predictions_list, idx_2_label)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument Parser for Evaluating Emotion Recognition Model")
    parser.add_argument('--dataset_dir', type=str, default='Dataset/IEMOCAP', 
                        help="Directory containing the dataset")
    parser.add_argument('--input_and_label_file', type=str, default='audio_and_label.json', 
                        help="File containing input input and label data")
    parser.add_argument('--idx_to_label_file', type=str, default='idx_2_label.json', 
                        help="File containing index to label mapping")
    parser.add_argument('--model_input', type=str, default='audio', 
                        help="Specify the input format of the model, either audio, text, or audio_to_text")
    parser.add_argument('--pretrained_processor_path', type=str, default='eric0708/finetuned_wav2vec_audio_emotion_recognition', 
                        help="Path to pretrained processor")
    parser.add_argument('--pretrained_tokenizer_path', type=str, default='roberta-base', 
                        help="Path to pretrained tokenizer")
    parser.add_argument('--pretrained_model_path', type=str, default='facebook/wav2vec2-base-960h', 
                        help="Path to pretrained model")
    parser.add_argument('--model_path_url', type=str, default='https://huggingface.co/eric0708/finetuned_wav2vec_audio_emotion_recognition/resolve/main/model.pth', 
                        help="URL to download the pretrained model.pth")
    parser.add_argument('--model_destination', type=str, default='Models/model.pth', 
                        help="Destination path to save the downloaded model")
    parser.add_argument('--pruning_method', type=str, default=None, 
                        help="Pruning method applied to the model before evaluation, either unstructured, structured, structured_to_encoder")
    parser.add_argument('--pruning_percentage', type=float, default=0.33, 
                        help="Pruning percentage that is used to perform pruning")
    parser.add_argument('--device', type=str, default='cuda:0', 
                        help="Device to use for evaluation")
    parser.add_argument('--batch_size', type=int, default=1, 
                        help="Batch size for evaluation")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    eval_model(args)