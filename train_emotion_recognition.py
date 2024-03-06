# Import Libraries
import os
import json
import numpy as np
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, AdamW
from transformers import RobertaTokenizer, RobertaForSequenceClassification

from utils import collate_fn
from dataset import IEMOCAP_Audio_Dataset, IEMOCAP_Text_Dataset
from model import Wav2VecClassifier

def train_model(args):
    # Generate Train and Validation Set 

    # load the list from the JSON file and separate text and label data
    with open(os.path.join(args.dataset_dir, args.input_and_label_file), 'r') as json_file:
        input_and_label = json.load(json_file)

    input_data = []
    label_data = []

    for input, label in input_and_label:
        input_data.append(input)
        label_data.append(label)

    # separate data into train and valid datasets
    train_input, valid_input, train_labels, valid_labels = train_test_split(input_data, label_data, test_size=0.12, random_state=42, stratify=label_data)

    # load idx to label dictionary
    with open(os.path.join(args.dataset_dir, args.idx_to_label_file), 'r') as json_file:
        idx_2_label = json.load(json_file)

    if args.model_input == 'audio':
        # load processor and model
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

        # create new model to add classification head to original model 
        num_classes = len(idx_2_label)
        model =  Wav2VecClassifier(model, num_classes)

        # define dataset
        train_dataset = IEMOCAP_Audio_Dataset(train_input, train_labels, processor)
        valid_dataset = IEMOCAP_Audio_Dataset(valid_input, valid_labels, processor)

        # define dataloader
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    elif args.model_input == 'text':
        # define tokenizer and model
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(idx_2_label))

        # define dataset
        train_dataset = IEMOCAP_Text_Dataset(train_input, train_labels, tokenizer)
        valid_dataset = IEMOCAP_Text_Dataset(valid_input, valid_labels, tokenizer)

        # define dataloader
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Finetune Model

    # define training parameters
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs

    # set up optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    # training loop
    device = args.device
    model.to(device)

    for epoch in range(num_epochs):
        # train
        print("Start Train for Epoch {}".format(epoch+1))
        model.train()
        train_loss = 0
        train_correct_preds = 0
        for batch in tqdm(train_dataloader):
            if args.model_input == 'audio':
                inputs = batch['input_values'].to(device)
                labels = batch['label'].to(device)
                outputs = model(inputs)
            elif args.model_input == 'text':
                inputs = {key: val.to(device) for key, val in batch.items()}
                outputs = model(**inputs).logits
                labels = inputs['labels'].to(device)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # calculate train accuracy
            _, train_predictions = torch.max(outputs, dim=1)
            train_correct_preds += torch.sum(train_predictions == labels).item()

        # validation
        print("Start Validation for Epoch {}".format(epoch+1))
        model.eval()
        valid_loss = 0
        valid_correct_preds = 0
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
                loss = loss_fn(outputs, labels)
                valid_loss += loss.item()

                # calculate validation accuracy
                _, valid_predictions = torch.max(outputs, dim=1)
                valid_correct_preds += torch.sum(valid_predictions == labels).item()

        avg_train_loss = train_loss / len(train_dataloader)
        train_accuracy = train_correct_preds / len(train_dataset)
        avg_valid_loss = valid_loss / len(valid_dataloader)
        valid_accuracy = valid_correct_preds / len(valid_dataset)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Valid Loss: {avg_valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}\n")

    # Save Model

    # specify the directory to save the models
    model_dir = args.model_dir

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    finetuned_model_dir = os.path.join(model_dir, args.checkpoint_dir)

    # save model and tokenizer to specified directory
    model.save_pretrained(finetuned_model_dir)
    processor.save_pretrained(finetuned_model_dir)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument Parser for Training Emotion Recognition Model")
    parser.add_argument('--dataset_dir', type=str, default='Dataset/IEMOCAP', 
                        help="Directory containing the dataset")
    parser.add_argument('--input_and_label_file', type=str, default='audio_and_label.json', 
                        help="File containing input and label data")
    parser.add_argument('--idx_to_label_file', type=str, default='idx_2_label.json', 
                        help="File containing index to label mapping")
    parser.add_argument('--model_input', type=str, default='audio', 
                        help="Specify the input format of the model, either audio or text")
    parser.add_argument('--device', type=str, default='cuda:0', 
                        help="Device to use for training")
    parser.add_argument('--model_dir', type=str, default='Models', 
                        help="Directory to save models")
    parser.add_argument('--checkpoint_dir', type=str, default='finetuned_wav2vec_IEMOCAP', 
                        help="Directory to save model checkpoints")
    parser.add_argument('--batch_size', type=int, default=4, 
                        help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, default=1e-5, 
                        help="Learning rate for training")
    parser.add_argument('--num_epochs', type=int, default=5, 
                        help="Number of epochs for training")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    train_model(args)