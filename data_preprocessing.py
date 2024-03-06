import os
import json
import argparse

from utils import id_2_text_construct, id_2_audio_construct, id_2_label_construct

def IEMOCAP_data_preprocessing(args):
    dataset_dir = args.dataset_dir 
    
    # initialize dictionaries
    id_2_audio = {}
    id_2_label = {}

    # iterate through all the session directories
    for session_num in range(1, 6):
        session_dir = '{}/Session{}'.format(dataset_dir, session_num)
        transcriptions_dir = os.path.join(session_dir, 'dialog/transcriptions')
        audio_dir = os.path.join(session_dir, 'sentences/wav')
        evaluation_dir = os.path.join(session_dir, 'dialog/EmoEvaluation')

        id_2_text = id_2_text_construct(id_2_text, transcriptions_dir)
        id_2_audio = id_2_audio_construct(id_2_audio, audio_dir)
        id_2_label = id_2_label_construct(id_2_label, evaluation_dir)

    # iterate through all id_2_label sentence ids and locate corresponding text
    idx_2_label = {}
    label_2_idx = {}
    text_and_label = []
    audio_and_label = []

    idx_count = 0

    for sentence_id in id_2_label:
        label = id_2_label[sentence_id]
        text = id_2_text[sentence_id]
        audio = id_2_audio[sentence_id]

        if label not in label_2_idx:
            label_2_idx[label] = idx_count
            idx_2_label[idx_count] = label
            idx_count += 1
        
        text_and_label.append((text, label_2_idx[label]))
        audio_and_label.append((audio, label_2_idx[label]))

    # save the dictionaries and lists to dataset directory
    with open(os.path.join(dataset_dir, args.idx_to_label_file), 'w') as json_file:
        json.dump(idx_2_label, json_file)

    with open(os.path.join(dataset_dir, args.label_to_idx_file), 'w') as json_file:
        json.dump(label_2_idx, json_file)

    with open(os.path.join(dataset_dir, args.text_and_label_file), 'w') as json_file:
        json.dump(text_and_label, json_file)

    with open(os.path.join(dataset_dir, args.audio_and_label_file), 'w') as json_file:
        json.dump(audio_and_label, json_file)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument Parser for Dataset Preprocessing")
    parser.add_argument('--dataset_dir', type=str, default='Dataset/IEMOCAP', 
                        help="Directory containing the dataset")
    parser.add_argument('--label_to_idx_file', type=str, default='label_2_idx.json', 
                        help="File containing label to indexed label mapping")
    parser.add_argument('--idx_to_label_file', type=str, default='idx_2_label.json', 
                        help="File containing label to indexed label mapping")
    parser.add_argument('--text_and_label_file', type=str, default='text_and_label.json', 
                        help="File containing paired text and label data")
    parser.add_argument('--audio_and_label_file', type=str, default='audio_and_label.json', 
                        help="File containing paired audio and label data")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    IEMOCAP_data_preprocessing(args)