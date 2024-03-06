# EmoSense 

😊😃🥲🤨😕

Welcome to EmoSense! This repository contains code for emotion recognition using audio and text based transformer models with pruning support. Presentation slides to the project can be viewed [here](https://docs.google.com/presentation/d/11L6VZKmYdf0F7yfEugzStx8l-P25POuB8DaTz-pau44/edit?usp=sharing).

## Description
EmoSense is a powerful tool designed to recognize emotions from audio and text data. Leveraging state-of-the-art transformer models, it offers accurate predictions for a wide range of applications. EmoSense also integrates pruning support into the pipeline to test the effect of pruning on emotion recognition performance. 

## File Structure

EmoSense/  
│    
│── README.md    
│── data_preprocessing.py  
│── dataset.py    
│── inference_emotion_recognition.py    
│── model.py   
│── requirements.txt  
│── train_emotion_recognition.py    
│── utils.py   

## Contents
- `data_preprocessing.py`: Script for preprocessing the dataset, including audio and text feature extraction along with the labels. The sample script is desgined to be used on the 'IEMOCAP' dataset and user defined datasets may have to be used in place of other datasets.
- `dataset.py`: Contains classes for dataset handling, including data loading and transformations for the IEMOCAP audio and text datasets
- `model.py`: Contains the emotion recognition model architectures for the audio and text based transformer models. 
- `train_emotion_recognition.py`: Script for training the emotion recognition model, including data splitting, model training, and validation.
- `inference_emotion_recognition.py`: Inference script to make predictions using trained emotion recognition models on validation audio or text samples. To use with own samples, self defined files with paired paths to audio or text files and labels has to be provided. Three kinds of inference are supported, predict using audio, predict using text, and predict using audio by transforming audio to text first. 
- `requirements.txt`: List of Python dependencies required to run the code.
- `utils.py`: Contains utility functions used across the codebase, including helper functions for data preprocessing and model evaluation.

## Installation

To run the code in this repository, you'll need to install the dependencies specified in `requirements.txt`:

You can install the required packages using pip:
```
pip install -r requirements.txt
```

## Usage
To use the code in this repository, follow these steps:

1. **Dataset Preparation**:  
Prepare the all the training data under a self defined directory, if the IEMOCAP dataset is to be used, please apply for access to the dataset first and store the 'IEMOCAP' directory under a 'Data' directory to match the default paths defined in the codebase. Use the `data_preprocessing.py` script to preprocess the dataset. The code can be run directly to use the default arguments provided in the script. By defalut, all the files that will be used during training and evaluation will be stored in the same directory as the data. 
```
python data_preprocessing.py
```

2. **Model Training**:  
Train the model using the `train_emotion_recognition.py` script. The list of arguments are included in the argument parser of the script. Adjust the hyperparameters as needed and specify the dataset directory. The code can be run directly to use the default arguments provided in the script. By default, an audio based model will be trained on the IEMOCAP dataset.
```
python train_emotion_recognition.py
```

3. **Inference**:  
Use the trained model for inference using the `inference_emotion_recognition.py` script. The list of arguments are included in the argument parser of the script. The code can be run directly to use the default arguments provided in the script. By default, an audio based model will be evaluated and inferenced on the IEMOCAP dataset. Furthermore, the pruning method can also be specified through arguments and the default is to not use pruning at all. 
```
python inference_emotion_recognition.py
```

## Contributing
Contributions to EmoSense are welcome! If you'd like to contribute, please follow these guidelines:

- Fork the repository
- Create a new branch for your feature or bug fix
- Make your changes and commit them with descriptive messages
- Push your changes to your fork
- Submit a pull request to the main repository

Feel free to reach out at [ericsuece@cmu.edu](mailto:ericsuece@cmu.edu) for any questions, discussions, or collaboration opportunities!

Happy emoting! 😊