import torch
import torchaudio
from torch.utils.data import Dataset

# define dataset for IEMOCAP audio emotion classification
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
        waveform, _ = torchaudio.load(audio_path, normalize=True)
        inputs = {"input_values": self.processor(waveform.squeeze().numpy(), return_tensors="pt", max_length=self.max_length, sampling_rate=16000).input_values[0]}
        inputs["label"] = self.labels[idx]

        return inputs
    
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