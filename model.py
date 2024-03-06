import torch
import torch.nn as nn

class Wav2VecClassifier(nn.Module):
    def __init__(self, wav2vec_model, num_classes):
        super(Wav2VecClassifier, self).__init__()
        self.wav2vec_model = wav2vec_model
        self.classification_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, num_classes)
        )

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.wav2vec_model(input_ids).logits
        pooled_output = torch.mean(hidden_states, dim=1)
        logits = self.classification_head(pooled_output)

        return logits