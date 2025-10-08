import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig

class TextEncoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        config = DistilBertConfig()
        self.bert = DistilBertModel(config)
        self.projection = nn.Linear(768, output_dim)
        
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = output.last_hidden_state[:, 0, :]
        projected = self.projection(cls_embedding)
        return projected
