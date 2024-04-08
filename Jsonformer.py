from transformers import AutoTokenizer, AutoModel
from jsonformer import Jsonformer

import torch
import torch.nn as nn

class CustomJsonformer(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b")
        self.model = AutoModel.from_pretrained("databricks/dolly-v2-3b")

        self.device = device
    
    def forward(self, prompt, json_schema):
        with torch.no_grad():
            jsonformer = Jsonformer(device=self.device)
            generated_data = jsonformer(self.model, self.tokenizer, prompt, json_schema)
        
        return generated_data
