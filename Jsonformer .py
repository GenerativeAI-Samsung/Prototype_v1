from transformers import AutoTokenizer, AutoModel
from jsonformer import Jsonformer

import torch
import torch.nn as nn

class Jsonformer(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b")
        self.model = AutoModel.from_pretrained("databricks/dolly-v2-12b")

        self.device = device
    
    def forward(self, prompt, json_schema):
        with torch.no_grad():
            jsonformer = Jsonformer(self.model, self.tokenizer, prompt, json_schema)
            generated_data = jsonformer()
        
        return generated_data
