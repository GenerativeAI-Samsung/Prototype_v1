from transformers import AutoTokenizer, AutoModelForCausalLM
from jsonformer import Jsonformer

import torch
import torch.nn as nn

class CustomJsonformer(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b")
        self.model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-3b")

        self.device = device
    
    def forward(self, prompt, json_schema):
        with torch.no_grad():
            jsonformer = Jsonformer(self.model, self.tokenizer, json_schema, prompt)
            generated_data = jsonformer()
        
        return generated_data
