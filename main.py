import os
import json

import torch
import torch.nn as nn

from SentenceEmbedding import SentenceEmbedding
from Jsonformer import Jsonformer 

if __name__ == '__main__':

    # Detect suitable function from Toolkit 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding = SentenceEmbedding(device=device).to(device)

    # Loading Function Vector
    # Index 0 -> "Matrix Multiplication of (2x2) by (2x2)"
    # Index 1 -> "Quadratic Equation"
    # Index 2 -> "Cubic Equations"
    if os.path.isfile("/content/FunctionVectors.json"):
        with open("/content/FunctionVectors.json") as f:
            function_vectors = json.load(f)
        function_vectors = torch.tensor(function_vectors).to(device)
    else:
        functions_list = ["Matrix Multiplication of (2x2) by (2x2)",
                          "Quadratic Equation",
                          "Cubic Equations"]
        function_vectors = embedding(functions_list)

        json_object = json.dumps(function_vectors.tolist())
        with open("/content/FunctionVectors.json", "w") as outfile:
            outfile.write(json_object)
    
    user_input = input("Please provide answer of math problem: ")
    user_input_vector = embedding(user_input)

    # Cosine similarity 
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    output = cos(user_input_vector, function_vectors)
    prediction = torch.argmax(output).item()
    
    if (prediction==0):
        print("Function: Matrix Multiplication of (2x2) by (2x2)")
        json_schema = {
            "properties": {
                "vector_1": {"type": "array"},
                "vector_2": {"type": "array"}
            } 
        }

    if (prediction==1):
        print("Function: Quadratic Equation")
        json_schema = {
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            } 
        }

    if (prediction==2):
        print("Function: Cubic Equation")
        json_schema = {
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
                "c": {"type": "number"}
            } 
        }

    # Fill Parameter
    filler = Jsonformer(device=device).to(device)
    generated_data = filler(prompt=user_input, json_schema=json_schema)
    print(f"Generated_data: {generated_data}")