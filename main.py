import os
import json

import torch
import torch.nn as nn

from SentenceEmbedding import SentenceEmbedding

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding = SentenceEmbedding().to(device)

    # Loading Function Vector
    # Index 0 -> "Matrix Multiplication of (2x2) by (2x2)"
    # Index 1 -> "Quadratic Equation"
    # Index 2 -> "Cubic Equations"
    if os.path.isfile("/content/FunctionVectors.json"):
        with open("/content/FunctionVectors.json") as f:
            function_vectors = json.load(f)
    else:
        functions_list = ["Matrix Multiplication of (2x2) by (2x2)",
                          "Quadratic Equation",
                          "Cubic Equations"]
        function_vectors = embedding(functions_list)

        json_object = json.dumps(function_vectors)
        with open("/content/FunctionVectors.json", "w") as outfile:
            outfile.write(json_object)
    
    user_input = input("Please provide answer of math problem: ")
    user_input_vector = embedding(user_input)

    # Cosine similarity 
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    output = cos(user_input_vector, function_vectors)
    prediction = torch.argmax(output)
    
    if (prediction==0):
        print("Matrix Multiplication of (2x2) by (2x2)")
    if (prediction==1):
        print("Quadratic Equation")
    if (prediction==1):
        print("Cubic Equation")