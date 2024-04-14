import os
import json

import torch 
import torch.nn as nn

from SentenceEmbedding import SentenceEmbedding
from utils import blockPrint, enablePrint

if __name__ == '__main__':
    
    print("Preparing for Function Detecting...")
    blockPrint()
    # Detect suitable function from Toolkit 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding = SentenceEmbedding(device=device).to(device)

    # Loading Function Vector
    # Index 0 -> "Matrix Multiplication of (2x2) by (2x2)"
    # Index 1 -> "Quadratic Equation"
    # Index 2 -> "Cubic Equations"
    if os.path.isfile("/content/Prototype_v1/FunctionVectors.json"):
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
    enablePrint()
    print("Done!")
    user_input = input("Please math problem: ")
    user_input_vector = embedding(user_input)

    # Cosine similarity 
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    output = cos(user_input_vector, function_vectors)
    prediction = torch.argmax(output).item()

    # Return result
    json_object = json.dumps({"prediction": prediction, "user_input": user_input})
    with open("/content/function_name.json", "w") as outfile:
        outfile.write(json_object)