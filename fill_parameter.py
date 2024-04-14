import os
import json

import torch

from Jsonformer import CustomJsonformer 
from utils import blockPrint, enablePrint

if __name__ == '__main__':

    print("Filling paramter...")

    blockPrint()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("/content/function_name.json") as f:
        function_name = json.load(f)
    
    prediction = function_name["prediction"]
    user_input = function_name["user_input"]

    if (prediction==0):
        function_name = "Matrix Multiplication of (2x2) by (2x2)"
        json_schema = {
            "type": "Matrix Multiplication of (2x2) by (2x2)",
            "properties": {
                "vector_1, x1": {"type": "number"},
                "vector_1, x2": {"type": "number"},
                "vector_1, x3": {"type": "number"},
                "vector_1, x4": {"type": "number"},
                "vector_2, x1": {"type": "number"},
                "vector_2, x2": {"type": "number"},
                "vector_2, x3": {"type": "number"},
                "vector_2, x4": {"type": "number"}
            } 
        }

    if (prediction==1):
        function_name = "Quadratic Equation"
        json_schema = {
            "type": "Quadratic Equation",
            "properties": {
                "constant of x^2": {"type": "number"},
                "constant of x": {"type": "number"},
                "scalar": {"type": "number"}}
            } 

    if (prediction==2):
        function_name = "Cubic Equation"
        json_schema = {
            "type": "Cubic Equation",
            "properties": {
                "constant of x^3": {"type": "number"},
                "constant of x^2": {"type": "number"},
                "constant of x": {"type": "number"},
                "scalar": {"type": "number"}}
            } 


    # Fill Parameter
    prompt = f'Context: "{user_input}". Generate constant of {function_name} based on the following schema'
    filler = CustomJsonformer(device=device).to(device)
    generated_data = filler(prompt=prompt, json_schema=json_schema)
    
    enablePrint()
    print("Done!")
 
    # Return result
    json_object = json.dumps([{"prediction": prediction}, generated_data])
    with open("/content/filled_parameter.json", "w") as outfile:
        outfile.write(json_object)