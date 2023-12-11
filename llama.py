#!/usr/bin/env python

from llama_cpp import Llama
import os
import json
from dotenv import load_dotenv

model = None

def create_model():
    model_path = os.getenv("MODEL_PATH", "./llama-2-7b-chat.Q2_K.gguf")
    # Create a llama model
    model = Llama(model_path=model_path)
    return model

def q(question: str):
    # Prompt creation
    system_message = "You are a helpful assistant"

    prompt = f"""<s>[INST] <<SYS>>
    {system_message}
    <</SYS>>
    {question} [/INST]"""

    # Model parameters
    max_tokens = 10000

    # Run the model
    output = model(prompt, max_tokens=max_tokens, echo=True)
    return output

def print_answer(answer):   
    print(answer["choices"][0]["text"])

def main():
    global model
    load_dotenv()
    model = create_model()

    answer = q("Make a JSON list of 3 random persons (with fields: name, age, gender, address, zip)")
    
    print_answer(answer)

if __name__ == '__main__':
    main()