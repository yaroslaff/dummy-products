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
    max_tokens = 1000000

    # Run the model
    output = model(prompt, max_tokens=max_tokens, echo=False, temperature=0.5)
    return output

def json_answer(answer):   
    return answer["choices"][0]["text"]

def main():
    global model
    load_dotenv()
    model = create_model()

    q1 = "Generate list of 20 top-level categories for marketplace, output in JSON"

    answer = q(q1)
    
    print(json_answer(answer))

if __name__ == '__main__':
    main()