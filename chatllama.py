#!/usr/bin/env python

from llama_cpp import Llama
import os
import json
import atexit
import sys
import time
import readline
from dotenv import load_dotenv

model = None

def create_model():
    model_path = os.getenv("MODEL_PATH", "./llama-2-7b-chat.Q2_K.gguf")
    # Create a llama model
    model = Llama(model_path=model_path)
    return model

def init_readline(path=None):
    path = path or '/tmp/.readline-history.txt'
    try:
        readline.read_history_file(path)
        # default history len is -1 (infinite), which may grow unruly
        readline.set_history_length(1000)
    except IOError:
        pass

    atexit.register(readline.write_history_file, path)


def q(question: str):
    # Prompt creation
    system_message = "You are a helpful assistant"
    # system_message = "You are an AI that strictly conforms to responses in JSON formatted strings. Your responses consist ONLY of valid JSON syntax, with no other comments, explainations, reasoninng, or dialogue not consisting of valid JSON."


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
    init_readline()

    try:
        q1 = sys.argv[1]
    except IndexError:
        q1 = "Generate list of 100 first names. Give only JSON in answer."

    print("Your first question:\n>>> ")

    while True:
        question = input(">>> ")
        t = time.time()
        answer = q(question)
        print(f"# time: {time.time() - t:.2f}")

        # print(answer)
        
        print(json_answer(answer))

    answer = q(q1)
    
    print(json_answer(answer))

if __name__ == '__main__':
    main()