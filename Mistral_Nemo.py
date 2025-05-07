from transformers import pipeline
import numpy as np
# from huggingface_hub import login

# login(token="your_access_token")

# messages = [
#     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#     {"role": "user", "content": "Who are you?"},
# ]
# chatbot = pipeline("text-generation", model="mistralai/Mistral-Nemo-Instruct-2407",max_new_tokens=128)
# chatbot(messages)

sentence_data = np.load("data/sentence_puzzle.npy", allow_pickle=True)

word_data = np.load("data/word_puzzle.npy", allow_pickle=True)

print(sentence_data[0])
print(len(sentence_data))
print(word_data[0])
print(len(word_data))
