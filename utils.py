import os
import random
import re
import numpy as np
import math
import torch
import torch.nn.functional as F
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    AutoTokenizer,
    AutoModelForCausalLM,
)

# os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-6


def load_neighbour_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
    model.eval()
    return model, tokenizer


def load_detect_model():
    tokenizer = AutoTokenizer.from_pretrained("detected_model")
    model = AutoModelForCausalLM.from_pretrained(
        "detected_model", torch_dtype=torch.bfloat16
    ).to(device)
    return model, tokenizer


def load_models():
    print("Loading models...")
    model_0, tokenizer_0 = load_neighbour_model()
    model_1, tokenizer_1 = load_detect_model()
    models = [model_0, tokenizer_0, model_1, tokenizer_1]
    return models

def loss_func(input_ids, model):
    loss = 0
    for id in input_ids:
        # inputs = tokenizer(text, return_tensors='pt')
        outputs = model(input_ids=id[None, :], labels=id[None, :])
        loss += outputs.loss.item()
    return loss / len(input_ids)

def score_func(delta_loss, a = 15, b = -0.6):
    delta_loss = np.array(delta_loss)
    return 1 / (1 + np.exp(a * delta_loss - b))


def neighbour(model, tokenizer, text, threshold):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    # tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    # Get embeddings
    with torch.no_grad():
        embeddings = model.bert.embeddings.word_embeddings(input_ids)
    # Compute swap probabilities
    def swap_probability(p_word, p_swap_word):
        return p_swap_word / (1 - p_word + EPS)
    # Find top m replacements for each token
    top_replacements = []
    for i, id in enumerate(input_ids[0]):
        with torch.no_grad():
            embed=embeddings.clone()
            embed[0][i] = F.dropout(embed[0][i], p=0.5, training=True)
            outputs = model(inputs_embeds=embed).logits
            probs = F.softmax(outputs, dim=-1)
        p_word = probs[0, i, input_ids[0, i]].item()
        probs[0, i, input_ids[0, i].item()] = 0
        max_j = torch.argmax(probs[0, i]).item()
        top_replacements.append(
            (i, (max_j, swap_probability(p_word, probs[0, i, max_j].item())))
        )
    # Generate neighbors
    top_replacements.sort(key=lambda x: x[1][1], reverse=True)
    top_replacements = list(filter(lambda x: x[1][1] > threshold, top_replacements))
    return top_replacements, input_ids


def neighbourhood_generation(model, tokenizer, input_text, n, m, threshold):
    texts = re.split(r'(?<=[.?!])', input_text)
    texts = [text.strip() for text in texts if text.strip()]
    top_replacements=[]
    ids=[]
    token_len=0
    for i,text in enumerate(texts):
        top_replacement, token = neighbour(model, tokenizer, text, threshold)
        for x in top_replacement:
            top_replacements.append((x[0]+token_len-(1 if i!=0 else 0), x[1]))
        if i == 0:
            ids += token[0][:-1].tolist()
            token_len += len(token[0][:-1])
        elif i == len(texts)-1:
            ids += token[0][1:].tolist()
            token_len += len(token[0][1:])
        else:
            ids += token[0][1:-1].tolist()
            token_len += len(token[0][1:-1])
    # Generate neighbors
    neighbors = []
    top_replacements.sort(key=lambda x: x[1][1], reverse=True)
    top_replacements = list(filter(lambda x: x[1][1] > threshold, top_replacements))
    if len(top_replacements) == 0: return [input_text] * n
    for i in range(n):
        new_text = ids[:]
        replacements = top_replacements[i:i+1]
        for k, (j, _) in replacements:
            new_text[k] = j
        neighbors.append(tokenizer.decode(torch.tensor(new_text[1:-1])))
    return neighbors




# model_0, tokenizer_0 = load_neighbour_model()
# text = "The animal didn't cross the street because it was too tired. It was too tired to cross the street."
# n = 10  # number of neighbors
# m = 2  # number of word replacements
# neighbors = neighbourhood_generation(model_0, tokenizer_0, text, n, m, 0)
# for neighbor in neighbors:
#     print(neighbor)
