import os
import random
import re
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
    tokenizer = BertTokenizer.from_pretrained("bert-base")
    model = BertForMaskedLM.from_pretrained("bert-base").to(device)
    model.eval()
    return model, tokenizer


def load_detect_model():
    tokenizer = AutoTokenizer.from_pretrained("detected_model")
    model = AutoModelForCausalLM.from_pretrained(
        "detected_model", torch_dtype=torch.bfloat16
    ).to(device).eval()
    return model, tokenizer


def loss_func(input_ids, model):
    loss = 0
    for id in input_ids:
        # inputs = tokenizer(text, return_tensors='pt')
        outputs = model(input_ids=id[None, :], labels=id[None, :])
        loss += outputs.loss.item()
    return loss / len(input_ids)


def neighbour(model, tokenizer, text, n, m, threshold):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    # Get embeddings
    with torch.no_grad():
        embeddings = model.bert.embeddings.word_embeddings(input_ids)
    # Compute swap probabilities
    def swap_probability(p_word, p_swap_word):
        return p_swap_word / (1 - p_word + EPS)
    # Find top m replacements for each token
    top_replacements = []
    for i, token in enumerate(tokens):
        if token in ["[CLS]", "[SEP]"]:
            continue
        with torch.no_grad():
            embeddings[0][i] = F.dropout(embeddings[0][i], p=0.5, training=True)
            outputs = model(inputs_embeds=embeddings).logits
            probs = F.softmax(outputs, dim=-1)
        p_word = probs[0, i, input_ids[0, i]].item()
        probs[0, i, input_ids[0, i].item()] = 0
        max_j = torch.argmax(probs[0, i]).item()
        top_replacements.append(
            (i, (max_j, swap_probability(p_word, probs[0, i, max_j].item())))
        )
    # Generate neighbors
    neighbors = []
    top_replacements.sort(key=lambda x: x[1][1], reverse=True)
    top_replacements = list(filter(lambda x: x[1][1] > threshold, top_replacements))
    if len(top_replacements) == 0: return [text] * n
    for i in range(n):
        new_text = tokens[:]
        replacements = random.choices(top_replacements, k=m, weights=[top_replacements[i][1][1] for i in range(len(top_replacements))])
        for k, (j, _) in replacements:
            new_text[k] = tokenizer.convert_ids_to_tokens(j)
        neighbors.append(" ".join((new_text)[1:-1]).replace(" ##", ""))
    return neighbors


def neighbourhood_generation(model, tokenizer, input_text, n, m, threshold):
    texts = re.split(r'[.?!":]', input_text)
    texts = [text.strip() for text in texts if text.strip()]
    neighbours = [""] * n
    for text in texts:
        neighbours_part = neighbour(model, tokenizer, text, n, 1, threshold)
        for i in range(n):
            neighbours[i] += neighbours_part[i]
    return neighbours


# model_0, tokenizer_0 = load_neighbour_model()
# text = "The animal didn't cross the street because it was too tired."
# n = 5  # number of neighbors
# m = 2  # number of word replacements
# neighbors = neighbourhood_generation(model_0, tokenizer_0, text, n, m)
# for neighbor in neighbors:
#     print(neighbor)
