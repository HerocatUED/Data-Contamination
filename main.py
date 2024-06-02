import json
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from utils import *
import matplotlib.pyplot as plt


def inference(models, data):
    """predict whether the data is clean:0 or dirty:1.
    Output: probability of dirty.
    """
    model_0, tokenizer_0, model_1, tokenizer_1 = models
    n, m, threshold = 10, 1, 0.9
    result = np.zeros(len(data))
    losses = np.zeros(len(data))
    delta_loss = np.zeros(len(data))
    with open("logs/loss_log1.txt", "w", encoding="utf-8") as f:
        f.write("loss log\n")
    for i in tqdm(range(len(data))):
        neighbours = neighbourhood_generation(model_0, tokenizer_0, data[i], n, m, threshold)
        assert len(neighbours) <= n
        loss = loss_func(tokenizer_1(data[i], return_tensors="pt").input_ids.to(device), model_1)
        batch = [
            tokenizer_1(neighbour, return_tensors="pt").input_ids[0].to(device)
            for neighbour in neighbours
        ]
        mean_loss = loss_func(batch, model_1)
        # TODO
        with open("logs/loss_log1.txt", "a", encoding="utf-8") as f:
            f.write(f"index: {i} loss: {loss} mean neighbour loss: {mean_loss}\n")
        if loss - mean_loss < -0.6:
            result[i] = 1
        delta_loss[i] = loss - mean_loss
        losses[i] = loss
        np.savez("logs/record1.npz", loss=losses, delta_loss=delta_loss)
    return result, delta_loss


def validation(models):
    # validation
    print("Validating...")
    with open("dataset/valid.json", "r", encoding="utf-8") as f:
        val_data = json.load(f)
        val_data = val_data[:200] + val_data[-200:]
        label = [data["label"] == "dirty" for data in val_data]
        label = np.array(label, dtype=int)
        predict, delta_loss = inference(models, [data["text"] for data in val_data])
        plt.scatter(x=delta_loss, y=label, s=5)
        plt.savefig("logs/delta_loss1.png")
        score = roc_auc_score(label, score_func(delta_loss))
        print(score)
        
        
def test(models):
    # test
    print("testing")
    with open("dataset/test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
        test_text = [data["text"] for data in test_data]
        predict, delta_loss = inference(models, test_text)
    with open("dataset/output.json", "w", encoding="utf-8") as f:
        output = []
        for data, d_loss in zip(test_text, delta_loss):
            tmp = {}
            tmp["text"] = data
            tmp["score"] = score_func(d_loss)
            output.append(tmp)
        json.dump(output, f, ensure_ascii=False, indent=4)


def plot_loss():
    print("Loading model")
    model_1, tokenizer_1 = load_detect_model()
    with open("dataset/valid.json", "r", encoding="utf-8") as f:
        val_data = json.load(f)
        label = [data["label"] == "dirty" for data in val_data]
        label = np.array(label, dtype=int)
        loss = np.zeros_like(label, dtype=float)
        for i in tqdm(range(len(val_data))):
            input_ids = tokenizer_1(
                val_data[i]["text"], return_tensors="pt"
            ).input_ids.to(device)
            loss[i] = loss_func(input_ids, model_1)
        plt.scatter(x=loss, y=label, s=5)
        plt.savefig("logs/loss.png")


if __name__ == "__main__":
    models = load_models()
    # validation(models)
    test(models)
    # plot_loss()
