import json
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from utils import *


def inference(models, data):
    """predict whether the data is clean:0 or dirty:1.
    Output: probability of dirty.
    """
    model_0, tokenizer_0, model_1, tokenizer_1 = models
    n, m, threshold = 10, 10, 0.8
    result = np.zeros(len(data))
    with open("loss_log.txt", "w", encoding='utf-8') as f:
        f.write('loss log\n')
    for i in tqdm(range(len(data))):
        neighbours = neighbourhood_generation(model_0, tokenizer_0, data[i], n, m, threshold)
        assert len(neighbours) <= n
        loss = loss_func(tokenizer_1(data[i],return_tensors='pt').input_ids.to(device), model_1)
        mean_loss = loss_func(torch.tensor([tokenizer_1.convert_tokens_to_ids(neighbour) for neighbour in neighbours],device=device), model_1)
        # TODO
        with open("loss_log.txt", "a", encoding='utf-8') as f:
            f.write(f'index: {i} loss: {loss} mean neighbour loss: {mean_loss}\n')
        if loss - mean_loss < - 0.2:
            result[i] = 1
    return result


def main():
    # load model
    print('Loading models...')
    model_0, tokenizer_0 = load_neighbour_model()
    model_1, tokenizer_1 = load_detect_model()
    models = [model_0, tokenizer_0, model_1, tokenizer_1]
    # validation
    print("Validating...")
    with open("dataset/valid.json", "r", encoding='utf-8') as f:
        val_data = json.load(f)
        val_data=val_data[:50]+val_data[-50:]
        label = [data['label']=='dirty' for data in val_data]
        label = np.array(label)
        predict = inference(models, [data['text'] for data in val_data])
        score = roc_auc_score(label, np.hstack(predict, 1-predict))
        print(score)
    # test
    print('testing')
    with open("dataset/test.json", "r", encoding='utf-8') as f:
        test_data = json.load(f)
        test_text = [data['text'] for data in test_data]
        predict = inference(models, test_text)
    with open("dataset/output.json", "w", encoding='utf-8') as f:
        output = [{'text': 'none', 'score': 'none'}] * len(test_data)
        for i, data, score in enumerate(zip(test_text, predict)):
            output[i]['text'] = data
            output[i]['score'] = score
        json.dump(output, f)
            
            
if __name__ == '__main__':
    main()