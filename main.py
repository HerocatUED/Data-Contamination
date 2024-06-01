import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from utils import *


def inference(models, data):
    """predict whether the data is clean:0 or dirty:1.
    Output: probability of dirty.
    """
    model_0, tokenizer_0, model_1, tokenizer_1 = models
    n, m, threshold = 100, 10, 0.8
    result = np.zeros(len(data))
    for i in tqdm(range(len(data))):
        neighbours = neighbourhood_generation(model_0, tokenizer_0, data[i], n, m, threshold)
        assert len(neighbours) < n
        loss = loss_func(data[i], model_1, tokenizer_1)
        mean_loss = np.mean(loss_func(neighbours, model_1, tokenizer_1))
        # TODO
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
        label = val_data[:]['label'] == 'dirty'
        label = np.array(label)
        predict = inference(models, val_data[:]['text'])
        score = roc_auc_score(label, np.hstack(predict, 1-predict))
        print(score)
    # test
    print('testing')
    with open("dataset/test.json", "r", encoding='utf-8') as f:
        test_data = json.load(f)
        predict = inference(test_data[:]['text'])
    with open("dataset/output.json", "w", encoding='utf-8') as f:
        output = list({'text': 'none', 'score': 'none'}) * len(test_data)
        for i, data, score in enumerate(zip(test_data[:]['text'], predict)):
            output[i]['text'] = data
            output[i]['score'] = score
        json.dump(output)
            
            
if __name__ == '__main__':
    main()