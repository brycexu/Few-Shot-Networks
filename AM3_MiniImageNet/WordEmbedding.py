# Author : Bryce Xu
# Time : 2019/12/31
# Function:

import io
import numpy as np

def load_model(model_addr):
    fin = io.open(model_addr, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        tmpv = [float(tokens[i]) for i in range(1, len(tokens))]
        data[tokens[0]] = tmpv
    return data

def load_embeddings(label_names, embeddings):
    emb_list = []
    for name in label_names:
        labels = name.split('_')
        tmpv = np.zeros(100)
        tmpl = []
        c = 0
        for l in labels:
            if l in embeddings.keys():
                tmpv += embeddings[l]
                tmpl.append(l)
                c += 1
        if len(labels) != 1:
            if c != len(labels):
                print(name, c, tmpl)
        if c != 0:
            emb_list.append(np.float32(tmpv / c))
        else:
            emb_list.append(np.float32(tmpv))
    return emb_list