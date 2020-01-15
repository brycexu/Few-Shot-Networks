## Prototype Network

### Evaluation on Omniglot with Nc=60 and Ns=5

|                  | 5-way 1-shot(Acc)  | 5-way 5-shot(Acc)  | 20-way 1-shot(Acc)| 20-way 5-shot(Acc)   |
| ---------------- |:------------------:|:------------------:|:-----------------:|:--------------------:|
| Paper            | 98.8%              | 99.7%              | 96.0%             | 98.9%                |
| Ours             | 98.4%              | 99.6%              | 94.7%             | 98.6%                |

### Evaluation on Omniglot with Nc=60 and Changable Ns

|                  | Ns = 1 (Acc)       | Ns = 3 (Acc)       | Ns = 5 (Acc)      | Ns = 10 (Acc)        | Ns = 15 ( Acc)     |
| ---------------- |:------------------:|:------------------:|:-----------------:|:--------------------:|:------------------:|
| 5-way-5-shot     | 99.61%             | 99.62%             | 99.60%            | 99.53%               | 99.52%             |
| 20-way-5-shot    | 98.60%             | 98.55%             | 98.53%            | 98.33%               | 98.38%             |

### Evaluation on Omniglot with Changable Nc and Ns=5

|                  | Nc = 20 (Acc)      | Nc = 40 (Acc)      | Nc = 60 (Acc)     | Nc = 80 (Acc)        | Ns = 100 ( Acc)    |
| ---------------- |:------------------:|:------------------:|:-----------------:|:--------------------:|:------------------:|
| 5-way-5-shot     | 99.51%             | 99.57%             | 99.54%            | 99.61%               | 99.62%             |
| 20-way-5-shot    | 98.10%             | 98.45%             | 98.44%            | 98.53%               | 98.58%             |

## Semantic-based Inter- and Intra- Class Feature Extractor

Backbone: 4 convolutional networks

Dataset: Mini-ImageNet

|                           | 5-way-5-shot       |
| ------------------------- |:------------------:|
| PN                        | 64.34%             |
| CTM                       | 66.86%             |
| Ours                      | 68.39%             |
