[TensorFlow 2] ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network
=====

Unofficial TensorFlow implementation of "ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network".  
Official PyTorch implementation is provided as the following link.  
https://github.com/clovaai/rexnet  

* Note that, the detailed neural network architecture is different with the original paper [1].  
* This repository only contains the shorten neural network reflecting the concept.  

## Concept
<div align="center">
  <img src="./figures/rexnet.png" width="600">  
  <p>Spec of ReXNet-v1 [1].</p>
</div>

## Performance

|Indicator|Value|
|:---|:---:|
|Accuracy|0.98180|
|Precision|0.98188|
|Recall|0.98177|
|F1-Score|0.98171|

```
Confusion Matrix
[[ 974    0    0    0    0    0    4    1    0    1]
 [   2 1120    1    1    1    1    2    2    3    2]
 [   6    0 1008    0    4    0    0    4    7    3]
 [   2    0    5  968    0   10    0    3    6   16]
 [   1    0    0    0  964    0    4    0    0   13]
 [   2    0    0    4    1  874    3    1    0    7]
 [   6    2    0    0    1    4  943    0    1    1]
 [   0    1    4    1    2    0    0 1010    1    9]
 [   7    0    2    0    0    2    0    1  954    8]
 [   0    2    0    1    1    2    0    0    0 1003]]
Class-0 | Precision: 0.97400, Recall: 0.99388, F1-Score: 0.98384
Class-1 | Precision: 0.99556, Recall: 0.98678, F1-Score: 0.99115
Class-2 | Precision: 0.98824, Recall: 0.97674, F1-Score: 0.98246
Class-3 | Precision: 0.99282, Recall: 0.95842, F1-Score: 0.97531
Class-4 | Precision: 0.98973, Recall: 0.98167, F1-Score: 0.98569
Class-5 | Precision: 0.97872, Recall: 0.97982, F1-Score: 0.97927
Class-6 | Precision: 0.98640, Recall: 0.98434, F1-Score: 0.98537
Class-7 | Precision: 0.98826, Recall: 0.98249, F1-Score: 0.98537
Class-8 | Precision: 0.98148, Recall: 0.97947, F1-Score: 0.98047
Class-9 | Precision: 0.94356, Recall: 0.99405, F1-Score: 0.96815

Total | Accuracy: 0.98180, Precision: 0.98188, Recall: 0.98177, F1-Score: 0.98171
```

## Requirements
* Python 3.7.6  
* Tensorflow 2.1.0  
* Numpy 1.18.1  
* Matplotlib 3.1.3  

## Reference
[1] Dongyoon Han et al. (2020). <a href="https://arxiv.org/abs/2007.00992">ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network</a>. arXiv preprint arXiv:2007.00992.
