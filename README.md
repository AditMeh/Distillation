### Interesting results
------
This table is for a model trained on classes 0-5, tested on classes 0-9. THe student network (all dense layers), has the following architecture:

``` 28*28 input -> layer 1 -> 14*14 output -> Relu -> Layer 2 -> (*, 10) output```

|  lr | T  | Epochs  | Weight  | 0  |  1 | 2  |  3 |  4 |  5 | 6  | 7  | 8  | 9  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  0.01 | 10.0  | 70  | 0.9  | 99.49%| 99.73% | 99.42% | 99.60% | 99.6% | 99.1% | 98.64% | 96% | 97.33% | 97.23% |

More tables to come. But as you can see above, the student model is able to learn how to classify images of a class that it did not see during training. Through these experiments, I have found that temperature values between 8-12 work the best for this task, along with a learning rate of 0.01 as 0.001 is too small.


So far:

- Distillation loss using torch 1.10 soft-label cross-entropy loss (Kullback-Leibler is the next best option)
- Teacher and student models setup, along with their respective training loops
- Visualization.
- Custom dataloaders that leave out certain classes (to test implicit learning from soft labels)
- Train teacher network. 14-16 errors on the val set.  
- Grid searcher.


To do:
- Abalation tests for performance with and without soft labels at different temperature values. Testing for changes in accuracy. 
- Test implicit learning from soft labels (IPR)
- Derive dC/dz for CE on softmax with temperature


### References: 
- https://arxiv.org/abs/1503.02531
- https://cs230.stanford.edu/files_winter_2018/projects/6940224.pdf 
