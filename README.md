### Interesting results
This performance table is for a model trained on classes 0-5, tested on classes 0-9. The student network (all dense layers), has the following architecture:

``` 28*28 input -> layer 1 -> 14*14 output -> Relu -> Layer 2 -> (*, 10) output```

---
|  lr | T  | Epochs  | Weight  | 0  |  1 | 2  |  3 |  4 |  5 | 6  | 7  | 8  | 9  |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  0.01 | 10.0  | 70  | 0.9  | 99.49%| 99.73% | 99.42% | 99.60% | 99.6% | 99.1% | 98.64% | 96% | 97.33% | 97.23% |

As you can see above, the student model is able to learn how to classify images of a class that it did not see during training. Through these experiments, I have found that temperature values between 8-12 work the best for this task, along with a learning rate of 0.01 as 0.001 is too small.

---
The below table shows the results for training the student network on all classes in MNIST (0-9). The first row represents a weight of 0 given to the soft-label loss term, which basically means a standard training loop for the student network using CE loss. The second row represents a student net trained on a mix of both the soft labels and hard labels using a temperature of 2.5. Both networks are trained with the same LR and epochs. As you can see, the loss on the net trained with the distillation loss is lower than that of the net trained with a regular CE loss. 

|lr|T|epochs|weight|0|1|2|3|4|5|6|7|8|9|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0.01|1.0|70|0.0|99.28%|98.59%|98.26%|98.81%|96.74%|95.29%|97.91%|97.86%|95.69%|97.03%|
|0.01|2.5|70|0.9|99.29%|99.74%|99.13%|99.60%|99.19%|98.77%|99.27%|98.83%|98.97%|98.32%|

----

I also tested the above experiments using the smallest possible student model with the following architecture:

``` 28*28 -> layer 1 -> (*, 10) output```

However, this resulted in poor accuracies in the 50% range for each of the unseen classes, which still is better than the baseline which gets accuracies of 0.0% for all unseen classes during training. But this also shows that this small student model is not capable of learning from the implicit labels, so I chose a larger architecture that would be able to fit to my data (and maybe even overfit!)


So far:

- Implement distillation loss using torch 1.10 soft-label cross-entropy loss (Kullback-Leibler is the next best option)
- Teacher and student models setup, along with their respective training loops
- Basic visualization of training curves.
- Custom MNIST dataloaders that leave out certain classes (to test implicit learning from soft labels)
- Train teacher network. 14-16 errors on the val set.  
- Grid searcher
- Abalation tests for regular classification performance with and without soft labels. Testing for changes in accuracy. 
- Test implicit learning from soft labels (IPR)

To do:
- Derive dC/dz for CE on softmax with temperature
- Try on CIFAR10
- Related to above, add flags for easy switching between datasets in the argparser 

### References: 
- https://arxiv.org/abs/1503.02531
- https://cs230.stanford.edu/files_winter_2018/projects/6940224.pdf 
