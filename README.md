# Distillation
Implementing and experimenting with knowledge distillation 

So far:

- Distillation loss using torch 1.10 soft-label cross-entropy loss (Kullback-Leibler is the next best option)
- Teacher and student models setup, along with their respective training loops
- Visualization is bare bones
- Custom dataloaders that leave out certain classes (to test implicit learning from soft labels)
- Train teacher network. 14-16 errors on the val set.  
- grid searcher is made


To do:
- Abalation tests for performance with and without soft labels at different temperature values.
- Test implicit learning from soft labels 
- Mathematically prove that minimizing the logsoftmax -> CE (torch) for soft labels is equivalent to bringing the softmax as close to the true soft-label distribution as possible (for my own understanding)


Remember to clean up all the if __name__ == "__main__" when done. 
