---
layout: post
title: Diary - Ignore value in Cross Entropy function of PyTorch
tags: archive
lang: en
---

In PyTorch, there are times when you don't want to compute the loss for certain labels. Since running a For loop while doing GPU computation is very inefficient, you use a different approach.  
If you look at the documentation for [CrossEntropy](https://pytorch.org/docs/1.12/generated/torch.nn.CrossEntropyLoss.html), there's an argument called ignore_index. The default value is set to -100, which means that if a label is -100, this part is not included in the computation.  
Simply putting in a negative value might cause an error, so to be safe let's put in the magic number -100.

## Reference

https://discuss.pytorch.org/t/negative-label-in-crossentropyloss/158873
