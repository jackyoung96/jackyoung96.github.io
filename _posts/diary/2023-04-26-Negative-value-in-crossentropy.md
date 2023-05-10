---
layout: post
title: Diary - Ignore value in Cross Entropy function of PyTorch
tags: archive
---

PyTorch에서 특정 label에 대해서는 loss 를 계산하고 싶지 않을 때가 있다. GPU 연산을 하는데 For 문을 돌리는 것은 굉장히 비효율적이기 때문에 다른 방법을 사용한다.  
[CrossEntropy](https://pytorch.org/docs/1.12/generated/torch.nn.CrossEntropyLoss.html) 의 document를 찾아보면 ignore_index 라는 argument가 있다. Default 값은 -100 으로 설정되어 있는데, label이 -100 이면 이부분은 계산을 안한다는 뜻이다.  
단순 negative value를 넣으면 에러가 발생할 수도 있을 것 같은데, 안전하게 -100이라는 magic number를 넣어주도록 하자.

## Reference

https://discuss.pytorch.org/t/negative-label-in-crossentropyloss/158873
