---
layout: post
title: Diary - Frequent mistake for using torch.distributions.Normal
tags: archive
---

To implement PPO algorithm with Pytorch, torch.distributions should be used. In case of your action space is continous space, you might use Normal distribution. At that time, **you must check your mean and standard deviation dimension**. 

Let's see the code below.

```python
import torch
from torch.distributions import Normal

action = torch.Tensor([2,3]) # (2,)

a1 = torch.Tensor([2,3]) # (2,)
b1 = torch.Tensor([1,1]) # (2,)
dist1 = Normal(a1,b1)
logprob1 = dist1.log_prob(action)

print(logprob1)

a2 = torch.Tensor([[2],[3]]) # (2,1)
b2 = torch.Tensor([[1],[1]]) # (2,1)
dist2 = Normal(a2,b2)
logprob2 = dist2.log_prob(action)

print(logprob2)

a3 = torch.Tensor([[2,3]]) # (1,2)
b3 = torch.Tensor([[1,1]]) # (1,2)
dist3 = Normal(a3,b3)
logprob3 = dist3.log_prob(action)

print(logprob2)
```

```python
# Result
tensor([-0.9189, -0.9189])
tensor([[-0.9189, -1.4189],
        [-1.4189, -0.9189]])
tensor([[-0.9189, -1.4189],
        [-1.4189, -0.9189]])
```

The main point is that **you should exactly match the action dimension and the mean, std dimensions. Dummy dimensions can cause issues because Normal realized it as multivariate Gaussian.  

This issue is really hard to debug because they don't give any errors. 