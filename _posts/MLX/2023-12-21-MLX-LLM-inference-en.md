---
layout: post
title: "MLX: A Machine Learning Framework for Apple Silicon - 04. LLM inference example"
tags: archive
lang: en
---

# LLM Inference with MLX

Let's check how much of a speed gain we can get from LLM inference using the MLX library.
This includes the work of implementing a Transformer-based LLaMA model from scratch.

### A sudden personal opinion

Having to implement the LLaMA model from scratch... this is where I see a downside. In the case of PyTorch, its tight integration with the amazing platform that is huggingface really stands out. With just 4-5 lines of code, you can download models from the huggingface hub and run inference on them. Well, MLX is a library that just came out not long ago, so there's plenty of room for improvement. But rather than aiming to be a framework at the same level as torch, wouldn't it be better to head in the direction of connecting code that's already implemented in torch to mlx (?). Otherwise, having to manually build the same model code every time a PyTorch architecture comes out....!! Just thinking about it sounds so inconvenient.

## Building the LLaMA model

First, let's import the relevant modules. One thing to be careful about: make sure to judge the size of the LLMs against your MacBook's RAM size. My M1 MacBook Pro has 16G of memory, so any model larger than 7B won't fit in memory.


```python
import mlx.core as mx
import mlx.nn as nn
import math
import gc
```

The LLaMA architecture is built by stacking blocks like the one shown in the figure below. It uses a technique called pre-normalization, which performs RMSNorm before attention.
![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/b4d542ca-a7e7-4d6f-a521-35cc3e157cbd)

It also uses SwiGLU as the activation function. SwiGLU is an activation function that combines the Swish activation function and GLU.
The mathematical formulas for Swish activation, GLU, and SwiGLU are as follows.
$$Swish(x) = x\sigma (\beta x)$$
$$GLU(x,W,V,b,c)=\sigma(xW+b)\otimes(xV+c)$$
$$SwiGLU(x,W,V,b,c,\beta)=Swish_\beta(xW+b)\otimes(xV+c)$$

The formulas look difficult, but when beta=1 it can be implemented simply with linear layers.

So let's define a total of 5 modules: LlamaAttention, LlamaMLPLayer, LlamaEncoderLayer, LlamaModel, and LlamaForCausalLM. The code is the same implementation as Huggingface llama. The module variable names are also set identically to the Huggingface llama model. This is to make it possible to download the huggingface weights and apply them directly.

### Llama Attention


```python
class LlamaAttention(nn.Module):
    def __init__(self, 
                 dims: int,
                 num_heads: int):
        super().__init__()
        
        self.num_heads = num_heads
        
        self.rope = nn.RoPE(dims // num_heads, traditional=True)
        self.q_proj = nn.Linear(dims, dims, bias=False)
        self.k_proj = nn.Linear(dims, dims, bias=False)
        self.v_proj = nn.Linear(dims, dims, bias=False)
        self.o_proj = nn.Linear(dims, dims, bias=False)
        
    def __call__(self, x, mask=None, cache=None):
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        
        # Shape values
        num_heads = self.num_heads
        B, L, D = queries.shape
        
        # Shape preprocessing (B, num_heads, L, D // num_heads)
        queries = queries.reshape(B, L, num_heads, -1).transpose(0,2,1,3)
        keys = keys.reshape(B, L, num_heads, -1).transpose(0,2,1,3)
        values = values.reshape(B, L, num_heads, -1).transpose(0,2,1,3)
        
        # query[0,0,0,0] -> -0.4308
        
        # RoPE processing
        if cache is not None:
            # axis = 2 -> Length (sequence position)
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
            
        # query[0,0,0,-1] = 0.6716
            
        scale = math.sqrt(1 / queries.shape[-1])
        # queries is (B, num_heads, L, D // num_heads)
        # keys is (B, num_heads, D//num_heads, L)
        # result is (B,num_heads, L, L)
        scores = (queries * scale) @ keys.transpose(0, 1, 3, 2)
        
        if mask is not None:
            scores = scores + mask
        scores = mx.softmax(scores, axis=-1)
        # score @ values is  (B, num_heads, L, D // num_heads)
        # after transpose it's (B, L, num_heads, D // num_heads)
        # value_hat is (B, L, D)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        
        # values_hat[0,0,0] = -0.0005
        
        # Return key, value (for the cache)
        return self.o_proj(values_hat), (keys, values)
```

### LLaMA MLP

This is the module where the swiGLU activation is implemented.


```python
class LlamaMLPLayer(nn.Module):
    def __init__(self,
                 dims: int,
                 mlp_dims: int):
        super().__init__()
        
        self.gate_proj = nn.Linear(dims, mlp_dims, bias=False)
        self.up_proj = nn.Linear(dims, mlp_dims, bias=False)
        self.down_proj = nn.Linear(mlp_dims, dims, bias=False)
        
    def __call__(self, x):
        a = self.gate_proj(x)
        b = self.up_proj(x)
        x = a * mx.sigmoid(a) * b # SwiGLU (the version using beta=1 in Swish)
        x = self.down_proj(x)
        
        return x
```

### LLaMA encoder


```python
class LlamaEncoderLayer(nn.Module):
    def __init__(self,
                 dims: int,
                 mlp_dims: int,
                 num_heads: int):
        super().__init__()
        
        self.self_attn = LlamaAttention(dims, num_heads)
        self.input_layernorm = nn.RMSNorm(dims)
        self.post_attention_layernorm = nn.RMSNorm(dims)
        self.mlp = LlamaMLPLayer(dims, mlp_dims)
        
    def __call__(self, x, mask=None, cache=None):
        y = self.input_layernorm(x)
        y, cache = self.self_attn(y, mask=mask, cache=cache)
        x = x + y
        
        y = self.post_attention_layernorm(x)
        y = self.mlp(y)
        x = x + y
        
        return x, cache
```

### LLaMA model


```python
class Llama(nn.Module):
    def __init__(
        self, 
        num_layers: int,
        vocab_size: int,
        dims: int,
        mlp_dims: int,
        num_heads: int
    ):
        super().__init__()
        
        self.embed_tokens = nn.Embedding(vocab_size, dims)
        self.layers = [
            LlamaEncoderLayer(dims, mlp_dims, num_heads) for _ in range(num_layers)
        ]
        self.norm = nn.RMSNorm(dims)
        
    def __call__(self, x, cache=None, return_cache=False):
        if cache:
            assert len(self.layers) == len(cache), "Length of cache must be equal to number of layers"
            mask = None
        else:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
            mask = mask.astype(self.embed_tokens.weight.dtype)
            cache = [None] * len(self.layers)
        
        x = self.embed_tokens(x)
            
        for i, layer in enumerate(self.layers):
            x, c = layer(x, mask, cache=cache[i])
            if return_cache:
                cache[i] = c
        x = self.norm(x)
        
        if return_cache:
            return x, cache
        
        return x
```

### LLaMA causal LM 


```python
class LlamaForCausalLM(nn.Module):
    def __init__(
        self, 
        num_layers: int,
        vocab_size: int,
        dims: int,
        mlp_dims: int,
        num_heads: int
    ):
        super().__init__()
        self.model = Llama(num_layers,
                            vocab_size,
                            dims,
                            mlp_dims,
                            num_heads)
        self.lm_head = nn.Linear(dims, vocab_size, bias=False)
        
    def __call__(self, x):
        x = self.model(x)
        
        return self.lm_head(x)
        
    def generate(self, x, temp=1.0):
        x, cache = self.model(x, return_cache=True)
        # Use only the last token (the previous ones are the prompt)
        y = self.lm_head(x[:, -1])
        y = mx.random.categorical(y * (1/temp))
        
        # Due to lazy evaluation, even if we yield, the computation isn't necessarily performed.
        yield y
        
        while True:
            x = y[:, None]
            x, cache = self.model(x, cache=cache, return_cache=True)
            y = self.lm_head(x[:, -1])
            y = mx.random.categorical(y * (1/temp))
            
            yield y
```

## Converting original Llama weight

So let's download an appropriate llama architecture model from the Huggingface hub, convert it to MLX weights, and then load it into the MLX model we built ourselves. Considering memory issues, I selected the 1.3b model [princeton-nlp/Sheared-LLaMA-1.3B](https://huggingface.co/princeton-nlp/Sheared-LLaMA-1.3B).

`map_torch_to_mlx` is for converting layer names when they are set differently. Since the model above has exactly the same variable names and structure as huggingface llama, no separate key name changes are needed. In the original code, [MLX LLM inference](https://ml-explore.github.io/mlx/build/html/examples/llama-inference.html), the structure and variable names were different, so conversion work was needed.


```python
from itertools import starmap

import numpy as np
import torch

def map_torch_to_mlx(key, value):
    # if "tok_embedding" in key:
    #     key = "embedding.weight"

    # elif "norm" in key:
    #     key = key.replace("attention_norm", "norm1").replace("ffn_norm", "norm2")

    # elif "wq" in key or "wk" in key or "wv" in key or "wo" in key:
    #     key = key.replace("wq", "query_proj")
    #     key = key.replace("wk", "key_proj")
    #     key = key.replace("wv", "value_proj")
    #     key = key.replace("wo", "out_proj")

    # elif "w1" in key or "w2" in key or "w3" in key:
    #     # The FFN is a separate submodule in PyTorch
    #     key = key.replace("feed_forward.w1", "linear1")
    #     key = key.replace("feed_forward.w3", "linear2")
    #     key = key.replace("feed_forward.w2", "linear3")

    # elif "output" in key:
    #     key = key.replace("output", "out_proj")

    # elif "rope" in key:
    #     return None, None
    
    return key, value.numpy()
```

For now, there is still no way to load a huggingface model into MLX by name alone. Therefore, we use an approach where we convert the huggingface torch weights into numpy arrays, save them to files, and then convert those numpy arrays into mlx arrays to load them. The MLX documentation states that this will be improved in the future.

Let's create a folder called `hf` and save the `weights.npz` files inside it. Since we could waste memory uselessly on numpy arrays, we use garbage collection to make it use as little memory as possible.


```python
from transformers import AutoModelForCausalLM
import os

hf_path = "princeton-nlp/Sheared-LLaMA-1.3B"
output_path = f"hf/{hf_path}"
output_file = output_path + "/{key}.npz"
if not os.path.exists(output_path):
    os.makedirs(output_path)

if not os.path.exists(output_path + "/DONE"):
    hf_model = AutoModelForCausalLM.from_pretrained(hf_path)
    hf_model.config.to_json_file(output_path + "/config.json")
    state = hf_model.state_dict()

    np.savez(
        output_file.format(key="weights"),
        **{k: v for k, v in starmap(map_torch_to_mlx, state.items()) if k is not None}
    )

    # Done indicate
    f = open(output_path + '/DONE', 'w')
    f.close()
    del hf_model
    
    gc.collect()
```

Using these files, we update the model weights.
A function called `mlx.utils.tree_unflatten` is used here. From the npz file made of {key: value}, if the keys are structured like `layers.2.attention.query_proj.weight`, it converts them into
```
{"layers": [..., ..., {"attention": {"query_proj": {"weight": ...}}}]}
```
this form. Converting into this form lets you directly update the corresponding weights through `model.update`.


```python
from mlx.utils import tree_unflatten
import json

# https://huggingface.co/beomi/llama-2-ko-7b/blob/main/config.json
with open(output_path + "/config.json", 'r') as f:
    config = json.load(f)

model = LlamaForCausalLM(num_layers=config['num_hidden_layers'],
              vocab_size=config['vocab_size'], # tokenizer.vocab_size
              dims=config['hidden_size'], 
              mlp_dims=config['intermediate_size'], 
              num_heads=config['num_attention_heads'])
for np_file in os.listdir(output_path):
    if np_file.endswith('.npz'):
        data = tree_unflatten(list(mx.load(output_path + f"/{np_file}").items()))
        # if "model" in data.keys():
        #     if "layers" in data["model"].keys():
        #         data['model']['layers'].extend([{}] * (config['num_hidden_layers']-len(data['model']['layers'])))
        model.update(data)
        del data
        
gc.collect()
mx.eval(model.parameters())
```

## LLaMA inference

All preparations are complete!! So let's run inference. For the tokenizer, we use Huggingface's tokenizer as-is.


```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(hf_path)
```


```python
from time import time

prompt = "My name is Julien and I like to"
max_token_length = 100

x = mx.array([tokenizer.encode(prompt)])
tokens = []

mx.eval(x)

tic = time()
for token in model.generate(x, temp=1.0):
    tokens.append(token)
    
    if len(tokens) == 1:
        mx.eval(token)
    
    if len(tokens) >= max_token_length:
        break

mx.eval(tokens)
toc = time()

s = tokenizer.decode([t.item() for t in tokens], skip_special_tokens=True)
print(s, flush=True)
print(f"Throughput (MLX): {len(tokens)/(toc-tic)} tokens/sec")
```

    write. I write about... lots of things. Unmannerstood, Difficult & Unniverse
     nova the unicorn, which makes me confidently and I am unable to deal with everything from fiction stories along with lol elfing, podcasting from 1sthing jobsearching
    com.Home about...Find backs.
    i and bjigg@nak..
    2tourn. Most dês.
    Matt. I'
    Throughput (MLX): 21.244782071635516 tokens/sec


We can confirm that inference works well. For throughput, we can see that we get about 20 tokens/sec.

## Comparing throughput with Huggingface

So let's compare with using Huggingface, i.e. PyTorch, to check how much of a speed gain we can get.


```python
# the existing model must be unloaded due to lack of memory
del model
gc.collect()
```
    734
```python
hf_model = AutoModelForCausalLM.from_pretrained(hf_path)

input_ids = tokenizer.encode('prompt', return_tensors='pt')
tic = time()
output = hf_model.generate(input_ids, max_length=max_token_length)
toc = time()
s = tokenizer.decode(output[0], skip_special_tokens=True)

print(s, flush=True)
print(f"Throughput (HF, cpu): {len(tokens)/(toc-tic)} tokens/sec")

print("\n" + "="*20 + "\n")

device = torch.device("mps:0")
hf_model.to(device)
input_ids = tokenizer.encode('prompt', return_tensors='pt').to(device)
tic = time()
output = hf_model.generate(input_ids, max_length=max_token_length)
toc = time()
s = tokenizer.decode(output[0], skip_special_tokens=True)

print(s, flush=True)
print(f"Throughput (HF, mps): {len(tokens)/(toc-tic)} tokens/sec")
```
    promptly and efficiently.
    We are a family owned and operated business that has been in business for over 20 years. We are a full service company that can handle all of your needs. We are a full service company that can handle all of your needs.
    We are a full service company that can handle all of your needs. We are a full service company that can handle all of your needs.
    We are a full service company that can handle all of your needs. We
    Throughput (HF, cpu): 4.973824978994109 tokens/sec
    
    ====================
    
    promptly and efficiently.
    We are a family owned and operated business that has been in business for over 20 years. We are a full service company that can handle all of your needs. We are a full service company that can handle all of your needs.
    We are a full service company that can handle all of your needs. We are a full service company that can handle all of your needs.
    We are a full service company that can handle all of your needs. We
    Throughput (HF, mps): 9.700159370346634 tokens/sec

```python
# For memory
del hf_model
gc.collect()
```

We can see that when using the CPU, the throughput is about 5 tokens/sec, and when accelerated using MPS, about 10 tokens/sec.
In other words, we were able to confirm that using MLX gives about a 2x speed improvement.

## References

- [https://ml-explore.github.io/mlx/build/html/examples/llama-inference.html](https://ml-explore.github.io/mlx/build/html/examples/llama-inference.html)
- [https://thecho7.tistory.com/entry/SwiGLU-Activation-Function-%EC%84%A4%EB%AA%85](https://thecho7.tistory.com/entry/SwiGLU-Activation-Function-%EC%84%A4%EB%AA%85)
- [https://huggingface.co/princeton-nlp/Sheared-LLaMA-1.3B](https://huggingface.co/princeton-nlp/Sheared-LLaMA-1.3B)
