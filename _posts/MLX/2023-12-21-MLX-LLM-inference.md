---
layout: post
title: "MLX: Apple silicon 용 Machine Learning 프레임워크 - 04.LLM inference example"
tags: archive
---

# LLM Inference with MLX

MLX 라이브러리를 이용한 LLM inference 에서 어느 정도 속도의 이득을 볼 수 있을지 확인해보도록 하겠습니다.  
Transformer를 기반으로 하는 LLaMA 모델을 직접 구현하는 작업이 포함되어 있습니다. 

### 갑자기 드는 개인적 의견

LLaMA 모델을 직접구현해야 한다...이런 부분에서 단점이 보이는데...PyTorch 같은 경우 huggingface 라는 엄청난 플랫폼과의 연계성이 돋보입니다. 코드 4~5줄만으로도 huggingface hub 에 있는 모델들을 다운받고 inference 할 수 있습니다. 뭐 MLX 는 아직 나온지 얼마 안된 라이브러리니까 개선의 여지는 많이 남아있습니다만, torch 와 동일한 레벨의 프레임워크로 방향을 잡기보다는 torch 로 이미 구현되어 있는 코드들을 mlx 로 연결하여 사용하는 방향(?)이 더 낫지 않을까 싶네요. 그렇지 않고서야 PyTorch 아키텍처가 나올 때마다 똑같은 모델 코드를 직접 만들어야 한다면....!! 생각만해도 너무 불편하네요.

## LLaMA 모델 구축

일단 관련 모듈들을 import 합니다. 주의할 점은 LLM 들의 사이즈와 맥북 RAM 사이즈를 잘 판단하시기 바랍니다. 제 M1 맥북 프로는 16G 메모리여서, 7B 이상 모델은 메모리에 올라가질 않습니다.


```python
import mlx.core as mx
import mlx.nn as nn
import math
import gc
```

LLaMA Architecture 는 아래 그림과 같은 block 들을 쌓아 만들어집니다. attention 전에 RMSNorm 을 수행하는 pre-normalization 이라는 테크닉을 사용합니다.
![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/b4d542ca-a7e7-4d6f-a521-35cc3e157cbd)

또 Activation function 으로는 SwiGLU를 사용합니다. SwiGLU 는 Swish activation function 과 GLU 를 합친 activation function 인데요.  
Swish activation, GLU, SwiGLU 의 수학적 식은 아래와 같습니다.
$$Swish(x) = x\sigma (\beta x)$$
$$GLU(x,W,V,b,c)=\sigma(xW+b)\otimes(xV+c)$$
$$SwiGLU(x,W,V,b,c,\beta)=Swish_\beta(xW+b)\otimes(xV+c)$$

수식은 어려워보이지만 beta=1 인 경우 linear layer들로 간단하게 구현이 가능합니다.  

그럼 LlamaAttention, LlamaMLPLayer, LlamaEncoderLayer, LlamaModel, LlamaForCausalLM 총 5개의 모듈을 정의하겠습니다. 코드는 Huggingface llama 와 동일한 구현입니다. 모듈들 변수명 또한 Huggingface llama model 과 동일하게 설정했는데요, 이는 huggingface weight 를 다운받아서 바로 적용할 수 있게 만들기 위함입니다.

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
        
        # Shape 값들
        num_heads = self.num_heads
        B, L, D = queries.shape
        
        # Shape 전처리 (B, num_heads, L, D // num_heads)
        queries = queries.reshape(B, L, num_heads, -1).transpose(0,2,1,3)
        keys = keys.reshape(B, L, num_heads, -1).transpose(0,2,1,3)
        values = values.reshape(B, L, num_heads, -1).transpose(0,2,1,3)
        
        # query[0,0,0,0] -> -0.4308
        
        # RoPE 처리
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
        # queries 는 (B, num_heads, L, D // num_heads)
        # keys 는 (B, num_heads, D//num_heads, L)
        # 결과는 (B,num_heads, L, L)
        scores = (queries * scale) @ keys.transpose(0, 1, 3, 2)
        
        if mask is not None:
            scores = scores + mask
        scores = mx.softmax(scores, axis=-1)
        # score @ values 는  (B, num_heads, L, D // num_heads)
        # transpose 하면 (B, L, num_heads, D // num_heads)
        # value_hat 은 (B, L, D)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        
        # values_hat[0,0,0] = -0.0005
        
        # Return key, value (cache 때문)
        return self.o_proj(values_hat), (keys, values)
```

### LLaMA MLP

swiGLU activation 이 구현된 모듈입니다.


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
        x = a * mx.sigmoid(a) * b # SwiGLU (Swish 에서 beta=1 을 사용한 버전)
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
        # 마지막 토큰만 사용함 (그 전꺼는 prompt 이므로)
        y = self.lm_head(x[:, -1])
        y = mx.random.categorical(y * (1/temp))
        
        # lazy evaluation 때문에 yield 하더라도 굳이 계산을 하지는 않는다.
        yield y
        
        while True:
            x = y[:, None]
            x, cache = self.model(x, cache=cache, return_cache=True)
            y = self.lm_head(x[:, -1])
            y = mx.random.categorical(y * (1/temp))
            
            yield y
```

## Converting original Llama weight

그럼 Huggingface hub 으로부터 적절한 llama architectur 모델을 다운받고 MLX weight로 변환한 후 직접 만든 MLX 모델에 load 해보겠습니다. 메모리 이슈를 고려하여 1.3b 모델인 [princeton-nlp/Sheared-LLaMA-1.3B](https://huggingface.co/princeton-nlp/Sheared-LLaMA-1.3B) 를 선정했습니다.

`map_torch_to_mlx` 의 경우 layer 이름이 다르게 설정된 경우 이를 변환하기 위함인데요. 위 모델은 huggingface llama 와 완전히 동일한 변수명과 구조로 되어있기 때문에 따로 key 이름 변경이 필요하지 않습니다. 원본 코드인 [MLX LLM inference](https://ml-explore.github.io/mlx/build/html/examples/llama-inference.html) 에서는 구조와 변수명이 달라서 변환 작업이 필요했습니다. 


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

일단 아직 huggingface 모델을 이름만으로 MLX 에 load 하는 방법은 없습니다. 따라서 huggingface 의 torch weight 를 numpy array 로 변환하여 파일로 저장한 후, 해당 numpy array 를 mlx array 로 바꾸어 load 하는 방식을 사용합니다. 추후 개선이 될거라고 MLX document 에 명시되어 있습니다.  

`hf` 라는 폴더를 만들어 그 안에 `weights.npz` 파일들을 저장해주도록 하겠습니다. 쓸모없이 numpy array 들로 memory 를 낭비할 수 있으니 garbage collection 을 이용해 메모리를 최대한 덜 사용하도록 만들어줍니다.  


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

해당 파일들을 이용해 model weight를 업데이트 해줍니다.  
`mlx.utils.tree_unflatten` 이라는 함수가 사용되는데요, {key: value} 로 이루어진 npz 파일에서 `layers.2.attention.query_proj.weight` 와 같이 key 가 구성되어 있으면
```
{"layers": [..., ..., {"attention": {"query_proj": {"weight": ...}}}]}
```
처럼 변환해줍니다. 이런 형태로 변환하면 `model.update` 를 통해 해당 weight 를 다이렉트로 업데이트 할 수 있습니다.


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

모든 준비가 끝났습니다!! 그럼 inference 를 해보겠습니다. Tokenizer는 Huggingface 의 tokenizer를 그대로 사용합니다.


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


Inference 가 잘 되는 것을 확인할 수 있습니다. Troughput 의 경우 20 tokens/sec 정도가 나오는 것을 확인할 수 있습니다.

## Huggingface 와 throughput 비교

그럼 Huggingface, 즉 PyTorch 를 사용했을 때와 비교해서 어느정도 속도의 이득을 볼 수 있는지 확인해보도록 하겠습니다.


```python
# memory 부족때문에 기존 model 내려야함
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

CPU 를 사용하는 경우 Throughput은 약 5 tokens/sec, MPS 를 이용해 가속한 경우 약 10 tokens/sec 정도가 나오는 것을 확인할 수 있습니다.
즉, MLX 를 사용했을 때 2배정도의 속도 개선이 있음을 확인할 수 있었습니다.

## References

- [https://ml-explore.github.io/mlx/build/html/examples/llama-inference.html](https://ml-explore.github.io/mlx/build/html/examples/llama-inference.html)
- [https://thecho7.tistory.com/entry/SwiGLU-Activation-Function-%EC%84%A4%EB%AA%85](https://thecho7.tistory.com/entry/SwiGLU-Activation-Function-%EC%84%A4%EB%AA%85)
- [https://huggingface.co/princeton-nlp/Sheared-LLaMA-1.3B](https://huggingface.co/princeton-nlp/Sheared-LLaMA-1.3B)