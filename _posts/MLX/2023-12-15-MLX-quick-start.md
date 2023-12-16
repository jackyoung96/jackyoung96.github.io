---
layout: post
title: "MLX: Apple silicon 용 Machine Learning 프레임워크 - 01.Quick-start"
tags: archive
---

- [MLX: Apple silicon 을 위한 ML 프레임워크](#mlx-apple-silicon-을-위한-ml-프레임워크)
  - [Installation](#installation)
  - [Basic quick start](#basic-quick-start)
  - [Function and Graph Transformations](#function-and-graph-transformations)
  - [Unified Memory](#unified-memory)

# MLX: Apple silicon 을 위한 ML 프레임워크

MLX 는 Apple silicon 에서 머신러닝을 수행하기 위해 만들어진 ML framework 입니다. Apple silicon 만의 CPU와 GPU를 사용하여 벡터와 그래프 연산 속도를 크게 높일 수 있습니다. 특히 맥북은 external GPU 를 허락하고 있지 않기 때문에 M1 이상의 맥북에서 ML/DL 연산 가속이 어려웠는데, MLX 를 통해 크게 개선할 수 있을 것으로 보입니다. Apple 의 On-device ML 의 서막이라고도 볼 수 있겠죠.

## Installation 

설치는 너무 간단합니다.


```python
# Requirements
# 1. M 시리즈 apple silicon 
# 2. native Python >= 3.8
# 3. MacOS >= 13.3
!python -c "import platform; print(platform.processor())" # It must be arm
!pip install mlx 
```

    arm
    Requirement already satisfied: mlx in /Users/1113506/.venv/mlx/lib/python3.10/site-packages (0.0.5)


## Basic quick start

array(배열)를 만들기 위해서y `mlx.core` 를 import 하고 array 를 만들어 보겠습니다.


```python
import mlx.core as mx

a = mx.array([1,2,3,4])
print(f"a shape: {a.shape}")
print(f"a dtype: {a.dtype}")

print()

b = mx.array([1.0, 2.0, 3.0, 4.0])
print(f"b shape: {b.shape}")
print(f"b dtype: {b.dtype}")
```

    a shape: [4]
    a dtype: mlx.core.int32
    
    b shape: [4]
    b dtype: mlx.core.float32


MLX 는 lazy evaluation 을 사용합니다. 
[lazy evaluation 이란?](https://medium.com/sjk5766/lazy-evaluation%EC%9D%84-%EC%95%8C%EC%95%84%EB%B3%B4%EC%9E%90-411651d5227b)  
lazy evaluation 은 실제로 연산 결과가 어딘가에 사용되기 전까지 연산을 미루는 프로그래밍 방법론입니다.
따라서 성능 관점에서 이득을 보거나 오류를 회피하거나 무한 자료구조를 사용할 수 있다는 장점이 있습니다.  
`mx.eval` 함수는 강제로 연산을 수행시킵니다.


```python
c = a + b # 실제로 연산이 일어나지 않는다. c 가 사용되지 않았기 때문
mx.eval(c) # 연산이 수행
print(c) # 연산이 수행
import numpy as np
np.array(c) # 연산이 수행
```

    array([2, 4, 6, 8], dtype=float32)
    array([2., 4., 6., 8.], dtype=float32)



실제로 연산 시간이 어떻게 소모되는지 확인해보겠습니다.


```python
import time

start = time.time()
for _ in range(100):
    c = a + b
print(f"lazy evaluation time: {time.time()-start}")

start = time.time()
for _ in range(100):
    c = a + b
    mx.eval(c)
print(f"forced evaluation time: {time.time()-start}")
```

    lazy evaluation time: 0.0009300708770751953
    forced evaluation time: 0.03633594512939453


## Function and Graph Transformations

MLX 는 기본적인 함수 변환인 `grad()`, `vmap()` 을 지원합니다. 각각 gradient 를 구하고 vectorize 하는 함수들입니다.


```python
x = mx.array(0.0)
print(mx.sin(x))
print(mx.grad(mx.sin)(x))
x = mx.array([0.0,0.1,0.2,3])
print(mx.vmap(mx.grad(mx.sin))(x))
```

    array(0, dtype=float32)
    array(1, dtype=float32)
    array([1, 0.995004, 0.980067, -0.989992], dtype=float32)


이외에도 vector-Jacobian products `vjp()` 나 Jacobian-vector products `jvp()`, forward-backward path 를 동시에 계산해주는 `value_and_grad()` 함수 등이 있습니다.

## Unified Memory

Apple silicon 은 CPU 와 GPU 가 별개의 장치로 존재하지 않습니다. 하나의 unifired memory architecture 로 구성되어 있습니다.  
따라서 CPU와 GPU가 동일한 memory pool 에서 직접적으로 접근 가능합니다. MLX 는 이러한 장점을 누릴 수 있도록 디자인 되었습니다.  

아래와 같이 두개의 array 를 만들 때, MLX 는 특정 위치를 지정하지 않습니다. CUDA 에서 `to(device)` 를 지정했던 것과는 다르죠.  
```python
a = mx.random.normal((100,))
b = mx.random.normal((100,))
```

MLX 에서는 operation 을 위한 device 를 지정해줍니다. 즉, memory 위치의 이동 없이 CPU 연산과 GPU 연산을 모두 할 수 있습니다.  
```python
mx.add(a, b, stream=mx.cpu)
mx.add(a, b, stream=mx.gpu)
```
위의 두 연산은 각각 CPU, GPU 를 통해 계산됩니다. 지금 두 연산은 어떠한 dependency 도 없기 때문에 완전히 병렬적으로 연산이 가능합니다.  
그러나 아래와 같이 변수에 대한 dependency가 존재할 때 MLX 는 알아서 첫번째 연산을 끝낸 뒤 두번째 연산을 끝냅니다.
```python
c = mx.add(a, b, stream=mx.cpu)
d = mx.add(a, c, stream=mx.gpu)
```

연산의 종류에 따라 CPU가 유리할 때도 GPU가 유리할 때도 있습니다.  
`matmul` 연산은 GPU에게 유리한 연산입니다. 그러나 for loop 으로 이루어진 연속된 연산은 CPU에게 유리한 연산입니다.  
따라서 아래 함수의 연산은 GPU와 CPU 를 적절히 사용했을 때 실행시간을 가장 줄일 수 있습니다.


```python
def fun(a, b, d1, d2):
    x = mx.matmul(a, b, stream=d1)
    mx.eval(x)
    for _ in range(500):
        b = mx.exp(b, stream=d2)
        mx.eval(b)
    return x, b

a = mx.random.uniform(shape=(4096, 512))
b = mx.random.uniform(shape=(512, 4))

start = time.time()
fun(a, b, mx.cpu, mx.cpu)
print(f"cpu elapsed time: {time.time()-start}")

start = time.time()
fun(a, b, mx.gpu, mx.gpu)
print(f"gpu elapsed time: {time.time()-start}")

start = time.time()
fun(a, b, mx.cpu, mx.gpu)
print(f"cpu-gpu elapsed time: {time.time()-start}")

start = time.time()
fun(a, b, mx.gpu, mx.cpu)
print(f"gpu-cpu elapsed time: {time.time()-start}")
```

    cpu elapsed time: 0.05551004409790039
    gpu elapsed time: 0.09125995635986328
    cpu-gpu elapsed time: 0.08087396621704102
    gpu-cpu elapsed time: 0.004935264587402344


MXL 은 stream 을 지정하지 않으면 default_device 로 설정되어 있습니다. M1 pro 기준은 GPU 로 설정되어 있네요.


```python
print(mx.default_device())
print(mx.default_stream(mx.default_device()))
```

    Device(gpu, 0)
    Stream(Device(gpu, 0), 0)

