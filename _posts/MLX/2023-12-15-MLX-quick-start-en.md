---
layout: post
title: "MLX: A Machine Learning framework for Apple silicon - 01. Quick-start"
tags: archive
lang: en
---

- [MLX: An ML framework for Apple silicon](#mlx-apple-silicon-을-위한-ml-프레임워크)
  - [Installation](#installation)
  - [Basic quick start](#basic-quick-start)
  - [Function and Graph Transformations](#function-and-graph-transformations)
  - [Unified Memory](#unified-memory)

# MLX: An ML framework for Apple silicon

MLX is an ML framework built to perform machine learning on Apple silicon. By using Apple silicon's own CPU and GPU, it can greatly increase the speed of vector and graph operations. In particular, since MacBooks don't allow external GPUs, ML/DL computation acceleration was difficult on M1 or later MacBooks, but it looks like MLX can greatly improve this. You could also see it as the prelude to Apple's On-device ML.

## Installation 

Installation is very simple.


```python
# Requirements
# 1. M-series apple silicon
# 2. native Python >= 3.8
# 3. MacOS >= 13.3
!python -c "import platform; print(platform.processor())" # It must be arm
!pip install mlx 
```

    arm
    Requirement already satisfied: mlx in /Users/1113506/.venv/mlx/lib/python3.10/site-packages (0.0.5)


## Basic quick start

To create an array, let's import `mlx.core` and make an array.


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


MLX uses lazy evaluation.
[What is lazy evaluation?](https://medium.com/sjk5766/lazy-evaluation%EC%9D%84-%EC%95%8C%EC%95%84%EB%B3%B4%EC%9E%90-411651d5227b)  
Lazy evaluation is a programming methodology that defers computation until the result is actually used somewhere.
Therefore, it has advantages such as gaining in performance, avoiding errors, or being able to use infinite data structures.
The `mx.eval` function forces the computation to be performed.


```python
c = a + b # The computation doesn't actually happen, because c hasn't been used
mx.eval(c) # The computation is performed
print(c) # The computation is performed
import numpy as np
np.array(c) # The computation is performed
```

    array([2, 4, 6, 8], dtype=float32)
    array([2., 4., 6., 8.], dtype=float32)



Let's check how computation time is actually consumed.


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

MLX supports the basic function transformations `grad()` and `vmap()`. These are functions that compute the gradient and vectorize, respectively.


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


Besides these, there are functions like vector-Jacobian products `vjp()`, Jacobian-vector products `jvp()`, and `value_and_grad()`, which computes the forward-backward path simultaneously.

## Unified Memory

On Apple silicon, the CPU and GPU don't exist as separate devices. They are organized into a single unified memory architecture.
Therefore, the CPU and GPU can directly access the same memory pool. MLX is designed to take advantage of this.

When creating two arrays as below, MLX doesn't specify a particular location. This is different from specifying `to(device)` in CUDA.
```python
a = mx.random.normal((100,))
b = mx.random.normal((100,))
```

In MLX, you specify the device for an operation. That is, you can do both CPU computation and GPU computation without moving the memory location.
```python
mx.add(a, b, stream=mx.cpu)
mx.add(a, b, stream=mx.gpu)
```
The two operations above are computed via the CPU and GPU respectively. Right now these two operations have no dependency at all, so they can be computed completely in parallel.
However, when there is a dependency on a variable as below, MLX automatically finishes the first operation and then finishes the second operation.
```python
c = mx.add(a, b, stream=mx.cpu)
d = mx.add(a, c, stream=mx.gpu)
```

Depending on the type of operation, sometimes the CPU is advantageous and sometimes the GPU is.
The `matmul` operation is one that's advantageous for the GPU. However, consecutive operations made up of a for loop are advantageous for the CPU.
Therefore, the computation of the function below can minimize execution time when the GPU and CPU are used appropriately.


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


If you don't specify a stream, MLX is set to the default_device. On the M1 Pro, it's set to the GPU.


```python
print(mx.default_device())
print(mx.default_stream(mx.default_device()))
```

    Device(gpu, 0)
    Stream(Device(gpu, 0), 0)
