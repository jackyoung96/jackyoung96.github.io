---
layout: post
title: Diary - LLM distillation envidence!!
tags: archive
---

I asked various LLMs (Large Language Models) for a BFS (Breadth-First Search) example in Korean.

> "Python으로 너비우선탐색(BFS) 함수 작성해줘. (=Write a BFS function in Python.)"

The responses from the models varied slightly, but the example graph provided was identical:

```python
graph = {
 'A': ['B', 'C'],
 'B': ['A', 'D', 'E'],
 'C': ['A', 'F'],
 'D': ['B'],
 'E': ['B', 'F'],
 'F': ['C', 'E']
}
```
The models confirmed so far are:

gpt-3.5-turbo-0125  
gpt-4o-2024-08-06  
gpt-4o-mini-2024-07-18  
Llama-3.1-405B  
llama-3.1-70b-instruct  
llama-3.1-8b-instruct  
llama-3-70b-instruct  
gemma-2-2b-it  
gemma-2-9b-it  
gemma-2-27b-it  
gemini-1.5-flash-api-0514  
gemini-1.5-pro-api-0514  
claude-3-5-sonnet-20240620  
claude-3-opus-20240229  
yi-large  
yi-large-preview  
phi-3-mini-4k-instruct-june-2024  
phi-3-medium-4k-instruct  

This totals 18 models. There might be more, but it’s challenging to check each one on LMSys. However, this sample likely covers the most popular models.

---

The Qwen series produced a different graph structure, but it’s essentially the same example.

```python
graph = {
    'A' : ['B','C'],
    'B' : ['D', 'E'],
    'C' : ['F'],
    'D' : [],
    'E' : ['F'],
    'F' : []
}
```
qwen-max-0428  
qwen2-72b-instruct  
qwen1.5-110b-chat  

---

Some models did produce different examples.

llama-3-8b-instruct  
yi-1.5-34b-chat  
mixtral-8x22b-instruct-v0.1  
phi-3-small-8k-instruct  

## What does it mean?

What could be causing this? (At least 18 of the most well-known models provided the same response.)

I think there are two hypothesis:

1. There is a famous code dataset that includes this BFS example.
2. Data augmentation using a specific model was performed (distillation).

I checked various datasets to confirm the first hypothesis, but I couldn’t find this example. Of course, it might still exist somewhere. However, even if such a dataset exists, why is the same example being used? Does this imply some form of overfitting?

Additionally, the same example appears for DFS (Depth-First Search) as well. When I asked GPT-4o for DFS instead of BFS, it returned the same graph as the Qwen series!

<img width="766" alt="image" src="https://github.com/user-attachments/assets/b0506188-aa26-4d67-8e7b-5207873c40da">

So, they’re essentially sharing the same examples.

### What is the root Model?

If the second hypothesis is true, what is the root model? If the root model is ChatGPT, does this reveal that GPT distillation has been happening behind the scenes?


## The results of the models

<img width="1531" alt="스크린샷 2024-08-08 오전 10 33 01" src="https://github.com/user-attachments/assets/5c0bbccb-0e52-45e2-9f58-89a4f8494145">
<img width="1530" alt="스크린샷 2024-08-08 오전 10 31 19" src="https://github.com/user-attachments/assets/a02322f7-a519-4329-9c96-909e0e241890">
<img width="1530" alt="스크린샷 2024-08-08 오전 10 26 22" src="https://github.com/user-attachments/assets/e787b9e4-35cb-4bd5-851f-2909731a3ba2">
<img width="1532" alt="스크린샷 2024-08-08 오전 10 25 09" src="https://github.com/user-attachments/assets/ab3f6e19-a0d6-4069-87df-99c1efbbcb0f">
<img width="1530" alt="스크린샷 2024-08-08 오전 10 21 57" src="https://github.com/user-attachments/assets/fc1211a0-e9a3-433e-9aff-acb18625d81f">
<img width="1532" alt="스크린샷 2024-08-08 오전 10 19 43" src="https://github.com/user-attachments/assets/7f258a3c-c1aa-4dba-9adb-a65499ed7f85">
<img width="1530" alt="스크린샷 2024-08-08 오전 10 18 39" src="https://github.com/user-attachments/assets/bbc0a879-f9ef-48a7-b6c9-48eaecdad623">
<img width="1527" alt="스크린샷 2024-08-08 오전 10 18 00" src="https://github.com/user-attachments/assets/c180b8a6-f4b5-4af1-8524-e6d38491f5fb">
<img width="1528" alt="스크린샷 2024-08-08 오전 10 15 33" src="https://github.com/user-attachments/assets/a7a96e3e-3084-413c-8221-de8206319ac6">
<img width="1530" alt="스크린샷 2024-08-08 오전 10 14 34" src="https://github.com/user-attachments/assets/082fe8c7-60ff-4771-8089-05eb1e3cf355">
<img width="1530" alt="스크린샷 2024-08-08 오전 10 13 39" src="https://github.com/user-attachments/assets/aed9da07-1e7a-4fc1-ad5b-456a7fe67594">
<img width="1527" alt="스크린샷 2024-08-08 오전 10 12 29" src="https://github.com/user-attachments/assets/e5967a66-8342-4612-9980-6d6d415033c8">