---
layout: post
title: Diary - LLM distillation 증거 찾기
tags: archive
---

다양한 LLM들에게 BFS 예제를 물었보았다.
> "Python으로 너비우선탐색(BFS) 함수 작성해줘."

그런데 모델들의 답변은 조금씩 달랐지만 예시 그래프의 형태가

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

로 완전히 동일했다. 일단 확인된 모델은 

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

까지 총 18개 모델이다. 물론 더 있을 수 있기는 한데 LMSys 에서 일일히 확인하기가 좀 어려운 관계로... 그리고 요정도면 유명한 모델들은 다 동일한 답변이라고 보면 된다.  

---

그리고 Qwen 시리즈는 형태는 다르지만 동일한 예제를 뱉기는 한다.

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

물론 다른 예제를 뱉는 모델도 있긴 있었다.

llama-3-8b-instruct  
yi-1.5-34b-chat  
mixtral-8x22b-instruct-v0.1  
phi-3-small-8k-instruct  

## 시사점

이게 대체 어떻게 된 일일까? (적어도 가장 유명한 18개 모델에서 동일한 답변이 나왔다)

가능성은 둘 중 하나라고 생각한다.
1. 해당 BFS 예제가 포함되어 있는 유명한 코드 데이터셋이 존재한다.
2. 특정 모델을 이용해 data augmentation 을 진행했다 (distillation)

일단 1번 가능성에 대해 확인해보기 위해서 여러 데이터셋들을 뒤졌는데, 해당 예제를 찾지 못했다. 물론 내가 못찾은 것일 수도 있다.
그러나 설사 데이터셋이 있다고 하더라도 왜 동일한 예제가 나오는 것인가? 이건 뭔가 overfitting 되었다는 것 아닐까?

심지어 BFS 뿐만 아니라 DFS 에서도 똑같은 예제가 나온다는 점이다. GPT-4o 에게 BFS가 아닌 DFS 를 물어보았다.

<img width="766" alt="image" src="https://github.com/user-attachments/assets/b0506188-aa26-4d67-8e7b-5207873c40da">

Qwen 시리즈에서 나온 그래프 시리즈가 아닌가!!!

결국 다 같은 녀석들을 공유하고 있는 셈이다. 

### Root model 은?

만약 가능성 2번이 사실이라면 root model 은 대체 뭘까? Root model 이 만약에 chatGPT 라면 암암리에 수행했던 gpt distillation 이 사실로 드러나는 건 아닐까나...