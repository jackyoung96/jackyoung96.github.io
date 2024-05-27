---
layout: post
title: Anthropic post - Mapping the Mind of a Large Language Model
tags: archive
---

이름부터가 아주 흥미로운 Anthropic 의 포스팅

## 포스팅의 주요 내용 요약

> 🏬 **ML 모델의 내부를 살펴보는 일은 크게 의미는 없는 일이었다. 의미가 명확하지 않은 숫자의 나열이기 때문이다.**  
> Opening the black box doesn't necessarily help: the internal state of the model—what the model is "thinking" before writing its response—consists of a long list of numbers ("neuron activations") without a clear meaning

> 🏬 **“Dictionary learning”
Many unclear active neuron 대신 few active features 의 dictionary 를 만들어 의미를 관측한다**  
> In turn, any internal state of the model can be represented in terms of a few active features instead of many active neurons. Just as every English word in a dictionary is made by combining letters, and every sentence is made by combining words, every feature in an AI model is made by combining neurons, and every internal state is made by combining features.

> 🏬 Base of dictionary learning  
> **Sparse autoEncoder 를 사용하여 neuron activation 으로부터 feature 를 추출한다**  
> Sparse AutoEncoder 란? [https://soki.tistory.com/64](https://soki.tistory.com/64)  
> Anthropic 의 feature decomposing: [https://transformer-circuits.pub/2023/monosemantic-features](https://transformer-circuits.pub/2023/monosemantic-features)  
> Superposition hypothesis: Model 이 Neuron 의 수보다 더 많은 특징을 표현하는 상태임을 가정 → 고차원 공간의 특성을 활용하여 더 큰 신경망을 estimate 한다는 가정  
> **feature 공간의 dimension 은 neuron activation demension 보다 커야 함**  
> 따라서 Sparse AutoEncoder 를 사용한다 → Latent dimension 가 input dimension 보다 256배 크다


> 🏬 **Feature distance  
> Neuron activation patterns 를 기반으로 feature 의 distance 를 측정할 수 있다.**  
> We were able to measure a kind of "distance" between features based on which neurons appeared in their activation patterns.  
> **Golden Gate Bridge 라는 명사는 Alcatraz Island, Ghirardelli Square, the Golden State Warriors 등의 명사와 유사한 feature 를 가진다**  
> This allowed us to look for features that are "close" to each other. Looking near a "Golden Gate Bridge" feature, we found features for Alcatraz Island, Ghirardelli Square, the Golden State Warriors, California Governor Gavin Newsom, the 1906 earthquake, and the San Francisco-set Alfred Hitchcock film *Vertigo*.  
> **Inner conflict 는 catch-22 (예상했던 결과를 얻을 수 없는 궁지상태' 나 '딜레마' 또는 '굉장히 어려운 상황’) 과 같은 비유적 표현과도 유사한 feature 를 가진다→ Claude 의 비유와 은유 능력을 어느정도 설명)**  
> looking near a feature related to the concept of "inner conflict", we find features related to relationship breakups, conflicting allegiances, logical inconsistencies, as well as the phrase "catch-22”

> 🏬 **Manipulate features  
> 특정 feature 를 activate 하면 model response를 control 할 수 있음**  
> But when we ask the same question with the feature artificially activated sufficiently strongly, this overcomes Claude's harmlessness training and it responds by drafting a scam email.  
> **Input context 에 반응하는 것이 이러한 feature activation 과 관련 있을 뿐 아니라, 모델이 feature 를 통해 world representation 을 이해하고, 이를 기반으로 behavior 를 사용한다는 것임**  
> The fact that manipulating these features causes corresponding changes to behavior validates that they aren't just correlated with the presence of concepts in input text, but also causally shape the model's behavior.


> 🏬 **Make the model safer**  
> **Feature 감지를 통해 AI monitoring 이 가능하고, 특정 주제를 제거하는 등의 용도로 사용 가능함**  
> For example, it might be possible to use the techniques described here to monitor AI systems for certain dangerous behaviors (such as deceiving the user), to steer them towards desirable outcomes (debiasing), or to remove certain dangerous subject matter entirely.

### 요약

- MLP layer의 activation pattern을 Sparse AutoEncoder 를 통해 feature 화 할 수 있다. 이는 고전 ML 에서 사용되던 Dictionary learning 방식이다
- LLM 에 Dictionary learning 방식을 사용했더니 연관된 단어 뿐만 아니라 유사한 high-level 개념이나 은유/비유 등의 모호한 개념들도 feature 공간에서 가까운 거리에 있음이 발견되었다.
- 특정 feature 를 일부러 강화하면 해당 feature 와 유사한 답변이 생성되는 것을 실험적으로 확인하였다. 이는 in-context learning 등이 특정 feature 를 강화시켜서 해당 내용과 연관된 답변을 하는 것을 설명할 수 있다. **LLM 은 feature 를 통해서 world representation 을 이해하고 있다!**
- 특정 Input/Output 이 어떤 feature 와 가까운지 알 수 있기 때문에 **LLM 을 더 safe 하게 만들 수 있다.**

### Anthropic 의 dictionary features

[https://transformer-circuits.pub/2023/monosemantic-features/vis/index.html](https://transformer-circuits.pub/2023/monosemantic-features/vis/index.html)

![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/49431c8f-959f-4bdc-a1f1-a21ad96e519c)
