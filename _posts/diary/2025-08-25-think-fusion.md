---
layout: post
title: Diary - DeepSeek-V3.1, GPT-5, 그리고 think-fusion
tags: archive
---


2025년 LLM 필드에서 가장 중요한 키워드 하나를 꼽으라면 바로 'Reasoning'일 것입니다. Test-time-computing을 통해서 LLM은 사람처럼 생각하는 과정을 거쳐 답변을 이끌어냈고, 이는 다양한 벤치마크 성능을 극적으로 끌어올렸습니다. 하지만 이 Reasoning 모델은 일반 사용자들 (단순한 태스크에 LLM을 적용하려는) 에게는 큰 영향을 주지 못했습니다. 금방 대답할 수 있는 질문들에도 쓸데없는 사고과정을 거치면서 답변을 기다리는 사용자들을 지치게 만들곤 했기 때문입니다.

그래서 제안된 방식 중 하나가 바로 Think-fusion입니다. 이전 블로그 글에서 [Qwen3에 적용된 Hybrid think mode](https://jackyoung96.github.io/2025/05/01/Qwen3-hybrid-think/) 에 대해서 살펴본 바 있습니다. Think-fusion이 적용된 LLM은 단일 모델 (unified model)이 reasoning과 non-reasoning 방식을 모두 지원합니다. 이 글에서는 think-fusion을 구현하는 여러 방식과, 최근 모델들의 think-fusion 방식을 통해 LLM의 방향성을 살펴보려 합니다.

### Think Fusion의 다양한 구현 방식

> 📌 **알아두어야 할 점**  
> LLM은 기본적으로 `<|im_start|>user\n[query]<|im_end|><|im_start|>assistant\n` 와 같이 사용자의 query 앞뒤에 turn indicator를 감싸고, `assistant` indicator 뒤에 응답을 생성하도록 학습됩니다. 

Think fusion은 기본적으로 사용자가 'Think 모드'를 활성화했을 때, 모델이 `<think>[reasoning]</think>[response]` 형태의 답변을 생성하도록 학습됩니다. 하지만 모델별로, 그리고 'Non-think 모드'일 때의 응답 형태에 따라 세 가지 구현 방식의 차이가 존재합니다.

| Case | Non-think 모드 | 사용 모델 |
| ---- | ------------- | ------- |
| 1   | `[response]`  | Llama-Nemotron |
| 2   | `<think></think>[response]` | Qwen3, EXAONE-4.0 |
| 3   | `</think>[response]` | DeepSeek-V3.1 |

각각의 방식을 조금 더 자세히 살펴보겠습니다.

**Case 1 (Llama-Nemotron 방식)**  
가장 직관적인 방식입니다. 사용자가 Think 모드를 끄면, 모델은 추론 과정을 생략하고 최종 답변([resp])만 내놓습니다. Think 모드를 사용하기 위해서는 `assistant` indicator 뒤에 `<think>` 를 미리 붙인 뒤 응답을 생성하도록 합니다.

**Case 2 (Qwen3, EXAONE-4.0 방식)**  
사용자가 Think 모드를 끄면, <think></think> 태그를 출력한 후 답변을 제공합니다. 이는 빈 reasoning을 생성한 것과 동일합니다. 이는 모델이 항상 '생각' 단계를 거치도록 구조적으로 템플릿을 유지하되, Non-think 모드에서는 그 내용을 비워두는 방식으로 응답 형태의 일관성을 유지하려는 시도입니다. Think 모드를 사용하기 위해서는 `assistant` indicator 뒤에 `<think>` 를 미리 붙인 뒤 응답을 생성하도록 하고, Non-think 모드를 사용하기 위해서는 `assistant` indicator 뒤에 `<think></think>` 를 미리 붙인 뒤 응답을 생성하도록 합니다.

**Case 3 (DeepSeek-V3.1 방식)**  
사용자가 Think 모드를 끄면 `assistant` 뒤에 바로 `</think>` 토큰이 나타나고, 그 이후에 답변 생성되는 다소 특이한 형태입니다. Think 모드를 사용하기 위해서는 `assistant` indicator 뒤에 `<think>` 를 미리 붙인 뒤 응답을 생성하도록 하고, Non-think 모드를 사용하기 위해서는 `assistant` indicator 뒤에 `</think>` 를 미리 붙인 뒤 응답을 생성하도록 합니다.

세 가지 방식이 별로 다를 것 없어 보이지만, 확률적으로 다음 토큰을 생성하는 LLM의 특성을 생각해보면 큰 차이가 있습니다.

### Think-fusion 구현 방식에 따른 차이점

LLM 은 next token prediction loss에 의해서 학습된 모델입니다. 따라서 학습 때 본 적 있는 sequence 라면 해당 token 이 생성될 가능성이 존재합니다.

Case 1의 경우, Non-think 모드를 사용하기 위해서 `assistant` 뒤에 아무것도 붙이지 않고 inference 를 수행하는 상황을 가정해보겠습니다. Think 모드를 위해서 `assistant` 뒤에 `<think>` 토큰이 있는 데이터가 학습된 적 있기 때문에 Non-think 모드라고 할지라도 `<think>` 토큰이 생성될 수 있습니다. 만약 생성된다면 think 모드로 답변하게 됩니다.

반면 Case 2의 경우, Think 모드를 사용하기 위해 `<think>` 만 붙여 inference 를 했는데, 바로 `</think>` 가 나오면서 Non-think 모드가 되어버릴 수 있습니다. 

Case 3의 경우, 앞의 경우와는 다르게 두 모드가 완전히 분리되어 있습니다. `<think>` 로 시작한다면 `</think>` 가 나올때까지 reasoning 을 하게 될 것이고, `</think>` 로 시작한다면 바로 응답이 생성될 것입니다.

따라서 Case 1과 Case 2의 경우, 각각 Non-think 모드와 Think 모드에서는 사용자가 아닌 모델이 모드를 선택하게 됩니다. 학습된 데이터의 비율이나 학습 레시피에 따라서 그 비중이 결정될겁니다.

| Case | Think 모드 사용시 | Non-think 모드 사용시 |
| ---- | ------------- | ------- |
| 1   | Think  | Think/Non-think 모델이 결정 |
| 2   | Think/Non-think 모델이 결정 | Non-think |
| 3   | Think | Non-think |

그렇다면 반드시 Case 3가 좋은 것일까요? 장담할 수는 없습니다. Case 1과 Case 2는 Think/Non-think 에서 사용하는 템플릿이 일관성을 유지합니다. 반면 Case 3는 특정 토큰이 나오면서 이후에 아예 다른 distribution 을 가지게 됩니다. 이는 모델 학습시에는 sample efficiency 를 떨어뜨릴 가능성이 있습니다. 다시 말해, Case 1과 Case 2가 학습이 더 효율적일 수도 있습니다. 반면 Case 1과 Case 2는 모델이 Think/Non-think 를 잘 결정할 수 있도록 하는 데이터 mixture 를 잘 찾아야 하겠습니다.


### Think-fusion을 사용하지 않고 다시 모델 분리
그런데 최근 공개된 [GPT-5](https://openai.com/index/gpt-5-system-card/)는 오히려 Reasoning 모델 (gpt-5-thinking) 과 Instruct 모델 (gpt-5-main) 을 아예 분리해 버렸습니다. 그리고 모델 앞에 **Router**를 붙였습니다. 사용자의 질문을 router가 먼저 분석하고 이 질문이 복잡한 추론을 요구하는지, 아니면 간단한 지시 수행이나 정보 검색으로 충분한지를 판단합니다. 그리고 판단에 따라 요청을 전문화된 각 모델로 전달합니다.

Alibaba의 [Qwen3-235BA22B도 2507 버전](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507)으로 Instruct-only 모델을 내놓았습니다. 기존의 think-fusion 을 업그레이드 하지 않고, instruct-only 모델로 변경하여 내놓은 것입니다.

### Think-fusion 의 단점?
GPT-5와 Qwen3는 Think/Non-think 모델을 분리하여 학습 하고 있고, think-fusion의 case 3방식을 사용하는 DeepSeek-V3.1 또한 Think/Non-think의 distribution 분리하려는 시도를 한 것을 볼 수 있습니다. 비교적 최신 모델들이 think-fusion 을 분리하려는 이유는 무엇일까요?

몇 가지 추론을 해볼 수 있습니다.

1. **성능 저하의 가능성**: 하나의 모델에게 복잡한 추론 태스크와 간결한 답변 생성이라는 두 가지 작업을 모두 잘 해내도록 학습시키는 것은 둘 중 하나의 태스크에서 성능 저하를 가져올 수 있습니다. 실제로 Think-fusion을 적용한 모델들은 Non-think모드 성능이 썩 좋다고 느껴지지는 않습니다. 실제로 Reasoning 은 attention 쪽 parameter에 많은 학습이 이루어지는 반면, 일반 Instruct 데이터는 MLP 쪽 parameter 에 업데이트가 많이 이루어집니다. 이러한 학습 불균형이 성능 저하를 가져올 수 있다고 봅니다.

2. **훈련의 복잡성**: 위에서 말했듯 Think와 Non-think 데이터의 적절한 비율을 찾아야 하는데, 이 작업은 굉장히 laborious 할 것으로 예상됩니다. 비율 조절에 실패하면 사용자가 의도하지 않은 상황에서 과도한 추론을 통해 서비스 속도가 저하되거나, 추론이 필요한 경우에도 일반 답변만 하여 성능이 떨어지는 경우가 발생할 수 있습니다.

3. **효율성 저하**: 사실 대부분의 태스크들은 reasoning 능력까지는 필요로 하지 않습니다. Instruct 모델에서도 CoT 를 적당히 활용한다면 좋은 답변을 얻어낼 수 있습니다. 또한 vllm 등에서 batch 처리를 하기 위해서는 response 길이가 유사해야 하는데, think-fusion 모델은 응답 길이가 천차만별이기 때문에 효율성이 떨어집니다. 차라리 모델을 분리하여 따로 서빙하는 것이 효율적일 수 있습니다.


### 끝으로 
2025년 상반기에 유행했던 Think-fusion은 단일 모델로 성능과 사용성이라는 두마리 토끼를 한번에 잡는 방법처럼 보였습니다. 하지만 최근 공개된 DeepSeek-V3.1과 GPT-5의 흐름에서 볼 수 있듯이, 다시 Think/Non-think의 분포를 분리하는 듯한 움직임을 관찰할 수 있었습니다. 앞으로 나오는 SOTA 모델들은 과연 어떤 방식을 사용하게 될까요?