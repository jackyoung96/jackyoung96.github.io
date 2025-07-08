---
layout: post
title: Diary - vLLM 의 cascade attention
tags: archive
---

어제, 모 회사의 LLM을 써보려고 벤치마크를 돌려봤다. 근데 이상하게, 답변이 반복이 엄청 생기고 다 깨지는 현상이 발생했다.  
분명 vLLM 그냥 띄웠을 때 숫자는 잘 나왔는데...?? 그래서 벤치마크에서 사용하는 프롬프트를 그대로 넣어봤다. 

**벤치마크 KO-IFEval 중 하나**
> Q: 용궁에 대한 수수께끼를 쉼표를 사용하지 말고 작성하세요.	

> A (벤치마크): 용궁에는 수수께끼가 있습니다. 용궁에 대한 수수께끼를 풀어보세요. 용궁에 대한 수수께끼를 풀어보세요. 용궁에 대한 수수께끼를 쉼표 없이 작성하세요. 쉼표를 사용하지 않고 답을 작성하세요. 용궁에 대한 수수께끼를 쉼표로 작성하세요. 쉼표를 사용하지 않고 답을 작성하세요. 용궁에 대한 수수께끼를 풀어보세요. 용궁에 대한 수수께끼를 쉼표를 사용하지 않고 작성하세요.

> A (vLLM): 깊은 바다 속에 숨겨진 용궁은 어떤 모습일까요? 그곳에는 황금빛 물고기들이 춤추고 진주가 하늘에서 내리는 궁전이 있다고 합니다. 그러나 아무나 들어갈 수 없는 이곳은 오직 바다의 비밀을 아는 자만이 방문할 수 있는 신비로운 장소입니다. 과연 용궁의 문을 열 수 있는 열쇠는 무엇일까요?

이런게 한 두 가지가 아니었다. Sampling parameter 차이인가 싶어서 이것저것 다 확인해봤는데, 전부 동일했다. 유일하게 달랐던 것 하나는 **tensor-parallel-size** 였다.

Tensor-parallel-size 는 GPU 한장에 올릴 수 없는 모델들을 GPU 여러장에 나눠서 올리기 위한 옵션이다. **절대로 성능에 영향을 주어서는 안되는** 옵션이라고 볼 수 있다. TP 자체가 문제는 아닐 것 같아서 이것저것 이슈를 찾다가 batch-size이 커지니 응답에 반복이 발생했다는 버그를 찾았다.

https://github.com/vllm-project/vllm/issues/17652

웃긴건 batch size 50 에서는 반복발생하다가 batch size 100 은 또 정상이라고 한다 ㅋㅋ

암튼 이 이슈에서 낸 결론은 `disable_cascade_attn=True` 를 줘야한다는 것이다.

### Cascade attention

Cascade attention 의 핵심은 Multi query 가 들어온 경우 겹치는 부분 (shared query prefix) 의 KV-Cache와 나머지 부분의 KV-Cache 를 분리하는 것이다. 

<img width="817" height="510" alt="Image" src="https://github.com/user-attachments/assets/21a8514e-7a22-4795-9829-1bfe1f98fdb6" />

[출처](https://flashinfer.ai/2024/02/02/cascade-inference.html)

이렇게 함으로써 얻는 이점은 2개인데, 우선 당연하게도 shared KV-Cache 를 N배 적게 사용하니까 메모리 이득이 발생한다는 점, 그리고 작아진 shared KV-Cache 를 L2 Cache 가 아닌 SMEM/Register 에 올려둠으로써 엑세스 속도를 훨씬 빠르게 가져갈 수 있다는 점이다. 

Devide-and-conquer 를 통해 중복되는 메모리를 줄이고, 추론속도를 향상시킨 알고리즘이라 볼 수 있겠다. 당연하게도, 결과에는 영향을 주면 안된다 (이론적으론)

### 이게 왜 발생?

웃긴건 H100 에서는 해당 버그가 발생하지 않는데 A100 GPU 에서만 버그가 발생한다.  
Tensor-parallel 도 결국은 devide-and-conquer 를 하는거고, cascade attention 도 마찬가지다.  
두 개를 동시에 하려다가 뭔가 연산이 꼬이는 것 같은데... 아주 심각한 버그라고 볼 수 있다.  
그리고 이 문제를 처음 발견한 이유가, 이번에 벤치마크 돌린 모델이 system message 가 꽤나 길게 박혀있는 모델이기 때문이다. (shared prompt prefix 가 길다는 것! cascade attention 이 반드시 activate 된다.) 

실제로 `disable_cascade_attn` 옵션 document 를 보면
> Disable cascade attention for V1. While cascade attention does not change the mathematical correctness, disabling it could be useful for preventing potential numerical issues. Note that even if this is set to False, cascade attention will be only used when the heuristic tells that it's beneficial.

라고 되어 있다. ~~아니 이러면 당연히 default 를 True 로 해놔야 하는거 아니야??~~

일단 적어도 2주 전까지는 해당 이슈가 해결되지 않았다.  
내부를 한 번 까볼까 싶다. 암튼 해결될 때 까지는 disable_cascade_attn=True 로 놓는걸로 (난 H100 이 없으니까...)