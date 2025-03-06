---
layout: post
title: Diary - LLM knowledge distillation
tags: archive
---



어느 순간부터 gpt-4o 대신 gpt-4o-mini 를 쓰기 시작했다. 가장 큰 차이는 속도다. gpt-4o 의 추론 속도가 느린편은 아니지만, gpt-4o-mini 에 비해서는 너무 답답하다. 그렇다고 성능 차이가 나느냐? 그렇지도 않다.  

어떻게 작은 모델이 큰 모델의 성능을 이렇게 따라잡게 된 것일까? 아마 그 이유는 Knowledge Distillation 기술에 있을 것이다.

Knowledge Distillation 에 대해 공부하던 중 간단하면서도 좋은 글을 발견했다. [Post-training distillation for LLMs](https://drive.google.com/file/d/1xMohjQcTmQuUd_OiZ3hB1r47WB1WM3Am/view)

원작자 Rishabh Agarwal 의 동의를 받고, 이 내용을 번역 및 내 방식대로 재구성해보았다.

- [Introduction](#introduction)
- [Knowledge distillation 기법들](#knowledge-distillation-기법들)
  - [Supervised KD (optimizes forward KL)](#supervised-kd-optimizes-forward-kl)
  - [Distillation using “Synthetic Data”](#distillation-using-synthetic-data)
  - [GKD: Generalized knowledge distillation](#gkd-generalized-knowledge-distillation)
- [Another advantages of Distillation](#another-advantages-of-distillation)
  - [Speculative decoding](#speculative-decoding)
  - [DistillSpec](#distillspec)
- [Advanced knowledge distillation](#advanced-knowledge-distillation)
  - [SKD: speculative knowledge distillation](#skd-speculative-knowledge-distillation)
- [Conclusion](#conclusion)























## Introduction

Knowledge distillation (KD) 이란 무엇인가?

> 크고 비싼 Teacher 모델이 가지고 있는 **지식(knowledge)**을 작은 student 모델에게 **전수(transfer)** 하는 방법론

<img width="736" alt="Image" src="https://github.com/user-attachments/assets/587ef672-db03-4e3a-92a3-ce2e3daab6fb" />
    
사실 이 distillation 기법은 Scaling law 의 파괴를 가져왔다. 기존에는 Model size 가 커질 수록 높은 성능을 내는 것이 일반적이었다. GPT-1, 2, 3 로 올라가면서 parameter 수는 급격히 커졌고, 한자리 수 사칙연산도 못하던 모델이 4자리 5자리 연산을 할 수 있게 되었다 (emergent ability)

그러나 LMsys 에서 공개한 performace vs Model pricing (model 크기와 직접적인 연관) 그래프를 보면 그 공식은 점점 깨지고 있다.
<img width="768" alt="Image" src="https://github.com/user-attachments/assets/db6fb90a-8f1a-4d7d-a3f8-58870b72f11d" />

작고 싼 모델들이 크고 비싼 모델들을 따라잡고 있는 것이다. 이게 가능한 이유는 아래 Jack Morris 의 포스트만 봐도 알 수 있다.
<img width="1117" alt="Image" src="https://github.com/user-attachments/assets/0c2257df-21a7-401b-b18b-33b485e69f3e" />

결국 강력한 큰 모델이 세상에 출시된 이후 knowledge distillation 기법을 이용해 작은 모델들을 학습시켰고, 성능 격차를 크게 줄였다고 볼 수 있다.

## Knowledge distillation 기법들

그렇다면 knowledge distillation 기법들은 어떤 것들이 있을지 알아보겠다.

### Supervised KD (optimizes forward KL)

(Hinton et al., 2015, [Distilling the knowledge in a Nueral Network](https://arxiv.org/abs/1503.02531))

큰 모델과 작은 모델을 동일한 데이터셋을 이용해 학습했을 때, 가장 먼저 발견되는 차이는 Classification confidence 다.
    
<img width="1244" alt="Image" src="https://github.com/user-attachments/assets/2f7aa5e0-91aa-496b-9ebe-4e5a5bf3a678" />

<img width="1239" alt="Image" src="https://github.com/user-attachments/assets/f6e4b446-3b1e-43a2-aa6f-20465e24fd8d" />
    
LLM 으로 따지면 가장 확률이 높은 토큰들 끼리 비교했을 때 teacher 모델의 토큰 확률이 student 모델의 토큰 확률보다 높다는 것이다.  
다시 말해 두 모델이 만약 같은 응답을 생성한다면 teacher 모델의 PPL 이 더 낮다고도 볼 수 있다 (PPL 이 LLM의 성능과 직결되는 건 아니지만… loss 가 더 낮으면 보통 classification 성능은 더 올라가니까, 그런 관점이다)

    
<img width="637" alt="Image" src="https://github.com/user-attachments/assets/ba8fba65-9c14-475a-9151-f69a96222bc8" />

만약 logit 의 분포 (token 의 확률 분포) 가 같다면 두 모델의 성능은 같을 것이다. 이렇게 만드는 것이 knowledge distillation 의 목표다.

> Teacher 모델의 logit 분포와 student 모델의 logit 분포를 같게 만드는 것  
> 분포를 같게 만드는 것 == KL divergence 를 최소화 하는 것

결국 hard label (ground truth) 이 아닌 teacher 모델에서 나온 soft label 에 대해서 cross-entropy loss 를 구해 최소화 하는 것이다. 보통 LLM 학습에 사용되는 loss 를 next token prediction 이라고 하는데, 이건 next token *distribution* prediction 이라고 볼 수 있다.
        

<img width="1387" alt="Image" src="https://github.com/user-attachments/assets/54e7be61-e4f2-4f1c-85a7-f04995af05e5" />
        
그렇다면 그 성능은 어떨까?

[Gemma 2 논문](https://arxiv.org/abs/2408.00118)에서는 Gemma 2B 를 학습(pre-training)할 때 7B distillation 과 from scratch 를 비교했다.
        
<img width="1085" alt="Image" src="https://github.com/user-attachments/assets/3b51d80d-fdb4-4d2b-a0a3-1cec280439e4" />
        
결국 Distillation 하면 성능도 더 높고 PPL 도 더 낮기 때문에 강력한 모델이 있다면 굳이 from scratch 학습을 할 필요가 없다

### Distillation using “Synthetic Data”

(Kim et al., 2016, [Sequence-Level Knowledge Distillation](https://aclanthology.org/D16-1139.pdf)) → ~~SKT분들에겐 익숙한 이름이...!!~~

사실 Supervised KD 는 logit 을 이용해야 하는데, 이게 간단해보여도 LLM 의 logit 은 token 1개당 최소 100K 사이즈의 embedding 이기 때문에, sequence 가 조금만 길어져도 메모리 이슈가 발생할 수 있다. (Top-K 만 사용하는 방법도 있지만) 그래서 더 간단하면서도 강력한 synthetic data 를 이용한 방법이 제안되었다.
    
<img width="1504" alt="Image" src="https://github.com/user-attachments/assets/d7dec726-1a08-4fcf-865d-91427bc20450" />
    
Prompt 데이터를 가지고 teacher 모델 응답을 뽑는다. 그리고 그 데이터로 student 모델에 대해 SFT를 수행한다.  
이 방법론의 최고 장점은…API access 만 있어도 distillation 이 가능하다는 것이다! (logit 이런거 몰라도 되니까 gpt-4o 같은 최신 LLM 으로 distillation 해서 성능을 올릴 수 있다)

<img width="449" alt="Image" src="https://github.com/user-attachments/assets/8a812891-caab-4ea9-b349-0d5b4220d01b" />
        
얼마전 출시된 DeepSeek에 대한 샘 알트만의 저격 포스팅도 비슷한 맥락이다.
            
<img width="401" alt="Image" src="https://github.com/user-attachments/assets/917fffbb-6c47-426b-b707-246120e44fab" />
            
그런데 이 synthetic data 방법론이 이래봬도 이론적 배경이 있다.
    
<img width="1177" alt="Image" src="https://github.com/user-attachments/assets/7d127bd5-b19f-4caa-8d62-1958d1f9ee28" />
    
이 최종 식의 기댓값 (expectation) 부분을 *Monte-Carlo approximation* 해주면 결국 synthetic data distillation 방법론인 것이다. 따라서 Synthetic data distillation 방법론을 사용하면 Supervised KD 와 동일하게 teacher 모델과 student 모델 사이의 KL divergence 를 최소화 하게 된다.


또 더 고퀄리티 답변을 얻어낼 수록 student 모델의 성능도 더 좋아진다

대표적인 방법으로 Rejection sampling (BoN) 이 있다. Teacher 모델로부터 N개의 답변을 생성해, 그 중 가장 좋은 답변만을 가지고 student 를 학습하는 것이다.
<img width="1414" alt="Image" src="https://github.com/user-attachments/assets/57464ec9-ce49-4095-8934-c8f86f6afa7c" />    

확장된 방법으로는 Compute-matched / Cost-matched sampling 이 있다. BoN을 쓸 때 더 작은 teacher 모델을 사용하고 N을 늘린다 (더 싼 모델을 사용하는 것과 동치) 
<img width="712" alt="Image" src="https://github.com/user-attachments/assets/de1bf72d-04a2-4646-90cf-106a78463be8" />    
웃긴건 심지어 Student 모델보다 작은 teacher모델을 사용해서 N 을 늘리는 것도 유의미한 성능 향상이 관측되었다
<img width="1540" alt="Image" src="https://github.com/user-attachments/assets/8201f4af-35c0-4d95-a5b7-e674b22a673a" />
        
이 방법론의 엄청난 장점은 두 가지다.
아까 말했듯, API access 로도 knowledge distillation이 가능하다. (DeepSeek R1 이전에 SOTA 모델들은 대부분 closed-source 였다) Open-source 모델이 아니더라도 API 사용을 막을 방법은 없다. 
그리고 Tokenizer 가 달라도 동작한다! (forward KL 은 logit 을 매칭해야 하므로 tokenizer 가 완전히 동일해야 함) 결국 아예 아키텍쳐나 근본이 다른 모델이더라도 distillation 이 가능하다는 것이다.

(이건 안쓸 이유가 없잖아...??)

### GKD: Generalized knowledge distillation

(Agarwal et al., 2023, [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://openreview.net/forum?id=3zKtaqxLhW))

위에서 소개한 두가지 distillation 방법론들에게는 한 가지 치명적인 문제가 있는데, 바로 train-inference mismatch 다.

Training distribution 과 inference distribution 이 다르면 OOD (Out-of-distribution) 상황에 빠지는 경우가 잦아진다. 특히 student 모델이 teacher 모델에 비해서 성능이 떨어지는 상황이기 때문에 잘못된 token 이 한 번 나오면 완전 이상한 답변으로 새버릴 수도 있다.
        
<img width="1196" alt="Image" src="https://github.com/user-attachments/assets/e8475889-c470-44af-9839-c4b0834f1f2a" />

Training distribution 은 teacher 모델의 응답, Inference distribution 은 student 모델의 응답이므로, 아무리 두 분포가 비슷해지도록 학습한다 해도 OOD 에 취약해지는 구조이다.

이를 해결하기 위해서 On-policy distillation 방법론이 제안되었다. 이 방법론의 목표는 Train-inference distribution 을 일치시키면서 학습하는 것이다.
        
<img width="1220" alt="Image" src="https://github.com/user-attachments/assets/8a0dfb1a-98d7-4227-b307-7fcf6d75d7a5" />
        
- **On-policy data**: Student 모델의 응답을 사용
- **Feedback**: Student 모델의 응답에 대한 Teacher 모델의 logit 을 사용
- **Supervised training**: Teacher - student token-level logit 분포가 같아지도록 학습

이걸 수학적으로 표현해 보면 **Reverse KL** 인 것을 알 수 있다.
        
<img width="938" alt="Image" src="https://github.com/user-attachments/assets/563d4a73-0400-4f92-b3ef-4f333889e46b" />

아까 synthetic data 방법론에서 사용된 KL 과 비슷하지만 studen 와 teacher 의 분포 위치가 바뀐 것을 볼 수 있다. (KL-divergence 는 asymmetry 하므로 같지 않다!)

물론 수식이 같지는 않더라도 KL divergence 를 최소화하는게, student 모델과 teacher 모델의 분포가 같아지도록 하는 목적이고, 이것 자체는 foward KL 과 같은데…그럼 무엇이 다른 것일까?

Teacher 모델과 student 모델은 expressiveness (model capacity) 가 근본적으로 다르다 **(표현할 수 있는 차원 == parameter 의 수)**

Forward KL을 사용하면 student 분포는 mode-covering 을 하도록 학습된다.
<img width="665" alt="Image" src="https://github.com/user-attachments/assets/5831c20b-69fa-49a7-b549-5409c902e222" />
반면 Reverse KL을 사용하면 mode-seeking 을 하도록 학습된다.                
<img width="637" alt="Image" src="https://github.com/user-attachments/assets/f9ad8d90-81d8-4cc4-85f6-1c81d20133d0" />
(forward KL, Reverse KL 에 대해서는 기똥찬 블로그 글이 하나 있다 [forward KL, Reverse KL 설명](https://process-mining.tistory.com/147))

Expressiveness 가 작은 student 모델에게는 mode-seeking 이 더 차라리 적합할 수도 있다. 왜냐하면 쌩뚱맞은 token 이 나오게 되는 것보다는 teacher 모델의 secondary token 이 나오는 게 낫기 때문이다. 그치만 teacher 의 mode 가 뾰족뾰족하지 않고 충분히 generalized 되어 있다면 mode-covering 이 나을 수도 있다.

<img width="1238" alt="Image" src="https://github.com/user-attachments/assets/936ce468-e52e-469f-9aa9-cb9b61fb5d6c" />

그래서 실험을 해보니, forward KL과 reverse KL 을 섞어서 쓰는게 가장 좋았다. (0.5 forward KL + 0.5 reverse KL = Jeffrey’s divergence)
                

아무튼 이론적인 부분들을 알아봤는데, 과연 GKD 방법론을 어떻게 구현할 수 있을까? 가장 간단한 방법으로는 [TRL 에 구현된 GKD](https://huggingface.co/docs/trl/main/en/gkd_trainer)를 쓰는 것이다. 또는 RLxF (RLHF, RLAIF) 코드를 응용하는 것이다.
        
<img width="1345" alt="Image" src="https://github.com/user-attachments/assets/9b675f3d-1802-4062-b207-2dec8bf0ab2e" />

        
RLxF 코드에는 RL을 수행할 때 SFT 분포와 멀어지지 않도록 하는 regularizor로 KL term 이 들어 있는데, SFT 대신 teacher 모델을 넣어준다. 그리고 Reward 관련 term 을 삭제하면 GKD 완성

Reward 관련 term 을 삭제하지 않으면 RL + distillation 두마리 토끼를 잡을 수 있다는 내용도 있다. (물론 RL 이 들어가면서 학습 불안정성이 높아질 것이다)
                
<img width="902" alt="Image" src="https://github.com/user-attachments/assets/e50f272d-e9a7-404d-8f0a-58bad724a8aa" />


## Another advantages of Distillation

사실 Distillation 의 장점은 student 모델의 성능 개선에만 있는 것이 아니다!

### Speculative decoding

(Leviathan et al., 2023, [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192))

Transformers는 학습 병렬화를 통해 거대 모델의 학습이 가능하게 했지만, 추론 할 때 엄청 느리다. Input sequence 에 대한 logit 은 한번에 구할 수 있는데, token 을 생성할 때에는 sequential 하게 생성해야 하기 때문이다.

그래서 나온 Speculative decoding은 큰 모델의 답변과 비슷하게 답변하는 작은 모델을 통해 큰 모델의 속도를 높이는 방법이다.
    
<img width="1439" alt="Image" src="https://github.com/user-attachments/assets/50b48e0b-86e2-47a6-922d-dc2aae18955a" />
    
- 작은 모델은 빠르게 생성할 수 있으니까 일단 token 여러개를 생성한다
- 큰 모델로 token 여러개에 대한 logit 을 한번에 뽑아서, 잘못 생성된 token 을 찾는다.
- 잘못 생성된 token 부터 다시 작은 모델로 토큰을 생성한다 -> 반복

이게 되려면 큰 모델과 작은 모델의 답변이 비슷해야 속도가 올라간다. 만약에 모든 토큰이 다 달라버리면, 괜히 작은모델을 태워서 리소스만 더 잡아먹는 것이다. 

### DistillSpec

(Zhou et al., 2024, [DistillSpec: Improving Speculative Decoding via Knowledge Distillation](https://arxiv.org/abs/2310.08461))

그런데 같은 데이터셋으로 학습했다고 하더라도 작은 모델과 큰 모델의 답변이 비슷할거라는 보장이 없다. 따라서 Distillation 을 통해서 speculative decoding 성능을 극대화하는 방법이 제안되었다.
    
<img width="752" alt="Image" src="https://github.com/user-attachments/assets/7e04f239-9a5b-4fa0-b97b-2c2a22890a61" />
    
Distillation을 하면 student model 의 답변들이 teacher 와 비슷해지면서 10~45% 추론 속도가 증가했다. 이 방법론은 실제로 Google 검색 페이지에 적용되어 있다
    
[Google에 적용된 speculative decoding 예시](https://storage.googleapis.com/gweb-research2023-media/media/SpeculativeDecoding-0-AIO.mp4)
    
## Advanced knowledge distillation

### SKD: speculative knowledge distillation

(Agarwal et al., 2025, [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649))

가장 최근에는 이 Speculative decoding을 응용한 하이브리드 방법론도 제안되었다.

GKD 가 성능이 좋기는 하지만, Student 의 on-policy response 가 썩 좋지 않은 상태에서 GKD 를 하는 게 괜찮을까에 대한 의문이 있다. 이런 상황에서는 off-policy 더라도 퀄리티 좋은 teacher 의 response 를 써서 Supervised KD 를 하는게 나을 수도 있다.  
이 두 방법론의 장점만을 취한 방법론이 SKD 다.
    
<img width="1288" alt="Image" src="https://github.com/user-attachments/assets/2d16d9b8-1753-460d-b1a2-200163024253" />

우선 Student 의 on-policy response 를 뽑는다. 그리고 각 token이 teacher 의 top-K token 에 포함되어 있는지 확인한다.
만약 모두 포함되어 있는 경우라면 on-policy 응답이 꽤 좋은 상황이므로 GKD를 수행한다. 
포함되지 않는 토큰이 있는 경우라면 해당 토큰을 Teacher 의 token 으로 교체하여 퀄리티를 높이고, 이를 이용해 Supervised KD를 수행한다. 

이 SDK 방법론을 적용했을 때 성능도 오르고 speculative decoding 속도도 올랐다고 한다.
    
<img width="1278" alt="Image" src="https://github.com/user-attachments/assets/716ee346-6a67-45e1-8ec4-95ef67f86505" />
    
왜냐면 Student 의 성능 관점에서는 on-policy distillation 이 낫고, Teacher 의 speculative decoding 관점에서는 teacher response 를 이용한 distillation 이 낫기 때문이다. 

## Conclusion

| Compute efficiency | Online < Offline |
| --- | --- |
| Sample efficiency | Online > Offline |
| 학습시 실시간 teacher 모델로 인한 리소스 낭비 | Online < Offline |
| Student sampling 으로 인한 시간 지연 | Online < Offline |
| Long horizon task 적합성 | Online > Offline |
| Train-test distribution mismatch | Online > Offline |
- Online KD 의 장점
    - 데이터가 많이 필요하지 않음
    - Train-test distribution mismatch 가 작기 때문에 Agent 와 같은 long horizon task 에 적합함 (OOD 에 쉽게 빠지지 않음)
- Offline KD 의 장점
    - Teacher 모델로 한번만 response 생성해두거나 logit 을 저장해두면 계속 사용할 수 있음
        - logit을 미리 뽑아두면 teacher model 에게 필요한 GPU 리소스도 아낄 수 있음
    - Student on-policy generation 과정이 없기 때문에 학습 속도가 빠름

Knowledge distillation 의 trade-off 는 결국 리소스와 성능이다. 성능이 올라가는 방법론일수록 컴퓨팅 리소스 / 시간 리소스를 더 많이 필요로 한다.  
따라서 상황에 맞춰 적당한 distillation 방법론을 선택하도록 하자