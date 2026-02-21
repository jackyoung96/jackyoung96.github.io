---
layout: post
title: "Diary - LLM에서 효율적인 강화학습이란 무엇일까"
tags: archive
---

지난주에 Microsoft에서 [Experiential Reinforcement Learning](https://arxiv.org/pdf/2602.13949) 이라는 재미있는 제목의 논문이 나온 걸 봤다. 사실 내용 자체에는 꽤나 허점이 있다고 생각하고 따로 코드도 없어서 돌려볼 수는 없었지만, 강화학습에 대해서 이것저것 생각해 볼 거리가 있었다. 리뷰와 함께 생각을 정리해 본다.

## 사실 강화학습은 비싸다

LLM 학습을 해보면 강화학습이 아니고서야 성능을 극한으로 끌어올릴 방법이 없다. 왜냐면 LLM의 학습방법 중 유일하게 **사용자가 생각하는 LLM의 성능**을 objective로 삼는 것이 강화학습이기 때문이다. *Pre-training 이랑 SFT는 PPL 이라는, LLM의 성능과 correlation은 높지만 성능 그 자체는 아닌, proximal objective를 학습한다.*

그러나 **Rollout은 비싸고, Reward는 너무 단순하다.**

On-policy RL을 LLM 에 적용할 때 가장 큰 병목은 Generation이다. 어떤 downstream task를 사용하고, reasoning/non-reasoning에 따라 다르겠지만 거의 50~60% 이상 차지한다고 보면 된다.  Autoregressive 방식의 한계 중 하나지만, 특히나 on-policy RL은 매 step 마다 데이터를 생성해야 하기 때문에 굉장히 병목이 심하다.
그런데 그 **결과물에서 얻는 피드백은** 단순한 **scalar reward**뿐이다.

이게 얼마나 낭비적인지 생각해보자. Reasoning model의 경우 생성된 response에는 사실 풍부한 정보가 담겨 있다: 어떤 reasoning path를 만들었는지, 어떤 실수가 있었는지, 어떤 점들이 reward를 제대로 받지 못했는지 등등. 일반적인 RL과는 다르게 LLM의 trajectory는 자연어로 구성되어 있기 때문에 분석할 수 있고, 그 분석 내용을 다시 입력 condition으로 줄 수도 있다. **모든 것이 자연어로 구성되어 있기 때문에 가능**한 것이다.

하지만 기존 RLVR(Reinforcement Learning with Verifiable Rewards)는 이 모든 정보를 무시하고 "맞았다/틀렸다"라는 단순한 신호만 사용한다. 결국 모델은 implicit하게 "대체 뭘 잘못했지?"를 추론해야 한다. 그리고 이는 더 다양한 exploration을, 더 많은 학습시간을 요구받게 된다.

## ERL (Experiential RL) 리뷰

*논문에 이 그림이 있는데, 진짜 잘그렸다고 생각한다. 직관적으로 아이디어를 잘 표현한듯!*

![image.png](https://github-production-user-asset-6210df.s3.amazonaws.com/57203764/553023510-1d4d0e1d-7dac-451b-97c9-136664fa8f77.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20260221%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260221T164542Z&X-Amz-Expires=300&X-Amz-Signature=3dfa87a5f898eedf9316dc6b578ec28b19174bc69c8a39e7ec149d76401bf2f0&X-Amz-SignedHeaders=host)

ERL은 결국 trajectory를 분석해서 exploration (trial & error) 을 줄일 수 있는 정보를 제공함으로써 학습 효율과 성능을 끌어올렸다. 논문에서는 이를 Experience-Reflection-Consolidation Loop 라고 부른다.

HotpotQA라는 reasoning task 를 기반으로 예시와 함께 ERL process 살펴보자. *참고로 HotpotQA는 tool-calling (information retrieval) 을 이용해 질문에 대한 정답을 찾는 QA라고 보면 되고, 다양한 정보를 retrieval 하고 조합해야 답변할 수 있는 multi-hop QA 문제다.*

### ERL process

**1단계: First Attempt**

$$y^{(1)} \sim \pi_\theta (\cdot | x) $$
$$ (f^{(1)}, r^{(1)}) \sim \text{Env}(x, y^{(1)}) $$

일단 첫 번째 시도를 한다. LLM policy $\pi_\theta$ 로부터 input $x$에 대한 response $y^{(1)}$을 뽑고, 이에 대한 reward $r^{(1)}$을 받는다. LLM judge 같은 걸 쓴다면 judge 과정에 대한 feedback $f^{(1)}$도 함께 얻는다 

HotpotQA에 사용되는 input (system prompt) 는 다음과 같다.
![image.png](https://github-production-user-asset-6210df.s3.amazonaws.com/57203764/553026075-99ef5e16-51db-4e5d-bc05-536eb4991fbf.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20260221%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260221T164620Z&X-Amz-Expires=300&X-Amz-Signature=c239b7b6ee1a498b8df354b0626b36e2ce7e860fa1fabf9b9502b5ace788d7a9&X-Amz-SignedHeaders=host)

**2단계: Self-Reflection**

$$ \Delta \sim \pi_{reflection} (\cdot | x, y^{(1)}, f^{(1)}, r^{(1)}, m) $$

모델의 첫 response에서 어떤 것이 잘못했고 어떻게 고쳐야 하는지 reflection을 생성한다. `m`은 cross-episode memory로, 이전에 효과적이었던 reflection들을 저장함으로써 전체 RL 학습 과정에서 얻은 정보들을 재활용한다.

논문에서는 $\pi_{reflection}$으로 $\pi_\theta$를 사용했다. 그러나 내 생각에는 외부 LLM을 써도 무방하다. 오히려 reflection의 퀄리티의 중요도를 생각하면, SOTA 모델을 사용하는 것이 가장 학습 효율을 올릴 수 있을 것이다.

Reflection 은 이런식으로 system prompt를 주고 정보를 뽑아내게 된다.

![image.png](https://github-production-user-asset-6210df.s3.amazonaws.com/57203764/553026416-8e8b79fa-b254-4ce8-ab7a-1094bacddf3c.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20260221%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260221T164631Z&X-Amz-Expires=300&X-Amz-Signature=c58f3cbbb18d5e4a279e432332c31fe1f30cf708a416393230d0e848758af4ab&X-Amz-SignedHeaders=host)

**3단계: Second Attempt**

$$y^{(2)} \sim \pi_\theta (\cdot | x, \Delta) $$
$$ (f^{(2)}, r^{(2)}) \sim \text{Env}(x, y^{(2)}) $$

Reflection을 conditioned input으로 주고 다시 response를 생성한다. Reflection $\Delta$가 policy의 behavior correction에 도움이 되었다면, 더 퀄리티 높은 (*i.e.* $r^{(2)} > r^{(1)}$) 답변이 생성되었을 것이다. 

그리고 $r^{(2)}$ 가 특정 threshold $\tau$ 이상이라면 reflection 을 memory `m`에 저장한다 ($m \leftarrow \Delta$)

**4단계: Internalization**

우선 현재까지 얻었던, first-attempt, second-attempt 데이터를 가지고 RL 업데이트를 한 번 한다. 그리고 Second attempt에서 얻은 데이터를 기반으로 SFT loss 를 적용해 업데이트를 한 번 더 한다.

$$\mathcal{L}_{\text{distill}}(\theta)
=
- \mathbb{E} \left[
\mathbb{I}\left( r^{(2)} > 0 \right)
\log \pi_{\theta}\left( y^{(2)} \mid x \right)
\right]$$

Conditioned input 이 아닌 original input을 사용하는 것으로 기존보다 더 나은 데이터로 SFT를 한다고 보면 된다. 사실 distillation 이 아니지만 $y^{(2)}$를 teacher model response로도 볼 수 있기 때문에 이런 말을 쓴 것 같다.

*이 distillation loss를 적용하는게 기존 RL이랑 비교할 때 반칙같이 보이긴 해도 RL과 SFT를 함께 적용하는건 LLM에서 흔히 있는 일이다. 학습 안정성 때문인데, SFT loss가 RL 성능 저하를 막는 케이스들도 있다.*

### 실험 결과

논문에서는 FrozenLake, Sokoban, 그리고 HotpotQA에서 RLVR과 결과를 비교했다. FrozenLake랑 Sokoban은 퍼즐문제같은건데, 여기서는 LLM policy를 사용해서 풀었다. LLM task가 아니기 떄문에 보통 LLM이 아니라 state-action model 정의해서 RL을 적용하기도 한다. RL알고리즘은 GRPO를 사용했다.

![image.png](https://github-production-user-asset-6210df.s3.amazonaws.com/57203764/553028228-ace74dea-7419-45e6-b849-96b738e51ecb.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20260221%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260221T164647Z&X-Amz-Expires=300&X-Amz-Signature=24a416e5590068e6518122e322650fcd3ae034e9fa98e6d79bc099b2f1acf29c&X-Amz-SignedHeaders=host)

실험 결과에서 주목할 점은 3가지다.

**1. 학습 초기의 빠른 성능 향상**

논문에는 자세히 안적혀 있지만 (Ablation study가 있었으면 좋았을텐데 아쉽다), 내 생각에는 Distillation loss 때문이라고 본다. SFT를 별도로 분리하지 않고 RL에서 생성되는 데이터들을 재활용 할 수 있는 방법으로 녹여낸 건 좋다고 생각한다.

일반적으로 RL은 성능을 쥐어짜내는데 사용하는게 좋다. SFT cold-start 라는 말이 괜히 있는게 아닌데, random policy 로 exploration 하는건 진짜 비효율의 끝판왕이기 때문이다. 초기에는 SFT를 적절히 레버리지 해서 성능을 어느정도 끌어올려둔 후 RL을 적용하는게 국룰이다. 

**2. 높은 최종 성능**

이건 좀 특이한데, 보통은 exploration에 의도적인 영향을 가할 경우 수렴이 빨라질 수는 있어도, local minimum에 빠질 수 있다고 생각한다. 그럼에도 불구하고 ERL이 성능이 높은 이유는 reward variance에 있다고 생각한다. 

경험적으로 RL에서 가장 중요한 시그널은 **동일한 prompt에 대한 완전히 다른 응답**이다. Reward가 높은 응답과 reward 낮은 응답이 밸런스있게 섞여 있어야 좋은 gradient를 받을 수 있다 (특히나 GRPO에서는). ERL은 강제로 second-attempt를 통해 동일 prompt 에 대해서 서로 다른 퀄리티의 응답 세트를 얻을 수 있게 만든다. 이게 포인트이지 않을까 싶다. 

**3. 가장 효과가 큰 곳**

FrozenLake랑 Sokoban은 사실 LLM 능력이 있어도 쉽사리 풀기 어렵다. 아마 사람도 이미지 보면서 하면 쉬운데, 텍스트로 줘버리면 비교적 풀기 쉽지 않을거라 생각한다. 특히 논문에서 사용한 7B급 모델들은 바로 풀어낼 수는 없다. 근데, 한번도 성공을 못하면 아예 reward를 못받는 sparse reward라서 exploration 도 엄청 걸린다. 이런 케이스들에서는 효과가 크다는 것이다. 오히려 HotpotQA는 기본적으로 어느정도 역량이 있다보니 exploration cost가 덜 들고, ERL을 통한 성능 개선도 크지 않았다.

따라서 LLM이 처음보는 unknown dynamics이고, reward가 정답을 맞고 틀리고 정도로 sparse 하다면 ERL이 매우 효과적인 방법이 될 수 있다. 보통 text로 푸는 문제들은 해당이 안된다고 보면 되겠다. (한계점)

## 마무리

LLM에서 강화학습을 사용할 떄 문제는 명확하다 

> **On-policy RL에서 rollout은 가장 비싼 연산인데, 거기서 나오는 풍부한 정보를 scalar reward로 압축해버리는 건 낭비다.**

ERL은 해결책으로 직관적인 아이디어를 제시했다

> **Generated response에 담긴 정보를 explicit reflection으로 추출하고, 성공한 교정을 base policy에 내재화하자.**

이건 크게 보면 AI 모델 업계에서 지금까지 일어나고 있는 일들과 다르지 않다.

1. 새로운 모델을 배포한다
2. 실제 환경에서 사용하면서 문제점을 발견한다
3. Prompting이나 agent system 같이, 사용자들은 스스로 reflection을 통해 성능을 끌어올린다
4. 이렇게 사용자들이 고도화시켜둔 데이터로 SFT를 하거나 RL을 돌린다
5. 개선된 모델을 다시 배포한다
6. **1-5를 반복한다**

ERL은 이 cycle을 self training loop로 만드는 방법 중 하나다. Anthropic이 이 사이클을 완전 자동화시켰다는 이야기가 있는데, 개인적으로 어떻게 했는지 너무너무너무너무 궁금하다.
