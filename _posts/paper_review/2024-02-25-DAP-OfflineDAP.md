---
layout: post
title: Direct Alignment from Preferences - Part 02. DAP
tags: archive
---

# Introdunction

지난 포스팅 "**Direct Alignment from Preferences Part - 01.RLHF**" 에서는 강화학습 (RL) 을 사용해 사람의 취향에 맞는 답변을 생성할 수 있게 된 언어모델을 학습하는 방법을 알아보았습니다. 하지만 RL 은 Supervised learning 과 비교했을 때 불안정한 학습 방법입니다. (학습이 hyperparameter 의 영향을 많이 받을 뿐더러 좋은 Reward 를 받을 수 있는 데이터를 확보할 수 있도록 엄청나게 많은 exploration 이 필요합니다. RL 의 불안정성에 대해서는 추후 기회가 되면 다뤄보도록 하겠습니다)

또한 RL 을 사용하는 방법론들은 Computation cost 가 높다는 단점이 있습니다. Policy, Reference-policy, Reward, Value Model 까지 4개의 모델을 학습에 사용해야 합니다. 즉, 모델이 커지면 커질수록 부담이 되는 방법론들입니다.

따라서 RL 을 사용하지 않고 학습 안정성을 높이는 여러 방법론들이 등장했고, 이들을 **DAP (Direct Alignment from Preference)** 라고 부릅니다. 이번 포스팅 3부작의 Part 2에서는 DAP 방법론들에 대해서 알아보도록 하겠습니다.

# Model alignment - DAP

## SLiC-HF (Deepmind)

Deepmind 가 2023 년에 내놓은 [SLiC-HF: Sequence Likelihood Calibration with Human Feedback](https://arxiv.org/abs/2305.10425) (Zhao et al., 2023) 논문에서 제안된 방법론입니다. 

고전 ML 방식인 SVM 은 Positive sample 과 Negative sample 이 잘 구분되도록 모델을 학습합니다. Positive sample 과 Negative sample 이 특정 margin 거리를 두고 Hyperplane 으로 구분될 수 있도록 만드는 것이죠.

![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/67d244f9-e01b-474a-aa5c-eec6d996a733)

SLiC-HF 는 SVM 에서 사용되는 loss function 으로 사람 취향에 대한 Positive sample 과 negative sample을 구분하도록 LLM 을 학습하면 LLM 이 positive sample, 즉 사람의 취향을 반영하는 답변을 생성할 수 있도록 만드는 것입니다. 

- SLiC-HF loss function 
    $$
    \mathcal{L}(\theta) = \max(0, \delta - \log P_\theta(y^+|x) + \log P_\theta(y^-|x)) - \lambda \log P_\theta(y_{\text{ref}}|x)
    $$
    - 앞부분의 loss 는 SVM 에서 사용하는 hinge-loss 입니다. Positive sample 의 likelihood 를 높이고 Negative sample 의 likelihood 를 낮춘다고도 이해할 수 있겠습니다.
    - 뒷부분의 loss 는 Reference model (SFT) 로부터 너무 달라지지 않도록 하는 regularizer 입니다. 보통은 KL-divergence 를 사용하는 것이 일반적이지만 Reference model 의 답변에 대한 NLL 값을 사용하더라도 같은 효과를 낼 수 있다고 합니다. 해당 방법을 사용하면 학습 중에 Reference model 을 띄워야 하는 computation cost 를 줄일 수 있습니다. (ref: [Calibrating Sequence likelihood Improves Conditional Language Generation](https://arxiv.org/abs/2210.00045))

- 결과
  - 외부에서 미리 태깅된 Preference 데이터를 쓰는지, SFT 모델로 만든 데이터를 태깅해서쓰는지에 따라서 결과가 조금씩 달라지는데요, 그 수치가 크게 달라지지는 않았습니다. 
    
    ![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/b5f0944a-5e56-4d48-b93e-4385669970ef)

    ![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/7faeca62-c80c-4406-92a6-6629b58fcc5b)
    
  - SLiC-HF 학습 방식은 SFT 모델과 비교해 80% 이상의 win rate 를 보였고, Human annotator가 평가했을 때에도 RLHF 보다 높은 점수를 얻었습니다.

본 논문에서는 SLiC-HF 방법론이 RLHF+PPO 보다 나은 성능을 보였다고 주장하고 있습니다. 다만 Summarization task 에 대해서만 성능을 평가했기 때문에 이 논문만 가지고 chatGPT 처럼 general task 에 대해 model alignment 가 가능하다고 판단하기는 어렵습니다.

## DPO (Stanford)

스탠포드에서 2023년에 나온 Direct Preference Optimization: Your Language Model is Secretly a Reward Model ([https://arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)) 논문은 RL objective function을 매만져 supervised learning 이 가능하도록 만든 방법론입니다. 

- Direct Preference Optimization (DPO)
  
    이 방법론은 KL-divergence regularizer가 붙은 RL objective 는 closed form 이 있다는 데에서 시작합니다.

    $$
        \max_{\pi_\theta} \mathbb{E}_{x \sim D, y \sim \pi_\theta(y|x)}[r_\phi (x, y)] - \beta D_{KL}[\pi_\theta(y | x) || \pi_{\text{ref}}(y | x)]
    $$

    위 RL objective 가 특정 값으로 수렴한다고 생각하면 optimal policy의 closed form 은 아래와 같습니다. 

    $$
        \pi_r(y | x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y | x) \exp\left(\frac{1}{\beta}r(x, y)\right)
    $$

    이 때 $Z(x)$ 는 partition function 으로 확률의 합을 1로 만들어주는 scaling 값이라고 생각하면 됩니다. 이제 이 수식을 잘 만져 reward 에 대한 수식으로 정리해 주겠습니다.

    $$
        r(x, y) = \beta \log \frac{\pi_r(y | x)}{\pi_{\text{ref}}(y | x)} + \beta \log Z(x)
    $$

    마지막으로 RLHF 에서 사용했던 Reward model loss 생각나시나요? Reward model loss 에 위 수식을 대입해 주면 

    $$
        \mathcal{L}_R(r_\phi, D) \newline
        = -\mathbb{E}_{(x,y_w,y_l) \sim D}[\log \sigma(r_\phi (x, y_w) - r_\phi (x, y_l))] \newline
        
        = -\mathbb{E}_{(x,y_w,y_l) \sim D}\left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right) \right] \newline

        = \mathcal{L}_{DPO}(\pi_\theta; \pi_{\text{ref}})
    $$

    재미있게도 위 loss 를 minimize 하기만 해도 reward function 의 별도 학습 없이 policy 학습이 가능하다는 겁니다. $y_w$, $y_l$ 는 외부의 human prefrence 데이터에 이미 태깅이 되어 있기 때문에 supervised learning 처럼 학습이 되는 것이죠.

    사실 RL objective 를 최대화하는 학습은 Reward Model 이 있어야 가능한건데, Reward Model 학습을 위한 loss 에 RL objective 에서 얻은 closed form 을 넣는게 좀 앞뒤가 안맞다는 생각이 들 수 있습니다. 그래서 논문의 저자들은 새로운 DPO 의 loss function 이 어떤 의미를 가지는지 서술해 두었습니다. 

    DPO loss function 의 gradient 를 구해보면 아래와 같은 식을 얻을 수 있습니다. 
        
    ![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/f8bc3199-4821-4a56-ad03-1fbe1978a07e)

    이걸 뜯어보면 3가지 의미를 가집니다.
    - $y_w$ 의 likelihood 를 높인다.
    - $y_l$ 의 likelihood 를 내린다.
    - Reward 평가가 잘못된 것일수록 더 gradient 가 크도록 큰 weight 를 곱해준다.

    이렇게 분석을 해보면 DPO loss 도 위에서 설명한 SLiC-HF 방식과 유사하게, positive sample 의 likelihood 를 높이고 negative sample 의 likelihood 를 낮춘다는 컨셉이 보이죠.

- 실험 결과
  
    ![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/801e4b14-cb21-45f4-8d5f-7e76717cef9b)

    DPO 방법론의 경우 PPO 에 비해 Reward 를 높게 받는 것을 알 수 있습니다. 또한 sampling temperature 에 둔감해서, PPO 가 높은 sampling temperature 상황에서 win rate 가 크게 떨어지는데 반해, DPO 는 그렇지 않음을 보여줍니다.

    ![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/385e9b41-1d9e-4672-8c5f-ee58ff8e7f6b)

    또 ground truth 데이터와의 win rate 를 사람이 평가했을 때에도 PPO보다 DPO가 더 높은 win rate 를 기록했습니다. 즉, **DPO 가 PPO 에 비해 학습이 간편하면서 성능이 더 높고, decoding parameter 에 대한 안정성도 높다**고 주장하는 것입니다.

SLiC-HF 와는 다르게, DPO 는 summarization task 이외에도 [Anthropic의 Helpful-Harmful 데이터](https://arxiv.org/abs/2204.05862)에 대해서도 그 성능을 검증했습니다. 현재 사용되고 있는 DAP 방법론 중 가장 널리 사용되는 방법론이 아닐까 싶네요.

## IPO (Deepmind)

딥마인드에서는 RLHF 에 근본적 문제가 있다고 주장하며, [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036) (Azar et al., 2023) 논문을 통해 IPO (Identity Preference Optimization) 라는 방법론을 제안했습니다. 그런데 이 논문은 굉장히 복잡한 수학이 많이 들어가 있습니다. 따라서 이 포스팅에서는 대략 어떤 내용을 담고 있는지 간략히 살펴보도록 하겠습니다.

이 논문에서는 RLHF 의 reward model 인 **Bradley-Terry Preference model** 이 근본적 문제를 가지고 있다고 합니다.

$$
    \mathcal{L}_R(r_\phi, D) = -\mathbb{E}_{(x,y_w,y_l) \sim D}[\log \sigma(r_\phi (x, y_w) - r_\phi (x, y_l))]
$$

위 loss 를 통해 모델이 완전히 학습되었다면 $r(y_w)-r(y_l)\to+\infty$ 를 만족해야 하므로, $\pi(y_w)=1$ 이 되고 $\pi(y_l)=0$ 이 되어야 합니다. 이 때 RLHF 의 RL 단계에서는 KL-divergence 를 regularizer 로 사용하는데, $\pi(y_l)=0$ 인 데이터가 사용되면 $D_{KL}\to\infty$ 가 됩니다. 

따라서 Reward model 은 필연적으로 underfitting 되어야 합니다. 이건 RLHF 논문에서도 나오는 내용인데요, Reward model 을 너무 많이 학습시키게 되면 학습 안정성이 떨어진다는 내용이 있습니다.

그런데 DPO 의 경우 Bradley-Terry Preference model 을 사용하고, RL 과정의 KL-divergence 와 같은 regularizer 가 별도로 존재하지 않기 때문에 학습 안정성이 떨어질 수 있다는 주장입니다. 즉, reward model 을 과도하게 학습해버리는 것을 막을 방법이 없다는거죠.

따라서 IPO 에서는 기존의 sigmoid 를 사용하던 reward model loss function 대신 MSE 를 사용하는 loss function 을 사용해 이 문제를 해결합니다. 논문에는 이 loss 를 사용해도 되는 이유에 대해서 장황하게 증명해 놓았습니다. 혹시 궁금하시다면...논문을 읽어보시는 것을 권장합니다.

- IPO reward
    
    $$
    \mathcal{L}_{IPO}(\pi_\theta; \pi_{\text{ref}}) = \mathbb{E}_{(x,y_w,y_l) \sim D}\left[ \left( \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} -\frac{1}{2\beta}\right)^2 \right]
    $$
    
    어쨋든 해당 loss 또한 분석을 해보면, 근본적으로는 1) positive sample 의 likelihood 를 높이고, 2) negative sample 의 likelihood 를 낮추며, 3) 그 차이가 reference sample 의 likelihood 들의 차이와 $\frac{1}{2\beta}$ 의 합만큼 날 정도로만 너무 커지지 않게 해주는 것이라 볼 수 있겠습니다.
- 실험 결과
    
    ![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/5e1edfba-f094-4653-884e-a12fa8406c59)
    
    위 그래프는 y1 > y2 > y3 순서의 preference 데이터에 대한 training curve 입니다. 
    - DPO 는 y1 만 생성하도록 greedy 하게 overfitting 되지만 IPO 는 y2, y3 도 때에 따라 생성할 수 있는 여지를 남겨둠. 즉, greedy policy 로 overfitting 되는 것을 피할 수 있음.
    - DPO 는 한 번도 이기지 못한 output 은 prob. 이 0이지만 IPO 는 한 번도 못이겨도 prob. 는 0이 아님. 즉 y2 의 prob 가 0이 되지 않음.
  
    이게 왜 중요하냐면 Real-world 에서는 human annotation error 가 있다보니 실제로 이겼어야 하는 퀄리티 좋은 데이터가 한 번도 이기지 못한 경우가 꽤나 발생합니다. 이런 경우에 DPO 는 해당 답변의 확률을 0으로 만들어버리지만 IPO 에서는 어느 정도 확률을 보장합니다. 즉, annotation error 로 인한 overfitting 을 적절히 sampling 방식으로 해소할 수 있다는 것이죠.

본 논문에서는 DPO 의 overfitting 문제를 해결할 수 있는 방법론을 새롭게 제안했는데요. 재미있는 건 win rate 이 높아졌다던거, 어떤 task 에서 잘 되었다던가 하는 별도의 Evaluation 이 없습니다. 그래서 얼마나 좋은 방법론인지 추가적인 실험을 해보지 않고서는 알기가 어렵네요. 그렇지만 Deepmind 니까 한 번 믿어봄직도...(개인적 의견입니다)

## RRHF (Alibaba)

Alibaba 도 2023 년 Qwen 이라는 고성능 LLM 을 출시했습니다. 해당 모델은 open-source LLM 의 성능을 비교하는 [Open-LLM leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) 에서 자주 발견되는 모델 중 하나인데요, 특히 다른 빅테크 LLM 들과 비교했을 때 중국어에는 강점이 있다보니 좀 많이 사용되는 것 같습니다. Alibaba 는 [RRHF: Rank Responses to Align Language Models with Human Feedback without tears](https://arxiv.org/abs/2304.05302) (Yuan et al., 2023) 논문을 통해 ranking loss 를 이용하는 방법론을 제안했습니다. 

![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/2aff87d7-de83-4d8c-911d-8973de2c3e28)

위에서 설명한 SLiC-HF, DPO, IPO 와 마찬가지로 별도의 value model, reward model, reference model 없이도 학습이 가능한 방법론입니다. 다만 win/loss 로 구성되는 pair 데이터가 아니라 여러 response 의 ranking 을 매겨 사용한다는 점에서 차이가 있습니다. (사실 OpenAI의 RLHF 도 pair 가 아닌 ranking 을 사용했습니다. 그치만 open-source 데이터들은 대부분 pair 입니다. 왜냐하면 annotation 하기가 더 쉬우니까요.)

- RRHF loss
    
    $$
    p_i = \frac{ \log P_{\pi} (y_{i\geq t} | x, y_{i,<t})}{\| y_i \|}
    $$
    
    $$
    L_{\text{rank}} = \sum_{r_i < r_j} \max(0, p_i - p_j)
    $$

    위 두 식이 RRHF 에서 사용한 Ranking loss 입니다. 단순히 생각하면 ranking 이 높은 sample의 likelihood 가, ranking 이 낮은 sample 의 likelihood 보다 높아지도록 하는 것이죠. 그냥 pair 로 구성된 데이터라고 생각한다면 SLiC-HF 에서 사용한 hinge-loss 에서 margin 을 사용하지 않는 것과 동일합니다.  
    
    $$
    i' = \underset{i}{\mathrm{argmax}} \ r_i
    \newline
    L_{ft} = -\sum_{t} \log P_{\pi} (y_{i',t} | x, y_{i',<t})
    $$
    
    여기서 그치지 않고, fine-tuning loss 를 추가해 주었는데요. 모은 데이터들 중 가장 ranking 이 높은 데이터에 대해서만 SFT loss 를 적용해줍니다. 가장 ranking 이 높은 데이터를 그냥 preference optimization 에만 사용하기는 아까우니까요. (이건 사실 Rejection-sampling 이라는 방법론의 일종인데요, Part 3 에서 다루도록 하겠습니다)

    $$
    L=L_{rank}+L_{ft}
    $$

    두 loss 를 합해 최종 loss function 을 구하게 됩니다. 
        
- 실험 결과
    
    ![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/a8f0af44-c455-4786-bd6c-360f62d5613c)

    PPO 와의 공정한 비교를 위해서 PPO 의 reward model 을 이용해 데이터의 ranking 을 매기고, 그 데이터를 이용해 RRHF 를 돌렸습니다. 그 결과 PPO 에 비해 win_rate 가 아주 조금 높아졌습니다. 

    또한 RRHF 로 학습된 모델을 이용해 데이터를 생성하고, 그걸 다시 ranking 을 매겨 RRHF iteration 한 번 더 돈 모델 ($\text{RRHF}_{\text{IP-2}}$) 가 RRHF 를 한번만 수행한 모델 ($\text{RRHF}_{\text{DP}}$) 보다 성능이 좋아졌습니다.

    다만 주의할 점은 학습된 모델의 답변만을 이용해 **iteration 을 계속 돌게 되면 Reward 는 증가하는 것처럼 보이지만 실제 PPL 과 win rate 는 떨어질 수 있다**고 합니다. 이는 **reward hacking** 이라는 현상으로 annotation 의 허점을 파고들도록 학습되는 현상입니다. 
    > “*That sounds great!”, “I appreciate your help.”, “Thanks for your help!”*

    과 같은 답변들, 틀리진 않았기 때문에 ranking 이 높게 잡힐 수는 있지만 아무 쓸모가 없는 답변들, 만 생성하는 현상입니다. 즉, 적절한 횟수의 Iteration 을 찾아야 할 것입니다.

    ![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/bf7ea2f8-1a30-49e5-9755-b182adefaf95)
    
    또 Alpaca 와 비교하는 실험도 진행했습니다. Alpaca 는 [self-instruct](https://arxiv.org/abs/2212.10560) 라는 방식을 사용하는 방법론으로 chatGPT 에게 답변을 생성하도록 하고, 해당 답변을 SFT 하는 knowledge distillation 방법론의 일종입니다. 

    본 논문에서는 Wombat - chatGPT로 답변을 여러개 생성하고, 해당 답변들의 ranking 도 chatGPT 를 사용해 매긴 후 RRHF 를 사용하여 학습한 모델 - 과 Alpaca 를 비교했습니다. 그 결과 Alpaca 보다는 더 나은 성능을 보였지만 chatGPT 데이터를 사용한 만큼 chatGPT 의 성능에는 미치지 못함을 보였습니다.

이후 칭화대에서 [CLICK: Controllable Text Generation with Sequence Likelihood Contrastive Learning](https://arxiv.org/pdf/2306.03350.pdf) 이라는 RRHF 에 몇가지 테크닉을 추가한 논문도 내놓았는데요, RRHF 를 사용하신다면 CLICK 논문도 보시는 것을 추천합니다. 

## SPIN (UCLA)

2024년 UCLA 에서 [Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://arxiv.org/pdf/2401.01335.pdf) (Chen et al., 2024) 라는 논문을 내놓았습니다. Self-Play 는 사실 RL 의 꽃입니다. Alpha-zero 가 self-play 방식으로 바둑을 정복했죠. (물론 Alpha-Go 에서 이미 인간을 넘었지만) AI 끼리의 대결을 통해 데이터를 얻고, 그 데이터를 통해 지속적으로 학습하는 방법입니다. OpenAI 에서는 과거 Hide-and-seek 이라는 게임에서 self-play 를 통해 흥미로운 현상들을 관찰한 바 있습니다. [OpenAI Hide-and-seek](https://www.youtube.com/watch?v=kopoLzvh5jY)

SFT 데이터가 있을 때, SFT response를 win sample, current model response 를 lose 로 해서 DPO 를 iterative 하게 수행하면 계속 성능이 향상된다는 컨셉인데요. Human preference 태깅이 전혀 필요하지 않고, open-source 로 엄청나게 많은 SFT 데이터들만 사용하면 된다는 장점이 있습니다.

- Self-Play Fine-Tuning (SPIN)
    
    ![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/d01a7484-5c12-42ae-a7d4-a8597e595533)

    컨셉 자체는 어이가 없을 정도로 간단하고 조금은 이상합니다. SFT 데이터가 완벽하다고 치고 현재 모델의 output 이 무조건 진다고 태깅을 한다는 것이니까요. 그렇다면 그냥 해당 데이터를 SFT 하는 것과 어떤 차이가 있을까요?
    
- 실험 결과

    Huggingface 에 공개된 모델 중 `zephyr-7b-sft-f` 라는 모델이 있는데요, 이건 Mistral-7B 를 Ultrachat200k 데이터로 SFT 한 모델입니다. 
    
    본 논문에서는 Ultrachat200k 데이터에서 50K 데이터를 샘플링해서 SPIN 을 수행했습니다. 그 결과는 조금 놀라운데요.

    ![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/19054d87-b9cc-4c5a-a300-ebd04a32afae)

    Ultrachat200K 데이터를 계속 SFT 해도 성능은 증가하지 않는데 비해 SPIN 을 사용하면 성능이 증가합니다. 같은 데이터인데도 성능이 증가하는건, 모델이 SFT 의 positive 데이터만을 학습하는것뿐만 아니라, negative 데이터를 보고 positive 데이터와의 차이를 배우는 것의 중요성을 시사합니다.

    ![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/657c241f-766d-42e1-b73e-67d87d97bb5b)

    또한 Preference 태깅이 필요한 DPO 모델과 비교했을 때에도 더 좋은 성능을 보였다고 주장하고 있습니다. 

# Conclusion

총 5가지의 DAP 방법론들에 대해 알아보았습니다. 사실 이 방법론들의 컨셉은 동일합니다. **사람의 취향에 맞았던 데이터의 likelihood 는 높이고, 그렇지 않았던 데이터의 likelihood 는 낮추는 것**입니다. 다만 RL 의 경우 학습이 불안정하다보니 그나마 학습안정성이 가장 높은 PPO 를 사용하는 수밖에 없었다면, DAP의 경우 supervised learning loss 를 사용하기 때문에 loss function 을 이것저것 시도해 볼 수 있었고, 다양한 방법론들과 논문이 나오는 결과로 이어지지 않았나 생각합니다.

# References

- [SLiC-HF: Sequence Likelihood Calibration with Human Feedback](https://arxiv.org/abs/2305.10425) (Zhao et al., 2023)
- [Calibrating Sequence likelihood Improves Conditional Language Generation](https://arxiv.org/abs/2210.00045)
- [https://arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)
- [Anthropic의 Helpful-Harmful 데이터](https://arxiv.org/abs/2204.05862)
- [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036)
- [Open-LLM leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [RRHF: Rank Responses to Align Language Models with Human Feedback without tears](https://arxiv.org/abs/2304.05302)
- [self-instruct](https://arxiv.org/abs/2212.10560)
- [CLICK: Controllable Text Generation with Sequence Likelihood Contrastive Learning](https://arxiv.org/pdf/2306.03350.pdf)
- [Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://arxiv.org/pdf/2401.01335.pdf)