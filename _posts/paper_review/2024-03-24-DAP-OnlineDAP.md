---
layout: post
title: "Direct Alignment from Preferences | Part - 03.Online DAP"
tags: archive
---

# Introdunction

지난 포스팅 "**Direct Alignment from Preferences | Part - 02.DAP**" 에서는 강화학습 (RL) 을 사용하지 않고 사람의 취향에 맞는 답변을 생성할 수 있게 된 언어모델을 학습하는 최신 DAP 방법론에 대해 알아보았습니다. 하지만 DAP 는 offline data 를 사용합니다. offline 데이터란, 현재 "내가 학습시키고자 하는 모델"이 생성한 데이터가 아닌 어디선가 수집된 데이터를 학습에 사용하는 방식입니다. 이는 학습시키고자 하는 모델이 생성하는 데이터와는 distribution shift 가 있기 때문에 모델이 학습되더라도 optimal 에 도달하지 못한다는 단점이 있습니다.

따라서 Offline 데이터를 사용하지 않고 Online 데이터를 이용해 optimal 에 도달할 수 있는 online DAP 방법론들이 등장했는데요. 이번 포스팅 3부작의 마지막, Part 3에서는 최신 online DAP 방법론들에 대해서 알아보도록 하겠습니다.

# Background

Online DAP 방법론에 대해 알아보기 전에 우선 online learning 과 offline learning 의 정의를 살펴보고 각각의 특징들에 대해 알아보겠습니다.

## On-line vs. Off-line learning

- Off-line learning
    - 미리 구축한 "**고정된 데이터**"를 이용해 모델을 학습
    - 학습과정에서 추가적인 데이터 수집이 없음
    - 따라서 학습 모델이 생성하는 데이터와, 학습용으로 사용되는 데이터 사이의 Data distribution 이 존재함
    - Offline RL 방법론들은 원래 distribution shift 보정을 위한 여러 가지 방법들을 적용함
    - **그러나 기존에 제안된 DPO 는 RL 을 offline 방식으로 풀어냈음에도 추가적인 보정을 사용하지 않았음** (LLM 의 특성, Massive corpus 를 이용해 학습했기 때문에 data distribution 이 크게 다르지 않아 다른 Domain 들에 비해 문제가 크게 부각되지 않음)
- On-line RL
    - 학습 과정에서 생성되는 데이터를 이용해 모델을 학습
    - 학습과정에서 생성되는 데이터들을 계속 수집하여 모델 학습에 지속적으로 사용함
    - Data distribution 이 동일하기 때문에 optimal 에 도달할 수 있음
    - 고퀄리티 데이터 생성에 어려움을 겪을 수 있음. 모델이 생성한 데이터의 퀄리티가 낮을 경우 역효과.

# Model alignment - online-DAP

## Rejection-sampling SFT (Meta)

Meta 가 2023 년에 내놓은 [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/pdf/2307.09288.pdf) (Touvron et al, 2023) 논문에서 사용된 방법론입니다. 

우선 offline 으로 구축된 Human feedback (HF) 데이터를 이용해 reward model 을 학습합니다. 그 이후 Best-of-N sample 을 모아 SFT 를 하는 엄청나게 간단한 방식입니다.

Best-of-N sample 이란 하나의 prompt 에 대해서 N 개의 답변을 생성합니다. 그리고 학습된 reward model 에 N 개의 답변을 던져서 가장 reward 가 높게 측정된 답변을 모아 SFT 데이터로 구축하는 것입니다.

이런 방법론이 가능한 이유를 논문에서 실험적으로 보였는데요. 

![스ㅔ](https://github.com/snulion-study/algorithm-adv/assets/57203764/0040e340-b0e9-405f-b826-21b4fa13130a)


Sampling 방식으로 답변을 생성하는 경우 N 이 커질수록 mean reward 는 크게 변하지 않았지만 max reward 는 크게 변화했습니다. 즉 sampling 결과는 꽤나 reward variance 가 크다는 것이죠. 따라서 Best-of-N 데이터의 SFT 만으로도 어느정도 human feedback 을 반영할 수 있게 됩니다.

Llama-2-chat 모델은 Pretrained model 로부터 Rejection-sampling SFT를 4번 반복한 후, iteration-5부터 PPO 를 수행하여 최종 성능을 끌어올리는 방식을 사용했다고 언급되어 있습니다.

## RSO (Deepmind)

Deepmind 에서는 2023 년에 [Statistical Rejection Sampling Improves Preference Optimization](https://arxiv.org/abs/2309.06657) (Liu et al., 2023) 논문을 통해 RSO 라는 방법론을 제안했습니다. 여기서 제안된 방법론은 Online learning 이라기보다는 offline data 를 수집할 때 "**data distribution 을 최종적으로 학습될 optimal model 이 생성할 data sample 과 최대한 비슷하게 하려는 것**" 입니다. 말이 엄청 어려운데, 자세히 살펴보겠습니다. 

우선 Offline 으로 수집된 HF 데이터를 이용해 reward model 을 학습합니다. 데이터 생성을 위해 reward model 을 지속적으로 사용하기 때문에 reward model 의 학습이 매우 중요합니다.

여기서는 3가지 human preference 데이터 수집 방법론을 비교합니다.
1) Direct: 
   
    Offline 으로 수집된 preference 데이터
2) SFT-sample-rank: 
   
    단일 프롬프트에 대해 하나의 SFT 모델로 여러개의 답변을 sampling 방식으로 생성한 뒤 미리 학습된 RM 을 이용해 rank를 태깅한 데이터
3) RSO-sample-rank:
   
    학습된 reward model $\rho_\psi$가 있을 때, $\pi_r(y | x) = \frac{1}{Z(x)} \pi_{\text{sft}}(y | x) \exp\left(\frac{1}{\beta}\rho_\psi(x,y)\right)$ 에서 데이터를 생성 ($\pi_{\text{sft}}(y | x)$ 가 아니라)

    생성된 답변들을 학습된 RM 으로 rank 를 태깅하여 데이터 구축

이 때 문제는 RSO-sample-rank 의 $\pi_r(y | x)$ 로 데이터를 어떻게 생성할 것인가? 가 문제입니다. 논문에서는 SFT 모델로 생성한 데이터들을 rejection sampling 을 통해 $\pi_r(y | x)$ 이 생성한 것과 최대한 비슷하게 distribution 을 맞춰주는 방식을 사용합니다.

Rejection sampling 은 아래와 같은 방식으로 진행됩니다.
    
![rs](https://github.com/snulion-study/algorithm-adv/assets/57203764/528d5981-d554-41e6-a136-a3db8c77a992)

이걸 더 쉽게 표현하면 $\mathbin{U}[0,1]$ 의 값이 아래 식의 값보다 작을 때, 해당 답변을 채택하는 방식을 취한다는 겁니다.
$$
\frac{\pi_{r_\phi}(y|x)}{M_{D_x} \pi_{\text{ref}}(y|x)} = \exp \left( \frac{1}{\beta} \left( r_\phi(x, y) - \max_{y' \in D_x} r_\phi(x, y') \right) \right)
$$

그 의미를 조금 더 살펴보면, reward 가 높을 수록 결국 채택할 확률이 높아지는 구조입니다. 이 때 max reward 를 알 수 없으니, Batch(64) 단위로 reward 를 계산해 최대값을 사용합니다.

이 식이 생소해 보이지만, 위에서 살펴본 Rejection sampling 의 일반화 버전이라고 볼 수 있습니다. $\beta=0$ 인 경우 Batch 에서 highest reward sample 만 채택하므로 rejection sampling 과 동일하고, $\beta=\infty$ 이면 모든 sample 을 채택하므로 SFT-sample-rank 와 동일합니다. 
    
RSO 논문에서는 reward model 을 이용해 데이터를 구축하는 RSO-sample-rank 방식이 Direct, SFT-sample-rank 방식의 데이터 구축보다 optimal policy data distribution 에 가깝고, 실험을 통해 그것을 보였습니다.
    
![exper1](https://github.com/snulion-study/algorithm-adv/assets/57203764/85f2de3d-5ca6-4cc5-8648-9611f184dc19)
![exper2](https://github.com/snulion-study/algorithm-adv/assets/57203764/1639b656-9911-4ef9-b091-bfa75cf61d55)
    
구축된 데이터는 DPO/SLiC/RSO loss 를 사용해 학습되었는데, 데이터셋마다 loss function 에 대한 우위는 다르게 나타났습니다. 그러나 모든 loss function 에서 RSO-sample-rank 방식이 가장 높은 성능을 보였습니다.

## Iterative DPO (Meta)

2023 년 Meta 가 [Some things are more CRINGE than others: Preference Optimization with the Pairwise Cringe Loss](https://arxiv.org/pdf/2312.16682.pdf) (Xu et al., 2023) 논문을 통해 Pairwise Cringe Loss 라는 걸 활용하는 방식을 새롭게 제안했는데요, 이 논문에서 Iterative DPO 가 처음으로 제안되었습니다. 

Offline 데이터를 이용한 DPO first iteration 이후, 자체 response 를 가지고 preference data update 를 진행하여 second iteration 진행하는 엄청 단순한 방식입니다. 그렇지만 이 방식으로도 성능이 개선되었다고 보고하고 있습니다.
    
![iterativeDPO](https://github.com/snulion-study/algorithm-adv/assets/57203764/74a5a8fe-8c24-490d-84ac-6a5d685554a0)

그러나 별도 reward model 을 학습해서 iteration 마다 preference tagging 을 하는 것인지, 학습된 DPO 모델을 이용해 iteration 마다 preference tagging 을 하는 것인지 명확히 공개하지 않았습니다. 학습 detail 등도 마찬가지로 공개되지 않았는데요, 아마 main contribution 이 아니었기 때문에 자세한 내용을 공개하지 않은 것 같습니다. 그러나 이후 online DAP 방법론들에서 꾸준하게 비교되는 모델 중 하나입니다.

## Self-rewarding (Meta)

2024년 또다시 Meta는 [Self-Rewarding Language Models](https://arxiv.org/pdf/2401.10020.pdf) (Yuan et al., 2024) 를 통해 새로운 online DAP 방식을 발표합니다. LLM 이 metric 의 평가 judge 로 사용될 수 있다는 것은 [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) 을 통해 널리 알려진 사실인데요, 이 능력을 online-DAP 방식에 적용했습니다.

![sr1](https://github.com/snulion-study/algorithm-adv/assets/57203764/2cd86c14-f22f-44fb-b564-e84ae4987a40)

방법은 간단합니다. 현재 모델을 이용해 prompt 의 답변을 sampling 방식으로 여러개 생성합니다. 그리고 그 답변들을 현재 모델에 다시 던져서 ranking 을 평가합니다. (여기서도 알 수 있듯이 모델이 judging 할 능력이 없는 작은 모델이면 애초에 사용할 수 없는 방식이긴 합니다.) 그렇게 모은 데이터로 DPO 를 수행합니다. 그리고 이 과정을 반복합니다. 간단하죠?

그럼 조금 더 자세히 살펴보겠습니다.

Self-rewarding 은 judge 로써의 능력이 필요하기 때문에 SFT 단계에서 judging 과 관련된 데이터가 학습되어야 합니다. 

![sr2](https://github.com/snulion-study/algorithm-adv/assets/57203764/26a057a4-86e4-49b9-bf37-f40afd052bfc)

따라서 다음과 같이 답변의 score 를 평가하는 데이터셋을 구축하고 SFT 에 함께 활용했다고 언급하고 있습니다. 이 데이터를 구축하는데 시간과 돈이 꽤나 들어갈 것으로 보이지만, 이 데이터를 추가하지 않으면 이후의 DPO 과정에서 성능개선이 크지 않다고 리포팅하고 있습니다.

이후 DPO 과정을 반복할 때 prompt 가 동일하면 그 효과가 떨어지기 때문에, self-instruction 방식을 이용해 prompt 를 새로 만들어줍니다. Self-instruction 이란 LLM 에게 몇 가지 prompt 를 few-shot 으로 주고 다양한 prompt 를 새로 만들어달라고 요청하는 방법론입니다. 재미있는 언급이 있는데, **DPO 이후 모델들은 self-instruction 방식이 잘 동작하지 않는다**고 합니다. 따라서 Self-reward iteration 을 수행할 때 prompt generation 은 초기 SFT 모델로만 수행합니다. 

Self-rewarding 방식으로 2번의 iteration 을 반복하였고, 성능이 지속적으로 개선되었음을 실험적으로 보였는데요. First iteration 에서는 4K pair 데이터 생성해서 사용했고, second iteration 에서는 7K pair 데이터 생성해서 사용했습니다.

- M1: Judging 데이터를 추가한 SFT 모델
- M2: Self-rewarding first iteration
- M3: Self-rewarding second iteration
  
![sr3](https://github.com/snulion-study/algorithm-adv/assets/57203764/0d78144d-c8e4-4b81-b094-d068d82e1ba6)

또한 이 과정을 통해 judge 로써의 성능도 지속적으로 향상됨을 보였습니다.

![sr4](https://github.com/snulion-study/algorithm-adv/assets/57203764/0848ed3c-1edf-4284-a626-fa80bc29f91a)

외부의 개입 (외부의 더 나은 LLM API 포함) 없이 스스로 데이터를 만들고 평가하고 개선될 수 있는 좋은 방법론이라고 생각됩니다. 다만 한계로는 Length-bias 가 심해졌다는 리포트가 있습니다. LLM-as-a-Judge 에서도 언급된 것으로 LLM 에게 평가를 계속 맡기면 긴 답변을 선호한다는 현상입니다.

## Online-DAP (Deepmind)

2024 년 Deepmind 가 [Direct Language Model Alignment from Online AI Feedback](https://arxiv.org/abs/2402.04792) (Guo et al., 2024) 논문을 통해 제안한 OAIF 방법론은 Self-reward 방식과 매우 유사하지만 자체 judging 능력을 신뢰하지 않고 off-the-shelve 모델을 사용하여 online DAP 를 수행한 방식입니다. 

<img width="985" alt="oaif1" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/878be16a-f794-4ccd-b69b-02417b5519db">

Self-rewarding 과의 차이는 두 가지입니다. 첫 번째로 Iteration 마다 동일한 prompt set 을 사용합니다. Self-instruction 을 통한 prompt generation 이 성능개선을 크게 만들지 못한다고 판단한 것 같습니다. 두 번째로 Self-annotation 이 아니라 외부의 off-the-shelve 모델을 사용합니다. 이 또한 학습하려는 LLM 이 judge 로써의 능력이 부족한 상황에서 사용할 수있는 대안으로 보입니다.

평가는 TL;DR 데이터 (Summarization task) 에 대해 진행했습니다. Offline DPO 의 경우 고정된 offline data 를 사용하기 때문에 reward hacking 이 발생합니다. 따라서 일정 step 이상에서 reward 는 상승할지 모르지만 성능이 급감합니다. 그러나 OAIF 의 경우 RLHF, RLAIF 와 유사한 성능을 보여줍니다.

<img width="472" alt="oaif2" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/de0e01ee-6252-4be8-9fce-e72b6427f24d">

그러나 몇 가지 한계가 있는데요, Annotator 로 작은 모델을 사용할 경우 RLAIF 보다 낮은 성능을 보입니다. 위 그래프는 Palm-2-XL 를 annotator 로 사용한 것이고 아래 그래프는 Palm-2-L 을 annotator 로 사용한 것입니다. 또, Palm-2-XL 대신 GPT-4-turbo 를 사용한다고 치면, 해당 그래프에서 필요한 데이터를 전부 태깅하려면 약 3000만원정도가 필요합니다. (너무 비싸네요...)

<img width="489" alt="oaif3" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/102577af-3d10-4e05-be3b-55d745e1f28c">

사람이 평가한 경우 Online-DPO 의 성능 개선이 더 두드러지게 나타났습니다. Offline-DPO 와의 비교 결과 win-rate 가 훨씬 높은데다가 quality 측면에서도 더 높은 점수를 받았습니다.

<img width="476" alt="oaif4" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/fad34705-8440-4a6e-8e95-c9f657d3c2d3">

또 RLHF, RLAIF 로 대표되는 RL 방식과 4-way 방식으로 비교했을 때에는 더 큰 성능 격차를 보였습니다.

<img width="949" alt="oaif5" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/e41da910-7da1-4b3d-9fe7-e6e94dfa10d5">

마지막으로 RM 대신 Off-the-shelve LLM 을 사용해 Online DAP 를 적용할 때의 장점으로, Reward model 의 학습 없이 "**자연어로 표현 가능한 human preference**" 를 반영할 수 있다는 점을 꼽았습니다. 예를 들어, 답변의 길이가 너무 길어서 짧은 답변을 선호하는 모델을 학습하고 싶다고 가정합시다. RM 을 사용하는 기법들은 동일 prompt 에 대해 짧은 답변을 win, 긴 답변을 lose 로 태깅한 데이터를 수집하고, RM 을 재학습해야 합니다. 그러나 Off-the-shelve LLM 을 사용하게 되면 프롬프트 엔지니어링을 통해 "동일한 의미를 가질 경우 답변이 짧을 수록 높은 점수를 부여합니다"와 같은 문장을 추가함으로써 별도의 재학습 없이도 preference 를 반영할 수 있습니다.

# Conclusion

3개의 포스팅을 통해 LLM 이 사람의 선호를 반영한 답변을 생성할 수 있도록 학습하는 방법 3가지 - RL, Offline DAP, Online DAP - 를 살펴보았습니다. 각 방식의 여러 가지 방법론들의 장단점, 비용, 컴퓨팅 리소스 등을 표로 정리해 보았습니다.

|  | Algorithm | NOT require human preference tagging | Unbiased data distribution (training data from training policy) | Online feedback (Iterative improvement) | NOT require external API cost | # of model for training |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RL | RLHF | X | O | O | O | 4 |
|  | RLAIF | O | O | O | X | 4 |
|  | ReST | X | O | O | O | 4 |
| Offline DAP | SLiC-HF | X | X | X | O | 2 (1) |
|  | DPO | X | X | X | O | 2 (1) |
|  | IPO | X | X | X | O | 2 (1) |
|  | RRHF | X | △ | X | O | 1 (2) |
|  | CLICK | X | △ | X | O | 1 (2) |
|  | SPIN | O | △ | O | O | 2 (1) |
| Online DAP | Rejection-sampling FT | X | O | O | O | 2 |
|  | RSO | X | O | X | O | 3 |
|  | Iterative DPO | X | O | O | O | 3 (2) |
|  | self-rewarding | △ (없어도 가능) | O | O | O | 2 (1) |
|  | OAIF (online DPO) | O | O | O | X | 2 (1) |

Model alignment 라고 불리는 이 방법론들은 아직까지도 활발하게 연구되고 있는 분야입니다. 물론 어떠한 방법론을 사용해야한다는 정답은 없습니다. 자신의 데이터 상황이나 컴퓨팅 자원에 맞게, 적절한 방법을 선택하면 됩니다. 

OpenAI 는 2023 년 새로운 팀을 만들었다고 공표했습니다. 그 팀의 이름은 "Super alignment", 사람보다 훨씬 더 똑똑한 AI 를 만들기 위한 팀입니다. 앞으로 Model alignment, 즉 사람의 선호를 반영하는 것을 넘어서, Super alignment, AI 보다 덜 똑똑한 사람으로 더 똑똑한 model 을 학습할 수 있는 방법론들이 등장할 것입니다. Model alignment 분야가 그 초석을 잘 닦아주리라 생각합니다.

<img width="489" alt="Untitled 45" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/a2688a9a-07a6-49ac-877d-490e57c54cb5">


# References

- [Llama 2: Open Foundation and Fine-Tuned Chat Models (Touvron et al, 2023)](https://arxiv.org/pdf/2307.09288.pdf)
- [Statistical Rejection Sampling Improves Preference Optimization (Liu et al., 2023)](https://arxiv.org/abs/2309.06657) 
- [Some things are more CRINGE than others: Preference Optimization with the Pairwise Cringe Loss (Xu et al., 2023)](https://arxiv.org/pdf/2312.16682.pdf) 
- [Self-Rewarding Language Models (Yuan et al., 2024)](https://arxiv.org/pdf/2401.10020.pdf) 
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
- [Direct Language Model Alignment from Online AI Feedback (Guo et al., 2024)](https://arxiv.org/abs/2402.04792) 