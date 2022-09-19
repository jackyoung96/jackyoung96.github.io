---
layout: post
title: Paper review - A survey of preference-based reinforcement learning methods
tags: archive
---

2022년 제 블로그의 첫 글이자 첫 논문 리뷰는 A survey of preference-based reinforcement learning methods 부터 시작해 보겠습니다. 처음 Preference-based RL를 알게 된 건 Berkeley에 계시는 Kimin Lee 박사님이 하신 2021년 여름방학에 진행된 SNU Summer AI 강연이었습니다 (강연을 잘하셔서 정말 재밌게 들었습니다. 관심 있으신 분들은 꼭 들어보세요! <a href="https://www.youtube.com/watch?v=MiwOvaywtew&t=569s" title="PEBBLE">SNU summer AI - PEBBLE</a>).
리뷰 논문부터 시작해서 한 번 흐름을 살펴보죠.  

![image](https://user-images.githubusercontent.com/57203764/147902365-1f6a9a32-b722-43f7-8b64-c2ba0a788e67.png)
<br />

Table of contents
   - [Introduction](#introduction)
   - [Preliminaries](#preliminaries)
     - [Preference learning](#preference-learning)
     - [Markov decision processes with preferences (MDPP)](#markov-decision-processes-with-preferences-mdpp)
     - [Objective](#objective)
     - [PbRL algorithms](#pbrl-algorithms)
     - [Related problem settings](#related-problem-settings)
   - [Design principles of PbRL](#design-principles-of-pbrl)
     - [Learning a policy](#learning-a-policy)
     - [Learning a preference model](#learning-a-preference-model)
     - [Learning a utility function](#learning-a-utility-function)
        - [Linear utility function](#linear-utility-function)
        - [Non-linear utility function](#non-linear-utility-function)
     - [The temporal credit assignment problem](#the-temporal-credit-assignment-problem)
     - [Trajectory preference elicitation](#trajectory-preference-elicitation)
        - [Trajectory generation](#trajectory-generation)
        - [Preference query generation](#preference-query-generation)

<br><br>

## Introduction
---
Preference-based reinforcement learning (PbRL)은 다음과 같은 동기에서 시작되었습니다. 
> PbRL have been proposed that can directly learn from an expert's preferences instead of a hand-designed numeric reward.

Reward 기반으로 학습하는 RL은 알파고를 기점으로 크게 발전되었지만, reward function을 정의함에 있어서 여러 가지 문제점을 노출하고 있습니다. (crucially depends on the prior knowledge that is put into the definition of the reward function) 그 중 가장 큰 부분을 차지하는 4가지 문제점을 살펴보겠습니다.
1. Reward hacking : The agent may maximize the given reward, without performing the intended task.   
   예를 들어 청소기에게 먼지가 없을 때 positive-reward를 주면 먼지를 없애는게 아니라 먼지가 없는 부분만 쳐다보고 있는 겁니다.
2. Reward shaping : The reward does not only define the goal but also guides the agent to the correct solution.   
   사실 optimal reward function을 모를뿐더라, (실제로 없을 수도 있음) 사람이 design 하다보니 문제가 있죠. 이게 사실 RL의 가장 큰 문제....
3. Infinite rewards : Some applications require infinite rewards.  
   예를 들어 자율주행 차가 사람을 치는 행동은 절대 하면 안되니 negative infinity reward를 줘야 하는데, 이는 classic RL을 성립시킬 수 없습니다. (RL의 가정 중 finite reward 가 있습니다)
4. Multi-objective trade-offs : The trade-off may not be explicitly known.   
   reward 간의 trade-off 관계는 제대로 파악하기가 어렵습니다. (자율주행 차의 승차감, 속도, safety 등의 balance를 numerical하게 설정하기는 어렵겠죠)

<br /><br />
어쨋든 문제는 agent의 behavior 안에 내재된 행동 동기 (Intrinsic motivation)를 numerical scalar value로 표현하는 것이 어렵다는 것입니다. 이를 해결하기 위해 여러 방법들이 연구되어 왔습니다. Inverse RL 이나 learning with advice 등이 있는데, PbRL은 한가지 측면에서 이들과는 다릅니다.  
> PbRL aims at rendering reinforcement learning applicable to a wider spectrum of tasks and **non-expert users**. 

즉 RL에 관해 아무것도 모르는 사람도 preference만 있으면 agent를 학습시킬 수 있게 만드는 것이 PbRL입니다. (일반인도 로봇을 가르칠 수 있는 세상...!!)
<br /><br /><br />

## Preliminaries
---
PbRL의 동기를 Introduction에서 알아보았고, 이제는 PbRL을 학습하기 위한 포인트들을 알아보겠습니다. 
> Preference learning is about inducing predictive preference models from empirical data.  

PbRL은 사용지의 데이터로부터 preferece model을 유추하는데 목적이 있습니다.   
  
<br />

### Preference learning
그럼 이제 수식을 좀 살펴볼까요? Preference는 아래와 같이 5가지로 나누어 표현됩니다. 일단 무조건 2가지 선택지를 비교하는 걸 가정합니다.
![image](https://user-images.githubusercontent.com/57203764/147896080-d91e785f-b76b-401b-ab08-977a2e0b18c0.png)

### Markov decision processes with preferences (MDPP)

MDPP는 sextuple로 구성됩니다.  
<center>
$$(S,A,\mu,\delta,\gamma,\rho)$$  
</center>
$$S,A$$는 state와 action space를 나타내고, $$\mu$$는 initial state distribution, $$\delta$$는 transition probability $$\delta(s'|s,a)$$ 를 나타냅니다. $$\gamma\in[0,1]$$는 discount factor겠죠.
<br />

이제 특이한 건 $$\rho$$ 인데요, 이는 probability distribution of preference 입니다. 사람은 언제나 stochasticity가 있기 때문에, 같은 선택지에서도 다른 선택을 할 수 있습니다. 이를 확률로 나타내는 거죠. 
즉 $$\rho(\tau_1\succ\tau_2)$$라고 하면 $$\tau_1$$을 $$\tau_2$$보다 선호할 확률인 것입니다. strict하게 접근하면 여집합이 성립하는 확률이고요, preference라는게 모호할 수도 있으니 여집합은 성립하지 않을수도 있습니다.  
이제 데이터 셋을 정의합니다. 모든 trajectory를 모아 놓은 것을 아래와 같이 정의합니다. 
<center> 
$$\zeta=\{\zeta_i\}=\{\tau_{i1}\succ\tau_{i2}\}_{i=1\dots N}$$  
</center>
  
<br />

### Objective
  
Objective function을 수식으로 정리하면 아래와 같습니다.
<center>
$$\boldsymbol{\tau}_{1} \succ \boldsymbol{\tau}_{2} \Leftrightarrow \operatorname{Pr}_{\pi}\left(\boldsymbol{\tau}_{1}\right)>\operatorname{Pr}_{\pi}\left(\boldsymbol{\tau}_{2}\right),$$
</center> 
<center> 
$$where\ \ \operatorname{Pr}_{\pi}(\boldsymbol{\tau})=\mu\left(s_{0}\right) \prod_{t=0}^{|\boldsymbol{\tau}|} \pi\left(a_{t} \mid s_{t}\right) \delta\left(s_{t+1} \mid s_{t}, a_{t}\right)$$
</center>  
이를 만족하는 $$\pi^*$$를 찾는 것이 목표가 되겠죠.   

<br />
근데 이건 preference의 차이가 아주 작을 때(게다가 사용자는 stochasticity까지 있을 때)에는 사용하기가 쉽지 않기 때문에, 조금 트릭을 사용하여 maximization 문제로 바꿔주겠습니다.

<center>
$$\boldsymbol{\tau}_{1} \succ \boldsymbol{\tau}_{2} \Leftrightarrow \pi^{*}=\arg \max _{\pi}\left(\operatorname{Pr}_{\pi}\left(\boldsymbol{\tau}_{1}\right)-\operatorname{Pr}_{\pi}\left(\boldsymbol{\tau}_{2}\right)\right)$$
</center>
이제 Deep learning의 전문 분야로 들어왔습니다. 아래의 Loss function을 사용해서 policy를 최적화시키면 된다는 얘기죠.  

<center>
$$L\left(\pi, \boldsymbol{\tau}_{1} \succ \boldsymbol{\tau}_{2}\right)=-\left(\operatorname{Pr}_{\pi}\left(\boldsymbol{\tau}_{1}\right)-\operatorname{Pr}_{\pi}\left(\boldsymbol{\tau}_{2}\right)\right)$$
</center>
추가적으로, 모든 dataset에 대해서 preference objective를 다 만족해야 하므로 weighted sum을 사용해서 최종 Loss function을 정의합니다.  
<center>
$$\mathcal{L}(\pi, \zeta)=\sum_{i=1}^{N} \alpha_{i} L\left(\pi, \zeta_{i}\right)$$
</center>
아주 단순하고 명확한 Loss function이 도출되었습니다. (물론 이제 시작)  
  
<br />
### PbRL algorithms

![image](https://user-images.githubusercontent.com/57203764/147902409-25d3600d-d2d0-4bd9-91cd-8a9aa63cfaf9.png)
PbRL의 알고리즘은 크게 3가지로 나뉩니다.
1. learning a policy computes a policy that tries to maximally comply with the preferences
2. learning a preference model learns a model for approximating the expert’s preference relation
3. learning a utility function estimates a numeric function for the expert’s evaluation criterion  

1번의 경우 direct policy learning, 다시 말해 Deep learning 방식으로 Loss function을 바로 minimization 해서 policy를 얻겠다는 겁니다. 그림에서는 위쪽 루프 (dashed line)가 되겠습니다.  
2,3번의 경우 preference를 가공해서 사용하겠다는 것입니다. 아래쪽 루프 (dotted line)이 되겠습니다.  
세가지 방식 모두 학습 이후에 new policy를 통해 sample을 얻고 다시 학습한다는 점에서 on-policy의 모습을 보여주고 있습니다. (처음에 언급한 Kimin Lee 박사님의 PEBBLE 논문에서 off-policy PbRL을 제안하였습니다)  

<br/>
### Related problem settings

RL의 문제점을 해결하기 위해 유사한 방법으로 접근한 연구들을 소개합니다.
1. Learning with advice  
   Classic RL + additional constraints(Rule or preference)
2. Ordinal feedback  
   numeric ranking instead of pairwise preference
3. IRL  
   Expert demo가 최적의 trajectory라는 아주 강력한 가정. 추가적인 feedback을 얻을 수 없다는 단점(GAIL이 있으니 이제는 가능한 것 같다.)
4. Unsupervised learning  
   학습할수록 policy가 더욱 **preferable**해 질 것이라는 강력한 가정.

전부 이해한 것은 아니지만 PbRL의 목적 중 하나인 non-expert의 interpretability 관점을 지닌 연구는 없음.  

<br/><br/><br/>

## Design principles of PbRL
---

이제 PbRL에서 어떤 방식으로 preference feedback을 주는지 알아보겠습니다. 본 논문은 3가지 type을 제안합니다.
1. action preference  
   같은 state에 대한 두가지 action preference 비교
2. state preference  
   다른 state에 대해서 각각 최고의 action을 비교
3. trajectory preference  
   (state, action)으로 구성된 sequential한 trajectory 전체를 비교
   
사실 3번이 1,2를 포함하므로 3번을 가정해도 무방합니다. (single step trajectory로 볼 수 있음) Trajectory preference를 사용할 경우 가장 큰 문제는 **credit assignment** 입니다.
> Yet, a difficulty with trajectory preferences is that the algorithm needs to determine which states or actions are responsible for the encountered preferences, which is also known as the temporal credit assignment problem

Objective function을 잘 짜서, trajectory를 비교했을 때 그 안에 어떤 state에서 취한 어떤 action에게 credit을 줄 것인지를 결정해야 한다는 뜻입니다. 

<br/>

### Learning a policy

**Policy distribution**
![image](https://user-images.githubusercontent.com/57203764/147904787-47d379c6-7168-4880-a5cf-bbb9a84d106f.png)
parameterized policy의 distribution을 구한 뒤 preference dataset에 대한 MAP(maximum-a-posterior)를 통해 최적의 policy를 구하는 방식입니다.  
특징은 policy distribution에서 2개의 policy를 sampling 하고, 그에 따른 trajectory pair를 buffer에 모아둡니다. 충분히 pair가 쌓인 뒤 expert feedback을 받습니다.  
이 때 Likelihood function은 아래와 같다고 하는데, **제 생각에는 negative가 붙어야 하지 않나 싶네요**.
> The likelihood is high if the realized trajectories $$\tau^\pi$$ of the policy $$\pi$$ are closer to preferred trajectory.  
<center>
$$\operatorname{Pr}\left(\boldsymbol{\tau}_{1} \succ \boldsymbol{\tau}_{2} \mid \pi\right)=\Phi\left(\frac{\mathbb{E}\left[d\left(\boldsymbol{\tau}_{1}, \boldsymbol{\tau}^{\pi}\right)\right]-\mathbb{E}\left[d\left(\boldsymbol{\tau}_{2}, \boldsymbol{\tau}^{\pi}\right)\right]}{\sqrt{2} \sigma_{p}}\right)$$
</center>
MAP나 MLE나 결과는 비슷하니까 MLE로 생각하면 직관적으로 이해는 됩니다.  
<br/>
이 방식의 문제는 distance function $$d$$가 필요하다는 것인데, Euclidean distance로는 high-dimenison continuous state space는 힘들다고 합니다.  

<br/>

**Ranking policy**

![image](https://user-images.githubusercontent.com/57203764/147905228-c23dbfa0-d49e-42a4-bf74-8110cf299d5c.png)
이번에는 policy를 대놓고 비교하는 겁니다. Multi-arm bandit과 비슷한 방식이라고 합니다. 특징은 trajectory rollout 즉시 preference feedback 과정을 거친다는 겁니다.  
일단 policy set을 잔뜩 준비하고, 비교한 후 EDPS(evolutionary direct policy search)라는 알고리즘을 통해서 더 나은 policy set으로 만들어줍니다.  
<center>
$$\operatorname{Pr}\left(\pi_{1} \succ \pi_{2}\right) \approx \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left(\boldsymbol{\tau}_{i}^{\pi_{1}} \succ \boldsymbol{\tau}_{i}^{\pi_{2}}\right)$$
</center>
딱봐도 그렇듯이 엄청난 양의 비교가 필요하다는 단점이 있습니다.  

<br/>

### Learning a preference model

Preference model은 classification 문제를 푸는 것과 같습니다.
<center>
$$C\left(a \succ a^{\prime} \mid s\right)$$
</center>
어떤 state $$s$$에 대해서 두 action의 preference를 알아낼 수 있는 모델이 되겠습니다. <a href="https://link.springer.com/content/pdf/10.1007/s10994-012-5313-8.pdf" title="PbRL-preference-model">Fürnkranz et al(2012)</a>가 제안한 알고리즘을 살펴보겠습니다. 

![image](https://user-images.githubusercontent.com/57203764/147988972-b8ed892e-59bd-4154-be5c-3059bcb16431.png)

line 5를 자세히 보겠습니다. initial state $$s$$에 대해서 모든 action을 다 훑는 것입니다. 그 이후로는 current policy로 trajectory를 형성합니다. 이 데이터를 바탕으로 preference model을 (classification model) 구성하게 되면, greedy policy를 이끌어낼 수 있습니다.
<center>
$$\pi^{*}(a \mid s)= \begin{cases}1 & \text { if } a=\arg \max _{a^{\prime}} k\left(s, a^{\prime}\right) \\ 0 & \text { else }\end{cases},$$  
$$where\ \ k(s, a)=\sum_{\forall a_{i} \in A(s), a_{j} \neq a} C\left(a_{i} \succ a_{j} \mid s\right)=\sum_{\forall a_{i} \in A(s), a_{j} \neq a} C_{i j}(s)$$
</center>
$$k(s,a)$$의 경우 resulting count라고 하는데, 높을 수록 가장 action set 내에서 preference가 높다고 보면 되겠습니다. 다만 continuous action set에서 어떻게 동작할 수 있을지는 의문입니다.  
또한 특징으로는 $$C$$를 countinuous function으로 사용할 경우, uncertainty를 얻어낼 수 있어서, exploration에 사용할 수 있다고 합니다. (Discrete이라고 안될 게 뭐지 싶지만 continuous가 더 잘되긴 하겠죠)  

<br/>

### Learning a utility function

Utility function은 RL에서 reward와 유사하지만 약간의 차이를 나타냅니다.
> However, in the PbRL case it is sufficient to find a reward function (=utility function) that induces the same, optimal policy as the true reward function.

Classic RL처럼 고정된 reward의 형태를 나타낼 필요가 없기 때문에 따로 이름을 붙여준 것입니다. 기본적으로는 Scalar utility를 사용하는데요, 아래와 같이 간단히 정의됩니다.
<center>
$$\boldsymbol{\tau}_{i 1} \succ \boldsymbol{\tau}_{i 2} \Leftrightarrow U\left(\boldsymbol{\tau}_{i 1}\right)>U\left(\boldsymbol{\tau}_{i 2}\right)$$
</center>

우리는 여기에서 utility를 최대화하는 policy를 선택하기만 하면 됩니다.
<center>
$$\pi^{*}=\max _{\pi} \mathbb{E}_{\operatorname{Pr}_{\pi}(\boldsymbol{\tau})}[U(\boldsymbol{\tau})]$$
</center>

![image](https://user-images.githubusercontent.com/57203764/147990022-43a4d7fa-f36c-4abe-a594-420b8715065e.png)

알고리즘을 살펴보겠습니다. 우선 모든 trajectory의 initial state는 sampling 된 값이므로, 서로 다른 곳에서 시작하는 trajectories임을 알 수 있습니다. 데이터셋을 만든 후 trajectory pair sampling을 통해서 **utility function을 학습하게 됩니다 (line 12)**. 그렇다면 utility function의 형태는 어떻게 되는 걸까요? 

#### Linear utility function

Linear utility function은 다음과 같이 정의될 수 있습니다.
<center>
$$U(\boldsymbol{\tau})=\boldsymbol{\theta}^{T} \boldsymbol{\psi}(\boldsymbol{\tau})$$
</center>
이 때 $$\psi$$는 feature function을 나타냅니다. prior knowledge가 있다면 이미 정의되어 있는 것일테고, DL을 사용하여 이를 추출할 수도 있겠습니다.  
최종 목표는 당연히 optimal utility function을 parameterize하는 $$\theta$$를 구하는 것입니다.  

Preference를 생각하려면 두 개의 trajectory를 비교해야겠죠. 그 utility 차이를 다음과 같이 정의합니다.
<center>
$$d\left(\boldsymbol{\theta}, \zeta_{i}\right)=\boldsymbol{\theta}^{T}\left(\boldsymbol{\psi}\left(\boldsymbol{\tau}_{i 1}\right)-\boldsymbol{\psi}\left(\boldsymbol{\tau}_{i 2}\right)\right)$$
</center>
우리는 이 차이가 커지도록 만들면 되겠습니다. 그래야 preference를 명확하게 구별해내는 utility function을 찾아낼 수 있을테니까요. 최적화 방법으로는 Loss를 사용하는 방법, 그리고 Log likelihood를 사용하는 방법이 있습니다. 근본적으로 큰 차이는 없습니다만 Loss를 사용하면 최소화, Log likelihood를 사용하면 최대화 문제이브로 function shape이 좌우 대칭 형태가 됩니다. 기존 연구들에서 사용한 방법들인데 크게 중요하지는 않을 것 같습니다. (저는 DL을 쓸거니까요)

![image](https://user-images.githubusercontent.com/57203764/148012135-75474499-d4ca-40ea-a2f1-a9966363085c.png)

![image](https://user-images.githubusercontent.com/57203764/148012159-4246e0ab-0f83-4c77-88c2-3b0516f6dfbf.png)  
<br/>

혹은 이마저도 귀찮다면 그냥 gradient descent를 사용해도 무방합니다. loss function으로 $$y=-x$$를 사용하면 $$\boldsymbol{\theta}_{k+1}=\boldsymbol{\theta}_{k}+\alpha\left(\boldsymbol{\psi}\left(\boldsymbol{\tau}_{k 1}\right)-\boldsymbol{\psi}\left(\boldsymbol{\tau}_{k 2}\right)\right)$$로 구할 수 있습니다.   

이러한 Linear utility function은 간단하지만 다음과 같은 경고를 포함하고 있습니다.
> However, it is **unclear** how the aggregated utility loss L(θ, ζ) is related to the policy loss L(π, ζ) (see Sec. 2.3), as the policy is subject to the system dynamics whereas the utility is not.

앞에서 살펴보았던 policy loss의 경우 policy가 실제로 action을 취하는 주체이다보니 system dynamics가 고려되었습니다만, 방금 utility function의 경우 주어진 trajectory에 대해서 단순한 utility 계산만 했죠. 그렇기 때문에 둘 사이의 관계가 **불명확**하다는 것입니다.  

#### Non-linear utility function

이런 저런 방법이 소개되었는데요, 가장 주목할 점은 DL을 사용하는 방식입니다. 앞에서 소개했던 PEBBLE 논문에서 baseline으로 삼기도 했던 방법입니다.  

<a href="https://arxiv.org/pdf/1706.03741.pdf" title="DRL_from_human_preference" >**DRL from human preference - Christiano et al.**</a>  

이 논문도 거쳐가야 하기 때문에, 추후에 따로 다루도록 하겠습니다.  

<br/>

### The temporal credit assignment problem

Classic RL에서도 temporal credit assignment는 커다란 문제였습니다. Policy의 value는 미래의 모든 가능성을 포함하기때문에, 현재 state에서 취하는 action이 얼마나 dominant한 영향을 끼칠지 알기가 어렵습니다. 이를 해결하기 위해서 Advantage를 사용하는 것이 표준적이라고 논문에서 설명하고 있습니다. 그러나 이를 **explicitly** 해결하기는 어렵습니다.
> Yet, if we try to solve the credit assignment problem explicitly, we also require the expert to comply with the Markov property. This assumption can easily be violated if we do not use a full state representation, i.e., if the expert has more knowledge about the state than the policy

Markov를 가정하기 위해서는 expert가 true state를 알아야 합니다. 하지만 세상의 모든 것이 Hidden Markov model인데 알 방법이 없겠죠. 

Utility를 정의하는 방법은 크게 세가지로 나뉩니다. 
1. Value-based utility  
   $$\pi^{*}(a \mid s)=\mathbb{I}\left(a=\underset{a^{\prime}}{\arg \max } \mathbb{E}_{\delta}\left[U\left(s^{\prime}\right) \mid s, a^{\prime}\right]\right)$$
   Transition model을 알 때만 사용 가능함.
2. Return-based utility  
   $$U(\boldsymbol{\tau})=\boldsymbol{\theta}^{T} \boldsymbol{\psi}(\boldsymbol{\tau})$$
   Reward-based utility로 일반화 가능함.
3. Reward-based utility
   $$U\left(s_{t}, a_{t}\right)=\boldsymbol{\theta}^{T} \boldsymbol{\varphi}\left(s_{t}, a_{t}\right)$$

3번의 경우 사실상 reward와 동일한 효과를 내는 것입니다. Preference 기반으로 reward를 알아낸다는 점에서 IRL과 유사한 형태를 띄게 되는 것이죠. 하지만 우리는 trajectory 사이의 preference를 비교할 것이기 때문에 return-based utility를 사용하는 것이나 마찬가지입니다.  
<center>
$$U(\boldsymbol{\tau})=\boldsymbol{\theta}^{T} \boldsymbol{\psi}(\boldsymbol{\tau})=\sum_{t=0}^{|\boldsymbol{\tau}|} \gamma^{t} U\left(s_{t}, a_{t}\right)$$
</center>

<a href="https://arxiv.org/pdf/1706.03741.pdf" title="DRL_from_human_preference" >**DRL from human preference - Christiano et al.**</a> 논문에서도 reward-based utility를 사용했습니다만 두 가지 특징이 있었습니다.

> In contrast to conventional reinforcement learning, a discount factor of γ = 1 is used for U(τ) in all approaches because the expert should not need to consider the effects of decay.

> A major problem as all considered trajectories have a finite length.

사람은 decay가 필요 없으니 안하겠다, 그리고 finite length를 사용할 수 밖에 없었다. 두 가지 모두 명확합니다.  

<br/>

### Trajectory preference elicitation

그렇다면 trajectory query를 어떻게 구성해야 expert로부터 좋은 feedback을 얻을 수 있을까요? Expert에게 preference를 얻어내는 것은 굉장히 cost가 높은 작업입니다. 자동화가 안되어 있는 작업이니까요. 최대한 적은 query를 사용해야 하는데, 그러면 exploration 문제가 발생해 local optimum에 빠지게 됩니다.   

#### Trajectory generation

PbRL의 Trajectory는 3가지의 특징을 가져야 합니다.
> In order to be informative, the obtained preferences should be different from existing trajectories. Yet, the trajectories should also be close to optimal in order to obtain useful information. Furthermore, the trajectories need to contain sufficient information about the transition function to compute an optimal policy.

1. Preference를 구분할 수 있을 정도로 구별되어야 하고
2. Optimal trajectory에 충분히 가까워야 하며
3. Transition function에 대한 정보를 충분히 담고 있어야 합니다.

Exploration을 충분히 해서 최대한 다양한 종류의 trajectory를 뽑아낸다면 3가지 조건을 만족할 수 있습니다. Preference query는 적게 할 수록 좋지만, trajectory generation 자체는 노동력이 들지 않으니 많으면 많을 수록 좋습니다.  
Exploration은 undirected, directed, heterogeneous, user-guided exploration으로 나뉩니다. 모든 연구마다 방법이 다르기 때문에, 자세히 설명하지는 않겠습니다만, DRL을 이용하는 경우 stochastic policy를 이용한 undirected exploration을 사용한다고 소개하고 있습니다. SAC 같은 알고리즘을 쓰면 Entropy가 높아지는 방향으로 exploration을 하게 되니까요.   
<br/>

#### Preference query generation

만들어진 trajectory로부터 어떤 query를 추출할 수 있을까요? 마찬가지로 3가지 방식이 있습니다.
1. Exhaustive generation : All possible queries (데이터셋 전부 사용)
2. Greedy generation : Use trajectories generated from the optimized policy (최신의 policy로 만들어낸 trajectory만 사용)
3. Interleaved generation : Several ways (여러가지 방식)

높은 utility를 가진 trajectory를 고르기도 하고, ensemble 방식을 사용해 variance가 큰 녀석을 고르기도 하고, 연구마다 다양한 방식이 존재합니다.  

<br/><br/><br/>

**계속 이어서 작성하는 중입니다..ㅠㅠ **