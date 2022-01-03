---
layout: post
title: Paper review - A survey of preference-based reinforcement learning methods
---

2022년 제 블로그의 첫 글이자 첫 논문 리뷰는 A survey of preference-based reinforcement learning methods 부터 시작해 보겠습니다. 처음 Preference-based RL를 알게 된 건 Berkeley에 계시는 Kimin Lee 박사님이 하신 2021년 여름방학에 진행된 SNU Summer AI 강연이었습니다 (강연을 잘하셔서 정말 재밌게 들었습니다. 관심 있으신 분들은 꼭 들어보세요! <a href="https://www.youtube.com/watch?v=MiwOvaywtew&t=569s" title="PEBBLE">SNU summer AI - PEBBLE</a>).
리뷰 논문부터 시작해서 한 번 흐름을 살펴보죠.
<br /><br /><br />

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

PbRL의 동기를 Introduction에서 알아보았고, 이제는 PbRL을 학습하기 위한 포인트들을 알아보겠습니다. 
> Preference learning is about inducing predictive preference models from empirical data.  

PbRL은 사용지의 데이터로부터 preferece model을 유추하는데 목적이 있습니다.   
  
<br />

### Preference learning
그럼 이제 수식을 좀 살펴볼까요? Preference는 아래와 같이 5가지로 나누어 표현됩니다. 일단 무조건 2가지 선택지를 비교하는 걸 가정합니다.
![image](https://user-images.githubusercontent.com/57203764/147896080-d91e785f-b76b-401b-ab08-977a2e0b18c0.png)

### Markov decision processes with preferences (MDPP)

MDPP는 sextuple로 구성됩니다.  
$$
(S,A,\mu,\delta,\gamma,\rho)
$$  
$$S,A$$는 state와 action space를 나타내고, $$\mu$$는 initial state distribution, $$\delta$$는 transition probability $$\delta(s'|s,a)$$ 를 나타냅니다. $$\gamma\in[0,1]$$는 discount factor겠죠.
<br />

이제 특이한 건 $$\rho$$ 인데요, 이는 probability distribution of preference 입니다. 사람은 언제나 stochasticity가 있기 때문에, 같은 선택지에서도 다른 선택을 할 수 있습니다. 이를 확률로 나타내는 거죠. 
즉 $$\rho(\tau_1\succ\tau_2)$$라고 하면 $$\tau_1$$을 $$\tau_2$$보다 선호할 확률인 것입니다. strict하게 접근하면 여집합이 성립하는 확률이고요, preference라는게 모호할 수도 있으니 여집합은 성립하지 않을수도 있습니다.  
이제 데이터 셋을 정의합니다. 모든 trajectory를 모아 놓은 것을 아래와 같이 정의합니다.  
$$
\zeta=\{\zeta_i\}=\{\tau_{i1}\succ\tau_{i2}\}_{i=1\dots N}
$$  
  
<br />

### Objective
  
$$
\boldsymbol{\tau}_{1} \succ \boldsymbol{\tau}_{2} \Leftrightarrow \operatorname{Pr}_{\pi}\left(\boldsymbol{\tau}_{1}\right)>\operatorname{Pr}_{\pi}\left(\boldsymbol{\tau}_{2}\right),
$$  
$$
where\ \ \operatorname{Pr}_{\pi}(\boldsymbol{\tau})=\mu\left(s_{0}\right) \prod_{t=0}^{|\boldsymbol{\tau}|} \pi\left(a_{t} \mid s_{t}\right) \delta\left(s_{t+1} \mid s_{t}, a_{t}\right)
$$  
을 만족하는 $$\pi^*$$를 찾는 것이 목표가 되겠죠.
