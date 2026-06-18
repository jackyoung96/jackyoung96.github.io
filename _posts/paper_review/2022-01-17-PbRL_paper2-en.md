---
layout: post
title: Paper review - Learning latent representations to influence multi-agent interaction
tags: archive
lang: en
---

This is my third paper review. It's a brilliant paper that won the Best Paper Award at CoRL 2020. It isn't actually about Preference-based RL, but I decided to review it because 1) it considers human interaction, and 2) it designed its experiments very well and led to impressive results.

Even if we train a robot, there are many people and other robots that interact with it. These agents will continually update their strategies, and the core of this research is to predict the next behavioral strategy from previous behavior and, based on that, continually update the ego agent's behavior as well. The authors claim that this can even be used to steer the opponent agent into taking specific actions.

<br>

Table of contents
   - [Introduction](#introduction)
   - [Related works](#related-works)
     - [Opponent modeling](#opponent-modeling)
     - [MARL](#marl)
     - [Influence through interactions](#influence-through-interactions)
     - [POMDP](#pomdp)
   - [Repeated Interactions with Non-stationary agents](#repeated-interactions-with-non-stationary-agents)
   - [Learning and Influencing Latent Intent (LILI)](#learning-and-influencing-latent-intent-lili)
     - [Influencing by optimizing for long-term rewards](#influencing-by-optimizing-for-long-term-rewards)
   - [Experiments](#experiments)
     - [Simulated environments ](#simulated-environments)
     - [Air Hockey result](#air-hockey-result)
     - [Playing against a Human Expert](#playing-against-a-human-expert)
   - [Conclusion](#conclusion)
<br><br>

## Introduction

This paper begins with a story containing a really excellent insight. (It's a very impressive way to start a paper, and personally I think it's great for grabbing attention.)
> . The first time you encounter this autonomous car, your intention is to act cautiously: when the autonomous car slows, you also slow down. But when it slowed without any apparent reason, your strategy changes: the next time you interact with this autonomous car, you drive around it without any hesitation. The autonomous car — which originally thought it had learned the right policy to slow your car — is left confused and defeated by your updated behavior

This is something that could actually happen. Once people start adapting to autonomous vehicles, they'll attempt aggressive cut-ins nearby, thinking "it'll dodge out of the way on its own anyway." To resolve situations like this, understanding the intentions of other agents is important. That said, this paper is a study that focuses on situations where this intention continually changes, so it's a little distant from the example above.
**In reality, there's no case where you repeatedly interact with a single vehicle on the road. Since only a single episode exists, I think understanding the more general intention — human preference — is actually more important.**

The representative scenario that appears in this research is shown in the figure below.

![image](https://user-images.githubusercontent.com/57203764/149857934-20c0b2f4-86d9-45f4-b452-ca7c904139ab.png?style=centerme){:width="60%"}

1. In the i-th round, the Opponent shot the hockey puck to the right, and the ego agent successfully defended.
2. In the i+1-th round, the Opponent thought the right side wouldn't work and shot to the center, but was defended again.
3. In the i+2-th round, the Opponent chose the left, which was neither of the others, and the ego agent predicted this in advance and defended. From the i-th and i+1-th attempts, it predicted the opponent's strategy in the form of a latent vector.

<br>

Now, before diving into the full content, let's look at the contributions to see exactly what research was carried out. The contributions claimed in this paper are as follows.
1. Learning latent representation  
    It can learn the opponent agents' strategies in the form of latent vectors.
2. Influencing other agents  
   It can steer opponent agents to have specific strategies.
3. Testing in multi-agent settings  
   It was validated with an actual real-world 7 DoF robot.

<br><br>

## Related works

### Opponent modeling
This is a method frequently used in MARL (Multi-Agent Reinforcement Learning). If there is no communication between agents, you have no choice but to predict exactly what action another agent will take. This is because the transition probability depends on the other agent's action, so if you don't know it, there's no way to know which action is optimal.
The first paper to claim this was the [Multi-agent actor-critic for mixed cooperative-competitive environment](https://arxiv.org/abs/1706.02275) paper published by OpenAI in 2017.

However, since it assumes the strategies of other agents are static, it differs from this paper.

<br>

### MARL
Generally, it uses a centralized training framework and assumes that communication is possible. This paper uses a decentralized, non-communication environment.
The reason this is possible is that it's based on a very strong assumption: **other agents have predictable strategies which can be captured by latent representations**. In fact, it's not clear whether this is true.

<br>

### Influence through interactions
> In particular, robots have demonstrated influence over humans by leveraging human trust [18, 32], generating legible motions [33], and identifying and manipulating structures that underlie human-robot teams [34]. Other works model the effects of robot actions on human behavior in driving [35, 36] and hangover [37] scenarios in order to learn behavior that influences humans

This is an area where a great deal of research has been done. I'll review these contents later. The key point is that, whereas the studies introduced here focus on recovering the opponent agent's strategy or recovering the reward function, this paper can operate more generally by representing it in the form of a latent vector.

<br>

### POMDP
Fittingly for a MARL problem, it assumes a POMDP. If we said it was an MDP, we'd also need to know the opponent's state, so it wouldn't hold. Even more surprisingly, the authors claim that **prediction is possible even if the opponent has a Non-Markovian strategy**. This is shown experimentally, and they didn't prove it theoretically.

<br><br>

## Repeated Interactions with Non-stationary agents
It assumes MARL, but for now it deals with an environment composed of 2 agents. They say it's scalable, but they didn't show it experimentally. I do think the scalability is actually high. After all, it increases linearly.

First, this problem is defined as a HiP-MDP (Hidden Parameter Markov Decision Process). The "parameter" means that a value whose actual meaning we cannot know is included in the MDP, and that is precisely the latent vector of the opponent's strategy. We define the latent vector of step i as $$z^i$$.

Two things are affected by this latent vector. The transition probability and the reward function, which is intuitively understandable.
<center>
$$T(s'|s,a,z^i),\\R(s,z^i)$$
</center>  

We can predict the next step's opponent latent strategy using the ego agent's trajectory. The ego car's trajectory is defined as follows.
  
<center>
$$\tau^i=\{(s_1,a_1,r_1),\dots(s_H,a_H,r_H)\}$$
</center>  

In conclusion, the Markovian latent dynamics model we want to learn is
<center>
$$z^{i+1}\sim f(\cdot|z^i,\tau^i)\sim f(\cdot|\tau^i)$$
</center>
We assume that $$z^i$$ is embedded within $$\tau^i$$.

<br><br>

## Learning and Influencing Latent Intent (LILI)

Actually, everything is well represented in the figure. (To win Best Paper, it seems you really do have to draw good figures.)

Our goal is to find $$\mathcal{E}_\phi$$ that predicts the latent strategy. However, since we can't know what $$z^i$$ is, we can't do supervised learning. Therefore, we learn the latent vector using the AutoEncoder structure in the representation learning part on the right side of the figure.

It's a very simple structure. $$\tau^{k-1},\tau^k$$ are each composed of $$(s,a,r,s')$$ tuples.
1. $$\mathcal{E}_\phi$$ takes the $$(s,a,r,s')$$ of $$\tau^{k-1}$$ as input and outputs $$z^k$$.
2. $$\mathcal{D}_\phi$$ takes $$z^k$$ as input and predicts the transition probability and reward. That is, it takes $$(z^k,s,a)$$ as input and outputs $$(r,s')$$.

The objective function for training this Autoencoder is as follows.
<center>
$$\mathcal{J}_{rep}=\max _{\phi, \psi} \sum_{i=2}^{N} \sum_{t=1}^{H} \log p_{\phi, \psi}\left(s_{t+1}^{i}, r_{t}^{i} \mid s_{t}^{i}, a_{t}^{i}, \tau^{i-1}\right)$$
</center>

![image](https://user-images.githubusercontent.com/57203764/149860515-e20390c3-fd24-4730-b5fa-79cca3683a06.png?style=centerme){:width="80%"}


The process of training the entire encoder, decoder, and RL networks is explained in detail in the algorithm below. It's a very simple algorithm. For the RL networks, they used SAC (Soft Actor Critic), and $$\mathcal{J}_\pi,\mathcal{J}_Q$$ mean the actor and critic losses respectively.

![image](https://user-images.githubusercontent.com/57203764/149862140-9964ba3e-8e4d-4008-a744-f3986bee83c0.png?style=centerme){:width="70%"}  

For the encoder and decoder, they used 2 fully-connected layers of size 128, and for the actor and critic, 2 fully-connected layers of size 256. Also, the latent strategy $$z^i$$ size is 8. Since these are simple problems being solved, that's sufficient.

<br>

### Influencing by optimizing for long-term rewards

If you look at the experience buffer, you can notice something odd. Since the latent strategy will keep being updated, you'd probably need to do on-policy learning, but it takes the form of off-policy learning. Isn't that strange?
In fact, the authors separate the two approaches in their explanation.
1. LILI (No influence) : on-policy  
   This is the approach of drawing out the maximum return based on the latent strategy predictor you currently have. That is, when learning, instead of looking at all trajectories, it learns using only the immediately preceding trajectory.
2. LILI : off-policy  
   It uses all trajectories. What this means is that it assumes a Non-Markovian strategy and aims to maximize the final return generated across all interactions. Expressed as a formula, it's as follows.
   <center>
   $$\max _{\theta} \sum_{i=1}^{\infty} \gamma^{i} \mathbb{E}_{\rho_{\pi_{\theta}}^{i}}\left[\sum_{t=1}^{H} R\left(s, z^{i}\right)\right]$$
   </center>

To get straight to the point, LILI is slow to learn in the early stages, but ultimately exhibits the characteristic of steering the opponent's strategy in a direction favorable to the ego agent.

<br><br>

## Experiments
The experiments consisted of 3 simulated-environment experiments and a real robot experiment.

### Simulated environments 

1. Point Mass  
   A task of reaching a target point, but the location of the target point is unknown. When an episode ends, the target point moves one step ccw or cw.
2. Lunar Lander  
   The destination keeps changing, and you have to land there. The location of the destination is unknown.
3. Driving (2D, CARLA)  
   You try to pass a vehicle, but the vehicle in front changes lanes right before being passed.

![image](https://user-images.githubusercontent.com/57203764/149891710-3cfd4df7-05b0-4a51-a2cf-7851b60f134b.png?style=centerme){:width="80%"}  

<br>
Four baselines were used.  

1. Soft Actor Critic
2. SLAC : assumes the opponent strategy is fixed
3. LILAC : assumes the opponent strategy changes according to the env
4. Oracle : assumes the opponent's state and policy are known

<br>
Shall we take a look at the results?

![image](https://user-images.githubusercontent.com/57203764/149892411-eb81754c-04f1-47d3-989f-490d8aa6695c.png?style=centerme){:width="80%"}

First of all, it shows overwhelming performance across the board. Also, in the Point Mass task, LILI produced a result where it steered the point's behavior and trapped it.

<br>

### Air Hockey result

The experiment was conducted with an actual robot. The game is structured as follows.
+ The Opponent shoots the puck to one of left, center, or right
+ The Ego agent takes an action toward one of left, center, or right
+ The input is the puck's vertical position (you can only tell whether it's getting closer)
+ Defending to the left gives +2 points, defending with the others gives +1 point, and failing to block gives 0 points

The Opponent's policy is as follows.
![image](https://user-images.githubusercontent.com/57203764/149893157-44047e42-13b6-494d-8040-5dbfd16f7c2f.png?style=centerme){:width="40%"}  

<br>

Below are the experimental results. [Result video](https://sites.google.com/view/latent-strategies/)

![image](https://user-images.githubusercontent.com/57203764/149892960-9742c232-6806-4c2d-b77e-935fd51d8038.png?style=centerme){:width="70%"}

Surprisingly, you can see that LILI steers the Opponent into throwing the puck to the left.

<br>

### Playing against a Human Expert
They experimented to see whether the latent strategy could be extracted against a human too. Honestly, intuitively I thought it wouldn't work. I thought that if a person threw randomly without thinking, there'd be no way to block it. But there was one assumption included.
> Like the robot striker, this human aims away from where the ego agent blocked during the previous interaction

With an assumption like this, it could be learned. In fact, while SAC blocked 45%, LILI blocked 73%.

<br><br>

## Conclusion

We looked at research on predicting the change in the opponent's latent strategy during interaction with a person or robot, and updating one's own policy accordingly. The authors describe that there were still many limitations in the case of human interaction, but I thought it was an interesting topic. I get the feeling that preference could actually be incorporated through a similar method.



