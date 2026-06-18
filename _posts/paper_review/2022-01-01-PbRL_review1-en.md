---
layout: post
title: Paper review - A survey of preference-based reinforcement learning methods
tags: archive
lang: en
---

The first post on my blog in 2022, and my first paper review, will start with A survey of preference-based reinforcement learning methods. The first time I came across Preference-based RL was at the SNU Summer AI lecture held during the summer break of 2021, given by Dr. Kimin Lee at Berkeley (he is a great speaker, so I really enjoyed it. If you're interested, definitely check it out! <a href="https://www.youtube.com/watch?v=MiwOvaywtew&t=569s" title="PEBBLE">SNU summer AI - PEBBLE</a>).
Let's start with the survey paper and take a look at the overall flow.

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
Preference-based reinforcement learning (PbRL) started from the following motivation.
> PbRL have been proposed that can directly learn from an expert's preferences instead of a hand-designed numeric reward.

RL based on rewards advanced dramatically with AlphaGo as a turning point, but it exposes several problems when it comes to defining the reward function. (crucially depends on the prior knowledge that is put into the definition of the reward function) Let's look at the four problems that account for the biggest part.
1. Reward hacking : The agent may maximize the given reward, without performing the intended task.
   For example, if you give a vacuum cleaner a positive reward when there is no dust, it won't remove the dust but will just stare at the spots where there is no dust.
2. Reward shaping : The reward does not only define the goal but also guides the agent to the correct solution.
   The thing is, not only do we not know the optimal reward function (it may not even exist in practice), but problems arise because a human has to design it. This is actually the biggest problem of RL....
3. Infinite rewards : Some applications require infinite rewards.
   For example, a self-driving car must never hit a person, so we would need to give it a negative infinity reward, but this breaks classic RL. (One of RL's assumptions is finite reward.)
4. Multi-objective trade-offs : The trade-off may not be explicitly known.
   The trade-off relationships between rewards are hard to grasp properly. (It would be difficult to set the balance of a self-driving car's ride comfort, speed, safety, etc. numerically.)

<br /><br />
In any case, the problem is that it's difficult to express the intrinsic motivation embedded in an agent's behavior as a numerical scalar value. Various methods have been studied to solve this. There are approaches like Inverse RL and learning with advice, but PbRL differs from them in one respect.
> PbRL aims at rendering reinforcement learning applicable to a wider spectrum of tasks and **non-expert users**.

In other words, PbRL aims to make it possible to train an agent with nothing but preferences, even for someone who knows nothing about RL. (A world where ordinary people can teach robots...!!)
<br /><br /><br />

## Preliminaries
---
We looked at the motivation of PbRL in the Introduction, and now let's look at the points needed to learn PbRL.
> Preference learning is about inducing predictive preference models from empirical data.

PbRL aims to infer a preference model from the user's data.

<br />

### Preference learning
So now shall we look at some equations? A preference is expressed in the following five ways. First, we unconditionally assume that we are always comparing two options.
![image](https://user-images.githubusercontent.com/57203764/147896080-d91e785f-b76b-401b-ab08-977a2e0b18c0.png)

### Markov decision processes with preferences (MDPP)

An MDPP consists of a sextuple.
<center>
$$(S,A,\mu,\delta,\gamma,\rho)$$  
</center>
$$S,A$$ denote the state and action spaces, $$\mu$$ is the initial state distribution, and $$\delta$$ denotes the transition probability $$\delta(s'|s,a)$$. $$\gamma\in[0,1]$$ is the discount factor, of course.
<br />

Now the unusual one is $$\rho$$, which is the probability distribution of preference. Since humans always have stochasticity, they may make different choices even given the same options. This expresses that as a probability.
That is, $$\rho(\tau_1\succ\tau_2)$$ is the probability of preferring $$\tau_1$$ over $$\tau_2$$. If approached strictly, it is a probability for which the complement holds, but since preference can be ambiguous, the complement may not hold.
Now let's define the dataset. We define the collection of all trajectories as follows.
<center> 
$$\zeta=\{\zeta_i\}=\{\tau_{i1}\succ\tau_{i2}\}_{i=1\dots N}$$  
</center>
  
<br />

### Objective
  
Organizing the objective function as an equation gives the following.
<center>
$$\boldsymbol{\tau}_{1} \succ \boldsymbol{\tau}_{2} \Leftrightarrow \operatorname{Pr}_{\pi}\left(\boldsymbol{\tau}_{1}\right)>\operatorname{Pr}_{\pi}\left(\boldsymbol{\tau}_{2}\right),$$
</center> 
<center> 
$$where\ \ \operatorname{Pr}_{\pi}(\boldsymbol{\tau})=\mu\left(s_{0}\right) \prod_{t=0}^{|\boldsymbol{\tau}|} \pi\left(a_{t} \mid s_{t}\right) \delta\left(s_{t+1} \mid s_{t}, a_{t}\right)$$
</center>  
The goal becomes finding $$\pi^*$$ that satisfies this.

<br />
But this is not easy to use when the difference in preference is very small (and on top of that the user has stochasticity), so we'll use a bit of a trick to turn it into a maximization problem.

<center>
$$\boldsymbol{\tau}_{1} \succ \boldsymbol{\tau}_{2} \Leftrightarrow \pi^{*}=\arg \max _{\pi}\left(\operatorname{Pr}_{\pi}\left(\boldsymbol{\tau}_{1}\right)-\operatorname{Pr}_{\pi}\left(\boldsymbol{\tau}_{2}\right)\right)$$
</center>
Now we've entered the specialty of Deep learning. The idea is that we just optimize the policy using the Loss function below.

<center>
$$L\left(\pi, \boldsymbol{\tau}_{1} \succ \boldsymbol{\tau}_{2}\right)=-\left(\operatorname{Pr}_{\pi}\left(\boldsymbol{\tau}_{1}\right)-\operatorname{Pr}_{\pi}\left(\boldsymbol{\tau}_{2}\right)\right)$$
</center>
Additionally, since the preference objective must be satisfied for the entire dataset, we use a weighted sum to define the final Loss function.
<center>
$$\mathcal{L}(\pi, \zeta)=\sum_{i=1}^{N} \alpha_{i} L\left(\pi, \zeta_{i}\right)$$
</center>
A very simple and clear Loss function has been derived. (Of course, this is just the beginning.)
  
<br />
### PbRL algorithms

![image](https://user-images.githubusercontent.com/57203764/147902409-25d3600d-d2d0-4bd9-91cd-8a9aa63cfaf9.png)
PbRL algorithms are broadly divided into three.
1. learning a policy computes a policy that tries to maximally comply with the preferences
2. learning a preference model learns a model for approximating the expert’s preference relation
3. learning a utility function estimates a numeric function for the expert’s evaluation criterion  

In case 1, it's direct policy learning—in other words, obtaining a policy by directly minimizing the Loss function in the Deep learning way. In the figure, this is the upper loop (dashed line).
In cases 2 and 3, the idea is to process the preference before using it. This is the lower loop (dotted line).
All three approaches show an on-policy character in that, after learning, they obtain samples through the new policy and learn again. (Dr. Kimin Lee's PEBBLE paper, mentioned at the start, proposed off-policy PbRL.)

<br/>
### Related problem settings

Here we introduce studies that approached RL's problems with similar methods.
1. Learning with advice  
   Classic RL + additional constraints (Rule or preference)
2. Ordinal feedback  
   numeric ranking instead of pairwise preference
3. IRL  
   A very strong assumption that the expert demo is the optimal trajectory. Its drawback is that you can't obtain additional feedback. (With GAIL now available, it seems possible nowadays.)
4. Unsupervised learning  
   A strong assumption that the more it learns, the more **preferable** the policy becomes.

I didn't fully understand everything, but none of these studies have the perspective of non-expert interpretability, which is one of PbRL's goals.

<br/><br/><br/>

## Design principles of PbRL
---

Now let's look at how preference feedback is given in PbRL. This paper proposes three types.
1. action preference  
   Comparing two action preferences for the same state
2. state preference  
   Comparing the respective best actions for different states
3. trajectory preference  
   Comparing the entire sequential trajectory composed of (state, action)
   
In fact, since case 3 includes cases 1 and 2, it's fine to assume case 3. (Cases 1 and 2 can be seen as single-step trajectories.) When using trajectory preference, the biggest problem is **credit assignment**.
> Yet, a difficulty with trajectory preferences is that the algorithm needs to determine which states or actions are responsible for the encountered preferences, which is also known as the temporal credit assignment problem

It means that we have to design the Objective function well so that, when comparing trajectories, we can determine which action taken in which state within them should be given credit.

<br/>

### Learning a policy

**Policy distribution**
![image](https://user-images.githubusercontent.com/57203764/147904787-47d379c6-7168-4880-a5cf-bbb9a84d106f.png)
This is a method that obtains the distribution of the parameterized policy and then finds the optimal policy via MAP (maximum-a-posterior) over the preference dataset.
A characteristic is that it samples two policies from the policy distribution and collects the resulting trajectory pair in a buffer. After enough pairs have accumulated, it receives expert feedback.
The Likelihood function here is said to be as follows, but **in my opinion, a negative sign should be attached**.
> The likelihood is high if the realized trajectories $$\tau^\pi$$ of the policy $$\pi$$ are closer to preferred trajectory.  
<center>
$$\operatorname{Pr}\left(\boldsymbol{\tau}_{1} \succ \boldsymbol{\tau}_{2} \mid \pi\right)=\Phi\left(\frac{\mathbb{E}\left[d\left(\boldsymbol{\tau}_{1}, \boldsymbol{\tau}^{\pi}\right)\right]-\mathbb{E}\left[d\left(\boldsymbol{\tau}_{2}, \boldsymbol{\tau}^{\pi}\right)\right]}{\sqrt{2} \sigma_{p}}\right)$$
</center>
Whether MAP or MLE, the result is similar, so thinking of it as MLE makes it intuitively understandable.
<br/>
The problem with this method is that it requires a distance function $$d$$, and Euclidean distance reportedly struggles with high-dimensional continuous state spaces.

<br/>

**Ranking policy**

![image](https://user-images.githubusercontent.com/57203764/147905228-c23dbfa0-d49e-42a4-bf74-8110cf299d5c.png)
This time, we compare policies head-on. It's said to be a method similar to multi-arm bandit. A characteristic is that it goes through the preference feedback process immediately upon trajectory rollout.
First, you prepare a whole set of policies, compare them, and then use an algorithm called EDPS (evolutionary direct policy search) to turn it into a better policy set.
<center>
$$\operatorname{Pr}\left(\pi_{1} \succ \pi_{2}\right) \approx \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left(\boldsymbol{\tau}_{i}^{\pi_{1}} \succ \boldsymbol{\tau}_{i}^{\pi_{2}}\right)$$
</center>
As is obvious, its drawback is that it requires an enormous amount of comparison.

<br/>

### Learning a preference model

Learning a preference model is the same as solving a classification problem.
<center>
$$C\left(a \succ a^{\prime} \mid s\right)$$
</center>
This becomes a model that can figure out the preference between two actions for a given state $$s$$. Let's look at the algorithm proposed by <a href="https://link.springer.com/content/pdf/10.1007/s10994-012-5313-8.pdf" title="PbRL-preference-model">Fürnkranz et al(2012)</a>.

![image](https://user-images.githubusercontent.com/57203764/147988972-b8ed892e-59bd-4154-be5c-3059bcb16431.png)

Let's look at line 5 in detail. For the initial state $$s$$, it sweeps through all actions. After that, it forms a trajectory with the current policy. Based on this data, if we build a preference model (a classification model), we can derive a greedy policy.
<center>
$$\pi^{*}(a \mid s)= \begin{cases}1 & \text { if } a=\arg \max _{a^{\prime}} k\left(s, a^{\prime}\right) \\ 0 & \text { else }\end{cases},$$  
$$where\ \ k(s, a)=\sum_{\forall a_{i} \in A(s), a_{j} \neq a} C\left(a_{i} \succ a_{j} \mid s\right)=\sum_{\forall a_{i} \in A(s), a_{j} \neq a} C_{i j}(s)$$
</center>
$$k(s,a)$$ is called the resulting count, and the higher it is, the higher the preference within the action set. However, it's questionable how this could work in a continuous action set.
Another characteristic is that if $$C$$ is used as a continuous function, you can obtain uncertainty, which can be used for exploration. (I wonder why it wouldn't work for discrete, but continuous probably works better.)

<br/>

### Learning a utility function

A utility function is similar to the reward in RL but shows a slight difference.
> However, in the PbRL case it is sufficient to find a reward function (=utility function) that induces the same, optimal policy as the true reward function.

Since it doesn't need to take the form of a fixed reward like in classic RL, it's given a separate name. Basically it uses a scalar utility, defined simply as follows.
<center>
$$\boldsymbol{\tau}_{i 1} \succ \boldsymbol{\tau}_{i 2} \Leftrightarrow U\left(\boldsymbol{\tau}_{i 1}\right)>U\left(\boldsymbol{\tau}_{i 2}\right)$$
</center>

Here, we just need to select the policy that maximizes utility.
<center>
$$\pi^{*}=\max _{\pi} \mathbb{E}_{\operatorname{Pr}_{\pi}(\boldsymbol{\tau})}[U(\boldsymbol{\tau})]$$
</center>

![image](https://user-images.githubusercontent.com/57203764/147990022-43a4d7fa-f36c-4abe-a594-420b8715065e.png)

Let's look at the algorithm. First, since the initial state of every trajectory is a sampled value, we can see that these are trajectories starting from different places. After creating the dataset, **we learn the utility function** through trajectory pair sampling (line 12). So what does the form of the utility function look like?

#### Linear utility function

A linear utility function can be defined as follows.
<center>
$$U(\boldsymbol{\tau})=\boldsymbol{\theta}^{T} \boldsymbol{\psi}(\boldsymbol{\tau})$$
</center>
Here $$\psi$$ denotes the feature function. If there's prior knowledge, it would already be defined; we could also extract it using DL.
The final goal is, of course, to find $$\theta$$ that parameterizes the optimal utility function.

To consider preference, we have to compare two trajectories. We define that utility difference as follows.
<center>
$$d\left(\boldsymbol{\theta}, \zeta_{i}\right)=\boldsymbol{\theta}^{T}\left(\boldsymbol{\psi}\left(\boldsymbol{\tau}_{i 1}\right)-\boldsymbol{\psi}\left(\boldsymbol{\tau}_{i 2}\right)\right)$$
</center>
We just need to make this difference larger. That way we can find a utility function that clearly distinguishes preferences. As for optimization methods, there's a method using Loss and a method using Log likelihood. There's no fundamentally big difference, but using Loss is a minimization problem while using Log likelihood is a maximization problem, so the function shape becomes left-right symmetric. These are methods used in prior studies, but they probably aren't very important. (Since I'm going to use DL.)

![image](https://user-images.githubusercontent.com/57203764/148012135-75474499-d4ca-40ea-a2f1-a9966363085c.png)

![image](https://user-images.githubusercontent.com/57203764/148012159-4246e0ab-0f83-4c77-88c2-3b0516f6dfbf.png)  
<br/>

Or if even that's a hassle, you can just use gradient descent. If you use $$y=-x$$ as the loss function, you can compute $$\boldsymbol{\theta}_{k+1}=\boldsymbol{\theta}_{k}+\alpha\left(\boldsymbol{\psi}\left(\boldsymbol{\tau}_{k 1}\right)-\boldsymbol{\psi}\left(\boldsymbol{\tau}_{k 2}\right)\right)$$.

Such a linear utility function is simple but contains the following caveat.
> However, it is **unclear** how the aggregated utility loss L(θ, ζ) is related to the policy loss L(π, ζ) (see Sec. 2.3), as the policy is subject to the system dynamics whereas the utility is not.

In the policy loss we looked at earlier, the policy is the entity that actually takes actions, so the system dynamics were taken into account. But the utility function we just saw only does a simple utility computation for a given trajectory. That's why the relationship between the two is **unclear**.

#### Non-linear utility function

Various methods are introduced, but the most notable one is the approach using DL. It's also the method that was used as a baseline in the PEBBLE paper introduced earlier.

<a href="https://arxiv.org/pdf/1706.03741.pdf" title="DRL_from_human_preference" >**DRL from human preference - Christiano et al.**</a>  

Since this paper also needs to be covered, I'll deal with it separately later.

<br/>

### The temporal credit assignment problem

In classic RL too, temporal credit assignment was a huge problem. Since a policy's value includes all future possibilities, it's hard to know how dominant an influence the action taken in the current state will have. To solve this, the paper explains that using Advantage is the standard approach. However, it's hard to solve this **explicitly**.
> Yet, if we try to solve the credit assignment problem explicitly, we also require the expert to comply with the Markov property. This assumption can easily be violated if we do not use a full state representation, i.e., if the expert has more knowledge about the state than the policy

To assume Markov, the expert has to know the true state. But everything in the world is a Hidden Markov model, so there's no way to know it.

The methods of defining utility are broadly divided into three.
1. Value-based utility  
   $$\pi^{*}(a \mid s)=\mathbb{I}\left(a=\underset{a^{\prime}}{\arg \max } \mathbb{E}_{\delta}\left[U\left(s^{\prime}\right) \mid s, a^{\prime}\right]\right)$$
   Usable only when the transition model is known.
2. Return-based utility  
   $$U(\boldsymbol{\tau})=\boldsymbol{\theta}^{T} \boldsymbol{\psi}(\boldsymbol{\tau})$$
   Can be generalized to reward-based utility.
3. Reward-based utility
   $$U\left(s_{t}, a_{t}\right)=\boldsymbol{\theta}^{T} \boldsymbol{\varphi}\left(s_{t}, a_{t}\right)$$

In case 3, this essentially produces the same effect as a reward. In that it figures out the reward based on preference, it takes a form similar to IRL. But since we will compare preferences between trajectories, it's effectively the same as using return-based utility.
<center>
$$U(\boldsymbol{\tau})=\boldsymbol{\theta}^{T} \boldsymbol{\psi}(\boldsymbol{\tau})=\sum_{t=0}^{|\boldsymbol{\tau}|} \gamma^{t} U\left(s_{t}, a_{t}\right)$$
</center>

The <a href="https://arxiv.org/pdf/1706.03741.pdf" title="DRL_from_human_preference" >**DRL from human preference - Christiano et al.**</a> paper also used reward-based utility, but it had two characteristics.

> In contrast to conventional reinforcement learning, a discount factor of γ = 1 is used for U(τ) in all approaches because the expert should not need to consider the effects of decay.

> A major problem as all considered trajectories have a finite length.

Humans don't need decay so they won't do it, and they had no choice but to use finite length. Both are clear.

<br/>

### Trajectory preference elicitation

So how should we construct the trajectory query to get good feedback from the expert? Obtaining preferences from an expert is a very high-cost task. It's a task that isn't automated, after all. We have to use as few queries as possible, but then an exploration problem arises and we fall into a local optimum.

#### Trajectory generation

A PbRL trajectory must have three characteristics.
> In order to be informative, the obtained preferences should be different from existing trajectories. Yet, the trajectories should also be close to optimal in order to obtain useful information. Furthermore, the trajectories need to contain sufficient information about the transition function to compute an optimal policy.

1. They must be distinct enough to distinguish preferences,
2. They must be sufficiently close to the optimal trajectory, and
3. They must contain sufficient information about the transition function.

If we explore sufficiently and extract as diverse a variety of trajectories as possible, we can satisfy all three conditions. The fewer preference queries the better, but trajectory generation itself takes no labor, so the more the better.
Exploration is divided into undirected, directed, heterogeneous, and user-guided exploration. Since the method differs for every study, I won't explain in detail, but it's introduced that in the case of using DRL, undirected exploration using a stochastic policy is used. If you use an algorithm like SAC, it explores in the direction of increasing entropy.
<br/>

#### Preference query generation

What kind of query can we extract from the created trajectories? There are likewise three approaches.
1. Exhaustive generation : All possible queries (uses the entire dataset)
2. Greedy generation : Use trajectories generated from the optimized policy (uses only trajectories created by the latest policy)
3. Interleaved generation : Several ways (various approaches)

Sometimes you pick trajectories with high utility, sometimes you use an ensemble approach to pick the one with high variance—various methods exist depending on the study.

<br/><br/><br/>

**Still in the process of writing this... ㅠㅠ**
