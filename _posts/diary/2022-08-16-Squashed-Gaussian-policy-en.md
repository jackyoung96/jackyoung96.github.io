---
layout: post
title: Diary - Sqaushed Gaussian policy for SAC
tags: archive
lang: en
---

SAC (Soft Actor-Critic) is a very famous model-free, stochastic policy gradient RL algorithm. Due to the nature of a stochastic policy, it doesn't derive perfectly optimal actions, but it can derive more robust actions.

I think this can be explained through the cliff walk problem.

<img width="500" alt="image" src="https://user-images.githubusercontent.com/57203764/184815427-d615ace7-d278-4a70-b560-2b284ad52769.png?style=centerme">
Source [https://dnddnjs.gitbooks.io/rl/content/q__learning.html](https://dnddnjs.gitbooks.io/rl/content/q__learning.html)

This figure compares SARSA and Q-learning, but it also holds if you put a stochastic policy in place of SARSA. It's obvious that a greedy algorithm (deterministic) is more optimal, but if you take into account the probability of falling off the cliff, it will choose a safer and more robust action.

The core of SAC is that it computes entropy and uses it as a regularizer for exploration. By maximizing the policy entropy, it makes exploration happen actively. A stochastic policy generally uses a Gaussian policy. It samples the action using the mean and std (standard deviation). In SAC, the standard deviation is directly used to compute the entropy, so it has the characteristic that it must be state-dependent. In other words, backpropagation must be possible, and that means we have to use the [reparameterization trick](https://simpling.tistory.com/34).

Up to here was something I already knew, but the problem was the method of using tanh at the end. The method I used at first was to use tanh when computing the mean and to use the output of the linear layer as-is for the std.

$$a_\theta(s,z)=tanh(\mu_\theta(s)) + \sigma_\theta(s)\odot z,\ \ \ \ \ z\sim N(0,1)$$

I'm not sure why I did it this way, but I think I just assumed "isn't this obviously how you do it?" However, the actual SAC was implemented a little differently. It said it uses a squashed Gaussian policy, and it generated the action in the following way.

$$a_\theta(s,z)=tanh(\mu_\theta(s) + \sigma_\theta(s)\odot z),\ \ \ \ \ z\sim N(0,1)$$

It's almost the same, but there's quite a performance difference. Instead, doing it this way requires using one trick when computing the entropy (negative log probability), which is given in the [SAC appendix](arxiv.org/pdf/1801.01290.pdf).

$$\pi(a|s)=\mu(u|s)\ |det(\frac{da}{du})|^{-1}$$

$$\log\pi(a|s)=\log\mu(u|s)-\sum_{i=1}^D\log(1-tanh^2(u_i))$$

$$where\ \ a=tanh(u)$$

I found this here. [Stable-baselines3 SAC document](https://spinningup.openai.com/en/latest/algorithms/sac.html#:~:text=we%20use%20a-,squashed,-Gaussian%20policy%2C%20which)
