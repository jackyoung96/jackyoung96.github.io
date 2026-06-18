---
layout: post
title: Diary - Network architecture for RL
tags: archive
lang: en
---

Reinforcement learning can be classified as a kind of supervised learning, in that it uses a **Label** called **Reward**. The thing to watch out for most in supervised learning is probably overfitting. To solve this, the most important thing is finding an appropriate network architecture (size, depth, hidden dimension, learning rate, regularizer, etc.). Yet the majority of RL studies, as if by agreement, use a fixed form of network. (2-layer feed forward networks) Of course, since the simulation environment is fixed to OpenAI gym or Mujoco, that's only natural...

The research I'm doing at UMD uses a somewhat unusual simulator built on the pybullet engine. It would be nice if there were many papers applying reinforcement learning using this environment, but there weren't many papers to use as references, so it was hard to settle on a network structure. After just recklessly making it big and training it, the learning seemed to be going well, but when I actually tested it the performance kept falling short. So a thought suddenly struck me: isn't this overfitting...??

Actually, in reinforcement learning it's hard to verify whether you're properly preventing overfitting. That's because the training environment and the test environment are often identical, so you can't satisfy the i.i.d. condition between training data and validation data. There's a paper that shows overfitting does occur in reinforcement learning too, by forcibly creating an i.i.d. environment.
[A study on overfitting in deep reinforcement learning](https://arxiv.org/pdf/1804.06893.pdf%C2%A0)

In the environment used in this study, a reward of +1 or -1 is given randomly depending on the environment's seed. With the same seed, the reward scheme is also the same. In other words, by separating the training environment seed and the test environment seed, you can satisfy the i.i.d. condition. As shown in the figure below, overfitting indeed occurs when training with too few episodes.

<figure>
<img width="556" alt="image" src="https://user-images.githubusercontent.com/57203764/179384991-5c9829dd-edbf-4b79-ac3b-3ab1bdf6ecd8.png?style=centerme">
<figcaption>Overfitting occurring in a random maze</figcaption>
</figure>

The interesting thing is that stochasticity for exploration doesn't necessarily solve overfitting either. It seems like you could just explore well to expand the data distribution, but the experimental results weren't like that. So how do you solve it? Let me quote a sentence from this paper.

**"Those blackbox policies are relatively poorly understood, and they might implicitly acquire certain kind of robustness due to the architectures or the training dynamics"**

In a word, they don't know.

But the Discussion explains that this is only natural, and I quite like the example. The "Backward Brain Bicycle" is a bicycle where everything is reversed. So if you steer right, it goes left. This is a kind of muscle memory, so they say it takes a person several months to relearn. It can be called biological overfitting. In that sense, reinforcement learning, which can only experience a limited state distribution, inevitably faces the overfitting problem.

So then, what network architecture should you use to squeeze out as much performance as possible? Thankfully, the Google research team used an enormous amount of GPU to run the experiments in advance for us. Unfortunately the analysis is only for the PPO algorithm, but off-policy algorithms probably aren't very different either (this is just my hunch). It naturally varies by environment, so it's probably best to focus on gaining intuition.
[WHAT MATTERS FOR ON-POLICY DEEP ACTORCRITIC METHODS? A LARGE-SCALE STUDY](https://openreview.net/pdf?id=nIAxjsniDzg)

They compared over 60 things, but I'll record only the ones I think are meaningful.
- Making the policy hidden dimension and depth blindly large degrades performance.
- Making the critic hidden dimension and depth large does not degrade performance. (Referring to a paper with a ridiculous name, performance actually doesn't change much when you keep the critic size while shrinking the actor size. [Honey. I Shrunk The Actor: A Case Study on Preserving Performance with Smaller Actors in Actor-Critic RL](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9619008&casa_token=vtwbej0Fo7QAAAAA:jfPsBslj17GfVvt1yYFBOmwrXY-B_cvctfXHFhG8pH9HrjJxSaU7yAHMa5RDkLl1sEcIueZ9HLY&tag=1))
- Using Tanh rather than ReLU showed higher performance.
