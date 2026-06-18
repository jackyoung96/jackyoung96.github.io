---
layout: post
title: Paper review - Interactive teaching strategies for agent training
tags: archive
lang: en
---

This is my fourth paper review. Like the previous ones, it's hard to call this Preference-based RL, but it seemed to be a paper showing one type of human-interaction RL, so I decided to review it.

Since it's a paper published at IJCAI in 2015, it doesn't use DRL (this was before DRL took off). The goal of this paper is for a teacher agent (Human) to provide help while the RL agent explores, and to minimize laborious human attention in that process.

<br>

Table of Contents
   - [Introduction](#introduction)
   - [Student-Teacher RL](#student-teacher-rl)
     * [Teacher-initiated advising](#teacher-initiated-advising)
     * [Student-initiated advising](#student-initiated-advising)
   - [Empirical evaluation](#empirical-evaluation)
   - [Conclusion](#conclusion)
     * [Pros](#pros)
     * [Cons](#cons)

<br><br>

## Introduction

Starting learning from a random policy is extremely inefficient. That's because it has to involve a process of exploring an enormously large state space. Humans, on the other hand, already have meta-knowledge accumulated over 20-plus years. The idea is to receive help from this so that the agent **can have a good experience buffer**.

> Agents learning how to act in new environments can benefit from input from more experienced agents or humans

It's an approach that enables smooth exploration by having actions guided by an expert who already knows the environment (which could be a Human, or another RL agent).

However, a human expert requires cognitive cost (the labor, time, etc. spent on recognizing). In other words, a method to minimize this is also needed. From the perspective of PbRL, this is in the same vein as efficient query generation.

In short, the goal is to find a way for the human to be involved for as little time as possible.
> Aiming to reduce the attention required from the teacher

<br><br>

## Student-Teacher RL

First, let's assume the Teacher has a fixed policy (you could say it has a particular preference). But then a question arises. Couldn't we just use the Teacher's policy as the Student's policy directly? There's a decisive difference between the Teacher and the Student: namely, their **input state representation** is different.

For example, when playing an Atari game, a Human plays based on visual information at 100 Hz, whereas the Student is based on a featurized state — that kind of difference. The state is the same, but the observation is different.

But even though the state differs, the action is the same. The gist is that if the Teacher selects the action the Student should take in a particular situation, better data will accumulate in the Buffer, and this can increase the learning speed.

So how should we interact? As mentioned, we have to choose a method that takes as little human effort as possible. Let's look at the example below.

> For example, if a person is helping an autonomous car to improve its policy, the overall duration of the teaching period does not matter because the person is always present when the car drives. However, the person might not pay at ention to the road at all times and therefore there is a cost as ociated with monitoring the car’s actions to decide whether to intervene. Moreover, if teaching in this setting requires the human to take control over the car, then providing advice incurs an additional cost beyond monitoring the agent’s behavior (i.e., deciding whether to intervene requires less effort than actually intervening

In other words, occasionally making a multiple-choice selection is easier for a human than directly intervening to operate the steering wheel. Picking actions one at a time is intuitively easier than providing an action sequence or continually giving Expert demos.

<br>
This way of providing actions is broadly divided into two methods.

1. Teacher-initiated advising

    The Teacher keeps observing, and when a particular situation arises, it provides advice.

2. Student-initiated advising

    When the Student requests advice from the teacher in a particular situation, the teacher provides it.

Let's briefly look at the two methods.

<br><br>

### Teacher-initiated advising

First, we define a value called Importance as follows.

<center>
$$I(s)=\max _{a} Q_{(s, a)}^{\text {teacher }}-\min _{a} Q_{(s, a)}^{\text {teacher }}$$
</center>

A large importance means that the action chosen in the current state has a large effect on the future cumulative reward, so it implies a state where a good decision must be made. However, this paper doesn't state how the teacher's Q value can be defined. After all, you wouldn't be able to define a Human Q value function. I couldn't figure it out no matter how much I looked, so I'll skip it for now.

The conditions for providing advice are divided into two. Because always giving an action would just be the same as Imitation learning.

1. Advise Importance

    Provides the teacher action when $$I(s)>t_{ti}$$

2. Correct importance

    Provides the teacher action when $$I(s)>t_{ti}$$ and $$\pi_{teacher}(s)\neq\pi_{student}(s)$$

There doesn't seem to be a big difference, but since Correct importance has one more condition attached, I think the attention will be a bit less.
<br>


### Student-initiated advising

This time it's a method where the Student requests help first. Since the Student is an RL agent, its functions are clearly defined.

The conditions for requesting help are divided into three.

1. Ask Importance

    Requests when the Student importance $$I(s)=\max _{a} Q_{(s, a)}^{\text {student}}-\min _{a} Q_{(s, a)}^{\text {student}} > t_{ti}$$
    → Requests at states where the Student thinks an important decision must be made

2. Ask Uncertainty

    Requests when $$\max _{a} Q_{(s, a)}^{\text {student }}-\min _{a} Q_{(s, a)}^{\text {student }}<t_{\text {unc }}$$
    → Requests when the Student is confused

3. Ask Unfamiliar

    Requests when $$\operatorname{distance}(s, N N(s))>t_{u n f}$$
    → Requests when it hasn't been there before

It's a very simple setup. That said, from the PbRL perspective, I think it would be a good answer to the question of how to select Queries. If the RL agent only generates Queries under particular conditions, that would definitely reduce Human labor.

<br><br>

## Empirical evaluation

Honestly, what I wanted to know was the formulas and the exact experimental methods, but they weren't introduced in detail in the paper. Only a simple empirical evaluation was conducted, which is a letdown.

The experiments were performed in the Pac-Man environment, and SARSA was used as the RL algorithm. Since it's a paper from before DRL appeared, it can't be helped. For the time, it was a state-of-the-art algorithm.

The conclusion is that the Ask Importance - Correct Importance method most effectively increased reward while decreasing human attention. You can confirm this in the graph below. (The graph is also drawn a bit half-heartedly, which makes me a little annoyed ^^)

![image](https://user-images.githubusercontent.com/57203764/150701733-b4d6e35c-f896-44c3-b20c-789474fe39a4.png?style=centerme){:width="70%"}


What the graph shows is clear. With the simple Correct Importance method, the Teacher has to keep watching, so the cumulative attention increases rapidly, whereas if you use the Student-initialized method, you can reach a similar reward level while clearly reducing attention.

One thing worth noting is that, comparing Fig 3 and 4, the same experiment was conducted with only the number of pre-training episodes differing. Here, when the pre-training episodes were increased, you can see that the scale of attention exploded. In other words, if you pre-train, it can be harder to steer things back to what the human wants. It's the same logic as how a blank-slate kid is easier to teach. (It reminds me of Kang Baek-ho.)

<br><br>

## Conclusion

The paper was simpler than I expected, so I'll write down the pros and cons I think of and wrap up. There's a slightly disappointing feeling. From next time, I think I should review a much more famous paper.

### Pros

1. It made me think that, from the perspective of Expert interaction, you could distill preference into it. A preference that directly specifies actions (though it would be enormously expensive).
2. You can gain insight into how to select preference queries well.
3. It made me think that pre-training might have an adverse effect on preference learning.

### Cons

1. It doesn't reveal how the Teacher Q value is defined.
2. It doesn't reveal exactly how the Teacher intervenes.
3. The Teacher's intervention would cause corner-case failures, but this isn't taken into account.


