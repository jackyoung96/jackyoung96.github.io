---
layout: post
title: "Paper review - Diff-DAC: Distributed Actor-Critic for Average Multitask Deep Reinforcement Learning"
tags: archive
lang: en
---

This is my sixth paper review, back after a month. It's a paper related to Multi-agent Reinforcement learning, and it deals with a very realistic problem. Let's suppose we train a robot with RL. If we put a policy trained on robot A directly into robot B, will it work well? Even if A and B are completely identical robots, there will be subtly different parts. The center of mass might be in a different location, the remaining battery level might differ, or there could be differences in the motor's torque curve. So how can we create a policy that works well no matter which robot you put it in? This paper found the answer to that in distributed MARL.

To be continue...

<br>

Table of Contents

<br><br>

## Introduction
