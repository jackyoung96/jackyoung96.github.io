---
layout: post
title: Paper review - Learning trajectory preferences for manipulators via iterative improvement
tags: archive
lang: en
---

This is my second paper review. I'll be reviewing the paper Learning trajectory preferences for manipulators via iterative improvement, submitted to NeurIPS in 2013.  
PbRL itself doesn't seem to be a very old research field, so a 2013 paper can be considered truly ancient. That said, since the survey paper mentioned that it used trajectory planning, I adopted it as my first one. It doesn't use deep learning at all, and carries out learning with a very simple linear parameterized optimization.  

Research using human demonstrations on a manipulator was actually well known. (I can't quite remember what the robot's name was, though...)

[![image](https://user-images.githubusercontent.com/57203764/148876845-b50a671f-558f-42cb-8691-8951700e2d8a.png?style=centerme){:width="50%"}](https://www.youtube.com/watch?v=M413lLWvrbI)

This paper argues that in **high-DoF environments**, giving a slightly improved trajectory preference rather than a human demo is more helpful for learning. Let's take a look at the method and the results.  
<br/>

Table of contents
   - [Introduction](#introduction)
   - [Related works](#related-works)
   - [Learning and Feedback model](#learning-and-feedback-model)
   - [Learning algorithm](#learning-algorithm)
   - [Features describing object-object interactions](#features-describing-object-object-interactions)
       - [Trajectory features](#trajectory-features)
       - [Learning the scoring function](#learning-the-scoring-function)
   - [Experiments](#experiments)
       - [Evaluation metric](#evaluation-metric)
       - [Results](#results)
   - [Conclusion](#conclusion)

<br><br>

## Introduction
---
This paper does a good job of defining, with several examples, why human preference is needed.
> a household robot should move a glass of water in an upright position without jerks while maintaining a safe distance from nearby electronic devices  

> For example, trajectories of heavy items should not pass over fragile items but rather move around them  

The gist of this research is that even when moving the same object, you should behave differently depending on what the object is and what is around it. This is a human preference that is hard to express as a reward function, and the goal is to let the robot learn it.  
> The robot learns a general model of the user's preferences in an online fashion.

In this process you can't really verify whether learning is going well, so the author defined something called regret. It's structured so that the more the ranking of trajectories that the human and the robot think of becomes the same, the smaller the regret (explained later).  

They used the Grocery checkout task — a problem of placing several objects and moving them with a robot arm — as the experimental environment.  

<br/><br/>

## Related works
---
This paper makes many comparisons with Learning from demonstration (LfD). The biggest problem with LfD is that you can't know whether the user demonstration is actually **optimal**.
> The user never discloses the optimal trajectory

**In other words, the goal of this research is to learn how to improve based on preference.**  
> Learning a score function representing the preferences in trajectories

Here, the goal is to obtain a score function, similar to the utility function that was one of the methods of PbRL.  

<br/><br/>

## Learning and Feedback model
---
First, let's call the scoring function we want to learn $$s(x,y;w)$$. Here $$x$$ is the context, $$y$$ is the trajectory, and $$w$$ is the parameter we want to learn. The optimal scoring function reflecting human preference is denoted $$s^*(x,y)$$.

The process of learning the scoring function is divided into 3 steps.

1. Step 1: The robot receives a context x. It then uses a planner to sample a set of trajectories, and ranks them according to its current approximate scoring function s(x, y; w).   
   Based on the context x, the robot forms multiple trajectories. Then it assigns them ranks with the scoring function. It runs trajectory generation using RRT, and since there is randomness, various kinds of trajectories will be produced.
2. Step 2: The user either lets the robot execute the top-ranked trajectory, or corrects the robot by providing an improved trajectory y¯. This provides feedback indicating that s∗(x, y¯) > s∗(x, y).   
    - Re-ranking : selects the top trajectory and gives feedback by correcting the ranking.
    - Zero-G : directly moves one of the trajectory waypoint positions.
3. Step 3: The robot now updates the parameter w of s(x, y; w) based on this preference feedback and returns to step  
    updates the scoring function.

It's a truly simple structure, clearly. Since this is actually a paper from before even the AlphaGo paper came out, it may look crude from the perspective of today's deep learning, but if this kind of theory hadn't laid the groundwork, deep learning wouldn't have been able to advance either.  

Finally, they define Regret for performance evaluation.
<center>
$$REG_T=\frac{1}{T}\sum_{t=1}^T[s^*(x_t,y_t^*)-s^*(x_t,y_t)]$$
$$where\ y^*_t=argmax_ys^*(x_t,y)$$
</center>

But something seems off. Since $$s^*$$ is actually an unknown value, you can't compute the Regret. So the author proves convergence using a regret bound. (Comes up later.)  

To receive human feedback, you need a UI/UX. The author says they used a program called OpenRave. It was set up so that clicking on one of the multiple trajectories makes its rank the highest.  

<br/><br/>

## Learning algorithm
---
First, let's keep in mind that this is a paper that doesn't use deep learning.  
The scoring function is defined as follows.
<center>
$$s(x,y;w_O,w_E)=w_O\dot\phi_O(x,y)+w_E\dot\phi_E(x,y)$$
</center>
Here O denotes the surrounding objects that the trajectory interacts with, and E denotes the object that must be manipulated and its environment.  

### Features describing object-object interactions
---
> We enumerate waypoints of trajectory y as $$y_1, .., y_N$$ and objects in the environment as O = {$$o_1$$, .., $$o_K$$}. The robot manipulates the object $$\bar{o}$$ ∈ O
> we connect an object ok to a trajectory waypoint if the minimum distance to collision is less than a threshold or if $$o_k$$ lies below

It defines the trajectory, the objects, and the manipulated object, and connects them. When they get close enough, it connects an edge. An example figure is shown below.


![image](https://user-images.githubusercontent.com/57203764/148962075-2a5c2e40-b400-4d3f-aeb7-f8953067ff7e.png?style=centerme){:width="50%"}

<br/>

First, the overall scoring function is as follows.
<center>
$$s_{O}\left(x, y ; w_{O}\right)=\sum_{\left(y_{j}, o_{k}\right) \in \mathcal{E}} \sum_{p, q=1}^{M} l_{k}^{p} l^{q}\left[w_{p q} \cdot \phi_{o o}\left(y_{j}, o_{k}\right)\right]$$
</center>
$$l_k^p$$ is the p-th attribute of the k-th object $$o_k$$. Every object has M attributes $$[l_k^1,\dots,l_k^M]$$, and each attribute is represented as binary. For example, a Laptop has the following attributes.
<center>
{heavy, fragile, sharp, hot, liquid, electronic} = [0,1,0,0,0,1]
</center>
Pretty naïve, isn't it?? It can't be helped. DL hadn't properly developed yet!  
$$l^q$$ is the q-th attribute of the manipulated object $$\bar{o}$$. That's because the distance to surrounding objects must be adjusted depending on which object you're moving.  
$$\phi_{oo}(y_j,o_k)$$ is the feature of the edge. It belongs to $$\phi_{oo}\in\mathcal{R}^4$$, consisting of the minimum x, y, z distances + (a binary for whether $$o_k$$ lies vertically with $$\bar{o}$$).  

<br/>
In the end, it's expressed as $$\phi_o(x,y)=\sum_{\left(y_{j}, o_{k}\right) \in \mathcal{E}} l_{k}^{u} l^{v}\left[\phi_{o o}\left(y_{j}, o_{k}\right)\right]$$.  

<br/>

### Trajectory features
The trajectory is first split into 3 parts. (No idea why 3.)  

![image](https://user-images.githubusercontent.com/57203764/148970025-214c5c22-64cc-4ea6-b933-384b57fcb318.png?style=centerme){:width="50%"}
In the figure it's split into 3 waypoints, namely 1, 2, 4. For each trajectory segment, 3 features are applied respectively and then concatenated for use.

1. Robot arm configuration $$\in \mathcal{R}^{27}$$  
    ($$r,\theta,\phi$$) of wrist and elbow w.r.t shoulder + elbow when the end effector attains maximum state (can indicate whether a joint lock occurs)
2. Orientation  and temporal behavior of the object to be manipulated $$\in \mathcal{R}^{28}$$  
  -> part we store the cosine of the object’s maximum deviation, along the vertical axis, from its final orientation at the goal location + maximum deviation along the whole trajectory

3. Object-environment interactions $$\in \mathcal{R}^{20}$$  
    (i) minimum vertical distance from the nearest surface below it. (ii) minimum horizontal distance from the surrounding surfaces; and (iii) minimum distance from the table, on which the task is being performed, and (iv) minimum distance from the goal location

In total they formed a trajectory feature of $$\phi_E(\cdot)\in\mathcal{R}^{75}$$. Since I'm not really interested in hand-made features, I'll move on quickly.  

<br/>

### Learning the scoring function
It learns the parameters in an absurdly simple way.
![image](https://user-images.githubusercontent.com/57203764/148972396-113e140b-147f-4b1d-99f6-16056cb06cb8.png?style=centerme){:width="50%"}
It's just a simple linear update, right? It's not even a random initialization.  

So then, how can this minimize Regret? The author used Expected $$\alpha$$-informative feedback.

<center>
$$E_{t}\left[s^{*}\left(x_{t}, \bar{y}_{t}\right)\right] \geq s^{*}\left(x_{t}, y_{t}\right)+\alpha\left(s^{*}\left(x_{t}, y_{t}^{*}\right)-s^{*}\left(x_{t}, y_{t}\right)\right)-\xi_{t}$$
</center>
They say that if you choose appropriate $$\alpha,\xi$$ here, it is bounded as $$E\left[R E G_{T}\right] \leq \mathcal{O}\left(\frac{1}{\alpha \sqrt{T}}+\frac{1}{\alpha T} \sum_{t=1}^{T} \xi_{t}\right)$$. (For details, see <a href="https://arxiv.org/pdf/1205.4213.pdf" title="Regret bound" >the reference Online Structured Prediction via Coactive Learning</a>.)  

<br/><br/>

## Experiments
---
![image](https://user-images.githubusercontent.com/57203764/148977450-d26a2a86-0f12-4697-9e7f-1461623dab31.png?style=centerme)
They ran experiments on the following 3 tasks.

1. Manipulator centric : just moving an object
2. Environment centric : moving a fragile object
3. Human centric : moving a sharp object while avoiding a person

As baselines they used BiRRT, Manual, Oracle-SVM, and MMP-online (maximum margin planning).

### Evaluation metric
To give a human preference, it has to be expressible as a number, whatever it may be. Likert scale and nDCG (normalized discounted cumulative gain) were used.  
The Likert scale gives a 5-way choice from 1–5 (5 being best).
nDCG is something often used in ranking recommendation algorithms and the like; it's scored so that recommending the high-ranked items well is more important than recommending the low-ranked ones. The details are well explained on the following blog. <a href="https://walwalgabu.tistory.com/entry/4-NDCG-Normalized-Discounted-Cumulative-Gain%ED%8F%89%EA%B0%80%EC%A7%80%ED%91%9C" title="nDCG" >Go to the nDCG explanation blog</a>


### Results
![image](https://user-images.githubusercontent.com/57203764/148978523-1399b503-a6e2-4406-ac9e-837a745de15c.png?style=centerme){:width="50%"}
First, the TPP method got the highest score on all tasks.

![image](https://user-images.githubusercontent.com/57203764/148978828-5e835a91-7102-440d-a195-94324e5cc191.png)
You can see it adapts well to new environments and objects too. Oracle-SVM produces high performance early on, but it has the drawback of being hard to use in real life since it requires knowing the entire trajectory space.
> This algorithm leverages the expert’s labels on trajectories (hence the name Oracle) 
and is trained using SVM-rank in a batch manner. This algorithm is not realizable in practice, as it requires labeling on the large space of trajectories  

![image](https://user-images.githubusercontent.com/57203764/148979499-28d66f8a-2c6c-40a9-b034-ecca89fccbda.png?style=centerme){:width="60%"}
They even conducted a user study. Viewed from an HCI perspective, you can see that as the task gets harder, the time and number of feedbacks increase. You can also confirm that the more feedback users give, the higher the score they give.  

<br/><br/>

## Conclusion
They studied a method for well-selecting a robot manipulator's trajectory using human preference. Rather than trajectory generation, this is really a paper that lets the robot judge which of the many trajectories formed by RRT is the best.  

The disappointing part is that it used hand-designed features, but I'm sure later papers have improvements on this point.
