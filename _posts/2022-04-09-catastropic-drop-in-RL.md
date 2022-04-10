---
layout: post
title: Diary - Catastropic performance drop of off-policy RL methods
---

On-policy RL methods (TRPO, PPO) are guaranteed of their performance increasing. Instead, off-policy RL methods (DQN, TD3, SAC) converge much faster because of their sample efficiency. However, I realized that these off-policy RL methods are vernerable by overfitting especially with lack of exploration. 
[This blog explained it as "Catastropic drop"](https://ai.stackexchange.com/questions/28079/deep-q-learning-catastrophic-drop-reasons)

<img width="343" alt="image" src="https://user-images.githubusercontent.com/57203764/162605084-ad6cf35d-28b7-47dc-9919-ccabd0c82d62.png?style=centerme">

I always suffered from this results when I trained SAC and DQN. Now we should add learning rate scheduler or dropout layers or much more exploration noise to solve this issue. If you don't do lazy things, just use PPO even if it needs more time to train.