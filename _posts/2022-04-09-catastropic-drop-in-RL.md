---
layout: post
title: Diary - Catastropic performance drop of off-policy RL methods
---

On-policy RL methods (TRPO, PPO) are guaranteed of their performance increasing. Instead, off-policy RL methods (DQN, TD3, SAC) converge much faster because of their sample efficiency. However, I realized that these off-policy RL methods are vernerable by overfitting especially with lack of exploration. 
[This blog explained it as "Catastropic drop"](https://ai.stackexchange.com/questions/28079/deep-q-learning-catastrophic-drop-reasons)
[Supplement: sudden exploding happended in RL](https://machinelearningmastery.com/exploding-gradients-in-neural-networks/)

<img width="343" alt="image" src="https://user-images.githubusercontent.com/57203764/162605084-ad6cf35d-28b7-47dc-9919-ccabd0c82d62.png?style=centerme">

I always suffered from this results when I trained SAC and DQN. Now we should add learning rate scheduler or regularizers or much more exploration noise to solve this issue. ([REGULARIZATION MATTERS IN POLICY
OPTIMIZATION](https://openreview.net/pdf?id=B1lqDertwr)) If you don't do lazy things, just use PPO even if it needs more time to train.

Especially, there is gradient exploding issue when the recurrent networks are used. [Exploding Gradients Problem with RNN](https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem) 

![exploding gradient](https://i0.wp.com/neptune.ai/wp-content/uploads/exploding-gradients.png?resize=513%2C350&ssl=1)
To solve this issue, we must use L2 normalization (most common answer) and gradient clipping. The next question is "How can we determine the threshold value for gradient clipping?".