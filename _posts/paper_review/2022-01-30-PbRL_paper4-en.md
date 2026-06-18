---
layout: post
title: "Paper review - Skill Preferences: Learning to Extract and Execute Robotic Skills from Human Feedback"
tags: archive
lang: en
---

This is my sixth paper review. It's one of the "offspring" papers based on PEBBLE, coming out of the Berkeley group led by Pieter Abbeel, who is actively conducting research in the Preference-based RL field. It's a fresh-off-the-press paper presented at PMRL 2022. To summarize it in one line, it's research that combines skill learning — proposed to solve the long-horizon problem in the RL field — with PbRL.

The authors newly proposed an algorithm called Skill Preference (SkiP).
> an algorithm that learns a model over human preferences and uses it to extract human-aligned skills from offline data

In fact, it's just a slight modification of PEBBLE to fit skill learning. I think it's a good example showing that once you write one good paper, papers pour out thanks to the leverage effect.

<br>

Table of Contents
   - [Introdunction](#introdunction)
     * [What is skill learning?](#skill-learning---)
     * [Contribution](#contribution)
   - [Background](#background)
     * [Skill](#skill)
     * [Method](#method)
     * [Skill extraction : Learning behavior priors with human feedback](#skill-extraction---learning-behavior-priors-with-human-feedback)
     * [Skill execution : Reward learning and human preference over skills](#skill-execution---reward-learning-and-human-preference-over-skills)
   -  [Experiment setup](#experiment-setup)
   - [Experimental results](#experimental-results)
   - [Conclusion](#conclusion)

<br><br>

## Introdunction

PbRL is a field that began in order to avoid doing reward engineering. However, existing research stopped at solving very simple tasks. In fact, for simple tasks, reward shaping isn't very difficult, so the utility of PbRL is not considered large. So this paper began in order to address the following question.

> How can we learn robotic control policies that are aligned with human intent and capable of solving complex real-world tasks?

The long-horizon problem is one that has long been dealt with in RL. If you imagine a person using only a single policy throughout their entire day, you'd need an enormous, high-performing policy capable of solving every task encountered during the day. But creating such a policy not only carries a heavy computation burden, it also has the problem of having to explore an enormous state space. The most widely studied field for solving this is skill learning.

<br>

### What is skill learning?

You introduce a latent vector called a skill, and make the policy dependent on the skill. You split policy execution into two stages and define the policy as: 1) Which skill will I use in the current state? 2) Based on this skill and the state, what action will I produce?

Let me give an even simpler example. Consider a task where a robot arm fries an egg in the kitchen. If we break this down into skills, we can split it into policies that 1) open the cupboard, 2) take out the frying pan, 3) crack the egg, 4) turn on the heat, 5) flip the egg, and 6) turn off the heat. Because each skill is represented by a latent vector, we can't know exactly which skill it is.

Usually, skills are extracted from an offline dataset composed of expert demonstrations. This is because experts presumably acted with skills, so the assumption is that skill extraction is possible from there. Skills are generally extracted using an autoencoder (AE), a representative method for extracting latent vectors.

![images](https://d1m75rqqgidzqn.cloudfront.net/wp-data/2020/04/29202749/Blog_info_29-04-2020-R-02-1024x522.png?style=centerme){:width="60%"}

The AE structure above is familiar, but how do we use an AE in RL? It's a structure where a state goes in as input and an action comes out as output. We reconstruct the action taken in each state. The main idea is that if we extend this to a state and action sequence, the latent vector in the middle will represent a skill.

Expressed as an equation, it's as follows. Here, $$p_\alpha(\mathbf{a}_t\|s_t)$$ denotes the generative model (AE), and $$\mathbf{a}_t=(a_t,\dots,a_{t+H-1}$$ denotes the action sequence. We find the latent vector — that is, the skill — that reconstructs the action sequence well.

<center>
$$p_{\alpha} \in \arg \max _{\alpha} \mathbb{E}_{\tau \sim \mathcal{D}}\left[\sum_{t=0} \log \left(p_{\alpha}\left(\mathbf{a}_{t} \mid s_{t}\right)\right)\right]$$
</center>

<br>

### Contribution

So how is human preference used in skill learning? The overall schematic is shown in the figure below.

![image](https://user-images.githubusercontent.com/57203764/151723906-242e0ca9-943e-4278-b549-8c059dbb5cb1.png?style=centerme){:width="70%"}

Human preference is used a total of two times. Once in the place where skills are extracted, and once in the place where the downstream task of skill execution is learned. The contributions of this paper are as follows.

- Proposed a new algorithm that combines skill learning and human preference.
- Used human preference in skill extraction so that it operates robustly even on a noisy offline dataset.
- Verified performance by experimenting in the actual D4RL environment.

<br><br>

## Background

### Skill

Let me just define skill very simply with equations and move on.

- Leanring skills $$z\in\mathcal{Z}$$

- encoder : $$q^{(e)}\left(z \mid s_{t}, a_{t}, \ldots, s_{t+H-1}, a_{t+H-1}\right)$$ → skill extraction

- decoder : $$q^{(d)}\left(a_{1}, a_{2}, \ldots, a_{H} \mid s, z\right)$$ → action sequence reconstruction (used only for training and unnecessary afterward)

<br><br>

### Method

The proposed algorithm SkiP is divided into 2 stages (skill extraction, skill execution). The full algorithm is as below, and let's look at each one.

![image](https://user-images.githubusercontent.com/57203764/151728233-e5345d8b-349c-4302-acae-e33384923c57.png?style=centerme){:width="70%"}


### Skill extraction : Learning behavior priors with human feedback

As mentioned in the Introduction, skill extraction extracts skills using an AE from an offline database composed of expert demos.
<center>
$$p_{\alpha} \in \arg \max _{\alpha} \mathbb{E}_{\tau \sim \mathcal{D}}\left[\sum_{t=0} \log \left(p_{\alpha}\left(\mathbf{a}_{t} \mid s_{t}\right)\right)\right]$$
</center>

The most vulnerable point of skill extraction using an AE is that it fails to extract proper skills when strange trajectories are included. There's no way a skill would be present in a trajectory produced by a policy that has nothing to do with skills and doesn't even solve the task properly, right? But collecting only good trajectories into the database is very labor-intensive and expensive work.

So the proposed method is something called **weighted behavioral priors**. Literally, the idea is to extract skills by putting weights on good trajectories. You just multiply the original equation by a weight.

<center>
$$p_{\alpha} \in \arg \max _{\alpha} \mathbb{E}_{\tau \sim \mathcal{D}}\left[\sum_{t=0}^{|\tau|} \exp \left(\omega\left(\tau_{t}\right) / T\right) \cdot \log \left(p_{\alpha}\left(\mathbf{a}_{t} \mid s_{t}\right)\right)\right]$$
</center>

So how do we know the weights? This is exactly where human preference comes in. We create a function such that the higher the human preference, the larger the weight.

<center>
Human preference predictor (Binary predictor)  
$$\omega\left(\tau_{t}\right):=\log P_{\psi}\left(\tau_{t}\right)$$
</center>

The method is crude but simple. We train the preference predictor using 10% of the data randomly selected from the database. Since it's binary classification, it trains very simply.

For the AE, a Variational AE is used. Actually, a VAE is not an AE. The structure is merely similar, but it is alike in the very meaning of extracting a latent vector. Above all, in that it extracts a latent space in the form of a Gaussian distribution, it can train a more powerful encoder.
A VAE is trained using a bounded loss function called the ELBO.

<center>
$$\log p\left(\mathbf{a}_{t} \mid s_{t}\right) \geq \mathbb{E}_{\tau \sim \mathcal{D}, z \sim q_{\phi_{2}}(z \mid \tau)}[\underbrace{\log p_{\phi_{1}}\left(\mathbf{a}_{t} \mid s_{t}, z\right)}_{\mathcal{L}_{\text {rec }}}+\beta \underbrace{\left(\log p(z)-\log q_{\phi_{2}}(z \mid \tau)\right.}_{\mathcal{L}_{\text {reg }}}]$$
</center>

The final loss, multiplied by the preference predictor weight, is as follows.

<center>
$$\mathcal{L}=\arg \max _{\phi_{1}, \phi_{2}} E_{\tau \sim \mathcal{D}, z \sim q_{\phi}(z \mid \tau)}\left[P_{\psi}(\tau)\left(\mathcal{L}_{\mathrm{rec}}+\mathcal{L}_{\mathrm{reg}}\right)\right]$$
</center>

<br>

### Skill execution : Reward learning and human preference over skills

The downstream task is learned in exactly the same way as conventional RL. Of course, here it uses not plain RL but PbRL, and brings in PEBBLE as-is (paper leveraging... I'm jealous). The only difference from PEBBLE is the length of the trajectory.

<center>
$$\tau^{(z)}=\left(s_{t}, z_{t}, s_{t+H}, z_{t+H}, \ldots, s_{(t+M) H}, z_{(t+M) H}\right)$$
</center>
Here, $$H$$ is the number of actions taken after selecting a skill once, and $$M$$ is the total number of skill changes.
This is something I asked a friend who is a skill-learning expert: the problem of deciding when to switch skills is apparently still unsolved. In fact, fixing it at a fixed length $$H$$ is, common-sensically, strange. After all, the number of optimal actions to take differs per skill.
Intuitively, it seems there should be one more policy that decides when to switch. However, all studies that used this method end up facing a collapse where the skill changes at every timestep.

Below is the reward function equation used in PbRL, specifically in PEBBLE. You compare 2 trajectories, do binary labeling, and derive the reward function $$R_{\eta}$$ that bakes this in. You then train the policy using the derived reward function and the SAC algorithm.

$$P_{\eta}\left[\tau_{1}^{(z)} \succ \tau_{0}^{(z)}\right]=\frac{\exp \sum_{t} \widehat{R}_{\eta}\left(s_{t}^{1}, z_{t}^{1}\right)}{\sum_{i \in\{0,1\}} \exp \sum_{t} \widehat{R}_{\eta}\left(s_{t}^{i}, z_{t}^{i}\right)}$$

$$\mathcal{L}^{\text {Reward }}=-\mathbb{E}_{\left(\tau^{0}, \tau^{1}, y\right) \sim \mathcal{D}}\left[y(0) \log P_{\eta}\left[\tau_{0}^{(z)} \succ \tau_{1}^{(z)}\right]+y(1) \log P_{\eta}\left[\tau_{1}^{(z)} \succ \tau_{0}^{(z)}\right]\right]$$

<br><br>

## Experiment setup

For experiments, the 7-DoF robot arm environment of the D4RL suite was used, specifically the kitchen task. The kitchen task includes opening and closing drawers, turning on lights, and so on.

![image](https://user-images.githubusercontent.com/57203764/151732135-6fd5398f-e9f9-4a8e-82e8-ce05e0c39254.png?style=centerme){:width="80%"}

<br>

**Offline dataset**  

For skill extraction, an offline dataset containing expert demos is needed. Because this research claims robustness to noisy datasets via human preference, it deliberately includes non-expert-demo trajectories in the DB. They built a total of 1202 data points by combining 601 trajectories obtained from a well-trained policy and 601 random-policy trajectories.

<br><br>

## Experimental results

The experimental results are also very short and simple.

First, they conducted an experiment on whether it can solve long-horizon tasks well. For SkiP, the preference predictor was trained using 10%, i.e., 120 trajectories, and for SkiP 3X, 30% was used. Both show higher performance than PEBBLE, and since the difference between the two is barely noticeable, they judged that 10% of the data is sufficient.
![image](https://user-images.githubusercontent.com/57203764/151732432-ca45cd90-60a2-4025-9c34-94f84c309fdb.png?style=centerme){:width="80%"}


Second, they experimented with how robustly SkiP performs skill extraction on noisy datasets. In fact, because as much as half were mixed in as random trajectories, ordinary skill extraction methods naturally don't work. (If you mix in a moderate amount, it might be similar....)
![image](https://user-images.githubusercontent.com/57203764/151732577-5202edb3-232c-4be5-85f8-d97633fd8575.png?style=centerme){:width="80%"}

Finally, they checked how effective human preference is for the downstream task. Actually, this was already shown to perform in PEBBLE; the difference is that the downstream tasks are very easy ones. Learning is sufficiently possible even with a sparse success/failure reward. I think that with just a little reward shaping, it would show higher performance than SkiP.
![image](https://user-images.githubusercontent.com/57203764/151732741-c428314e-d34a-4a07-b091-2e43a5c9f6da.png?style=centerme){:width="80%"}

<br><br>

## Conclusion

This was research that solved long-horizon tasks with PbRL and skill learning. Honestly, if you've read the PEBBLE paper first, I don't think you'll find anything that special about it (I don't mean it's bad research, just that it may not be a sparkling, dazzling piece of work...). Still, if there's a distinctive point, it would be the attempt to use human preference split across 2 stages, I suppose.  