---
layout: post
title: Direct Alignment from Preferences - Part 02. DAP
tags: archive
lang: en
---

# Introdunction

In the previous post, "**Direct Alignment from Preferences Part - 01. RLHF**", we looked at how to train a language model that can generate responses matching human preferences using reinforcement learning (RL). However, compared to supervised learning, RL is an unstable training method. (Training is heavily affected by hyperparameters, and it also requires an enormous amount of exploration to secure data that can earn good rewards. I'll cover the instability of RL in more detail in a future post if I get the chance.)

In addition, methods that use RL have the drawback of high computation cost. You need to use four models for training: the Policy, the Reference policy, the Reward, and the Value Model. In other words, the bigger the model gets, the more burdensome these methods become.

For this reason, various methods that improve training stability without using RL have emerged, and these are called **DAP (Direct Alignment from Preference)**. In Part 2 of this three-part series, we'll take a look at DAP methods.

# Model alignment - DAP

## SLiC-HF (Deepmind)

This is a method proposed in the [SLiC-HF: Sequence Likelihood Calibration with Human Feedback](https://arxiv.org/abs/2305.10425) (Zhao et al., 2023) paper, which Deepmind released in 2023.

SVM, a classic ML approach, trains a model so that positive samples and negative samples are well separated. The idea is to make it possible to separate positive samples and negative samples with a hyperplane, keeping them a certain margin distance apart.

![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/67d244f9-e01b-474a-aa5c-eec6d996a733)

SLiC-HF uses the loss function used in SVM to train the LLM to distinguish positive samples from negative samples with respect to human preference, so that the LLM can generate positive samples, i.e., responses that reflect human preferences.

- SLiC-HF loss function 
    $$
    \mathcal{L}(\theta) = \max(0, \delta - \log P_\theta(y^+|x) + \log P_\theta(y^-|x)) - \lambda \log P_\theta(y_{\text{ref}}|x)
    $$
    - The first part of the loss is the hinge-loss used in SVM. You can also understand it as raising the likelihood of positive samples and lowering the likelihood of negative samples.
    - The latter part of the loss is a regularizer that keeps the model from diverging too far from the Reference model (SFT). Usually it's common to use KL-divergence, but apparently you can achieve the same effect even using the NLL value for the Reference model's responses. Using this approach lets you reduce the computation cost of having to keep the Reference model loaded during training. (ref: [Calibrating Sequence likelihood Improves Conditional Language Generation](https://arxiv.org/abs/2210.00045))

- Results
  - The results vary slightly depending on whether you use externally pre-tagged preference data or data tagged from the SFT model, but the numbers didn't change much.
    
    ![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/b5f0944a-5e56-4d48-b93e-4385669970ef)

    ![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/7faeca62-c80c-4406-92a6-6629b58fcc5b)
    
  - The SLiC-HF training method showed a win rate of over 80% compared to the SFT model, and even when evaluated by human annotators it earned higher scores than RLHF.

This paper claims that the SLiC-HF method showed better performance than RLHF+PPO. However, since performance was only evaluated on the summarization task, it's hard to conclude from this paper alone that model alignment for general tasks like chatGPT is possible.

## DPO (Stanford)

The Direct Preference Optimization: Your Language Model is Secretly a Reward Model ([https://arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)) paper that came out of Stanford in 2023 is a method that reworks the RL objective function to make supervised learning possible.

- Direct Preference Optimization (DPO)
  
    This method starts from the fact that the RL objective with a KL-divergence regularizer attached has a closed form.

    $$
        \max_{\pi_\theta} \mathbb{E}_{x \sim D, y \sim \pi_\theta(y|x)}[r_\phi (x, y)] - \beta D_{KL}[\pi_\theta(y | x) || \pi_{\text{ref}}(y | x)]
    $$

    If we assume the RL objective above converges to a certain value, the closed form of the optimal policy is as follows.

    $$
        \pi_r(y | x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y | x) \exp\left(\frac{1}{\beta}r(x, y)\right)
    $$

    Here $Z(x)$ is the partition function, which you can think of as a scaling value that makes the probabilities sum to 1. Now let's rework this equation to rearrange it into an expression for the reward.

    $$
        r(x, y) = \beta \log \frac{\pi_r(y | x)}{\pi_{\text{ref}}(y | x)} + \beta \log Z(x)
    $$

    Finally, do you remember the Reward model loss we used in RLHF? If we substitute the equation above into the Reward model loss,

    $$
        \mathcal{L}_R(r_\phi, D) \newline
        = -\mathbb{E}_{(x,y_w,y_l) \sim D}[\log \sigma(r_\phi (x, y_w) - r_\phi (x, y_l))] \newline
        
        = -\mathbb{E}_{(x,y_w,y_l) \sim D}\left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right) \right] \newline

        = \mathcal{L}_{DPO}(\pi_\theta; \pi_{\text{ref}})
    $$

    Interestingly, simply minimizing the loss above makes it possible to train the policy without separately training a reward function. Since $y_w$ and $y_l$ are already tagged in the external human preference data, training proceeds just like supervised learning.

    In fact, training that maximizes the RL objective is only possible when you have a Reward Model, so you might feel it's a bit contradictory to plug the closed form obtained from the RL objective into the loss for training the Reward Model. So the paper's authors describe what meaning the new DPO loss function carries.

    If we compute the gradient of the DPO loss function, we get the following expression.
        
    ![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/f8bc3199-4821-4a56-ad03-1fbe1978a07e)

    Dissecting this, it carries three meanings.
    - It raises the likelihood of $y_w$.
    - It lowers the likelihood of $y_l$.
    - The more wrong the reward evaluation was, the larger the gradient, by multiplying with a larger weight.

    Analyzing it this way, you can see that the DPO loss, similar to the SLiC-HF method explained above, follows the concept of raising the likelihood of positive samples and lowering the likelihood of negative samples.

- Experimental results
  
    ![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/801e4b14-cb21-45f4-8d5f-7e76717cef9b)

    We can see that the DPO method earns higher rewards than PPO. It also shows that DPO is insensitive to sampling temperature: whereas PPO's win rate drops sharply under high sampling temperature, DPO does not.

    ![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/385e9b41-1d9e-4672-8c5f-ee58ff8e7f6b)

    Also, when humans evaluated the win rate against the ground truth data, DPO recorded a higher win rate than PPO. In other words, the claim is that **DPO is easier to train than PPO while delivering higher performance, and also has higher stability with respect to decoding parameters**.

Unlike SLiC-HF, DPO verified its performance not only on the summarization task but also on [Anthropic's Helpful-Harmful data](https://arxiv.org/abs/2204.05862). I'd guess it's probably the most widely used DAP method currently in use.

## IPO (Deepmind)

Deepmind argued that there's a fundamental problem with RLHF, and through the [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036) (Azar et al., 2023) paper, proposed a method called IPO (Identity Preference Optimization). However, this paper contains a great deal of very complex math. So in this post we'll briefly look at roughly what it covers.

This paper argues that the **Bradley-Terry Preference model**, which is RLHF's reward model, has a fundamental problem.

$$
    \mathcal{L}_R(r_\phi, D) = -\mathbb{E}_{(x,y_w,y_l) \sim D}[\log \sigma(r_\phi (x, y_w) - r_\phi (x, y_l))]
$$

If the model is fully trained through the loss above, it should satisfy $r(y_w)-r(y_l)\to+\infty$, so $\pi(y_w)=1$ and $\pi(y_l)=0$ must hold. Now, in the RL stage of RLHF, KL-divergence is used as the regularizer, and if data where $\pi(y_l)=0$ is used, then $D_{KL}\to\infty$.

Therefore, the Reward model must inevitably be underfit. This is content that also appears in the RLHF paper, where it's stated that if you train the Reward model too much, training stability drops.

But in the case of DPO, it uses the Bradley-Terry Preference model and there's no separate regularizer like the KL-divergence in the RL process, so the argument is that training stability can drop. In other words, there's no way to prevent the reward model from being trained excessively.

So IPO solves this problem by using a loss function that uses MSE instead of the existing reward model loss function that used sigmoid. The paper provides a lengthy proof of why it's okay to use this loss. If you're curious... I recommend reading the paper.

- IPO reward
    
    $$
    \mathcal{L}_{IPO}(\pi_\theta; \pi_{\text{ref}}) = \mathbb{E}_{(x,y_w,y_l) \sim D}\left[ \left( \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} -\frac{1}{2\beta}\right)^2 \right]
    $$
    
    In any case, if we analyze this loss too, fundamentally it can be seen as 1) raising the likelihood of positive samples, 2) lowering the likelihood of negative samples, and 3) keeping the difference from becoming too large—only large enough that the difference equals the sum of the difference of the reference samples' likelihoods and $\frac{1}{2\beta}$.
- Experimental results
    
    ![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/5e1edfba-f094-4653-884e-a12fa8406c59)
    
    The graph above is a training curve for preference data in the order y1 > y2 > y3.
    - DPO greedily overfits to generate only y1, but IPO leaves room to generate y2 and y3 depending on the case. In other words, it can avoid overfitting to a greedy policy.
    - With DPO, an output that never won has prob. 0, but with IPO even an output that never won doesn't have prob. 0. That is, the prob of y2 doesn't become 0.
  
    Why does this matter? In the real world, there are human annotation errors, so it happens quite often that high-quality data that should have won never wins. In such cases, DPO drives the probability of that response to 0, but IPO guarantees a certain amount of probability. In other words, overfitting due to annotation error can be appropriately resolved through the sampling approach.

This paper newly proposed a method that can solve DPO's overfitting problem. Interestingly, there's no separate evaluation showing the win rate went up or that it worked well on some task. So it's hard to know how good a method it is without doing additional experiments. But since it's Deepmind, it might be worth giving it the benefit of the doubt... (this is my personal opinion)

## RRHF (Alibaba)

Alibaba also released a high-performance LLM called Qwen in 2023. This model is one frequently found on the [Open-LLM leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard), which compares the performance of open-source LLMs. In particular, since it has strengths in Chinese compared to other Big Tech LLMs, it seems to be used quite a lot. Through the [RRHF: Rank Responses to Align Language Models with Human Feedback without tears](https://arxiv.org/abs/2304.05302) (Yuan et al., 2023) paper, Alibaba proposed a method that uses ranking loss.

![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/2aff87d7-de83-4d8c-911d-8973de2c3e28)

Like SLiC-HF, DPO, and IPO explained above, this is a method that can be trained without a separate value model, reward model, or reference model. However, it differs in that rather than pair data composed of win/loss, it ranks multiple responses and uses that. (In fact, OpenAI's RLHF also used ranking rather than pairs. But most open-source data is pairs, because they're easier to annotate.)

- RRHF loss
    
    $$
    p_i = \frac{ \log P_{\pi} (y_{i\geq t} | x, y_{i,<t})}{\| y_i \|}
    $$
    
    $$
    L_{\text{rank}} = \sum_{r_i < r_j} \max(0, p_i - p_j)
    $$

    These two expressions are the Ranking loss used in RRHF. Put simply, it makes the likelihood of a higher-ranked sample higher than the likelihood of a lower-ranked sample. If you just think of it as pair-structured data, it's identical to the hinge-loss used in SLiC-HF but without using a margin.
    
    $$
    i' = \underset{i}{\mathrm{argmax}} \ r_i
    \newline
    L_{ft} = -\sum_{t} \log P_{\pi} (y_{i',t} | x, y_{i',<t})
    $$
    
    It doesn't stop there—a fine-tuning loss was added. Among the collected data, the SFT loss is applied only to the highest-ranked data. It would be a waste to use the highest-ranked data only for preference optimization. (This is actually a type of method called Rejection-sampling, which I'll cover in Part 3.)

    $$
    L=L_{rank}+L_{ft}
    $$

    Combining the two losses, we get the final loss function.
        
- Experimental results
    
    ![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/a8f0af44-c455-4786-bd6c-360f62d5613c)

    For a fair comparison with PPO, they used PPO's reward model to rank the data and ran RRHF using that data. As a result, the win_rate increased ever so slightly compared to PPO.

    Also, a model that generated data using the RRHF-trained model, re-ranked it, and ran one more RRHF iteration ($\text{RRHF}_{\text{IP-2}}$) performed better than a model that ran RRHF only once ($\text{RRHF}_{\text{DP}}$).

    However, one thing to note is that if you **keep running iterations using only the trained model's responses, the Reward appears to increase, but the actual PPL and win rate can drop**. This is a phenomenon called **reward hacking**, where the model is trained to exploit loopholes in the annotation.
    > “*That sounds great!”, “I appreciate your help.”, “Thanks for your help!”*

    This is the phenomenon of generating only responses like the above—responses that aren't wrong so they can get a high ranking, but are completely useless. In other words, you'll have to find an appropriate number of iterations.

    ![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/bf7ea2f8-1a30-49e5-9755-b182adefaf95)
    
    They also ran an experiment comparing with Alpaca. Alpaca is a method that uses an approach called [self-instruct](https://arxiv.org/abs/2212.10560), a type of knowledge distillation method where chatGPT is made to generate responses and those responses are then SFT'd.

    In this paper, they compared Alpaca with Wombat—a model trained using RRHF after generating multiple responses with chatGPT and ranking those responses also using chatGPT. As a result, it showed better performance than Alpaca, but since it used chatGPT data, it fell short of chatGPT's performance.

Afterward, Tsinghua University also released a paper called [CLICK: Controllable Text Generation with Sequence Likelihood Contrastive Learning](https://arxiv.org/pdf/2306.03350.pdf), which adds a few techniques to RRHF. If you're going to use RRHF, I recommend also looking at the CLICK paper.

## SPIN (UCLA)

In 2024, UCLA released a paper called [Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://arxiv.org/pdf/2401.01335.pdf) (Chen et al., 2024). Self-Play is, in fact, the flower of RL. Alpha-zero conquered Go via the self-play approach. (Of course, Alpha-Go had already surpassed humans.) It's a method of obtaining data through AI-vs-AI matches and continuously training on that data. OpenAI has previously observed interesting phenomena through self-play in a game called Hide-and-seek. [OpenAI Hide-and-seek](https://www.youtube.com/watch?v=kopoLzvh5jY)

The concept is that, given SFT data, if you iteratively perform DPO with the SFT response as the win sample and the current model response as the lose, performance keeps improving. It has the advantage that no human preference tagging is needed at all—you only need the enormous amount of SFT data available open-source.

- Self-Play Fine-Tuning (SPIN)
    
    ![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/d01a7484-5c12-42ae-a7d4-a8597e595533)

    The concept itself is absurdly simple and a little strange. It assumes the SFT data is perfect and tags the current model's output as unconditionally losing. So then, what difference is there from just SFT-ing that data?
    
- Experimental results

    Among the models released on Huggingface, there's a model called `zephyr-7b-sft-f`, which is Mistral-7B SFT'd on the Ultrachat200k data.
    
    In this paper, they sampled 50K data from the Ultrachat200k data and performed SPIN. The result is a bit surprising.

    ![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/19054d87-b9cc-4c5a-a300-ebd04a32afae)

    Whereas continuing to SFT on the Ultrachat200K data doesn't increase performance, using SPIN does increase performance. The fact that performance increases even with the same data suggests the importance of the model not only learning the positive data of SFT, but also looking at the negative data and learning the difference from the positive data.

    ![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/657c241f-766d-42e1-b73e-67d87d97bb5b)

    They also claim it showed better performance even when compared with the DPO model, which requires preference tagging.

# Conclusion

We've looked at a total of five DAP methods. In fact, the concept of these methods is identical: **raise the likelihood of data that matched human preferences, and lower the likelihood of data that didn't**. The difference is that, since RL is unstable to train, you had no choice but to use PPO, which has the highest training stability among them; whereas with DAP, because it uses a supervised learning loss, you could try out various loss functions, which I think led to the appearance of diverse methods and papers.

# References

- [SLiC-HF: Sequence Likelihood Calibration with Human Feedback](https://arxiv.org/abs/2305.10425) (Zhao et al., 2023)
- [Calibrating Sequence likelihood Improves Conditional Language Generation](https://arxiv.org/abs/2210.00045)
- [https://arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)
- [Anthropic's Helpful-Harmful data](https://arxiv.org/abs/2204.05862)
- [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036)
- [Open-LLM leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [RRHF: Rank Responses to Align Language Models with Human Feedback without tears](https://arxiv.org/abs/2304.05302)
- [self-instruct](https://arxiv.org/abs/2212.10560)
- [CLICK: Controllable Text Generation with Sequence Likelihood Contrastive Learning](https://arxiv.org/pdf/2306.03350.pdf)
- [Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://arxiv.org/pdf/2401.01335.pdf)
