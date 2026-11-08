---
layout: post
title: "Diary - LLM Orchestration (Sakana Fugu)"
tags: archive
lang: en
---

Around the time Mythos and Fable were banned, I started seeing a lot of news about models with fairly strong performance. They probably saw an opportunity. Among them, Sakana AI's Fugu caught my eye. I had heard that Sakana was doing good work in AI for Science, but then it suddenly announced a frontier AI model, which made me curious. After looking into it, I found that Fugu was less a fully proprietary LLM and more a model focused on maximizing end-to-end performance by training its own router to combine responses from frontier LLMs effectively.

[🐡 Sakana Fugu](https://sakana.ai/fugu/) / [📄 Technical Paper](https://arxiv.org/pdf/2606.21228)

## Let's take a look at Sakana Fugu

First, "fugu" means pufferfish. (I have no idea why they chose a pufferfish. Because it tastes good? Because it is tough?) To the user, it looks like they are calling a single Sakana Fugu model. Internally, however, several frontier LLMs, such as GPT-5.5, Claude Opus 4.8, and Gemini 3.1 Pro, collaborate as a team, while a single orchestrator coordinates them. This orchestrator is Fugu's core module. It does not solve the problem itself. Instead, it decides "who should handle this task" and "in what order should they work on it." Naturally, **the large LLMs used as workers are not trained at all.**

Fugu comes in two versions:

- **Fugu**: The everyday version. It balances latency and performance by selecting only one worker for each input.
- **Fugu-Ultra**: The version for difficult problems. It prioritizes answer quality and constructs workflows involving multiple agents.

Fugu-Ultra outperforms frontier models on benchmarks such as SWE-Bench Pro, LiveCodeBench, and GPQA-Diamond. In a sense, it is not that different from a harness. Once you formalize a better way for frontier models to work together, it is only natural for benchmark performance to rise. The subtle difference, however, is that the workflow is not created through a human abstraction process, as a harness would be. It is learned to maximize a specific objective. It is not merely saying, "This works well!" It is saying, "We can mathematically guarantee that this works better!" (Those are decidedly different claims.)

In any case, these models did not appear out of nowhere. Fugu grew out of a paper called Trinity, while Fugu-Ultra came from another paper called Conductor. Let me unpack them one by one at a conceptual level.

## Sakana Trinity

[Sakana Trinity](https://arxiv.org/html/2512.04695v3) is the paper that laid the foundation for Sakana Fugu. Its orchestrator uses a small Qwen3-0.6B backbone to classify which LLM to call and which role to assign to that LLM.

![alt text](/imgs/posts/2026-07-01-llm-orchestration-router-0.png)

![alt text](/imgs/posts/2026-07-01-llm-orchestration-router-1.png)

The available roles are Thinker (T), Worker (W), and Verifier (V). The LLM pool consists of GPT-5, Gemini 2.5 Pro, Claude 4 Sonnet, Gemma 3 27B IT, DeepSeek-R1-Distill-Qwen-32B, and Qwen3-32B.

The Thinker devises a strategy, the Worker writes the actual code or performs the calculation, and the Verifier checks the result and decides whether it passes. At each turn, the orchestrator assigns one of these three roles, for up to five turns. The process ends when the Verifier approves the result.

It is an incredibly simple method, but at first glance, it also feels like something that would either overfit or fail to learn altogether. Because it does not predict a token sequence, a full episode produces only five trainable classification signals at most. In situations like this, it can be easier to use a more fundamental ML algorithm rather than a complex one. Instead of reinforcement learning (RL), Trinity uses an **evolution strategy (sep-CMA-ES)**. The paper experimentally shows that RL fails to learn properly because the signal is too sparse. In contrast, Trinity trained through an evolution strategy does not need gradients. It only asks, "Was this parameter combination better than that one?" and moves according to the ranking, which makes it more robust in this setting.

Sakana Fugu directly adopts Trinity's model architecture. It therefore solves problems through a simple sequential workflow, and because router latency is also low, it is well suited to tasks that require a quick response.

## Sakana Conductor

[Sakana Conductor](https://arxiv.org/abs/2512.04388) can be understood as an orchestrator that **creates workflows by generating text**. Unlike Trinity, which selects only one model and one role at each step, Conductor can invoke multiple models and roles simultaneously.

![alt text](/imgs/posts/2026-07-01-llm-orchestration-router-2.png)

Each step in a workflow consists of three elements:

- **Model ID**: Which model should handle this subtask
- **Subtask**: What specific instruction should be given to that model
- **Access list**: Which outputs from previous steps should be included in this worker's context

Thanks to the access list, Conductor can build a wide variety of collaboration topologies, from simple best-of-N arrangements to sequential chains and tree structures. In an example from the paper, Gemini and GPT independently attempt the problem at the leaf nodes, and Gemini at the root synthesizes both responses into a complete answer. There are also workflows that bring in Opus only for particularly difficult debugging points. In the figure above, for example, agent 2 is asked to design an algorithm, while agent 0 receives the previous response as context and implements it in Python.

Conductor is trained with RL, specifically GRPO in the paper. The sparse reward problem still exists here. It receives a binary reward based on whether the final result is correct, along with an intermediate format reward based on whether the workflow steps can be parsed. Since this is still an ORM (Outcome-Based Reward Model), the training itself is inevitably inefficient. Even so, the reason Sakana did not choose an evolution strategy as it did for Trinity is that Conductor uses a 7B model, Qwen 2.5 7B. The model is simply too large to train with an evolution strategy. More importantly, this is not a classification task but the learning of a token-sequence distribution. Since it also fine-tunes an already instruction-tuned model, applying RL was sufficient.

The frontier models were the same ones used in Trinity: GPT-5, Gemini 2.5 Pro, Claude 4 Sonnet, Gemma 3 27B IT, DeepSeek-R1-Distill-Qwen-32B, and Qwen3-32B. RL training used only around 960 samples and ran for roughly 200 steps. To reduce cost, they reportedly limited the frontier models' output tokens very aggressively. (It is surprising that performance was still so strong.)

Sakana Fugu-Ultra is based on Sakana Conductor. Conceptually, Conductor actually subsumes Trinity. Because it can construct graph-shaped workflows, a sequential workflow is simply a subset of what it can represent. In any case, both generating the orchestrator response and executing the more complex workflow take additional time. It is therefore best suited to tasks where performance matters more than latency.

## Sakana Fugu and my own thoughts

In short, the Sakana Fugu models train an orchestrator to route among existing frontier models effectively and achieve SOTA performance in a **mathematically optimal** way. At a time when access to SOTA models could suddenly become restricted, this kind of ensemble method may eventually offer a robust way to preserve performance. However...

### A trained router is vulnerable to model changes

Both Trinity and Conductor are trained around a fixed set of LLMs. You can see this from the fact that workers are identified by **integer indices**. GPT and Claude receive new versions every few months; models disappear, and new ones are released. Even a version upgrade can therefore shift the system away from the distribution it was trained on. Moreover, the papers do not provide an ablation study showing whether Trinity or Conductor would continue to work properly if a particular model were masked out of the router's LLM pool.

### Is harnessing the answer, then?

In practice, trained routers are not the industry mainstream. Anthropic's "[Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)" presents routing as a workflow pattern implemented through LLM prompting. The [handoff](https://openai.github.io/openai-agents-python/handoffs/) pattern in the OpenAI Agents SDK and the subagent delegation used by [OMC](https://github.com/yeachan-heo/oh-my-claudecode) both amount to the same thing: "Give an LLM instructions and let it decide." There is no training involved. [Devin Fusion](https://cognition.com/blog/devin-fusion) takes the same approach, creating a sidekick architecture through harness design and prompt engineering without fine-tuning or end-to-end training.

The advantage of this approach is clear. Because the available LLMs are **identified through natural-language descriptions**, when a model changes, you only need to replace its description. No retraining is required. The vulnerability to model changes that I mentioned earlier naturally disappears. It is also possible to construct workflows with roles other than Thinker, Worker, and Verifier. Overall, this approach offers much more freedom in workflow design. But **there is no guarantee that a system constructed this way is actually good**. It merely demonstrates something practical that happens to work.

So I suspect the realistic path forward looks something like this: **Start by getting the system running quickly through harnessing and prompt engineering. Meanwhile, collect data about which routing choices worked well for which inputs. Once enough data has accumulated, train the router.** Compared with trying to train a router from scratch using sparse rewards, generating labels with a harness and then applying supervised learning would be much faster and more stable. After all, even Sakana warms up the base Fugu model with SFT before fine-tuning it with ES or RL.

## Closing thoughts

I have been thinking about this for a long time, but harnessing and prompt engineering have a fundamental weakness: **there is absolutely no guarantee that the result is optimal.** In the end, it is difficult to explain the process more rigorously than "Oh, this works" or "This should probably work." The Devin Fusion blog itself uses phrases such as "many rounds of tuning," "artful engineering," and "with the right prompting." Ultimately, they kept shaping the system by hand until they reached a point where they could say, "This is good enough." That stands in contrast to learning-based methods, which at least have a direction defined by maximizing an objective function. A harness has no such direction; it simply stops at a point that works well.

This is where I ultimately landed. Training a router from scratch makes it vulnerable to model changes, while the reward is far too sparse. It is therefore more practical to begin with harnessing, collect data, and then move to SFT. Yet I cannot stop thinking about the fact that there is no guarantee that the initial harness or prompt is itself optimal. What I really want, then, is **a way to find an optimal harness or prompt.** Today, people shape these systems by intuition. If we found a principled way to search for or guarantee better designs, much of the dilemma above could be resolved. We could use a harness to discover optimal routing, then train a router on those labels, producing a system that is both resilient to model changes and at least somewhat grounded in optimality.
