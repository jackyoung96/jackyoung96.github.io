---
layout: post
title: "Diary - DeepSeek-V3.1, GPT-5, and think-fusion"
tags: archive
lang: en
---


If I had to pick one of the most important keywords in the LLM field in 2025, it would be 'Reasoning'. Through test-time-computing, LLMs arrived at answers by going through a thinking process like humans, and this dramatically boosted performance on various benchmarks. However, this Reasoning model didn't have much impact on ordinary users (those who want to apply LLMs to simple tasks). That's because, even for questions that can be answered right away, it would go through a pointless thinking process and wear out users waiting for the answer.

So one of the proposed approaches is Think-fusion. In a previous blog post, I looked at the [Hybrid think mode applied to Qwen3](https://jackyoung96.github.io/2025/05/01/Qwen3-hybrid-think/). In an LLM with Think-fusion applied, a single unified model supports both reasoning and non-reasoning modes. In this post, I want to look at the various ways to implement think-fusion, and examine the direction of LLMs through recent models' think-fusion approaches.

### Various Ways to Implement Think Fusion

> 📌 **What you need to know**  
> An LLM is fundamentally trained to wrap a turn indicator around the front and back of the user's query, like `<|im_start|>user\n[query]<|im_end|><|im_start|>assistant\n`, and to generate the response after the `assistant` indicator.

Think fusion is basically trained so that when the user activates 'Think mode', the model generates an answer in the form `<think>[reasoning]</think>[response]`. However, there are three differences in implementation depending on the model and on the response form in 'Non-think mode'.

| Case | Non-think mode | Models used |
| ---- | ------------- | ------- |
| 1   | `[response]`  | Llama-Nemotron |
| 2   | `<think></think>[response]` | Qwen3, EXAONE-4.0 |
| 3   | `</think>[response]` | DeepSeek-V3.1 |

Let's look at each approach in a bit more detail.

**Case 1 (Llama-Nemotron approach)**  
This is the most intuitive approach. When the user turns off Think mode, the model omits the reasoning process and produces only the final answer ([resp]). To use Think mode, it prepends `<think>` after the `assistant` indicator before generating the response.

**Case 2 (Qwen3, EXAONE-4.0 approach)**  
When the user turns off Think mode, it outputs the <think></think> tags and then provides the answer. This is identical to generating empty reasoning. This is an attempt to maintain consistency in the response form by structurally keeping the template such that the model always goes through a 'thinking' step, but leaving its content empty in Non-think mode. To use Think mode, it prepends `<think>` after the `assistant` indicator before generating the response, and to use Non-think mode, it prepends `<think></think>` after the `assistant` indicator before generating the response.

**Case 3 (DeepSeek-V3.1 approach)**  
When the user turns off Think mode, the `</think>` token appears immediately after `assistant`, and the answer is generated after that—a somewhat peculiar form. To use Think mode, it prepends `<think>` after the `assistant` indicator before generating the response, and to use Non-think mode, it prepends `</think>` after the `assistant` indicator before generating the response.

The three approaches may seem hardly different, but considering the nature of LLMs that generate the next token probabilistically, there's a big difference.

### Differences According to the Think-fusion Implementation Approach

An LLM is a model trained by next token prediction loss. Therefore, if it's a sequence the model has seen during training, there's a possibility that token will be generated.

In the case of Case 1, let's assume a situation where, to use Non-think mode, you perform inference appending nothing after `assistant`. Because data with a `<think>` token after `assistant` for Think mode has been trained on, the `<think>` token can be generated even in Non-think mode. If it is generated, the model answers in think mode.

On the other hand, in the case of Case 2, you did inference appending only `<think>` to use Think mode, but `</think>` could come out right away, turning it into Non-think mode.

In the case of Case 3, unlike the previous cases, the two modes are completely separated. If it starts with `<think>`, it will reason until `</think>` appears, and if it starts with `</think>`, the response will be generated right away.

Therefore, in the case of Case 1 and Case 2, in Non-think mode and Think mode respectively, the model—not the user—chooses the mode. The proportion will be determined by the ratio of trained data or the training recipe.

| Case | When using Think mode | When using Non-think mode |
| ---- | ------------- | ------- |
| 1   | Think  | The model decides Think/Non-think |
| 2   | The model decides Think/Non-think | Non-think |
| 3   | Think | Non-think |

So then, is Case 3 necessarily the good one? I can't say for sure. Case 1 and Case 2 maintain consistency in the templates used in Think/Non-think. On the other hand, Case 3, once a specific token appears, takes on an entirely different distribution afterward. This may lower sample efficiency during model training. In other words, Case 1 and Case 2 might be more efficient to train. On the other hand, Case 1 and Case 2 need to find a good data mixture so that the model can decide Think/Non-think well.


### Splitting the Models Again Without Using Think-fusion
However, the recently released [GPT-5](https://openai.com/index/gpt-5-system-card/) instead completely separated the Reasoning model (gpt-5-thinking) and the Instruct model (gpt-5-main). And it attached a **Router** in front of the models. The router first analyzes the user's question and judges whether the question requires complex reasoning or whether simple instruction-following or information retrieval is sufficient. Then, based on that judgment, it forwards the request to each specialized model.

Alibaba also released an Instruct-only model with the [2507 version of Qwen3-235BA22B](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507). Instead of upgrading the existing think-fusion, they changed it to an instruct-only model and released it.

### The downsides of Think-fusion?
GPT-5 and Qwen3 are training Think/Non-think models separately, and DeepSeek-V3.1, which uses the Case 3 approach of think-fusion, can also be seen to have attempted to separate the distribution of Think/Non-think. What's the reason relatively recent models are trying to separate think-fusion?

We can make a few conjectures.

1. **Possibility of performance degradation**: Training a single model to do well at both complex reasoning tasks and concise answer generation can bring about performance degradation in one of the two tasks. Indeed, with models that apply Think-fusion, the Non-think mode performance doesn't feel particularly good. In fact, Reasoning involves a lot of training on the attention-side parameters, whereas general Instruct data involves a lot of updates to the MLP-side parameters. I think this training imbalance can bring about performance degradation.

2. **Complexity of training**: As mentioned above, you have to find the appropriate ratio of Think and Non-think data, and this work is expected to be very laborious. If you fail at adjusting the ratio, situations can arise where, in cases the user didn't intend, excessive reasoning degrades service speed, or where even in cases that require reasoning, it only gives a general answer and performance drops.

3. **Reduced efficiency**: In fact, most tasks don't even require reasoning ability. Even with an Instruct model, you can obtain good answers if you make reasonable use of CoT. Also, to do batch processing in vllm and the like, the response lengths need to be similar, but think-fusion models have wildly varying response lengths, so efficiency drops. It can be more efficient to just separate the models and serve them separately.


### In closing 
Think-fusion, which was popular in the first half of 2025, looked like a way to catch two rabbits—performance and usability—at once with a single model. However, as we can see in the recent trend of DeepSeek-V3.1 and GPT-5, we could observe a movement that seems to once again separate the distributions of Think/Non-think. What approach will the SOTA models that come out from now on end up using?
