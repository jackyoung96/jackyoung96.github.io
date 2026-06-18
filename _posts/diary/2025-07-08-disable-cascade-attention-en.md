---
layout: post
title: Diary - vLLM's cascade attention
tags: archive
lang: en
---

Yesterday, I tried running a benchmark to test out a certain company's LLM. But strangely, a phenomenon occurred where the answers had tons of repetition and were all broken.
The numbers clearly came out fine when I just spun up vLLM normally...?? So I tried feeding in the exact prompts used in the benchmark.

**One of the benchmarks, KO-IFEval**
> Q: Write a riddle about the Dragon Palace without using any commas.

> A (benchmark): There is a riddle in the Dragon Palace. Try to solve the riddle about the Dragon Palace. Try to solve the riddle about the Dragon Palace. Write the riddle about the Dragon Palace without commas. Write the answer without using commas. Write the riddle about the Dragon Palace with commas. Write the answer without using commas. Try to solve the riddle about the Dragon Palace. Write the riddle about the Dragon Palace without using commas.

> A (vLLM): What does the Dragon Palace, hidden deep in the sea, look like? They say there is a palace there where golden fish dance and pearls fall from the sky. But this place, which not just anyone can enter, is a mysterious spot that only those who know the secrets of the sea can visit. What on earth could be the key that opens the gate of the Dragon Palace?

There were not just one or two cases like this. Wondering if it was a difference in sampling parameters, I checked all sorts of things, but they were all identical. The only thing that was different was **tensor-parallel-size**.

Tensor-parallel-size is an option for splitting models that can't fit on a single GPU across multiple GPUs. It's an option that **should absolutely never affect performance**. Thinking TP itself probably wasn't the problem, I searched around for issues and found a bug where repetition occurred in responses as the batch size grew.

https://github.com/vllm-project/vllm/issues/17652

The funny thing is, they say repetition occurs at batch size 50, but batch size 100 is fine again lol

Anyway, the conclusion this issue reached is that you have to set `disable_cascade_attn=True`.

### Cascade attention

The core of cascade attention is that, when multiple queries come in, it separates the KV-Cache of the overlapping part (the shared query prefix) from the KV-Cache of the rest.

[Source](https://flashinfer.ai/2024/02/02/cascade-inference.html)

There are two benefits gained by doing this. First, naturally, since the shared KV-Cache is used N times less, a memory benefit arises. Second, by placing the now-smaller shared KV-Cache in SMEM/Register rather than the L2 Cache, access speed can be made much faster.

It can be seen as an algorithm that reduces redundant memory and improves inference speed through divide-and-conquer. Naturally, it should not affect the results (in theory).

### Why does this happen?

The funny thing is that this bug does not occur on the H100, only on the A100 GPU.
Tensor-parallel is, in the end, also doing divide-and-conquer, and so is cascade attention.
It seems like something in the computation gets tangled when trying to do the two at the same time... It can be considered a very serious bug.
And the reason I first discovered this problem is that the model I benchmarked this time is one with a fairly long system message baked in. (That means the shared prompt prefix is long! Cascade attention is definitely activated.)

In fact, looking at the [official docs for the `disable_cascade_attn` option](https://docs.vllm.ai/en/stable/api/vllm/config.html#vllm.config.ModelConfig.disable_cascade_attn):
> Disable cascade attention for V1. While cascade attention does not change the mathematical correctness, disabling it could be useful for preventing potential numerical issues. Note that even if this is set to False, cascade attention will be only used when the heuristic tells that it's beneficial.

That's what it says. ~~Wait, if that's the case, shouldn't the default obviously be set to True??~~

For now, at least as of two weeks ago, this issue had not been resolved.
I'm thinking of cracking open the internals sometime. Anyway, until it's resolved, I'll go with disable_cascade_attn=True (since I don't have an H100...).