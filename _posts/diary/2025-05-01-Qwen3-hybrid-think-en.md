---
layout: post
title: Diary - Qwen3's Hybrid Thinking Mode
tags: archive
lang: en
---


In the early hours of April 29, 2025, Alibaba's Qwen3 was released.
[Qwen3 official blog](https://qwenlm.github.io/blog/qwen3/)

Since no technical report was published and it was only revealed through a blog post, how the training was done was disclosed only very partially.

The lineup consists of a total of models — 2 MoE models and 6 Dense models.
The MoE models are 235B (22B activated) and 30B (3B activated), and the Dense models come in sizes of 32B, 14B, 8B, 4B, 1.7B, and 0.6B.
According to the blog post, it appears that they trained only the largest model in each case and then trained the remaining models via strong-to-weak distillation. The base models were probably reduced in size through pruning techniques. (My guess is they cut it down roughly by half each step.)

Honestly, since no technical report was published, it's still hard to pin down the training method with certainty. However, among what was disclosed there was one interesting part: the so-called **Hybrid Thinking mode.**

## Hybrid Thinking mode

According to the Qwen3 official blog, Qwen3 was trained to use test-time reasoning when solving hard problems that require thinking — that is, reasoning — or tasks that demand accuracy, and to simply provide an answer with thinking turned off when a fast response is needed. This is the approach first offered in Claude-sonnet-3.7, and as far as I know, this is the first time it has been explicitly made possible in open source.

So how did they implement the on/off? Qwen3 controls the on/off in two ways.

### 1. `/think`, `/no_think` commands

You can control thinking by explicitly using the `/think` or `/no_think` commands in a user or system turn. Let's look at an example.

First, when using the `/no_think` option:
> How many r's in blueberries? /no_think

The answer to this is
> Let's count the number of r's in the word "blueberries" step by step:
>
> 1. Write out the word: b l u e b e r r i e s
> 2. Go through each letter one by one:
> - b – no
> - l – no
> - u – no
> - e – no
> - b – no
> - e – no
> - r – yes (1st r)
> - r – yes (2nd r)
> - i – no
> - e – no
> - s – no
>
> So, there are 2 r's in blueberries.
>
> ✅ Answer: 2.

And when using the `/think` option:
> How many r's in blueberries? /think

The answer to this is
> \<think\>
> 
> Okay, so the user is asking how many times the letter "r" appears in the word "blueberries". Let me start by writing down the word to visualize it better. The word is "blueberries".
>
> First, I need to break down the word into individual letters. Let's spell it out: B-L-U-E-B-E-R-R-I-E-S. Wait, is that correct? Let me double-check. B-L-U-E for "blue", then "berries". So "berries" is B-E-R-R-I-E-S. So combining them, "blueberries" is B-L-U-E-B-E-R-R-I-E-S. Let me count each letter one by one.
>
> Starting with the first letter: B (1). Then L (2), U (3), E (4), B (5), E (6), R (7), R (8), I (9), E (10), S (11). Wait, that's 11 letters total. Now, I need to check how many "r"s are there. Let's go through each position:

> 1. B - no
> 2. L - no
> 3. U - no
> 4. E - no
> 5. B - no
> 6. E - no
> 7. R - yes, that's one
> 8. R - yes, that's two
> 9. I - no
> 10. E - no
> 11. S - no
>
> So according to this count, there are two "r"s in "blueberries". But wait, let me make sure I didn't miss any. The word is spelled B-L-U-E-B-E-R-R-I-E-S. The "berries" part has two "r"s in a row, right? So between the two "e"s, there are two "r"s. Let me confirm the spelling again. Yes, "blueberries" is spelled with two "r"s in the middle.
>
> Is there a possibility that I miscounted? Let's see. The letters after the first "blue" are "berries". "Berries" is spelled B-E-R-R-I-E-S. So the two "r"s are consecutive. Therefore, in the entire word "blueberries", there are exactly two instances of the letter "r".
>
> I think that's correct. But just to be thorough, maybe I should write down each letter with their positions:
>
> 1. B
> 2. L
> 3. U
> 4. E
> 5. B
> 6. E
> 7. R
> 8. R
> 9. I
> 10. E
> 11. S
>
> Yes, letters 7 and 8 are both "R", so two "r"s. Therefore, the answer should be 2.
>
> ✅ Answer: 2.
> 
> \</think\>
>
> The word blueberries contains 2 instances of the letter r.
>
> Here's the breakdown:
>
> B L U E B E R R I E S
>
> The letter r appears at the 7th and 8th positions.
>
> Answer: There are 2 r's in "blueberries".

You can see that it reasons using the `\<think\>` and `\</think\>` tokens and then gives a final answer.

This was probably implemented by training with `/no_think` on data without thinking and `/think` on data with thinking.

In other words, **you can view it as a method of controlling reasoning on/off by using switch tokens during training.**

### 2. enable_thinking (apply_chat_template)

Another method is to use a parameter called `enable_thinking`. The Qwen3 blog shows the following approach.

```python
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switch between thinking and non-thinking modes. Default is True.
)
```

So what exactly changes when you give the `enable_thinking=True` option? If you look at `tokenizer_config.json`, you'll find Qwen3's chat_template. There you can see the part that changes via this option.

```
if enable_thinking is defined and enable_thinking is false
    '<think>\\n\\n</think>\\n\\n'
```

In other words, when enable_thinking is false, it forcibly appends an empty \<think\>\</think\> reasoning part with no reasoning right after the assistant indicator, so that it generates an answer without reasoning. This takes advantage of the fact that reasoning starts right after the assistant indicator, and personally I think it's a very smart method.

## Conclusion

Qwen3 implemented Hybrid thinking mode in two ways. The first method, the `/think` and `/no_think` commands, is a way for the user to explicitly control it when using the API or a chat service, but since it's implemented through training, there is a possibility that reasoning happens even when you use `/no_think` (the probability is low, but the possibility exists nonetheless). The second method, `enable_thinking`, on the other hand, forcibly appends an empty thinking part to strongly turn reasoning off (of course, even this has a probabilistic chance of producing two thinking sections).

In any case, both methods implemented hybrid think mode in a way that is very simple yet smart. I should refer to these and try using them in my future model training.
