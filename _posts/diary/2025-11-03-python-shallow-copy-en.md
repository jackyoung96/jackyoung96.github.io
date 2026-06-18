---
layout: post
title: "Diary - A story about Python shallow-copy: The problem with top-down learning"
tags: archive
lang: en
---

I work as an AI engineer, but I came from a mechanical engineering department. In my first year, while learning the C++ language in an intro-to-computers class, I wondered why on earth we were doing this. (I should have started coding back then...)

After coming back from the military, I took an AI class and came to think it looked really fun, and after passing through autonomous driving I've now entered the LLM field, but in all that time I've never taken a lecture on Python. I studied everything top-down, building projects and looking things up whenever there was something I didn't know. So it's not that I can't code at all, but sometimes I feel that my fundamentals are lacking.

Then a while ago, a problem occurred. The code I wrote was something like this:

```python
num_chunk = ...
world_size = ... 
dataset = ... # [data_1, data_2, ... data_n_c]

gathered_data = [[None] * world_size] * num_chunk
for i_c in range(num_chunk):
    torch.distributed.all_gather(gathered_data[i_c], dataset[i_c])
```

Roughly something like this. It's a situation where each process loaded a different dataset, and I was trying to all-gather the whole thing, but doing it all at once was too much data and caused a GPU-OOM, so I was trying to do the all-gather by chunking it. [There's a post that dealt with a similar implementation.](_posts/diary/2025-03-31-all-gather-object.md)

People with solid fundamentals might spot the problem right away. First of all, this code doesn't throw an error. The problem is that **the multiplication operation on a List is a shallow copy**.

What is a shallow copy? It's not copying the value but copying the memory address. [Detailed explanation](https://wikidocs.net/16038) It's the concept of a pointer.

So the problem right now is that, because I shallow-copied the list `[None, None, ..., None]` `num_chunk` times, the all-gather operation for each chunk kept overwriting. In the end, it ended up holding `num_chunk` copies of only the very last chunk.

How did I happen to find this out? After implementing this chunked loading, overfitting kept occurring somehow. Naturally, since I was using the same data N times over, overfitting was bound to happen.

The solution is to do:

```python
gathered_data = [[None] * world_size for _ in range(num_chunk)] 
```

This is fine because it's not the concept of copy but creating `num_chunk` instances. Or you could use deepcopy.

In any case, it became an occasion to feel my lack of fundamentals once again.
**Even complex AI model training can be ruined by confusion over a very simple concept like shallow/deep copy.** Let me keep this in mind.
