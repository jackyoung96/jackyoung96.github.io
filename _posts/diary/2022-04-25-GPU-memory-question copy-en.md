---
layout: post
title: Diary - Curious about GPU memory
tags: archive
lang: en
---

While training DRL, I discovered a curious phenomenon. There are a total of 3 servers in the lab, and I mainly use server 1. This time, when I got to use the third server, I moved the code I'd been running on server 1 over to server 3 as-is, and a curious thing happened on the GPU.

**The size of the memory loaded onto the GPU is different.**  
<figure>
<img width="556" alt="image" src="https://user-images.githubusercontent.com/57203764/165021666-5300eace-7da5-4d64-b7bc-702841119170.png?style=centerme">
<figcaption>GPU memory status of server 1</figcaption>
</figure>
<figure>
<img width="560" alt="image" src="https://user-images.githubusercontent.com/57203764/165021435-197a295c-5c1c-4443-8820-f49cf58fad92.png?style=centerme">
<figcaption>GPU memory status of server 3</figcaption>
</figure>

First of all, even though I loaded the exact same network, it eats up nearly twice as much memory on server 3.
I believe server 1 is a Titan X, and server 3 is the latest 3090 ti.
My guess is that since the 3090 has well over 20GB, maybe it sets up something like swap memory separately to boost the computation speed. If anyone knows, please let me know.

Second, memory suddenly got allocated on GPU 0. I have no idea why. I had applied the device to everything that uses pytorch, and this never happened on server 1, so I don't understand how this is possible. Or maybe the GYM environment is using the GPU. If anyone knows this one too, please let me know.
