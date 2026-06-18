---
layout: post
title: "Diary - EFA and InfiniBand, the hidden infrastructure story beneath LLMs"
tags: archive
lang: en
---

When training LLMs, I've built up a lot of experience on the architecture design and training code side, but the truth is there's infrastructure beneath all of that, including the GPU cluster.
While I was at SKT, there were experts in charge of the infrastructure, so I didn't look closely at that part.
But AWS is a cloud company, and my position seems to require me to handle the full pipeline as a full-stack engineer.
So I figure I'll take this opportunity to study infrastructure a bit.

![image.png](https://github.com/user-attachments/assets/e8a42e0f-3102-4799-8e0d-997a54ecb257)

And in fact, LLM architectures and training code should really be designed in an infra-wise manner that takes the configuration and conditions of the infrastructure into account.
In my heart I thought I should study this ahead of time, but it always seemed like I only studied a little bit and scrambled to fix things after the training code's speed wasn't coming out or errors occurred.
And the key point is, **no matter what model it is, models are going to keep getting bigger going forward, but it'll be hard for VRAM capacity to grow explosively**, so I think I need to build up my infrastructure-related knowledge base more.
That way, when a new architecture comes out, having these fundamentals well in hand will let me apply it faster, and won't that make me a more outstanding and rare AI engineer?

The first topics I'll study are InfiniBand, the core of large-scale GPU clusters, and EFA, which AWS uses instead of InfiniBand.
In the process of looking into them, there was truly a mountain of things I didn't know.
There were a lot of slightly embarrassing "you didn't even know this?" things, but anyway, this top-down approach sticks in my memory better, so…

### So why am I curious about InfiniBand and EFA in the first place?

There's a library called DeepEP that DeepSeek used to accelerate its MoE model.
But when I tried to use it on an AWS cluster, a problem came up.
That's because [the DeepEP library assumes the use of InfiniBand](https://github.com/deepseek-ai/DeepEP/issues/369)!
Surprisingly, even though AWS was a large cluster of over 100 nodes, it wasn't using InfiniBand and was connecting everything with Ethernet.
Up until then, I had thought InfiniBand was a must-have for building a GPU cluster, but AWS was using its own technology, EFA, to reduce inter-node network bottlenecks in a way similar to InfiniBand.
With the feeling of "I just joined AWS, how can I not even know this?", I started studying!

### Inter-node network

When building GPU servers, due to power issues you can only fit 8 GPUs in a single rack.
A rack unit is also commonly called a node, and communication within a node, i.e. the intra-node network, is all connected so it's extremely fast.
However, communication between nodes, i.e. the inter-node network, is inevitably comparatively slow.
The reasons are 1) delays caused by data copying and 2) the bandwidth is comparatively small, so the amount of data that can move at once is small.

I referred a lot to the Toss tech blog ([Introducing our high-performance GPU cluster #2: migrating data](https://toss.tech/article/30767)), which explains data copying and bandwidth incredibly well.
First, during AI training almost all computation happens on the GPU.
That is, reading and writing the data in the GPU's VRAM is almost the entirety of the computation, and the CPU only needs to run code and handle data processing.
But when moving data from node 1's GPU VRAM to node 2's GPU VRAM, if you just move it, you have to go through the cumbersome process of copying that memory to the CPU, having the CPUs communicate with each other, and then copying back to the GPU.
This is the delay caused by data copying.

![image.png](https://github.com/user-attachments/assets/04433c2b-3166-405e-978c-c709fdd64dd2)
> Original GPU-to-GPU communication

![image.png](https://github.com/user-attachments/assets/2142681b-75b2-47f6-b572-33b4f050b589)
> OS-bypass GPU-to-GPU communication

Bandwidth is a somewhat simpler issue. For the intra-node network, the chips are all attached to the motherboard so it's fine, but inter-node, since they're physically separated, you have to connect them with cables like fiber optics.
So there can be signal distortion and packet loss, and going through processes to handle these slows things down a bit.

Therefore, the key to reducing inter-node GPU communication bottlenecks is 1) eliminating unnecessary GPU-to-CPU data copying and 2) using a network with large bandwidth.

### InfiniBand

Surprisingly, InfiniBand has a fairly long history.
The first InfiniBand architecture was released in 2000. It was proposed by the IBTA, which anticipated that a low-latency yet high-bandwidth network would be needed, and its spec was decided by a steering committee composed of **HP, IBM, Intel**, and others.
A representative manufacturer is Mellanox, which was acquired by Nvidia in 2019 at a valuation of $6.9 billion.

One important point about InfiniBand is that it's a network dedicated to RDMA (Remote Direct Memory Access).
To put RDMA simply, it's a **technology for directly reading and writing memory**.
As we saw earlier, to read from or write to the GPU's memory, VRAM, you originally have to go through the CPU or OS.
What that means is that you have to copy the data to the CPU first before you can read or write it.
But RDMA is a network where you can move data directly from GPU memory to GPU memory without doing that.

You might then ask whether RDMA only exists with InfiniBand, but that's not the case.
Nvidia developed a solution called RoCE (RDMA over Converged Ethernet), which does GPU RDMA using Ethernet.
In terms of speed, using high-speed Ethernet cables you get about 800Gbps, while InfiniBand gets about 1.6Tbps, so there's roughly a 2x difference.
In the old days the gap was bigger, but high-speed Ethernet technology has advanced a lot, so the gap has narrowed considerably.

Nvidia implemented RoCE by **transmitting InfiniBand's packets, which were defined for RDMA communication, over Ethernet**.
It was originally developed by Microsoft as an Ethernet communication technology used in data centers, but Nvidia established it for GPU RDMA communication.
The reason RoCE is important is Ethernet's high compatibility and low price.
I once heard that InfiniBand costs over $100 per meter, but Ethernet is far cheaper than this and has many manufacturers, so the likelihood of it getting cheaper is also much higher.

### EFA

So then what is EFA?
EFA (Elastic Fabric Adapter) is a network interface that uses **SRD (Scalable Reliable Datagram)**, AWS's in-house developed network protocol.
SRD is a method that works only on AWS infrastructure, developed to guarantee reliability as much as TCP while sending and receiving information over various network paths rather than a single path like UDP ([reference on TCP/UDP](https://mangkyu.tistory.com/15)).
SRD is a technology based on a [paper published by AWS's Annapurna Labs](https://ieeexplore.ieee.org/document/9167399), and it has an interesting history.

Annapurna Labs is an Israeli semiconductor company that AWS acquired in 2015 for $370 million, and it's currently the organization within AWS developing in-house semiconductors.
After being acquired, it developed various chips like Graviton and Trainium, contributing to the cost-efficiency of AWS's computing.
The founders of Annapurna Labs previously founded a semiconductor chip company called Galileo Tech, and the founders of Mellanox, the InfiniBand manufacturer acquired by Nvidia, also came from the Galileo Tech executive ranks.
Maybe because their roots are similar, their thinking seems similar too.

Anyway, EFA used SRD to dramatically increase network bandwidth, and **enabled RDMA by leveraging an open-source library called libfabric**.
You can think of the structure as Application → MPI/NCCL → **Libfabric** → EFA, where it's a library that enables RDMA through OS-bypass communication.
AWS started supporting EFA on EC2 instances in 2018. For CPU resources, just note that it's available from c5n onward, and for GPU resources you can assume it's all supported. ([reference on EFA-supported instances](https://docs.aws.amazon.com/ko_kr/AWSEC2/latest/UserGuide/efa.html))

**In the end, you can say AWS chose generality by using EFA rather than InfiniBand.**
If there were lots of people who wanted GPU instances in multi-node setups, that'd be one thing, but if you rent a single node, the inter-node network is useless.
Also, AWS handles a wide variety of non-GPU instances too, so it has no choice but to lay down an Ethernet-based general-purpose network.
That said, **companies with developers building ultra-large models usually build and use On-prem clusters**, so it seems like there's still a lot of code with a dependency on InfiniBand (DeepEP is one of them…).
This seems like a problem that will naturally resolve itself once multi-node training open-source contributors start to appear.

### NVLink

In fact, InfiniBand and EFA are solutions for connecting physically separated servers, so they can't possibly be faster than putting all the GPUs in a single rack.
As long as the GPUs are in the same rack, communication is possible at extreme speeds, and that technology is exactly NVLink. NVLink is an **ultra-high-speed data communication technology between GPUs** created by Nvidia.
It transmits and receives data directly between GPUs without going through the CPU, and on the Blackwell GPU it gets nearly 14Tbps, so you can consider it incomparably faster than InfiniBand.

Originally this is a point-to-point network that directly connects two GPUs bidirectionally, but using something called an NVLink Switch you can bundle multiple GPUs together.
There's actually a special rack called NVL72, which is a monstrous thing that bundles 72 B200s into a single rack with NVLink Switches.
The total bandwidth is said to be 1040Tbps, so… that says it all.
However, it requires a tremendous amount of power, and since cooling only works via liquid cooling, for now it has to be used in a limited way.

### Organizing terms I didn't know / found confusing

Lastly, here are some terms I came across while studying that I didn't properly understand or found confusing, briefly organized.

- NIC (Network Interface Controller)
    - A hardware device that connects a computer to a network
    - A general term for various cards like the network card, LAN card, Ethernet card, etc.
    - Network identification via MAC address / hardware-based encryption, etc.
    - Speed of about 1~100Gbps
- ENI (Elastic Network Interface)
    - A virtual network interface for network communication between EC2 instances
    - If a NIC is a network interface that connects physical servers, an ENI connects EC2 instances, which are virtual servers
    - 10Gbps or less (above 10Gbps is ENA: Elastic Network Adapter)
        - ENA raises network speed through **SR-IOV** (Single Root I/O Virtualization) → **a technology that makes a single PCIe device appear as multiple virtual PCIe devices** (since PCIe is a parallel method without interference, increasing the count can raise speed)
- HCA (Host Channel Adapter)
    - A special interface card for InfiniBand network connections
    - Latency 1~2 µs / 200Gbps / RDMA
- HBA (Host Bus Adapter)
    - An interface card that connects servers and storage devices
    - Hardware-based offloading to reduce the server CPU's load (independent I/O without burdening the CPU)
    - 8~64Gbps
- **PCIe (Peripheral Component Interconnect express)**
    - **A general-purpose interface → a switch that supports connecting various devices such as CPU, GPU, storage, Ethernet, IB, etc.**
    - A parallel method where interference-free transmission and reception happen per lane (each lane is serial)
    - PCIe x16 means an allocation of 16 lanes
    - There are PCIe slots, and cards are installed into those slots
- **NCCL (Nvidia Collective Communication Library)**
    - Communication between multiple processes: collective communication
        - Broadcast, scatter, gather, all-gather, all-to-all, reduce, all-reduce
    - It's been proven that all collective communication is nearly optimal if you think of it as a Ring topology
- MPI (Message Passing Interface) / GLOO
    - Libraries used for CPU-based parallel computing
    - Since GLOO was made by Meta, it's set as the default backend of torch distributed
    - OpenMPI has the advantage of supporting various languages and being feature-rich
- Storage (SSD) ↔ CPU communication
    - SATA (Serial ATA)
        - A serial interface that uses the AHCI communication protocol
        - Speed of about 600MB/s
    - **NVMe (Non-Volatile Memory express)**
        - A high-speed protocol that directly uses the PCIe bus to maximize NAND (SSD storage) speed
        - Speed of about 6GB/s

### Reference

- [https://computing-jhson.tistory.com/81](https://computing-jhson.tistory.com/81)
- [https://toss.tech/article/30767](https://toss.tech/article/30767)
- [https://aws.amazon.com/ko/blogs/tech/aws-efaelastic-fabric-adaptor/](https://aws.amazon.com/ko/blogs/tech/aws-efaelastic-fabric-adaptor/)
- This might turn out to be really useful on the coding side
