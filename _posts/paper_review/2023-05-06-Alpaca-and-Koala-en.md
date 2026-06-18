---
layout: post
title: "Project review - Knowledge distillation from powerful LLM, Alpaca and Koala"
tags: archive
lang: en
---

This post summarizes a talk I gave as the host of DEEPEST Season 13.
This post has two main goals:
- A basic explanation of text generation models for people who have heard of Transformer and GPT but have never read the papers or looked into the details
- A proposal for a new methodology for people who want to solve a downstream task with an LLM but are stuck because there is no well-trained, publicly available model

<br>

- [Large Language Model (LLM)](#large-language-model-llm)
  - [Transformer](#transformer)
  - [GPT-1](#gpt-1)
  - [GPT-2](#gpt-2)
  - [GPT-3](#gpt-3)
  - [Instruct-GPT](#instruct-gpt)
  - [GPT-4](#gpt-4)
- [Closed-source LLMs](#closed-source-llms)
- [Knowledge distillation from openAI's LLM](#knowledge-distillation-from-openais-llm)
  - [Alpaca](#alpaca)
  - [Koala](#koala)
- [Conclusion](#conclusion)

<br><br>

## Large Language Model (LLM)

Let's look at Large Language Models (LLMs) in chronological order. Actually, when we say LLM, there are not only GPT but also the BERT and T5 families, but for now let's focus on text generation models and look only at the GPT models.

### Transformer

The Transformer model proposed in the [Attention is all you needs](https://arxiv.org/abs/1706.03762) (NeurIPS17') paper presented a methodology that solved the parallelization problem that was impossible with the existing Recurrent Neural Network (RNN).

<img width="513" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/96e7e461-97d4-42b6-9df0-81222b89dce9?style=centerme">

Unlike the RNN structure, where the input must be fed in sequentially, the Transformer uses an approach that computes the attention of the input sequence **simultaneously**. Explaining the Transformer alone would take half a day, so I'll skip it here.

<img width="604" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/cbfad0c7-9e52-4bf3-bb5f-b980fcb3b634?style=centerme">

Anyway, the benefit gained from using the Transformer is that, because parallelization is possible, it became feasible to handle longer sequences and to build larger models. Since RNNs can't be parallelized, if you made the model big, not only training but also inference would take an enormous amount of time.
Also, when people tried it, the performance turned out to be good too (after all, in deep learning all that matters is good performance). As a result, in the NLP field (and even extending to vision), Transformer-based models came to dominate.

### GPT-1

Through the [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) (Arxiv18') paper, OpenAI released the GPT-1 model. This model uses only the Decoder structure of the Transformer.

<img width="377" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/c508f7cc-afb0-4a02-a0f0-9ca0615d8281?style=centerme">

As shown in the figure above, training was conducted in two stages: Unsupervised pre-training (PT) and Supervised fine-tuning (SFT). In PT, since training simply involves feeding in a sequence of tokens and predicting the next token, no annotation work is needed. In SFT, a labeled dataset is used depending on the task.
Actually, this structure is not much different from the approach used in Computer Vision. It directly borrows the method of freezing a pre-training model like ResNet and attaching a linear model at the end to solve the downstream task.
<br>
What's interesting is that something called zero-shot behavior began to emerge. Without fine-tuning, just with pre-training and some reasonable heuristic methods, it showed decent performance on NLP tasks.

<img width="415" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/642b4c77-5c8b-4dea-a564-04d32d72a3cb?style=centerme">

Taking sentiment analysis as an example, the approach was to feed in a sentence and then judge based only on comparing the probability of the words positive/negative appearing as the next token. With just this, it showed performance approaching nearly 70%.

This shows the possibility that GPT-family language models can achieve good performance while skipping expensive annotation work.

### GPT-2

The [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (Arxiv18') paper shows an attempt to maximize the zero-shot behavior discovered in GPT-1. They increased the model's capacity further and built the data used for pre-training to be larger and higher quality.

First, they scaled the model size by nearly 13x. Compared to GPT-1, which was 117M in size, GPT-2 used a 1.5B model.
They also built a new dataset called WebText. Task-specific data was not needed, but the number of data points had to be large, so they obtained data through web crawling. They also constructed the dataset based on three criteria to maintain quality.
- Data that received 3 or more Karma (upvotes) on Reddit
- Wikipedia was excluded because the Training/Test overlap is severe
- Only English data
The Wikipedia point is impressive, as it apparently has too much influence on evaluation performance. This is because too many sentences in Reddit replies are quoted from Wikipedia. (In effect, the answers to the questions are all on Wikipedia...)

<img width="931" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/4d05f52a-c032-4096-9125-4a0046949969?style=centerme">

As expected, when they increased the model size, the zero-shot behavior also began to increase.

<img width="677" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/8edcf541-1d32-4441-a41f-a88af656f829?style=centerme">

Beyond simply increasing performance, GPT-2's performance shows tremendous results, **surpassing existing SOTA methods in a zero-shot manner** on some tasks. Of course, it wasn't that outstanding on Summarize, Translate, Factual QA, etc. In particular, on Factual QA, only 4.3% of the answers were correct. (From this point you can start to smell Hallucination...)

### GPT-3

With the publication of the [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (NeurIPS20') paper, OpenAI opened the prelude to the current era of ChatGPT. They confirmed from GPT-2 the fact that zero-shot behavior is proportional to the model size and dataset quality, and they directly put this into practice to create GPT-3. With much larger capacity, a higher-quality dataset, and the few-shot learning technique.

First, they increased the model size all the way to 175B. They suddenly scaled it up by 100x. At 175B, the model size alone is roughly 800GB, and to train it you need at least 160 A100s. Since an A100 costs about 15 million won, the GPU price alone would be around 24 billion won. On top of that, at that scale the electricity cost alone would be about 6 million won per day in Korea. A research lab can't handle that.

It's the same with the data. They carefully refined a 45GB dataset scraped from the internet to create a 570GB high-quality dataset. In doing so, they intensively collected high-quality reference corpora such as books and Wikipedia, and performed work to remove duplication as much as possible.

Finally, the few-shot learning technique is a method where, when solving a downstream task, instead of zero-shot, you present a few examples along with the problem and have it solve the problem.

<img width="514" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/f66f1be3-0ba8-48ea-befd-3185c3d7c27c?style=centerme">

Even when translating "cheese" into French, giving a few examples would help it understand the problem better. (At this point it's no different from human metacognition.)

GPT-3 began to solve parts that existing language models had not been able to solve, parts that were thought to be the domain of humanity. It became able to do addition and subtraction, and able to solve anagrams.

<img width="487" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/48242e0f-d211-4022-960d-749ffbf027ce?style=centerme">

<img width="619" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/5e77489e-a44a-4427-81fd-47cfbc25215a?style=centerme">

### Instruct-GPT

The [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (Arxiv22') paper takes one step further from GPT-3. GPT-3 shows amazing things, but in reality it was a bit different from what people actually want. The reason is actually that the objective is different. GPT looks at the preceding token sequence and predicts the next token that will appear, whereas the language model that people want produces answers that are helpful, harmless, and truthful to them. So OpenAI tried to create a user-aligned LLM using reinforcement learning, and as a result they proposed InstructGPT.

<img width="611" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/4cad218e-a0fa-475f-894c-1586ea17bb1e?style=centerme">

InstructGPT consists of three stages in total: Supervised Fine-tuning (SFT), Reward model training, and Optimizing policy. Let's look at each in detail.

**SFT**

First, OpenAI decided to turn all tasks into Instructions. An Instruction has the following form.

<img width="799" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/1c9db7b3-a320-4d46-8295-fa6a0b92e1cb?style=centerme">

Let me give an example of asking a language model to write an essay about school safety. In this case, the Instruction would be "writing an essay about following topic" and the instance input would be "school safety".

OpenAI turns the scraped data into this kind of form. They hired 40 labelers full-time from Upwork and ScaleAI, and created a set of 13K Instructions. (It probably cost quite a lot.)

They fine-tuned on the created Instruction set. Apparently it was done for 16 epochs.

**Reward model training**

The Reward model is trained using PbRL as proposed in [Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741). The approach is that human labelers rank the multiple outputs generated by the SFT-ed model, and this is used to estimate the reward for each prompt. (For details on PbRL, see [here](https://jackyoung96.github.io/2022/01/01/PbRL_review1/).)

For this too, 40 human labelers estimated rankings for 33K data points. It must have taken a tremendous amount of money, time, and effort, right? For the Reward model, they apparently used a 6B LLM. There was an attempt to use 175B, but apparently the reward model becomes unstable as its size grows.

**Optimize policy**

Finally, they optimize the LLM using the trained reward model and a reinforcement learning algorithm (PPO). At this point no separate labeling is needed, and they conducted training using 31K prompts obtained from users on OpenAI's GPT-3.

I'll skip the results for InstructGPT. This is because it's effectively only a qualitative evaluation. I'll just note the point that InstructGPT's results had **much higher human preference** compared to GPT-3.

InstructGPT is a language model that generates answers well-aligned to humans while minimizing the decrease in general performance. However, quite a lot of resources were needed in that training process. It seems like work that's hard to do at the research-lab level.

### GPT-4

And in February 2023, GPT-4 was released along with the [GPT-4 Technical Report](https://cdn.openai.com/papers/gpt-4.pdf) (Arxiv23'). It's a model capable of processing multi-modal data with images added, that improved the hallucination problem, and that produces higher-quality answers. Unfortunately, they did not disclose any of the model size, architecture, HW info, dataset, or training process. Speculation puts it somewhere between a 1~100T model, and apparently around 30K GPUs are being used for model serving.

## Closed-source LLMs

<img width="778" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/83151f4e-f790-4765-ad88-4992df4d3ae0?style=centerme">

The three LLMs currently provided as services that are known to have the best performance - OpenAI's ChatGPT, Google's Bard, and Anthropic AI's Claude - are all closed-source. That is, they don't release the models and only provide services in the form of API calls.

In a situation where the pre-trained weights are not released, the training dataset is not released, and SFT and RLHF are also very expensive (annotators composed of highly educated people with master's degrees or higher), it's difficult to do research on various downstream tasks using pre-trained LLMs.

Also, these LLMs actually use a very inefficient training method. The amount of text data a human sees in a lifetime is said to be about 0.16GB, but GPT-3 used 450GB of data. Humans learn through educational curricula, or learn from teachers.

Therefore, I'll introduce two projects that are open-source LLMs and that quickly trained models using an efficient training method (knowledge distillation).

## Knowledge distillation from openAI's LLM

<img width="1176" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/d2363719-cb76-4644-92c5-5e49374ceee7?style=centerme">

Both models are really fresh off the press. Stanford's Alpaca was released on March 13, 2023, and Berkeley's Koala was released on April 3, 2023.

Both models are based on [LLaMA](https://arxiv.org/abs/2302.13971) (Arxiv23'), an open-source LLM released by Meta. It's a very fresh model released on February 4, 2023. Surprisingly, Meta provided this model for research purposes, not as a service via API. They used the expression "further democratizing access." It feels a bit like the company doing this because things were going badly for them lately... (shh...)

LLaMA was released in 7B, 13B, 33B, and 65B models. According to the paper, it used 4x more refined data, and the 13B model showed performance similar to GPT-3 (175B) on some tasks. But it still shows lower performance when compared to GPT-3.5 or GPT-4.

Alpaca and Koala are results of applying knowledge distillation to LLaMA to train smaller models that have performance rivaling OpenAI's models. Both models start from the hypothesis that if you have 1) a reasonably powerful open-source pre-trained LLM and 2) high-quality Instruction data, you can create a powerful open-source LLM.

<img width="989" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/1a9134d2-d7cd-442b-b462-da95699935fd?style=centerme">

### Alpaca

Alpaca created its Instruction dataset using a method called self-instruct. [SELF-INSTRUCT: Aligning Language Model with Self Generated Instructions](https://arxiv.org/abs/2212.10560) (Arxiv22') proposes a method of generating data using an LLM.

<img width="682" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/9fecfe70-413e-422a-a68e-bbddeda20d34?style=centerme">

Using the method above, they created 52K instruction data points using only 175 seed instructions. Data generation is carried out through 5 steps.

***Step 1 : Seed Instruction***

A human directly writes 175 Instruction datasets.

<img width="814" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/0feb7fb4-7065-41ad-bdb5-4c310a9ac35b?style=centerme">

***Step 2 : Task generation***

Using 6 seed tasks and 2 generated tasks, they generate 8 new tasks. They use the Template below and use `text-davinci-003`, GPT-3's completion model.

<img width="478" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/d0ce63e0-2d39-48f7-af7f-24a6bb80275f?style=centerme">

You can see that new tasks are generated well, as shown below.

<img width="592" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/0c094496-71ef-49d4-a2ca-36aa2fc00b2f?style=centerme">

***Step 3 : Classification task identification***

Using 12 classification and 19 non-classification tasks, they determine whether the generated task is a classification task or not. They use the few-shot learning methodology (although with 31 examples it's hardly "few-shot"). Below is an example using just about 3.

<img width="312" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/90973b3d-bf10-4da5-8697-de03d5761fea?style=centerme">

***Step 4 : Instance generation***

Based on the given task, they generate instance input-output. However, for non-classification problems they construct the template to generate in input-output order, and for classification problems in output-input order.

<img width="1109" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/dbf2e8de-1a24-4fb5-bad5-597656ed32e2?style=centerme">

The reason is that for classification tasks, when the input is generated first, there were many cases where only one output was generated. It's a heuristic approach.

***Step 5 : Filtering***

Finally, among the generated instructions, only those satisfying the conditions below are added to the instruction pool.
- Add the instruction to the dataset pool only when ROUGE-L < 0.7 (ROUGE-L: a metric indicating the degree of string matching)
- Exclude cases containing the keywords images, pictures, graphs (there's a possibility it couldn't be properly expressed in text)
- Exclude cases where the input of an instance is the same but the output is different

<br>
The 52K self-instruct dataset generated this way apparently guarantees sufficient diversity and sufficient quality.

<img width="376" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/3164b523-f4ca-46a9-ab9b-b09a20b8c9a8?style=centerme">

<img width="647" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/049c494b-6718-42cc-9ae5-b33bf6bc5d61?style=centerme">

Alpaca fine-tuned the LLaMA 7B model using self-instruct data generated with text-davinci-003 (GPT-3). Generating 52K Instructions via the self-instruction method cost about \$500 in OpenAI API call costs, and fine-tuning LLaMA on eight 80GB A100s for 3 hours cost about \$100 on GCS.

<img width="667" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/998625f4-230b-496b-a21d-1f6129e5f407?style=centerme">

In their own qualitative evaluation, when they had 5 student authors do a blind test against GPT-3, Alpaca won (?) by 90:89. However, there were drawbacks: the answers were relatively short and hallucinations occurred frequently.

But Alpaca showed that you can own an LLM with performance rivaling GPT-3 for just \$600. It's not at all burdensome to use in a research lab, and it could solve countless downstream tasks.

### Koala

Koala took an even simpler approach than Alpaca. They directly used ChatGPT distillation data on the LLaMA 13B model. Since there wasn't even a self-instruct data generation process, they performed only 6 hours of fine-tuning using eight 80GB A100s and trained the Koala model using only about \$100.

<img width="737" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/926c9198-11b2-44f2-8780-4eeece635d21?style=centerme">

There were two kinds of ChatGPT distillation data: [shareGPT](https://sharegpt.com/) and [HC3](https://arxiv.org/abs/2301.07597). shareGPT is a public dataset with 60K conversations from ChatGPT made publicly available, and HC3 is a dataset gathering 24K human answers and 27K ChatGPT answers for 60K questions. The authors of the Koala project named the model fine-tuned on LLaMA using ChatGPT distillation data "Koala-distill."

Additionally, they also tested a model that used various open-source data for the fine-tuning work. The model trained by adding the data below was named "Koala-all."

<img width="569" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/c2fffbaf-b99e-48db-9a1d-85631524d5bd?style=centerme">

Unlike Alpaca, Koala conducted a qualitative evaluation by forming a panel of 100 evaluators on the Amazon Mechanical Turk platform.

<img width="657" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/20b76c5b-90bc-4b75-b273-06d51d287dd4?style=centerme">

As a result, it showed slightly better preference than Alpaca, but still lower preference than ChatGPT. But to rationalize (?) it, you can see it as an advantage given that the model is more than 10x smaller.

The surprising point was that Koala-Distill showed higher preference than Koala-All. This shows that rather than roughly scraping together data, **performing only knowledge distillation using a powerful model** yields much better performance.

## Conclusion

There's a movement in academia to create powerful yet open-source text generation LLMs using LLaMA. It's showing that methods of generating data from closed-source LLMs, such as self-instruction and ChatGPT distillation datasets, work very efficiently. In other words, if you have a pre-trained LLM (the initial condition) and data created by a good teacher, you can easily create a powerful language model.

However, one thing to be careful about is that there are license issues, so it can't be used commercially. LLaMA's license is non-commercial bespoke, and the ChatGPT terms of use have a condition that it can't be used for models competing with OpenAI, so please be careful about using these in a company!
