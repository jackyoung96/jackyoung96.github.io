---
layout: post
title: Diary - LLM knowledge distillation
tags: archive
lang: en
---



At some point I started using gpt-4o-mini instead of gpt-4o. The biggest difference is speed. gpt-4o's inference speed isn't exactly slow, but compared to gpt-4o-mini it feels frustratingly sluggish. And is there a performance difference? Not really.

How did a small model end up catching up to a large model's performance like this? The reason probably lies in the Knowledge Distillation technique.

While studying Knowledge Distillation, I came across a simple yet excellent piece of writing. [Post-training distillation for LLMs](https://drive.google.com/file/d/1xMohjQcTmQuUd_OiZ3hB1r47WB1WM3Am/view)

With the consent of the original author, Rishabh Agarwal, I have translated and reorganized this content in my own way.

- [Introduction](#introduction)
- [Knowledge distillation 기법들](#knowledge-distillation-기법들)
  - [Supervised KD (optimizes forward KL)](#supervised-kd-optimizes-forward-kl)
  - [Distillation using “Synthetic Data”](#distillation-using-synthetic-data)
  - [GKD: Generalized knowledge distillation](#gkd-generalized-knowledge-distillation)
- [Another advantages of Distillation](#another-advantages-of-distillation)
  - [Speculative decoding](#speculative-decoding)
  - [DistillSpec](#distillspec)
- [Advanced knowledge distillation](#advanced-knowledge-distillation)
  - [SKD: speculative knowledge distillation](#skd-speculative-knowledge-distillation)
- [Conclusion](#conclusion)




















## Introduction

What is Knowledge distillation (KD)?

> A methodology for **transferring** the **knowledge** held by a large, expensive Teacher model to a small student model

<img width="736" alt="Image" src="https://github.com/user-attachments/assets/587ef672-db03-4e3a-92a3-ce2e3daab6fb" />
    
In fact, this distillation technique brought about the breaking of the Scaling law. Previously, it was generally true that the larger the Model size, the higher the performance. As we went from GPT-1 to 2 to 3, the number of parameters grew dramatically, and a model that couldn't even do single-digit arithmetic became able to perform 4-digit and 5-digit operations (emergent ability).

However, if you look at the performance vs. Model pricing graph (directly correlated with model size) released by LMsys, that formula is gradually breaking down.
<img width="768" alt="Image" src="https://github.com/user-attachments/assets/db6fb90a-8f1a-4d7d-a3f8-58870b72f11d" />

Small, cheap models are catching up to large, expensive ones. You can see why this is possible just by looking at Jack Morris's post below.
<img width="1117" alt="Image" src="https://github.com/user-attachments/assets/0c2257df-21a7-401b-b18b-33b485e69f3e" />

In the end, you can say that after a powerful large model is released into the world, knowledge distillation techniques are used to train small models, greatly narrowing the performance gap.

## Knowledge distillation 기법들

So let's take a look at what kinds of knowledge distillation techniques exist.

### Supervised KD (optimizes forward KL)

(Hinton et al., 2015, [Distilling the knowledge in a Nueral Network](https://arxiv.org/abs/1503.02531))

When a large model and a small model are trained on the same dataset, the first difference you notice is Classification confidence.
    
<img width="1244" alt="Image" src="https://github.com/user-attachments/assets/2f7aa5e0-91aa-496b-9ebe-4e5a5bf3a678" />

<img width="1239" alt="Image" src="https://github.com/user-attachments/assets/f6e4b446-3b1e-43a2-aa6f-20465e24fd8d" />
    
In LLM terms, when comparing the highest-probability tokens, the teacher model's token probability is higher than the student model's token probability.  
In other words, if the two models generate the same response, you could say the teacher model has a lower PPL (PPL isn't directly tied to LLM performance, but… a lower loss usually means higher classification performance, so it's that kind of perspective).

    
<img width="637" alt="Image" src="https://github.com/user-attachments/assets/ba8fba65-9c14-475a-9151-f69a96222bc8" />

If the distribution of the logits (the probability distribution over tokens) is the same, then the two models will have the same performance. Making this happen is the goal of knowledge distillation.

> Making the teacher model's logit distribution and the student model's logit distribution the same  
> Making the distributions the same == minimizing KL divergence

In the end, instead of hard labels (ground truth), you compute and minimize the cross-entropy loss against the soft labels coming out of the teacher model. The loss usually used in LLM training is called next token prediction, but this can be seen as next token *distribution* prediction.
        

<img width="1387" alt="Image" src="https://github.com/user-attachments/assets/54e7be61-e4f2-4f1c-85a7-f04995af05e5" />
        
So how does it perform?

In the [Gemma 2 paper](https://arxiv.org/abs/2408.00118), they compared 7B distillation against from-scratch training when training (pre-training) Gemma 2B.
        
<img width="1085" alt="Image" src="https://github.com/user-attachments/assets/3b51d80d-fdb4-4d2b-a0a3-1cec280439e4" />
        
In the end, distillation gives both higher performance and lower PPL, so if you have a powerful model there's no need to bother with from-scratch training.

### Distillation using “Synthetic Data”

(Kim et al., 2016, [Sequence-Level Knowledge Distillation](https://aclanthology.org/D16-1139.pdf)) → ~~A familiar name to the folks at SKT...!!~~

Actually, Supervised KD requires using logits, and while this looks simple, an LLM's logit is at minimum a 100K-sized embedding per token, so memory issues can arise as soon as the sequence gets a bit longer. (There's also a method that uses only Top-K.) So a simpler yet powerful method using synthetic data was proposed.
    
<img width="1504" alt="Image" src="https://github.com/user-attachments/assets/d7dec726-1a08-4fcf-865d-91427bc20450" />
    
You take prompt data and extract responses from the teacher model. Then you perform SFT on the student model using that data.  
The greatest advantage of this methodology is… distillation is possible with just API access! (You don't need to know anything about logits, so you can distill from the latest LLMs like gpt-4o to boost performance.)

<img width="449" alt="Image" src="https://github.com/user-attachments/assets/8a812891-caab-4ea9-b349-0d5b4220d01b" />
        
Sam Altman's recent post taking a shot at the newly released DeepSeek is in a similar vein.
            
<img width="401" alt="Image" src="https://github.com/user-attachments/assets/917fffbb-6c47-426b-b707-246120e44fab" />
            
But this synthetic data methodology actually has a solid theoretical foundation.
    
<img width="1177" alt="Image" src="https://github.com/user-attachments/assets/7d127bd5-b19f-4caa-8d62-1958d1f9ee28" />
    
If you do a *Monte-Carlo approximation* of the expectation part of this final equation, you end up with exactly the synthetic data distillation methodology. Therefore, using the synthetic data distillation methodology minimizes the KL divergence between the teacher model and the student model, just like Supervised KD.


Also, the higher the quality of the answers you obtain, the better the student model's performance becomes.

A representative method is Rejection sampling (BoN). You generate N answers from the teacher model and train the student using only the best answer among them.
<img width="1414" alt="Image" src="https://github.com/user-attachments/assets/57464ec9-ce49-4095-8934-c8f86f6afa7c" />    

An extended approach is Compute-matched / Cost-matched sampling. When using BoN, you use a smaller teacher model and increase N (equivalent to using a cheaper model).
<img width="712" alt="Image" src="https://github.com/user-attachments/assets/de1bf72d-04a2-4646-90cf-106a78463be8" />    
The funny thing is that even using a teacher model smaller than the student model and increasing N showed a meaningful performance improvement.
<img width="1540" alt="Image" src="https://github.com/user-attachments/assets/8201f4af-35c0-4d95-a5b7-e674b22a673a" />
        
This methodology has two huge advantages.
As I said earlier, knowledge distillation is possible even with just API access. (Before DeepSeek R1, most SOTA models were closed-source.) Even if a model isn't open-source, there's no way to block API usage.
And it works even with different tokenizers! (Forward KL requires matching logits, so the tokenizer has to be exactly identical.) In the end, it means distillation is possible even between models with completely different architectures or foundations.

(There's no reason not to use this...??)

### GKD: Generalized knowledge distillation

(Agarwal et al., 2023, [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://openreview.net/forum?id=3zKtaqxLhW))

The two distillation methodologies introduced above have one fatal problem: train-inference mismatch.

When the training distribution and the inference distribution differ, you frequently fall into OOD (Out-of-distribution) situations. Especially since the student model performs worse than the teacher model, once a wrong token comes out, the answer can veer off into something completely bizarre.
        
<img width="1196" alt="Image" src="https://github.com/user-attachments/assets/e8475889-c470-44af-9839-c4b0834f1f2a" />

The training distribution is the teacher model's responses, and the inference distribution is the student model's responses, so no matter how much you train the two distributions to become similar, the structure remains vulnerable to OOD.

To solve this, the On-policy distillation methodology was proposed. The goal of this methodology is to train while aligning the train and inference distributions.
        
<img width="1220" alt="Image" src="https://github.com/user-attachments/assets/8a0dfb1a-98d7-4227-b307-7fcf6d75d7a5" />
        
- **On-policy data**: Use the student model's responses
- **Feedback**: Use the teacher model's logits for the student model's responses
- **Supervised training**: Train so that the teacher–student token-level logit distributions become the same

If you express this mathematically, you can see it's **Reverse KL**.
        
<img width="938" alt="Image" src="https://github.com/user-attachments/assets/563d4a73-0400-4f92-b3ef-4f333889e46b" />

It's similar to the KL used in the synthetic data methodology earlier, but you can see the positions of the student and teacher distributions have been swapped. (KL-divergence is asymmetric, so they're not the same!)

Of course, even though the formulas aren't the same, the purpose of minimizing the KL divergence is to make the student model's and teacher model's distributions become the same, and this itself is the same as forward KL… so what is different?

The teacher model and the student model are fundamentally different in expressiveness (model capacity) **(the number of dimensions that can be represented == the number of parameters)**.

If you use Forward KL, the student distribution is trained to do mode-covering.
<img width="665" alt="Image" src="https://github.com/user-attachments/assets/5831c20b-69fa-49a7-b549-5409c902e222" />
On the other hand, if you use Reverse KL, it's trained to do mode-seeking.                
<img width="637" alt="Image" src="https://github.com/user-attachments/assets/f9ad8d90-81d8-4cc4-85f6-1c81d20133d0" />
(There's a fantastic blog post about forward KL and Reverse KL [forward KL, Reverse KL explanation](https://process-mining.tistory.com/147))

For a student model with small expressiveness, mode-seeking may actually be more suitable. Because rather than having a random out-of-place token come out, it's better to get the teacher model's secondary token. But if the teacher's modes aren't sharp and spiky and are sufficiently generalized, mode-covering might be better.

<img width="1238" alt="Image" src="https://github.com/user-attachments/assets/936ce468-e52e-469f-9aa9-cb9b61fb5d6c" />

So when they ran experiments, mixing forward KL and reverse KL worked best. (0.5 forward KL + 0.5 reverse KL = Jeffrey's divergence)
                

Anyway, we've covered the theoretical parts, but how exactly can we implement the GKD methodology? The simplest way is to use [GKD implemented in TRL](https://huggingface.co/docs/trl/main/en/gkd_trainer). Or you can adapt RLxF (RLHF, RLAIF) code.
        
<img width="1345" alt="Image" src="https://github.com/user-attachments/assets/9b675f3d-1802-4062-b207-2dec8bf0ab2e" />

        
RLxF code includes a KL term as a regularizer that keeps the policy from drifting too far from the SFT distribution when performing RL, and you plug in the teacher model in place of SFT. Then you delete the Reward-related term and GKD is complete.

There's also content saying that if you don't delete the Reward-related term, you can catch two rabbits at once: RL + distillation. (Of course, as RL gets involved, training instability will increase.)
                
<img width="902" alt="Image" src="https://github.com/user-attachments/assets/e50f272d-e9a7-404d-8f0a-58bad724a8aa" />


## Another advantages of Distillation

In fact, the advantages of distillation aren't only in improving the student model's performance!

### Speculative decoding

(Leviathan et al., 2023, [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192))

Transformers made it possible to train massive models through training parallelization, but they're extremely slow at inference. You can compute the logits for the input sequence all at once, but when generating tokens you have to generate them sequentially.

So Speculative decoding, which came out of this, is a method of speeding up a large model by using a small model that answers similarly to the large model.
    
<img width="1439" alt="Image" src="https://github.com/user-attachments/assets/50b48e0b-86e2-47a6-922d-dc2aae18955a" />
    
- Since the small model can generate quickly, it first generates several tokens
- The large model extracts the logits for those several tokens all at once and finds the wrongly generated tokens
- Starting from the wrongly generated token, the small model generates tokens again -> repeat

For this to work, the large model's and small model's answers have to be similar for the speed to go up. If all the tokens turn out different, you're just burning the small model and consuming more resources for nothing.

### DistillSpec

(Zhou et al., 2024, [DistillSpec: Improving Speculative Decoding via Knowledge Distillation](https://arxiv.org/abs/2310.08461))

But even if they were trained on the same dataset, there's no guarantee that the small model's and large model's answers will be similar. So a method to maximize speculative decoding performance through distillation was proposed.
    
<img width="752" alt="Image" src="https://github.com/user-attachments/assets/7e04f239-9a5b-4fa0-b97b-2c2a22890a61" />
    
With distillation, the student model's answers become similar to the teacher's, increasing inference speed by 10~45%. This methodology is actually applied to Google's search page.
    
[Example of speculative decoding applied to Google](https://storage.googleapis.com/gweb-research2023-media/media/SpeculativeDecoding-0-AIO.mp4)
    
## Advanced knowledge distillation

### SKD: speculative knowledge distillation

(Agarwal et al., 2025, [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649))

Most recently, a hybrid methodology that applies this Speculative decoding was also proposed.

GKD performs well, but there's a question of whether it's okay to do GKD when the student's on-policy responses aren't that good. In such a situation, it might be better to do Supervised KD using the teacher's high-quality responses, even if they're off-policy.  
The methodology that takes only the advantages of these two is SKD.
    
<img width="1288" alt="Image" src="https://github.com/user-attachments/assets/2d16d9b8-1753-460d-b1a2-200163024253" />

First, you extract the student's on-policy responses. Then you check whether each token is included in the teacher's top-K tokens.
If they're all included, the on-policy response is in pretty good shape, so you perform GKD.
If there are tokens that aren't included, you replace those tokens with the teacher's tokens to raise the quality, and use this to perform Supervised KD.

When this SKD methodology is applied, they say both performance and speculative decoding speed improved.
    
<img width="1278" alt="Image" src="https://github.com/user-attachments/assets/716ee346-6a67-45e1-8ec4-95ef67f86505" />
    
This is because from the perspective of the student's performance, on-policy distillation is better, while from the perspective of the teacher's speculative decoding, distillation using teacher responses is better.

## Conclusion

| Compute efficiency | Online < Offline |
| --- | --- |
| Sample efficiency | Online > Offline |
| Resource waste from a real-time teacher model during training | Online < Offline |
| Time delay from student sampling | Online < Offline |
| Suitability for long horizon tasks | Online > Offline |
| Train-test distribution mismatch | Online > Offline |
- Advantages of Online KD
    - Doesn't require much data
    - Suitable for long horizon tasks like Agents because the train-test distribution mismatch is small (doesn't easily fall into OOD)
- Advantages of Offline KD
    - Once you generate the teacher model's responses or store the logits just once, you can keep reusing them
        - If you extract the logits in advance, you can also save the GPU resources needed for the teacher model
    - Training is fast because there's no student on-policy generation process

The trade-off of knowledge distillation ultimately comes down to resources versus performance. The more a methodology improves performance, the more computing resources / time resources it requires.  
So let's choose an appropriate distillation methodology depending on the situation.
