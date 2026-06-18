---
layout: post
title: Anthropic post - Mapping the Mind of a Large Language Model
tags: archive
lang: en
---

An Anthropic post that's very intriguing right from the title.

## Summary of the post's main content

> 🏬 **Looking inside an ML model wasn't really meaningful work. That's because it's a list of numbers with no clear meaning.**  
> Opening the black box doesn't necessarily help: the internal state of the model—what the model is "thinking" before writing its response—consists of a long list of numbers ("neuron activations") without a clear meaning

> 🏬 **"Dictionary learning"
Instead of many unclear active neurons, you build a dictionary of a few active features to observe meaning**  
> In turn, any internal state of the model can be represented in terms of a few active features instead of many active neurons. Just as every English word in a dictionary is made by combining letters, and every sentence is made by combining words, every feature in an AI model is made by combining neurons, and every internal state is made by combining features.

> 🏬 Base of dictionary learning  
> **They use a Sparse AutoEncoder to extract features from neuron activations**  
> What is a Sparse AutoEncoder? [https://soki.tistory.com/64](https://soki.tistory.com/64)  
> Anthropic's feature decomposing: [https://transformer-circuits.pub/2023/monosemantic-features](https://transformer-circuits.pub/2023/monosemantic-features)  
> Superposition hypothesis: Assumes a state where the model represents more features than the number of neurons → the assumption that it estimates a larger neural network by leveraging the properties of high-dimensional space  
> **The dimension of the feature space must be larger than the neuron activation dimension**  
> So they use a Sparse AutoEncoder → the latent dimension is 256 times larger than the input dimension

> 🏬 **Feature distance  
> You can measure the distance between features based on neuron activation patterns.**  
> We were able to measure a kind of "distance" between features based on which neurons appeared in their activation patterns.  
> **The noun "Golden Gate Bridge" has features similar to nouns like Alcatraz Island, Ghirardelli Square, the Golden State Warriors, etc.**  
> This allowed us to look for features that are "close" to each other. Looking near a "Golden Gate Bridge" feature, we found features for Alcatraz Island, Ghirardelli Square, the Golden State Warriors, California Governor Gavin Newsom, the 1906 earthquake, and the San Francisco-set Alfred Hitchcock film *Vertigo*.  
> **"Inner conflict" also has features similar to figurative expressions like catch-22 (a 'bind' where you can't get the outcome you expected, or a 'dilemma', or a 'very difficult situation') → this somewhat explains Claude's capacity for analogy and metaphor)**  
> looking near a feature related to the concept of "inner conflict", we find features related to relationship breakups, conflicting allegiances, logical inconsistencies, as well as the phrase "catch-22”

> 🏬 **Manipulate features  
> Activating a specific feature lets you control the model's response**  
> But when we ask the same question with the feature artificially activated sufficiently strongly, this overcomes Claude's harmlessness training and it responds by drafting a scam email.  
> **Not only is responding to the input context related to this feature activation, but the model understands a world representation through features and uses behavior based on it**  
> The fact that manipulating these features causes corresponding changes to behavior validates that they aren't just correlated with the presence of concepts in input text, but also causally shape the model's behavior.

> 🏬 **Make the model safer**  
> **Through feature detection, AI monitoring becomes possible, and it can be used for purposes such as removing specific topics**  
> For example, it might be possible to use the techniques described here to monitor AI systems for certain dangerous behaviors (such as deceiving the user), to steer them towards desirable outcomes (debiasing), or to remove certain dangerous subject matter entirely.

### Summary

- The activation pattern of an MLP layer can be turned into features through a Sparse AutoEncoder. This is the Dictionary learning method used in classical ML.
- When the Dictionary learning method was applied to an LLM, it was found that not only related words but also similar high-level concepts and ambiguous concepts like metaphors/analogies sit close together in the feature space.
- It was experimentally confirmed that deliberately strengthening a specific feature generates an answer similar to that feature. This can explain how things like in-context learning strengthen a specific feature to produce an answer related to that content. **The LLM understands a world representation through features!**
- Because you can tell which features a given input/output is close to, you can **make the LLM safer.**

### P.S. Anthropic's dictionary features

[https://transformer-circuits.pub/2023/monosemantic-features/vis/index.html](https://transformer-circuits.pub/2023/monosemantic-features/vis/index.html)

![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/49431c8f-959f-4bdc-a1f1-a21ad96e519c)
