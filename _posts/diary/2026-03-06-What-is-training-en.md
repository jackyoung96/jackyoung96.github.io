---
layout: post
title: "Diary - What Is AI Training, Really? (feat. Claude Code)"
tags: archive
lang: en
---

These days I've been helping the only customer of "Jack's AI Center" — my wife and her older sister — set up Claude Code. The request was to build a bot that writes blog posts, and as I was building it, I had a strange realization. It felt like things I had taken for granted as an AI scientist were being shaken. That's the story I want to tell.

## Building a writing bot

At first, I agonized a lot over how to approach it. A blog post is something where everyone has a different style, a different tone, a different structure. And critically, even after roughly putting together an agent, it wasn't perfectly to my liking. A last mile problem existed, and while pondering how to improve this part, I chose the most fundamental and simple method below.

**Step 1: Initial pipeline setting**  
First, I explain in detail what kind of writing I want, and give a few example posts. Then I tell Claude Code, "Build a writing pipeline based on this." Surprisingly, it completes the project on its own. It organizes a style guide in the CLAUDE.md file, creates the necessary scripts, and so on.

**Step 2: Test pipeline**  
I tell it to produce a test output once. I have it run the built pipeline and actually write one post.

**Step 3: Feedback by natural language**  
Looking at that result, I tell Claude Code the parts I don't like. Fixing things by saying, "The tone here is a bit stiff," "I want to change this part like so."

**Step 4: Feedback by giving the ground-truth**  
(This is the part I think is key.) I give the answer. Even if I keep talking to Claude Code, something isn't quite perfect. About 5–10% will keep differing from what I want. For this part, I edit it directly and create the **ground truth**.

**Step 5: Training (?)**  
I give the corrected **ground truth** back to Claude Code and tell it, "Compare it to what you wrote, analyze what changed, and update the pipeline."

**Step 6: Repeat step 2 ~ step 5**  
After just a few repetitions, a nearly perfect writing bot comes out.


## So what on earth is Training?

### "Training" that runs as a black box

But during this process, I felt something curious. I never looked inside at how the prompt is composed or how the internals change at all. I just looked at the output, gave feedback, and let Claude Code fix it on its own. So this is running as a complete black box.

For me, AI training had always been backpropagation. Defining a loss function, computing gradients, updating weights. But that's only because I know the fact that an LLM is made up of parameters; from a non-expert's perspective, backprop is a black box too.

### Is backpropagation the only form of learning?

Actually, I'd been having this thought continuously.

> "What is an Agent system, in the end? Isn't it just a wrapper that changes a few files inside a project and modifies prompts to leverage the underlying LLM's performance?"

That's true. But what if we view this as a black box? From the outside, it's just a single AI system where you put in an input and an output comes out. Whether the prompt changes inside, or files get modified, you can't tell from the outside. And when you give feedback, a better output comes out next time.

This is... learning, isn't it?

Let's think about what we do when we train a neural network.

1. Put in an input
2. Get an output
3. Compare with the answer and identify what's wrong
4. Update the internal parameters based on that information
5. A better output comes out next time

Let's look again at what I did when building the writing bot with Claude Code.

1. Put in a topic
2. Get a post
3. A human edits the parts they don't like
4. Update the prompt/files based on that information
5. A better post comes out next time

In fact, the two have the exact same structure. The difference is that what gets updated isn't weights but text (prompts, config files). And the updating happens not through gradient descent but through Claude Code's `Claude.md` and the files inside the project.

Of course, text updates have no mathematical rigor. So you can't **guarantee** learning. **But does our brain really learn with mathematical rigor?** Perhaps in the future this will come to be called "AI learning."

### The "learnable system" that an LLM generates

I used to think an LLM was simply powerful as a Generative model. Good at producing text. But what I felt this time is different. The **entire project** that the LLM generates felt like a single **learnable** AI.

The writing bot that Claude Code created:
- improves when it receives feedback
- learns the style when given new examples
- produces output increasingly tailored to the user's taste

This no longer felt like simple "prompt engineering," but like a system capable of modifying itself.


## Wrapping up

### Breaking out of mental frameworks

> In the end, what matters isn't "how it works" but "what it can do."

To be honest, I think I was somewhat trapped in a mental framework. "Learning is backprop," "An AI system is defined by model weights," "Prompt engineering isn't real AI development"... these fixed ideas. But from the outside, you can't tell them apart. Whether gradients flow inside the black box or text gets modified, if the system improves as a result, isn't that exactly the definition of learning?

### Unnecessary ego in the great AI era

As an AI scientist, I'm familiar with things like model architectures, learning algorithms, and loss functions. So I think I had a habit of trying to distinguish "real AI" from "just a wrapper." But at this inflection point of the era, does such a distinction even mean anything? From the user's standpoint, a writing bot made with Claude Code and a model fine-tuned on its parameters are the same as long as they write well. If anything, the former gets personalized faster and is easier to modify.

I find myself thinking I should let go of my ego a bit. The thought that I should have a broad perspective.

### To sum up

I came all the way to these thoughts while making a writing bot for my wife. These days things change so fast that I sometimes get FOMO, but because I believe that changing is always better than standing still, I'm also grateful to have been born in an era like this.

Anyway, maybe a few years from now the very definition of "AI training" will have changed. An era where we train not just via gradient descent, but entire self-modifiable systems. Beyond combinations of parameters and matrix operations, an era may come where the combination of text and pipelines inside it is what's called intelligence.
