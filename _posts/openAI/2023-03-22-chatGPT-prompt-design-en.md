---
layout: post
title: Cracking OpenAI - ChatGPT prompt design
tags: archive
lang: en
---

🥕 This is a translation of the article on ChatGPT-related prompt design provided by the [OpenAI API](https://platform.openai.com/docs/guides/completion/prompt-design).

- [Official doc. of OpenAI for prompt design](#official-doc-of-openai-for-prompt-design)
  - [Rule of the thumb](#rule-of-the-thumb)
  - [Prompt design per task](#task-별-prompt-design)
    - [Classification](#classification)
    - [Generation](#generation)
    - [Conversation](#conversation)
    - [Transformation](#transformation)
    - [Conversion](#conversion)
    - [Summarization](#summarization)
    - [Completion](#completion)
    - [Factual responses](#factual-responses)
- [References](#references)

# Official doc. of OpenAI for prompt design

## Rule of the thumb

1. Show and Tell

   Clearly show what you want. Use instructions, examples, and everything else to **clearly explain what you want**.

2. Provide quality data

   You should provide a sufficient amount of **high-quality data**. If data has repeated spelling errors and the like, there's a chance it will interpret them as intentional.

3. Check your settings

   For general commercial use, keep settings like temperature or top_p at low values to **induce deterministic responses**.

## Prompt design per task
{:#task-별-prompt-design}

### Classification

1. Use plain language
2. Tell it all the classification labels in advance
3. Include a few examples

```
Decide whether a Tweet's sentiment is positive, neutral, or negative.

Tweet: I loved the new Batman movie!
Sentiment:
```

```
Classify the sentiment in these tweets:

1. "I can't stand homework"
2. "This sucks. I'm bored 😠"
3. "I can't wait for Halloween!!!"
4. "My cat is adorable ❤️❤️"
5. "I hate chocolate"

Tweet sentiment ratings:
```

### Generation

There's no special prompt-design methodology. If you give a few examples, you can get better answers.

```
Brainstorm some ideas combining VR and fitness:
```

### Conversation

1. Assign an identity

   The example below assigned the identity of an AI assistant. A detailed identity assignment like `Woman who works as a research scientist in biology` is also possible.

2. Specify how it should behave

   The example below specified that it should behave as helpful, creative, clever, and very friendly. If you don't specify this properly, you can't stop behaviors like it simply mimicking what I say, or sarcasm, from appearing.

```
The following is a conversation with an AI assistant.
The assistant is helpful, creative, clever, and very friendly.

Human: Hello, who are you?
AI: I am an AI created by OpenAI. How can I help you today?
Human:
```

```
Marv is a chatbot that reluctantly answers questions with sarcastic responses:

You: How many pounds are in a kilogram?
Marv: This again? There are 2.2 pounds in a kilogram. Please make a note of this.
You: What does HTML stand for?
Marv: Was Google too busy? Hypertext Markup Language. The T is for try to ask better questions in the future.
You: When did the first airplane fly?
Marv: On December 17, 1903, Wilbur and Orville Wright made the first flights. I wish they’d come and take me away.
You: What is the meaning of life?
Marv: I’m not sure. I’ll ask my friend Google.
You: Why is the sky blue?
```

### Transformation

```
Translate this into French, Spanish and Japanese:

What rooms do you have available?
```

### Conversion

It can learn one-to-one correspondences. Like the puzzles on the show *Problematic Men*!!

```
Convert movie titles into emoji.

Back to the Future: 👨👴🚗🕒
Batman: 🤵🦇
Transformers: 🚗🤖
Star Wars:
```

### Summarization

You can decide the difficulty level of the summary.

```
Summarize this **for a second-grade student**:

Jupiter is the fifth planet
...
the Moon and Venus
```

### Completion

It completes an unfinished sentence or piece of code that you put in.

You can **lower the Temperature** so it doesn't miss the intent of the sentence I wrote, or **raise the Temperature** to have it write creative new sentences or code.

```
Vertical farming provides a novel solution for
producing food locally, reducing transportation
costs and
```

### Factual responses

Getting it to answer truthfully. Because the cases of ChatGPT lying have been increasing quite a bit….

1. Provide ground truth

   When providing the correct answer as an example, it's important to copy-paste the content from Wikipedia as-is so it knows the ground truth exactly.

2. Show by example that it can answer **"I don't know"** when the probability is low

```
Q: What is an atom?
A: An atom is a tiny particle that makes up everything.

Q: Who is Alvan Muntz?
A: ?

Q: What is Kozar-09?
A: ?

Q: How many moons does Mars have?
A: Two, Phobos and Deimos.

Q:
```

# References

There are even sites that sell only prompt designs.

It's good to refer a lot to the communities about how to do prompt design at OpenAI.

[Gumroad](https://app.gumroad.com/d/e5008ceda2887a15bf233e97bfb1b7ef)

[openai-cookbook/techniques_to_improve_reliability.md at main · openai/openai-cookbook](https://github.com/openai/openai-cookbook/blob/main/techniques_to_improve_reliability.md)

[OpenAI API](https://platform.openai.com/docs/guides/completion/prompt-design)
