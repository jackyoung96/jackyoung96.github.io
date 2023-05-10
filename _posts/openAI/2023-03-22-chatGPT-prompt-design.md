---
layout: post
title: OpenAI 뿌수기 - ChatGPT prompt design
tags: archive
---

🥕 [OpenAI API](https://platform.openai.com/docs/guides/completion/prompt-design) 에서 제공하고 있는 chatGPT 관련 prompt design에 대한 글을 번역한 것입니다.

- [Official doc. of OpenAI for prompt design](#official-doc-of-openai-for-prompt-design)
  - [Rule of the thumb](#rule-of-the-thumb)
  - [Task 별 prompt design](#task-별-prompt-design)
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

   무엇을 원하는지 분명하게 보여줄 것. Instruction, example 등을 모두 활용하여 **원하는 것을 명확하게 설명해야 함**

2. Provide quality data

   충분한 양의 **양질 데이터**를 제공해야 함. 철자 오류 등이 반복되는 데이터는 이를 의도한 것으로 파악할 가능성이 있음

3. Check your settings

   일반적인 상용화를 위해서는 temperature나 top_p 등의 설정을 낮은 값으로 유지해, **deterministic 한 응답을 유도**할 것.

## Task 별 prompt design

### Classification

1. Plain language 를 사용할 것
2. 모든 classification label을 미리 알려줄 것
3. 몇 가지 예시를 포함할 것

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

별다른 prompt design 방법론은 없다. 몇 가지 예시를 준다면 더 좋은 답변을 받을 수 있다.

```
Brainstorm some ideas combining VR and fitness:
```

### Conversation

1. 정체성을 부여하기

   아래의 예시는 AI assistant 라는 정체성을 부여했다. `Woman who works as a research scientist in biology` 와 같은 디테일한 정체성 부여도 가능하다.

2. 동작 방법 지정하기

   아래의 예시는 helpful, creative, clever, and very friendly 로 동작할 것을 지정했다. 이를 제대로 지정해주지 않으면 그냥 단순히 내 말을 따라하거나(mimic), 비꼬는(sarcasm) 등의 행동들이 나오는 것을 막을 수 없다.

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

일대일 대응에 대한 학습이 가능하다. 문제적 남자에 나오는 문제들 같은 것!!

```
Convert movie titles into emoji.

Back to the Future: 👨👴🚗🕒
Batman: 🤵🦇
Transformers: 🚗🤖
Star Wars:
```

### Summarization

Summarize 의 난이도를 결정해 줄 수 있다.

```
Summarize this **for a second-grade student**:

Jupiter is the fifth planet
...
the Moon and Venus
```

### Completion

완성되지 않은 문장 혹은 code 를 넣으면 완성시켜 주는 것.

**Temperature를 낮춰서** 내가 쓴 문장의 intent를 놓치지 않게 하거나 **Temperature를 높여서** 창의적인 새로운 문장이나 code를 작성하게 할 수 있다.

```
Vertical farming provides a novel solution for
producing food locally, reducing transportation
costs and
```

### Factual responses

진실을 대답하도록 하는 것. chatGPT 가 거짓말을 하는 경우가 꽤나 늘어나고 있기 때문에….

1. Ground truth 를 제공 할 것

   예시로 정답을 제공할 때에는 Wikipedia 의 내용을 그대로 복붙해서 ground truth를 정확히 알려주는 것이 중요하다.

2. 낮은 확률일 경우 **“I don’t know”** 라고 대답할 수 있다는 것을 예시로 보여 줄 것

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

Prompt 디자인만 판매하는 사이트도 있음

OpenAI에서 prompt 디자인을 어떻게 해야 하는지 community들을 많이 참고하는 것이 좋음

[Gumroad](https://app.gumroad.com/d/e5008ceda2887a15bf233e97bfb1b7ef)

[openai-cookbook/techniques_to_improve_reliability.md at main · openai/openai-cookbook](https://github.com/openai/openai-cookbook/blob/main/techniques_to_improve_reliability.md)

[OpenAI API](https://platform.openai.com/docs/guides/completion/prompt-design)
