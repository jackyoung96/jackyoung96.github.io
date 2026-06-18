---
layout: post
title: Diary - 2024 OpenAI Dev Day Attendance Review
tags: archive
lang: en
---

OpenAI Dev Day 2023. My company said they would send me, but OpenAI cut me because too many people had applied. This year it was held in London and Singapore in addition to SF, and I attended OpenAI Dev Day 2024, partly to meet a friend who is doing a postdoc in Singapore. Maybe because I went on my own dime, it felt like an even more meaningful experience, so I'm leaving a review.

## Intro

OpenAI Dev Day 2024 was held at the Pan Pacific Hotel. It looked like this. Singapore has a lot of these distinctive hotels with holes punched all over them and tons of plants growing inside.

![image](https://github.com/user-attachments/assets/0f39df5b-7ef1-40af-926c-b4a90cd555ef)

I arrived at the hotel around 11, and OpenAI staff warmly greeted me right from the lobby. The Dev Day held in Singapore apparently doubled as the opening event for OpenAI's Singapore unit. They said "We are hiring!!" but... rumor has it they only take PhDs who'll answer the phone 24 hours a day...

![image](https://github.com/user-attachments/assets/99a2fab7-5c01-4e34-be32-9cbdb102a8d1)

The event was held on (I think) the 19th floor, and as soon as you walked in there was a box prepared for you to write down any questions and drop them in. I didn't write down a question, but I did grab a piece of paper.

![image](https://github.com/user-attachments/assets/993a3b02-fb05-4218-b384-9ea1a370b40d)

## Demo booths

The event started at 12:30, and from 11:00 to 12:30 there was time to walk around the demo booths or wait while having a light lunch in the networking zone. There were 6 demos in total: the video generation model Sora, the image generation model Dall-e, the audio mode of gpt-4o, and demos related to the chatGPT API covering distillation, evaluation, reasoning and coding ability, and the realtime API. The chatGPT-related parts were also covered in the later talks, and I asked about various things I was curious about.

### Sora

Sora is a video generation model, and you can find detailed information at [https://openai.com/index/sora/](https://openai.com/index/sora/). The demo just played the videos that are already posted on the actual homepage. I asked them what on earth the model size is, and they answered,

> **Each video here uses a different model. The simple videos with one person use a small model, and things like the road video with a crowd of people use a big model.**

That's what they told me. Anyway, Sora hasn't been released via API, and if you look very closely there are spots where the detail breaks down a bit. They probably cherry-picked the videos posted on the homepage, and I suspect they plan to develop it further before the API is released.

### Dall-e and audio mode

This one is... it feels like it has become an ordinary technology by now. It was just a simple image generation and a demo of generating jokes and haiku with gpt-4o, but the funny thing is they put a lot of effort into it.

![image](https://github.com/user-attachments/assets/c2e2d70b-0a87-4255-8f79-6e1d47f94914)

Was it really worth going this far?? lol You press a button labeled "Joke" and it generates a joke. Did they really need to...??

### distillation and evaluation

[Distillation](https://platform.openai.com/docs/guides/distillation) and [Evaluation](https://platform.openai.com/docs/guides/evals) are features released this time. Distillation is a feature for training a small model using a big model, with the advantage that you can train with data made up of prompts only. Evaluation is a feature where you upload a dataset for measuring the performance of a fine-tuned model, select or write your own metric, and it does the evaluation for you.
Honestly, if you have some LLM knowledge, these features aren't strictly necessary. The fine-tune API alone is enough. But they genuinely made it possible to do training and evaluation with no code. The Evaluation side in particular has quite a good UX. I find myself thinking I should refer to this a lot when updating our internal LLM evaluation page.
Distillation can also be done by hand, but it reduces the hassle of generating all the answers with gpt-4o, building the dataset again, and feeding it into the fine-tune API. And the feature I thought was the best is that you can perform distillation using your API call history data! If I had built an application using the OpenAI API, then without any separate work of collecting log data, it lets me store the call records on OpenAI's server and go straight to fine-tuning with just a few clicks.

### real-time API

gpt-4o apparently processes input and output based on audio tokens (I had guessed this, but I confirmed it's true!!). The demo used [https://github.com/openai/openai-realtime-console](https://github.com/openai/openai-realtime-console) as-is.
The most puzzling thing when they said it uses audio tokens is that text comes out alongside the output, and the real-time API has a separate audio-only mode and an audio-text output mode. One audio token is apparently about 500 ms, and since the per-second token generation speed is much faster than that, audio tokens keep piling up in the buffer. So I figured it might also be possible to generate text tokens alternately with them.

<img width="488" alt="image" src="https://github.com/user-attachments/assets/a9608036-0ab5-4e7c-96d9-b272d4fd0043">

If you actually watch the demo, the text output comes out before the sound is generated. This is because while the audio tokens are piling up in the buffer, the text just gets streamed out.
It was really interesting that you could interrupt it in the middle, and this part was simply done through threshold handling. On the input side too, it piles up audio in a buffer and when it exceeds the threshold it stops the output and pushes in the data in the buffer.

### Reasoning

The reasoning ability, which improved enormously starting with o1-preview, shows considerable prominence in coding. The demo was creating a simple snake game with prompting only, but honestly it looked like a boring game, so it was a bit disappointing.
I asked what they think the difference is between o1-preview and CoT, and it was interesting that they answered it's in consistency. In the case of CoT, it generates the reasoning and the final answer step by step, maintaining consistency. But o1-preview, even after producing an answer once, will admit that it was wrong and revise it. **Easy problems can be answered in one shot, but for hard ones, isn't moving forward by continually getting it wrong and revising the way humans solve problems?** Explaining the difference with a single word, consistency, gave me a good insight.

## Speech sessions

From 12:30, the main event began with a kick-off. After the kick-off, there were sessions introducing 3 features for 30 minutes each.

![image](https://github.com/user-attachments/assets/1fdb404e-6175-4604-b397-6ca9722bb466)

### Kick-off

It started with a brief greeting and a demo, and the demo was insane... They prompted it to write code that could fly a drone with the keyboard, and they actually flew it right there in the session hall!!

![image](https://github.com/user-attachments/assets/56e5ae21-ae9a-4e2e-9b1c-7ea4cd0eeb1b)

It made me think they really know how to make an impact. Of course, anyone who has flown a DJI drone knows the driver is quite well-built, so the code isn't hard. Still, it was enough to draw out those cheers.

![image](https://github.com/user-attachments/assets/55ae091f-a0d8-4ec1-ad13-9eae6055d4ce)

They also announced a version update of gpt-4o (gpt-4o-1120). It gets updated roughly every 3 months, and I'm curious what the meaning of the 3-month cycle is. Anyway, they said it took 1st place again on Chat Arena.

### Structured output

[https://platform.openai.com/docs/guides/structured-outputs](https://platform.openai.com/docs/guides/structured-outputs)
As agent development becomes more important, there are increasingly many cases where you need to receive JSON-type output from the LLM that can execute an API, rather than natural language output. The OpenAI API supports something called JSON mode, which forces the answer into JSON form no matter what.
In this case, hallucination becomes a problem. For example, if only numbers should go into a num field but it spits out output with a string in it, the application layer using that API has to handle all the exceptions. To solve this, they introduced structured output strict mode.

Let's say we call the gpt API using a tool like the one below.
```json
{
    "name": "get_weather",
    "description": "Fetches the weather in the given location",
    "strict": true,
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The location to get the weather for"
            },
            "unit": {
                "type": "string",
                "description": "The unit to return the temperature in",
                "enum": ["F", "C"]
            }
        },
        "additionalProperties": false,
        "required": ["location", "unit"]
    }
}
```
If you put in `strict = true` here, location gets filled with a string, and unit gets filled with either "F" or "C".

As for how on earth this is possible, it's apparently because they use a mask decoding strategy. For example, the token that comes after the `"unit": ` token must necessarily be `F` or `C`, so the idea is to **sample the next token only from those two tokens**.
Likewise, the token after `F` or `C` can be `}`, `}\t`, `}\n`, and so on. They didn't explain it in detail, but up to what I heard, I get the sense they approached it quite heuristically. The supported format types are only String, Number, Boolean, Integer, Object, Array, Enum, and anyOf.

While developing this feature, one research-side problem arose: when doing mask decoding, tokens like `\t` and `\n` are always included, so apparently there was a phenomenon where these came out infinitely. Repetition phenomena occur frequently in the LLM field too, and usually this kind of problem is caused by underfitting, so increasing epochs or increasing data naturally solves it. It can also occur when the model is too small, but surely that's not the case for gpt...

### Distillation

Unfortunately, there was no research-side talk about the distillation methodology. I'm curious whether alignment tuning is done via distillation inside OpenAI... such a shame...
The distillation side is also used a lot in inference optimization these days, like speculative decoding, and it's used a lot for shipping small models into commercial services.

There was mention of situations where it's good to apply this distillation API to an application.

![image](https://github.com/user-attachments/assets/394efa86-b62d-4751-b7b2-5c8d80f8ec83)

In the end, knowledge or reasoning ability inevitably decreases significantly when distilled, but for simple and limited situations like sentiment analysis or keyword extraction, it can play a big role in cost savings. (This is obvious, of course.) Another interesting point was the mention that **distillation performance was highest with around 1K of data, and conversely there's a high chance performance drops when you put in 1M of data**. Inferring from this statement, putting in too much data from a specific domain can cause a lot of forgetting that drops overall performance, and given that gpt-4o-mini's model size is estimated to be between 8B and 60B, we could guess that the room for additional learning in the SFT stage of a model of that size is around 1K of data.
Ultimately, the significance of the distillation feature is that application developers can create, with just a few clicks, a model that approaches gpt-4o while costing as much as gpt-4o-mini. The price is a whopping 1/16, after all. However, since you can't know how the distillation happens, there's a limit to trying out various techniques.

### Realtime API

The Realtime API is a feature that emerged while building an LLM (gpt-4o) capable of audio input / audio output to handle the existing STT -> LLM -> TTS audio input/audio output pipeline. The biggest problem with the existing methodology was that the LLM had to wait until STT finished, and TTS had to wait until the LLM's output came out. This produced a latency of at least 1 second, which worsened the user experience in things like voice conversational chatbots. That's why you could give a task instruction to an AI speaker, but natural conversation was impossible.

However, as audio tokens became usable in gpt-4o, latency converged to almost 0. gpt-4o's throughput is about 109 tokens/second, meaning it takes as little as about 10ms until the first audio token comes out. We usually call it realtime when it comes in within 100ms, considering that humans don't feel the latency much, so it became an API capable of genuinely realtime voice conversation.

At Dev Day, they showed a demo of directly ordering a mango tart using this API. They input a prompt for the API to call the tart shop and have it delivered to here by such-and-such time. At this point, they also input a tool for calling the tart shop (written like the below; it wasn't exactly in this form, this is reconstructed from memory).

```json
"tools": [
    {
        "name": "call",
        "description": "Make a call",
        "parameters": {
        "type": "object",
        "properties": {
            "number": {
            "type": "string",
            "description": "Number of target phone",
            }
        },
        "required": ["number"]
        }
    }
]
```

So it immediately made the call, and the call came to the cell phone of the tart shop owner who was on stage. Then it began a real-time conversation, and based on the delivery location, time, quantity, and budget limit written in the prompt, it went through about 10 turns of conversation with the tart shop owner and eventually completed the order.

The first thought I had watching this was: instead of voicemail, you could use the realtime API to make a bot that answers the phone for me. The second thought was: automating voice phishing would become possible... (which is the worst). Anyway, this API seemed to have quite limitless potential. And beyond a simple chatbot, there's also a gpt-4o-realtime-preview API with enhanced reasoning capabilities. (Though I'll have to think about where it would be used.)

The price is a bit of a problem. Output tokens are apparently about 24 cents per minute. Combined with input tokens, it's nearly 40 cents per minute, i.e. about 500 won. I think it's quite pricey to ship as a large-scale user service. Because unlike text, voice can just keep going on as a conversation? You'd have to use it well, in a limited way.

### Firechat

There was a fireside chat session with Mark Chen, OpenAI's SVP. Last year in SF it was Sam Altman, so I had high hopes, what a shame. There was about 20 minutes of Q&A with the host, and they didn't take live questions separately. I've organized the questions and answers that left an impression.

#### *Q1*. When and in what form do you think AGI will arrive?
A1. When OpenAI first released chatGPT, people said it would be AGI if it solved math problems. When gpt-4 started solving math problems, they said it's AGI if it solves PhD-level problems. When o1-preview started solving PhD-level problems, they bring up domains o1-preview can't solve and say it's AGI if it solves those. As models advance and the times change, the definition and criteria for AGI keep changing. We will keep repeating these iterative processes, and I think if we conquer all the benchmarks one by one, the AGI era can someday arrive.

#### *Q2*. How did you come up with the idea for o1?
A2. Two years ago, I thought about what was lacking as an AGI. (Two whole years ago...?? Amazing.) When I pondered whether humans think the way gpt-4 does, I concluded they don't. In the end, I figured we'd have to repeat the process of thinking, getting it wrong, correcting it, and producing the right answer again, and we were able to produce a result by going through the process of collecting and training on that data.

#### *Q3*. What is the most important thing when doing research at OpenAI?
A3. Actually, when doing moon-shot projects that newly create something that didn't exist in the world, it's important to protect the researchers. For at least 3-4 months while the project is ongoing, results that feel meaningless will keep coming out, and funding will keep being spent without any progress. Even so, it's about continually maintaining people's excitement level for the project within the organization. As you do that, suddenly someone finds an enormous breakthrough. You have to create an environment where you keep believing and waiting.

#### *Q4*. With the advent of o1, human coding ability won't matter much, so what do you think?
A4. Coding doesn't matter? I'm not sure. It's obvious, but a person who understands more deeply uses it better. Of course, with the advent of o1, the difficulty of coding will drop a lot, and productivity will rise even more. Shouldn't we study coding in the direction of using these tools to boost productivity even further?

#### *Q5*. What do you think is OpenAI's biggest strength?
A5. Many people probably think OpenAI is technology-centric, but actually I think it's an extremely human place. OpenAI runs on excitement. I think it's because everyone is deeply excited that we can produce breakthroughs. The biggest strength is that everyone is so passionate and excited.

## Reception

When all the events wrapped up, everyone moved together to the reception venue. The location was LAVO, situated at the Marina Bay observatory!! It really made me think OpenAI is a rich company. They provided buffet-style food and unlimited alcohol, and we had time to network with the people around us.

![image](https://github.com/user-attachments/assets/c9832a02-236c-46d1-b86c-2b4ac927eee1)
![image](https://github.com/user-attachments/assets/767acefb-8200-434c-a071-9da79c1bc2ca)

The slightly disappointing part was that it was a bit hard to talk with the OpenAI folks? The OpenAI people were clustered together talking among themselves, so it was hard to have more conversations. (Though honestly I still could have gone up and talked.)
Luckily, in the middle I got to briefly meet Jillian Khoo, who was a speaker for the Distillation session, and we chatted about this and that and talked about how the future might unfold. I couldn't ask deeply about the technical content of LLMs (she was in charge of the application side of the distillation API), but I remember things like the fact that they're keeping an eye on agent-related features going forward, and that OpenAI's Singapore unit opened today and is hiring.

## Conclusion

It was a really fun experience and an opportunity to broaden my horizons. Of course, it was an event for people who turn OpenAI's APIs and products into applications rather than for AI researchers, but it felt like a chance to indirectly find out what the front-runner of the LLM field is thinking. So that someday I can catch up to OpenAI! Let's work hard.
