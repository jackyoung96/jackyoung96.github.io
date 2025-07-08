---
layout: post
title: Diary - Qwen3의 Hybrid thinking mode
tags: archive
---


2025년 4월 29일 새벽에 Alibaba의 Qwen3 가 릴리즈되었다.
[Qwen3 공식 블로그](https://qwenlm.github.io/blog/qwen3/)

Technical report 가 공개되지 않고, 블로그로만 공개되었기 때문에 학습이 어떻게 이루어졌는지는 아주 제한적으로 공개되었다. 

모델은 총 6개로 MoE 모델 2개와 Dense 모델 6개로 구성되었다.
MoE 모델은 235B (22B activate), 30B (3B activate), Dense 모델은 32B, 14B, 8B, 4B, 1.7B, 0.6B 의 사이즈다.
블로그 글에 따르면 각각 가장 큰 모델만 학습한 뒤, strong-to-weak distillation 을 통해 나머지 모델들을 학습한 것으로 보인다. 아마도 base model 들은 pruning 기법들을 통해서 사이즈를 줄인 것 같다. (예상컨데 절반정도씩 줄여나간 것이 아닐까 싶다)

사실 technical report 가 공개되지 않아서 학습 방법에 대해서는 아직 확실하게 파악하기 어렵다. 그러나 공개된 것 중에 흥미로운 부분이 있었는데, 바로 **Hybrid Thinking mode** 이라는 것이다.

## Hybrid Thinking mode

Qwen3 공식 블로그에 따르면 Qwen3 는 thinking, 그러니까 reasoning 이 필요한 어려운 문제나 정확도를 요구하는 task 를 해결할 때에는 test-time reasoning 을 사용하고, 빠르게 답변이 필요한 경우에는 think 를 끄고 그냥 답변을 제공하도록 학습되었다. 이는 Claude-sonnet-3.7 에서 처음으로 제공했던 방식인데, 내가 알기로 명시적으로 open-source 에서 이를 가능하게 한 것은 처음이다.

그렇다면 어떻게 on/off 를 구현했는가? Qwen3 는 두 가지 방법으로 on/off 를 제어하고 있다.

### 1. `/think`, `/no_think` 명령어

유저나 시스템 턴에 `/think`, `/no_think` 명령어를 명시적으로 사용하면 thinking 을 제어할 수 있다. 예시를 살펴보자.

우선 `/no_think` 옵션 사용한 경우에
> How many r's in blueberries? /no_think

이에 대한 답변은 
> Let's count the number of r's in the word "blueberries" step by step:
>
> 1. Write out the word: b l u e b e r r i e s
> 2. Go through each letter one by one:
> - b – no
> - l – no
> - u – no
> - e – no
> - b – no
> - e – no
> - r – yes (1st r)
> - r – yes (2nd r)
> - i – no
> - e – no
> - s – no
>
> So, there are 2 r's in blueberries.
>
> ✅ Answer: 2.

그리고 `/think` 옵션 사용한 경우에는
> How many r's in blueberries? /think

이에 대한 답변은
> \<think\>
> 
> Okay, so the user is asking how many times the letter "r" appears in the word "blueberries". Let me start by writing down the word to visualize it better. The word is "blueberries".
>
> First, I need to break down the word into individual letters. Let's spell it out: B-L-U-E-B-E-R-R-I-E-S. Wait, is that correct? Let me double-check. B-L-U-E for "blue", then "berries". So "berries" is B-E-R-R-I-E-S. So combining them, "blueberries" is B-L-U-E-B-E-R-R-I-E-S. Let me count each letter one by one.
>
> Starting with the first letter: B (1). Then L (2), U (3), E (4), B (5), E (6), R (7), R (8), I (9), E (10), S (11). Wait, that's 11 letters total. Now, I need to check how many "r"s are there. Let's go through each position:

> 1. B - no
> 2. L - no
> 3. U - no
> 4. E - no
> 5. B - no
> 6. E - no
> 7. R - yes, that's one
> 8. R - yes, that's two
> 9. I - no
> 10. E - no
> 11. S - no
>
> So according to this count, there are two "r"s in "blueberries". But wait, let me make sure I didn't miss any. The word is spelled B-L-U-E-B-E-R-R-I-E-S. The "berries" part has two "r"s in a row, right? So between the two "e"s, there are two "r"s. Let me confirm the spelling again. Yes, "blueberries" is spelled with two "r"s in the middle.
>
> Is there a possibility that I miscounted? Let's see. The letters after the first "blue" are "berries". "Berries" is spelled B-E-R-R-I-E-S. So the two "r"s are consecutive. Therefore, in the entire word "blueberries", there are exactly two instances of the letter "r".
>
> I think that's correct. But just to be thorough, maybe I should write down each letter with their positions:
>
> 1. B
> 2. L
> 3. U
> 4. E
> 5. B
> 6. E
> 7. R
> 8. R
> 9. I
> 10. E
> 11. S
>
> Yes, letters 7 and 8 are both "R", so two "r"s. Therefore, the answer should be 2.
>
> ✅ Answer: 2.
> 
> \</think\>
>
> The word blueberries contains 2 instances of the letter r.
>
> Here's the breakdown:
>
> B L U E B E R R I E S
>
> The letter r appears at the 7th and 8th positions.
>
> Answer: There are 2 r's in "blueberries".

`\<think\>`, `\</think\>` 토큰을 사용하여 reasoning 을 하고, 최종 답변하는 것을 볼 수 있다.

아마도 이는 학습 시 thinking 없는 데이터는 `/no_think`를, 있는 데이터는 `/think` 을 사용하여 학습한 것으로 보인다.

즉, **스위치 토큰을 학습에 사용하여, reasoning On/Off 를 제어하는 방법이라고 볼 수 있다.**

### 2. enable_thinking (apply_chat_template)

또 다른 방법은 `enable_thinking` 이라는 파라미터를 사용하는 것이다. Qwen3 블로그를 보면 아래와 같은 방법이 나와 있다.

```python
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switch between thinking and non-thinking modes. Default is True.
)
```

과연 `enable_thinking=True` 옵션을 주면 무엇이 달라지는 것일까? `tokenizer_config.json` 을 보면 Qwen3 의 chat_template 이 있다. 여기서 해당 옵션을 통해 달라지는 부분을 확인할 수 있다.

```
{%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\\n\\n</think>\\n\\n' }}
{%- endif %}
```

즉 enable_thinking 이 false 인 경우 \<think\>\</think\> reasoning 이 없는 reasoning 파트를 assistant indicator 뒤에 강제로 붙여 넣어서, reasoning 이 없는 답변을 생성하는 것이다. 이는 reasoning 이 assistant indicator 바로 뒤부터 시작한다는 점을 활용한 것인데, 개인적으로 굉장히 스마트한 방법이라고 생각한다.

## Conclusion

Qwen3 는 두 가지 방법으로 Hybrid thinking mode 를 구현하였다. 첫 번째 방법인 `/think`, `/no_think` 명령어는 유저가 API 를 사용하거나 chat 서비스를 활용할 때 명시적으로 제어할 수 있는 방법이지만, 학습을 통해 구현되는 것이기 때문에 실제로 `/no_think` 를 사용하더라도 reasoning 을 할 가능성이 있다 (확률적으로는 낮겠지만, 어디까지나 가능성은 있다). 반면 두 번째 방법인 `enable_thinking` 은 빈 thinking 부분을 강제로 붙여버림으로써 reasoning 을 강력하게 꺼버리는 방법이다 (물론 이 또한 확률적으로 두 번의 thinking 이 나와버릴 가능성은 있다) 

아무튼 두 방법 모두 아주 간단하면서도 스마트하게 hybrid think mode 를 구현하였다. 이후 모델 학습에 참고해서 사용해 봐야겠다.