---
layout: post
title: "Project review - Knowledge distillation from powerful LLM, Alpaca and Koala"
tags: archive
---

디피스트 Season 13 의 호스팅으로 발표했던 내용을 정리합니다.  
본 글의 목표는 크게 2가지입니다.
- Transformer, GPT에 대해 알고만 있었지 논문이나 내용은 보지 않은 사람에게 text generation model의 기초 설명
- LLM 으로 downstream task를 풀고 싶은데 잘 학습된 공개된 모델이 없어서 고민중인 사람을 위한 새로운 방법론 제안

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

Large Language Model (LLM) 에 대해서 시간 순서대로 알아보겠습니다. 사실 LLM이라 하면 GPT 뿐만 아니라 BERT나 T5 계열도 있지만, 일단은 text generation model에 집중해서 GPT 모델만을 살펴보겠습니다.

### Transformer

[Attention is all you needs](https://arxiv.org/abs/1706.03762) (NeurIPS17’) 논문에서 제시된 Transformer 모델은 기존 Recurrent Neural Network (RNN) 에서는 불가능했던 병렬화 문제를 해결하는 방법론을 제시했습니다. 

<img width="513" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/96e7e461-97d4-42b6-9df0-81222b89dce9?style=centerme">

Input 순차적으로 입력해야 하는 RNN 구조와는 다르게, Transformer는 input sequence의 attention을 **동시**에 계산하는 방식을 사용합니다. Transformer 설명만 한나절이니 여기서는 패스하도록 하겠습니다.

<img width="604" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/cbfad0c7-9e52-4bf3-bb5f-b980fcb3b634?style=centerme">

아무튼 Transformer를 사용하면서 얻은 이점은 병렬화가 가능하기 때문에 더 긴 sequence에 대한 대응이 가능하게 되었고, 모델을 더 크게 만들 수 있게 되었습니다. RNN은 병렬화가 안되기 때문에 모델을 크게 만들었다가는 학습도 물론이고 inference도 엄청나게 오래 걸렸죠.  
또 해보니까 성능도 좋았습니다 (역시 딥러닝은 성능만 좋으면 됩니다). 따라서 NLP 쪽은 (vision 까지도 확장) Transformer 기반의 모델들이 득세하게 됩니다.

### GPT-1

[Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) (Arxiv18’) 논문을 통해 OpenAI는 GPT-1 모델을 공개합니다. 이 모델은 Transformer의 Decoder 구조만을 사용한 모델입니다.

<img width="377" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/c508f7cc-afb0-4a02-a0f0-9ca0615d8281?style=centerme">

위 그림처럼 Unsupervised pre-training (PT) 과 Supervised fine-tuning (SFT) 두 단계로 나누어 학습을 진행하였습니다. PT의 경우 단순히 token의 sequence를 넣고 그 다음에 나올 token을 맞추는 방식으로 학습이 진행되기 때문에 annotation 작업이 필요하지 않습니다. SFT의 경우 task에 따라서 labeling이 된 dataset을 사용합니다.  
사실 이 구조는 Computer vision에서 사용하는 방식과 크게 다르지 않습니다. ResNet 같은 pre-training 모델을 freeze 하고 뒷단에 linear 모델을 붙여서 downstream task 를 푸는 방식을 그대로 차용한 겁니다.  
<br>
재미있는 것은 zero-shot behavior 라는 것이 등장하기 시작했습니다. Fine-tuning 없이 Pre-training과 적당한 heuristic 방법론만으로도 NLP task 들에서 대한 적당한 성능을 보여준 것입니다.

<img width="415" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/642b4c77-5c8b-4dea-a564-04d32d72a3cb?style=centerme">

Sentiment analysis를 예시로 들자면 문장을 넣은 후 그 다음 token에 positive/negative 라는 단어가 등장할 probability 만을 비교하여 판단하는 방식을 사용했습니다. 이것만으로도 거의 70%에 육박하는 성능을 보여줍니다.  

이는 GPT 계열의 언어모델이 값비싼 annotation 작업을 생략하면서도 좋은 성능을 낼 수 있다는 가능성을 보여줍니다.

### GPT-2

[Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (Arxiv18’) 논문은 GPT-1에서 발견되었던 zero-shot behavior를 극대화하려는 시도를 보여줍니다. 모델의 capacity를 더 키우고, pre-training에 사용되는 데이터를 더 크고 퀄리티있게 구성했죠.  

우선 모델의 크기는 거의 13배 가까이 키웠습니다. 기존의 117M 사이즈였던 GPT-1에 비해 GPT-2 는 1.5B 모델을 사용했습니다.  
또한 WebText라는 새로운 데이터셋을 구축했습니다. Task-specific data가 필요하지는 않았지만, 데이터의 개수는 많아야 했기 때문에 web crawling을 통해 데이터를 얻었습니다. 또한 퀄리티 유지를 위해 3가지 기준을 통해 데이터셋을 구성했습니다.
- Reddit 에서 Karma(따봉) 3개 이상 받은 데이터
- Wikipedia는 Training/Test overlapping 이 심하므로 제외
- Only English data
Wikipedia 가 인상적인데, 이게 evaluation 성능에 영향을 너무 많이 준다고 합니다. 왜냐면 reddit의 답글들에도 wikipedia에서 인용한 문장들이 너무 많기 때문입니다. (사실상 질문들에 대한 정답은 다 wikipedia에 있죠...)  

<img width="931" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/4d05f52a-c032-4096-9125-4a0046949969?style=centerme">

예상대로 모델의 사이즈를 키우자 zero-shot behavior도 함께 증가하기 시작했습니다.

<img width="677" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/8edcf541-1d32-4441-a41f-a88af656f829?style=centerme">

단순히 성능이 증가한 것을 넘어서 GPT-2의 성능은 몇몇 task에서 기존 SOTA 방식들을 **zero-shot으로 능가**하는 엄청난 퍼포먼스를 보여줍니다. 물론 Summarize, Translate, Fatual QA 등에서는 그리 뛰어나지는 않았습니다. 특히 Fatual QA에서 대답은 4.3%만 정답이었습니다. (이때부터 Hallucination 냄새가...)

### GPT-3

[Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (NeurIPS20’) 논문을 발표하면서 OpenAI는 지금의 chatGPT 시대의 서막을 열었습니다. GPT-2에서 zero-shot behavior가 모델의 크기와 데이터셋의 퀄리티에 비례한 사실을 확인했고, 이를 직접적으로 실천하여 GPT-3를 만들어 냈습니다. 훨씬 더 큰 capacity, 더 고퀄의 데이터셋, 그리고 few-shot learning 기법을 사용해서 말이죠.  

우선 모델의 크기는 175B 까지 늘렸습니다. 갑자기 100배를 늘려버린 건데요. 175B 면 대충 모델크기만 800GB 정도고, 이를 학습시키기 위해서는 최소 A100 160장이 필요합니다. A100이 천오백만원 정도니까, GPU 가격만 약 240억 정도가 들겠네요. 게다가 저정도면 하루 전기료만 한국 기준으로 600만원 정도 나옵니다. 연구실에서는 감당할 수 없죠.  

데이터도 마찬가지입니다. 인터넷에서 긁어모은 45GB 데이터셋을 잘 정재하여 570GB의 고퀄리티 데이터셋을 만들어냈습니다. 이 때 책, Wikipedia 등의 high-quality reference corpora 를 집중적으로 수집하고, duplication 을 최대한 제거하는 작업을 수행합니다.  

마지막으로 few-shot learning 기법은 downstream task를 풀 때 zero-shot이 아니라 몇가지 example을 함께 제시하고 문제를 풀게 하는 방식입니다. 

<img width="514" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/f66f1be3-0ba8-48ea-befd-3185c3d7c27c?style=centerme">

cheese를 프랑스어로 번역하더라도, 몇 가지 예시를 주면 문제를 더 잘 이해할 수 있겠죠. (사람의 메타인지나 다름없죠 이제는)  

GPT-3는 기존의 언어모델들이 해결하지 못했던, 인류의 영역이라고 생각되었던 부분들을 해결하기 시작합니다. 덧셈뺄셈이 가능해지고, Anagram 을 풀 수 있게 되었습니다.

<img width="487" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/48242e0f-d211-4022-960d-749ffbf027ce?style=centerme">

<img width="619" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/5e77489e-a44a-4427-81fd-47cfbc25215a?style=centerme">

### Instruct-GPT

[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (Arxiv22’) 논문은 GPT-3 에서 한 발 더 나아갑니다. GPT-3 가 놀라운 모습을 보여주기는 하지만, 사실 사람이 원하는 결과물과는 약간 달랐습니다. 그 이유는 사실 objective 가 다르기 때문인데요. GPT는 앞의 token sequence를 보고 그 다음에 나올 token을 예측하는 것이고, 사람이 원하는 언어 모델은 자신들에게 도움이 되거나 (helpful), 해롭지 않고 (harmless), 진실된 (truthful) 답변을 내놓는 것입니다. 따라서 OpenAI는 강화학습을 사용하여 User aligned LLM을 만들고자 했고, 그 결과물로 InstructGPT를 제안합니다.  

<img width="611" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/4cad218e-a0fa-475f-894c-1586ea17bb1e?style=centerme">

InstructGPT는 총 세 단계로 구성됩니다. Supervised Fine-tuning (SFT), Reward model training, Optimizing policy 입니다. 각각을 자세히 살펴보겠습니다.

**SFT**  

일단 OpenAI는 모든 task를 Instruction으로 만들기로 합니다. Instruction 이란 다음과 같은 형태입니다. 

<img width="799" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/1c9db7b3-a320-4d46-8295-fa6a0b92e1cb?style=centerme">

학교 안전에 대한 글을 쓰도록 언어 모델에게 요청하는 예시를 들어보겠습니다. 이 때 Instruction은 "writing an essay about following topic", instance input은 "school safety"가 되겠습니다.  

OpenAI는 긁어모은 데이터를 이러한 형태로 만들어냅니다. Upwork와 ScaleAI에서 40명의 labeler를 full-time으로 고용했고, 13K개의 Instruction set을 만들어냅니다. (꽤나 비쌌을 것 같네요)  

만들어진 Instruction set을 Fine-tuning 시켰습니다. 16 epoch 동안 진행되었다고 하네요.

**Reward model training**  

Reward model은 [Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741) 에서 주장한 PbRL을 이용해 학습합니다. SFT를 거친 모델이 생성하는 여러개의 output들에 대해서 human labeler가 순위를 매기면, 이를 이용해 각 prompt의 reward를 추정하는 방식입니다. (PbRL에 대한 자세한 내용은 [여기](https://jackyoung96.github.io/2022/01/01/PbRL_review1/))  

이 또한 Human labeler 40명이서 33K개의 데이터에 대해 순위를 추정했습니다. 엄청나게 돈과 시간과 노력이 들어갔겠죠? Reward model의 경우 6B LLM 을 사용했다고 합니다. 175B를 사용하려는 시도가 있었으나 reward model은 크기가 커지면 unstable해지는 문제가 발생한다고 하네요.

**Optimize policy**  

마지막으로 학습된 reward model과 강화학습 알고리즘 (PPO) 를 이용하여 LLM 을 optimize 합니다. 이 때는 따로 labeling은 필요 없고, OpenAI GPT-3에서 사용자들에게 얻어낸 prompot 31K개를 사용해 학습을 진행했습니다.  

InstructGPT에 대한 결과는 생략하겠습니다. 왜냐면 사실상 정성적인 평가밖에 되지 않기 때문입니다. InstructGPT의 결과가 GPT-3에 비해 훨씬 **사람 선호도가 높았다**는 점만 짚고 넘어가도로 하겠습니다.

InstructGPT는 general performance의 감소를 최소화하면서도 사람에게 잘 aligned 된 형태의 답변을 생성하는 언어 모델입니다. 다만 그 학습과정에서 필요한 리소스가 꽤나 많았죠. 연구실 레벨에서는 하기 힘든 작업들로 생각됩니다.

### GPT-4

그리고 2023년 2월 [GPT-4 Technical Report](https://cdn.openai.com/papers/gpt-4.pdf) (Arxiv23’)와 함께 GPT-4가 공개되었습니다. Image 가 추가된 Multi-modal 데이터 처리가 가능하고 Hallucination 문제를 개선했으며 더 높은 퀄리티의 답변을 내놓는 모델입니다. 아쉽게도 Model size, architecture, HW info, Dataset, Training process 전부 공개하지 않았습니다. 추측으로는 1~100T 모델 사이가 거론되고 있고, GPU 약 30K개를 model serving 에 사용하고 있다고 합니다.

## Closed-source LLMs

<img width="778" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/83151f4e-f790-4765-ad88-4992df4d3ae0?style=centerme">

현재 서비스로 제공되고 있는 가장 성능이 좋다고 알려진 LLM 3가지 - OpenAI의 chatGPT, Google의 Bard, Anthropic AI의 Claude - 는 모두 closed-source 입니다. 즉 모델을 공개하지 않고 API call 형태의 서비스만 제공합니다.  

Pre-trained weight 공개하지 않고, 학습용 데이터셋 공개하지 않으며, SFT, RLHF 도 굉장히 Expensive (석사 이상 고학력자로 annotator 구성) 한 상황에서 Pre-trained LLM을 이용한 다양한 downstream task 연구가 어려운 상황입니다.  

또한 이러한 LLM은 사실 굉장히 비효율적인 학습 방식을 사용하고 있습니다. 인간이 평생 보는 Text 데이터의 양이 약 0.16GB 정도라고 하는데, GPT-3는 450GB의 데이터를 사용했습니다. 사람은 교육과정을 통한 학습, 또는 선생님으로부터 배우는 방식을 사용합니다.  

따라서, Open-source LLM 이면서 효율적인 학습 방식 (knowledge distillation) 으로 모델을 빠르게 학습시킨 두 가지의 프로젝트를 소개해 보겠습니다.

## Knowledge distillation from openAI's LLM

<img width="1176" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/d2363719-cb76-4644-92c5-5e49374ceee7?style=centerme">

두 모델 모두 정말 따끈따끈하네요. Stanford의 Alpaca는 2023년 3월 13일에 공개되었고, Berkeley의 Koala 는 2023년 4월 3일에 공개되었습니다.  

두 모델은 모두 [LLaMA](https://arxiv.org/abs/2302.13971) (Arxiv23') 라는 Meta가 공개한 open-source LLM에 기반을 두고 있습니다. 2023년 2월 4일에 공개된 아주 따끈한 모델입니다. 놀랍게도 Meta는 이 모델을 API 를 통한 서비스 제공이 아닌, 연구용으로 제공했습니다. “further democratizing access” 라는 표현 사용하면서요. 약간 요새 회사가 말리니까 이런 방식을....(읍읍..)  

LLaMA는 7B, 13B, 33B, 65B 모델이 공개되어 있습니다. 논문에 따르면 4배의 정제된 데이터를 사용했고, 13B 모델이 일부 task에서 GPT-3 (175B) 와 유사한 성능을 보였다고 하네요. 하지만 여전히 GPT-3.5나 GPT-4와 비교했을 때에는 낮은 성능을 보여줍니다.  

Alpaca와 Koala는 LLaMA에 knowledge distillation을 적용해 OpenAI의 모델에 필적하는 성능을 갖추면서도 크기가 작은 모델을 학습해낸 결과물입니다. 두 모델 모두 1) 적당히 강력한 open-source인 pre-trained LLM과 2) 고품질의 Instruction 데이터 이 있다면, 강력한 open-source LLM 을 만들 수 있을 것이라는 가설로부터 시작합니다. 

<img width="989" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/1a9134d2-d7cd-442b-b462-da95699935fd?style=centerme">

### Alpaca

Alpaca는 self-instruct 라는 방식을 사용해 Instruction dataset을 만들어 냈습니다. [SELF-INSTRUCT: Aligning Language Model with Self Generated Instructions](https://arxiv.org/abs/2212.10560) (Arxiv22’) 은 LLM을 이용해 data을 생성해내는 방식을 제안합니다. 

<img width="682" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/9fecfe70-413e-422a-a68e-bbddeda20d34?style=centerme">

위와 같은 방법으로 175개의 seed instruction 만을 사용하여 52K개의 instruction 데이터를 만들어냈는데요. 데이터 생성은 5가지 step을 통해 이루어집니다. 

***Step 1 : Seed Instruction***  

사람이 직접 175개의 Instruction dataset를 작성합니다. 

<img width="814" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/0feb7fb4-7065-41ad-bdb5-4c310a9ac35b?style=centerme">

***Step 2 : Task generation***  

6개 seed task, 2개 generated task를 이용해 새로운 task 8개를 생성합니다. 아래와 같은 Template을 사용하고 GPT-3 의 completion 모델인 `text-davinci-003` 을 사용합니다. 

<img width="478" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/d0ce63e0-2d39-48f7-af7f-24a6bb80275f?style=centerme">

아래와 같은 모습으로 새로운 task 들이 잘 생성되는 것을 볼 수 있습니다.

<img width="592" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/0c094496-71ef-49d4-a2ca-36aa2fc00b2f?style=centerme">

***Step 3 : Classification task identification***  

12개의 classification, 19개의 non-classification task 이용해 생성된 task가 classification task 인지 아닌지 판단합니다. few-shot learning (few-shot이라기엔 31개나 쓰지만) 방법론을 사용합니다. 아래는 3개정도만 써 본 예시입니다.

<img width="312" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/90973b3d-bf10-4da5-8697-de03d5761fea?style=centerme">

***Step 4 : Instance generation***

주어진 task를 기반으로 instance input-output을 생성합니다. 다만 Non-classification 문제의 경우 input-output 순서로 생성하고, classification 문제의 경우 output-input 순서로 생성하도록 template을 구성합니다. 

<img width="1109" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/dbf2e8de-1a24-4fb5-bad5-597656ed32e2?style=centerme">

그 이유는 classification task의 경우 input을 먼저 생성하면 output이 한 개만 생성되는 경우가 많이 발생했다고 합니다. Heuristic 한 접근법이네요.  

***Step 5 : Filtering***

마지막으로 생성된 instruction 중 아래의 조건을 만족하는 것들만 instruction pool 에 추가합니다. 
- ROUGE-L < 0.7 일 때에만 dataset pool에 해당 instruction을 추가 (ROUGE-L : 문자열 매칭 정도를 나타내는 metric)
- images, pictures, graphs 키워드가 있는 경우 제외 (Text로 제대로 표현되지 못했을 가능성 있음)
- Instance 의 input이 같은데 output이 다른 경우 제외

<br>
이렇게 생성된 52K의 self-instruct 데이터셋은 충분한 diversity와 충분한 quality를 보장한다고 합니다. 

<img width="376" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/3164b523-f4ca-46a9-ab9b-b09a20b8c9a8?style=centerme">

<img width="647" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/049c494b-6718-42cc-9ae5-b33bf6bc5d61?style=centerme">

Alpaca는 LLaMA 7B 모델에 text-davinci-003 (GPT-3) 를 이용한 self-instruct 데이터를 활용해 fine-tuning 작업을 진행하였습니다. 52K 의 Instruction을 self-instruction 방식으로 생성하는데 OpenAI API call 비용으로 약 500\$를 사용하였고, LLaMA 8개의 80GB A100, 3시간 fine-tuning하는데 GCS 기준으로 약 100\$ 를 사용했습니다. 

<img width="667" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/998625f4-230b-496b-a21d-1f6129e5f407?style=centerme">

자체적으로 진행한 정성평가에서는, 5명의 학생 저자에게 GPT-3 와 의 Blind test 를 했더니 90:89 로 Alpaca가 승리 (?) 했다고 합니다. 다만 답변의 길이가 상대적으로 짧고 Hallucination이 자주 발생한다는 단점은 있었다고 하네요. 

하지만 Alpaca는 단 돈 600\$ 로 GPT-3 에 필적하는 성능을 가진 LLM을 보유할 수 있는 모습을 보여주었습니다. 연구실에서 사용하기에 전혀 부담스럽지 않으면서 수많은 downsteam task를 풀어나갈 수 있겠습니다.

### Koala

Koala는 Alpaca 보다도 더 간단한 방법론으로 접근했습니다. LLaMA 13B 모델에 chatGPT distillation data를 직접 사용한 건데요, Self-instruct 데이터 생성과정조차 없었기 때문에 8개의 80GB A100을 이용한 6시간 fine-tuning만 수행했고 약 100\$ 만을 사용해 Koala 모델을 학습시켰습니다.

<img width="737" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/926c9198-11b2-44f2-8780-4eeece635d21?style=centerme">

chatGPT distillation data는 2가지가 있었는데요, [shareGPT](https://sharegpt.com/)와 [HC3](https://arxiv.org/abs/2301.07597) 입니다. shareGPT는 chatGPT에서 발생한 대화 60K개가 공개되어 있는 public dataset이고, HC3는 60K개의 질문에 대한 24K개의 사람 답변과 27K개의 chatGPT 답변이 모여있는 dataset입니다. Koala project의 저자들은 LLaMA에 chatGPT distillation data를 이용해 fine-tuning 을 진행한 모델을 Koala-distill 이라고 명명했습니다.  

추가적으로 여러 open-source data들을 fine-tuning 작업에 사용한 모델도 테스트했습니다. 아래와 같은 데이터들을 추가하여 학습한 모델을 Koala-all 이라고 명명했습니다.

<img width="569" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/c2fffbaf-b99e-48db-9a1d-85631524d5bd?style=centerme">

Koala는 Alpaca와는 다르게 Amazon Mechanical Turk platform 에서 100명의 평가인단을 구성해 정성평가를 진행했습니다.

<img width="657" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/20b76c5b-90bc-4b75-b273-06d51d287dd4?style=centerme">

그 결과 Alpaca 보다는 조금 나은 선호도를 보였지만 chatGPT보다는 여전히 낮은 선호도를 보였습니다. 하지만 합리화(?)를 해보자면 모델이 10배 이상 작다는 점에서 이득이라고 볼 수 있습니다.  

놀라운 점은 Koala-Distill가 Koala-All보다 선호도가 높게 나타났다는 점이었는데요. 이는 대충 데이터를 긁어 모으는 것보다, **강력한 모델을 이용해 knowledge distillation 만을 진행하는 것**이 훨씬 좋은 성능을 나타낸다는 것을 보여줍니다.

## Conclusion
 
LLaMA 를 활용한 강력하면서도 Open-source 인 text generation LLM 을 만들고자 하는 학계의 움직임이 있습니다. Self-instruction, chatGPT distillation dataset 등, closed-source LLM으로부터 데이터를 만들어내는 방식이 굉장히 효율적으로 동작한다는 것을 보여주고 있고요. 다시 말해 Pre-trained LLM (초기 조건)과 좋은 선생님으로부터 만들어진 데이터가 있으면 강력한 언어모델을 쉽게 만들어낼 수 있다는 것입니다.  

다만 주의할 점은 License issue가 있기 때문에, 상업적으로 사용할 수 없습니다. LLaMA의 License는 Non-commercial bespoke, chatGPT 사용약관은 OpenAI 와 경쟁하는 모델에 사용할 수 없다는 조건이 있기 때문에, 회사에서의 사용은 주의해 주세요!
