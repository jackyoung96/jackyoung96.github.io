---
layout: post
title: Survey - Multimodal LLM fine-tuning dataset
tags: archive
---

[Multimodal LLM survey](https://jackyoung96.github.io/2024/07/02/Multimodal-LLM-survey/)에 이어서 사용된 dataset detail 과 링크들을 정리해 보았습니다.

### Instruction-tuning dataset

| 분류 | 이름 | 개수 | 사용 모델 | 비고 | 링크 |
| --- | --- | --- | --- | --- | --- |
| Instruction | LLaVA-Instruct-150K | 150K | LLaVA-1.5, InstructBLIP, MiniGPT4-v2 | Conversation, detail description, complex reasoning 으로 구성 | https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K |
|  | M4-Instruct | Multi-image 594K<br>Single-image 307K<br>Video 262K<br>3D 99.5K | LLaVA-NeXT | SOTA 모델의 dataset | https://huggingface.co/datasets/lmms-lab/M4-Instruct-Data |
|  | Shikra (GPT4Gen_BoxCoT) | 6K | Shikra |  | https://github.com/shikras/shikra/tree/main?tab=readme-ov-file |
|  | MiniGPT4 | 3.5K | MiniGPT4 | Simple caption describing instruction | https://github.com/Vision-CAIR/MiniGPT-4/blob/main/MiniGPT4_Train.md |
|  | MiniGPT4-v2 multitask conversation | 12K | MiniGPT4-v2 | Multi-turn multi-task | https://github.com/Vision-CAIR/MiniGPT-4/blob/main/dataset/README_MINIGPTv2_FINETUNE.md |
|  | VQA | 4.4M |  | Short answer question | https://visualqa.org/download.html |
|  | OK-VQA | 14K |  | VQA requiring external knowledge | https://okvqa.allenai.org/ |
| Image-caption-grounded pair | GRIT | 20M | Ferret | Bounding box link to both “noun phrase” and “image segment” | https://huggingface.co/datasets/zzliang/GRIT?row=0 |
|  | VG (Visual Genome) | 1.7M |  | Object region, name, bounding box 명시 | https://huggingface.co/datasets/ranjaykrishna/visual_genome |
| Image-answer pair | CLEVR | 16K |  | Visual reasoning (Color, size, count…) | https://cs.stanford.edu/people/jcjohns/clevr/ |
| Image-caption pair | Cap COCO | 330K |  |  | https://huggingface.co/datasets/HuggingFaceM4/COCO |
|  | SBU captions | 860K |  |  | https://huggingface.co/datasets/vicenteor/sbu_captions |
|  | Flickr30K | 30K |  |  | https://huggingface.co/datasets/nlphuji/flickr30k |

| 분류 | 이름 | 개수 | 사용 모델 | 비고 | 링크 |
| --- | --- | --- | --- | --- | --- |
| Instruction | KoLLaVA-Instruct-150K | 150K | koLLaVA | LLaVA-Instruct-150K 를 DeepL 로 번역 | https://huggingface.co/datasets/tabtoyou/KoLLaVA-Instruct-150k |
|  | M3IT-80 | 1K |  | M3IT 번역 | https://huggingface.co/datasets/MMInstruction/M3IT-80 |
|  | 시각정보 기반 질의응답 | 7.5M |  | AI hub | https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=104 |
|  | KVQA | 10K |  | SKT 데이터셋 (더 있을 수도) | https://github.com/SKTBrain/KVQA?tab=readme-ov-file |
| Image-caption pair | KoCC12M | 12M |  | CC12M 번역 | https://huggingface.co/datasets/QuoQA-NLP/KoCC12M |

### Preference alignment methods

> Annotated: 두 개의 답변을 비교하는 형태  
> Synthetic: 하나의 답변을 기반으로 rule-based/LLM-based 방법을 통해 나머지 하나를 생성한 형태

| 분류 | 이름 | 개수 | 사용 모델 | 비고 | 링크 |
| --- | --- | --- | --- | --- | --- |
| Human Annotated | LLaVA-Human-Preference-10K | 9.42K |  |  | https://huggingface.co/datasets/zhiqings/LLaVA-Human-Preference-10K |
|  | RLHF-V | 5.73 | RLHF-V |  | https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset |
| AI annotated | RLAIF-V | 33.8K | RLAIF-V | Divide & Conquer 방식의 LLaVA-NeXT annotation | https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset |
|  | Silkie | 80K | Silkie, mDPO | GPT-4V annotation | https://huggingface.co/datasets/MMInstruction/VLFeedback |
|  | LLaVA-Hound-DPO | 17K | LLaVA-Hound-DPO | Video support<br>GPT-4V video captioning<br>chatGPT annotation | https://huggingface.co/ShareGPTVideo/LLaVA-Hound-DPO |
| Synthetic modified | STIC-coco-preference-6k | 6K | STIC | Only image description instruction | https://huggingface.co/datasets/STIC-LVLM/stic-coco-preference-6k |
|  | POVID-preference-data | 17.2K | POVID | LLaVA-Instruct 기반 | https://huggingface.co/datasets/YiyangAiLab/POVID_preference_data_for_VLLMs |
|  | BPO | 188K | BPO | ShareGPT4V, COCO, LLaVA-Instruct 기반 | https://huggingface.co/datasets/renjiepi/BPO?row=0 |
|  | HALVA | 21.7K | HALVA | Visual Genome 기반 | https://github.com/FuxiaoLiu/LRV-Instruction/blob/main/download.txt#L28 |