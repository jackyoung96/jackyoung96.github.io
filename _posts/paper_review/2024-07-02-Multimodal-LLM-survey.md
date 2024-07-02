---
layout: post
title: Paper survey - Multimodal LLM
tags: archive
---

슬슬 LLM 의 시대가 Multimodal LLM 의 시대로 넘어가고 있기 때문에, follow-up 이 필요한 시점이라고 생각한다.

간단하게 

1. Multimodal LLM architecture 들에는 어떤 것들이 있는지,  
2. 사용성을 위한 instruction tuning 은 어떻게 진행되는지,  
3. chatGPT hype 을 촉발한 RLHF 는 어떻게 적용되는지  

정리해보았다.

## Summary

### Base multimodal LLM architectures

| 대분류 | 아키텍처 | 방식 |
| --- | --- | --- |
| Projection matrix | Frozen | vision encoder 만 따로 학습 |
|  | Kosmos | visual encoder last layer, LLM 학습 |
|  | FROMAGe | projection matrix 만 학습<br>embedding to image 도 가능 |
|  | LLaVA | projection layer, LLM  학습<br>2-stage training |
| Cross attention | Flamingo | visual feature resampler, LLM cross attention 만 학습 |
|  | BLIP | visual feature 를 llm embedding 으로 변환하는 Q-former 학습 |
| Adaptation prompt | LLaMA-Adapter | projection matrix, prompt embedding 학습 |
| Image tokenizer | Chameleon | Codebook 을 이용한 image latent vector tokenize |

### Instruction-tuning methods

- Pre-trained mLLM 기반 vision-text instruction dataset 의 SFT 수행
- Multi-stage training / Freezing module / Architecture 등에서 차이
- Image grounded task 의 경우 bounding box (or free-form) 정보를 지원하는 방식

| 대분류 | 이름 | 아키텍처 | 방식 | 데이터셋 / 수량 |
| --- | --- | --- | --- | --- |
| Projection matrix | LLaVA-1.5 | LLaVA | MLP projection layer<br>High resolution (336x336) | LLaVA-Instruct-150K / 150K |
|  | LLaVA-NeXT | LLaVA | video input 지원<br>Backbone upgrade (Qwen1.5 110B 까지 지원) | M4Instruct / 1M |
|  | Shikra | LLaVA | image grounded task 에 집중 | LLaVA-Instruct-150K + Shikra RD 6K / 156K |
|  | Ferret (Apple) | LLaMA + visual encoder | Free-form region image grounded task 에 집중 | GRIT / 34K |
|  | Fuyu-8B | Fuyu | Arbitrary image resolution 에 대응 | Unknown |
| Cross attention | OpenFlamingo (Deepmind) | Flamingo | Flamingo 의 open-source replica | LAION-2B / 2B<br>Multimodal C4 / 101M<br>Synthetic / 417K |
|  | InstructBLIP (Salesforce) | BLIP | Q-former instruct tuning | COCO caption / 82K<br>Web CapFilt / 14M<br>TextCaps / 21K<br>VQAv2 / 82K<br>OKVQA / 9K<br>A-OKVQA / 17K<br>OCR-VQA / 800K |
|  | MiniGPT-4 | BLIP | Q-former freeze, additional linear layer training | Curated image-description pair / 3.5K |
|  | MiniGPT-v2 (Meta) | BLIP | 7개의 task identifier token 을 통해 task-specific performance 강화 | LLaVA-Instruct / 81K<br>Filcker / 5.5K |
|  | Qwen-VL | BLIP (비슷) | 3 stage training | Custom / 350K |
| Adaptation prompt | LLaMA-Adapter V2 | LLaMA-Adapter | 2 stage training<br>Visual / instruction learning parameter 분리 | Text-only instruction / 52K<br>COCO caption / 567K |

### Preference alignment methods

- 전부 Hallucination 을 줄이기 위한 방법으로 alignment method 들이 사용됨
- 전부 LLaVA 기반의 모델 사용 (다른 architecture 를 사용하지 않는 이유가 engineering 이슈 때문인지 LLaVA 의 성능이 압도적이어서 그런 것인지 알 수 없음)
- LoRA finetuning 이 일반적으로 사용됨

| 학습 방식 분류 | 소분류 | 이름 | 방식 | 데이터셋 / 수량 |
| --- | --- | --- | --- | --- |
| PPO |  | LLaVA-RLHF (Berkeley) | 10K Human annotated preference dataset ($3000)<br>RLHF pipeline 수행 | \<SFT\><br>Conversation / 98K<br>VQA-v2 / 83K<br>A-OKVQA / 16K<br>Flicker / 23K<br>\<RLHF\><br>Human annotated / 10K |
| DPO |  | RLHF-V (Tsinghua) | 1.4K Human modified preference dataset<br>Hallucination 발생한 부분만 직접 수정<br>DDPO (changed token 에 높은 가중치를 둔 DPO) | Human modified / 1.4K |
|  |  | Silkie (CUHK) | 80K GPT-4V annotation → 380K pairs (VLFeedback dataset) | GPT-4V annotated / 80K |
|  |  | LLaVA-Hound-DPO (CMU, Bytedance) | Video 지원하는 LLaVA-NeXT<br>80K GPT-4V video captioning<br>240K chatGPT question gen<br>240K chatGPT reward tagging (replace video to caption) → 20$ | chatGPT generated / 17K |
|  |  | mDPO (MS) | 기존 DPO 가 image conditioning 이 안되는 것을 해결<br>Image / non-image pair 를 이용해 conditioning loss 추가 | Silkie / 10K |
|  |  | RLAIF-V (Tsinghua) | Reward tagging 시 Divide & Conquer 를 활용하면 GPT-4V 대신 LLaVA-NeXT 로도 충분함을 보임 | Diverse set / 4K |
|  | Synthetic pair data | STIC (UCLA) | Good prompt → chosen 생성<br>Bad prompt / corrupt image → rejected 생성<br>DPO 이후 일반 prompt 로 SFT | \<DPO\><br>MSCOCO / 6K<br>\<SFT\><br>LLaVA’s SFT / 5K |
|  |  | POVID (Stanford) | Dis-preferred prompt / noisy image → rejected 생성<br>Text Hal DPO (3 epochs) → Noisy image Hal DPO (1 epoch) | LLaVA-Instruct / 17K |
|  |  | BPO (HKUST) | Image gaussian noise / text error injection → rejected 생성<br>negative 는 이렇게 hallucination 부분만 차이가 나도록 만드는게 성능이 가장 높음 | ShareGPT-V / 58K<br><br>LLaVAR / 55K<br>LLaVA-Instruct / 54K |
|  |  | HSA-DPO (Zhejiang Univ.) | 6K GPT-4V 를 이용해 hallucination detection and scoring<br>detection and scoring model 학습<br>HSA-DPO 수행 (Hallucination score 높으면 가중치를 높인 DPO) | Visual Genome / 8K |
| Contrastive learning |  | HALVA (Google) | Chosen 의 object word 들을 바꿔치기 (LLM 사용) → rejected 생성<br>Contrastive loss 사용 | Visual Genome / 21.5K |
