---
layout: post
title: Paper survey - Multimodal LLM
tags: archive
lang: en
---

Since the era of LLMs is gradually shifting into the era of Multimodal LLMs, I think it's time for a follow-up.

I briefly organized

1. what kinds of Multimodal LLM architectures there are,
2. how instruction tuning for usability is carried out, and
3. how RLHF, which sparked the chatGPT hype, is applied.

## Summary

### Base multimodal LLM architectures

| Category | Architecture | Method |
| --- | --- | --- |
| Projection matrix | Frozen | trains only the vision encoder separately |
|  | Kosmos | trains the visual encoder last layer and the LLM |
|  | FROMAGe | trains only the projection matrix<br>embedding to image is also possible |
|  | LLaVA | trains the projection layer and the LLM<br>2-stage training |
| Cross attention | Flamingo | trains only the visual feature resampler and the LLM cross attention |
|  | BLIP | trains a Q-former that converts visual features into LLM embeddings |
| Adaptation prompt | LLaMA-Adapter | trains the projection matrix and prompt embedding |
| Image tokenizer | Chameleon | tokenizes the image latent vector using a Codebook |

### Instruction-tuning methods

- Performs SFT of a vision-text instruction dataset based on a pre-trained mLLM
- Differs in multi-stage training / freezing modules / architecture, etc.
- For image grounded tasks, methods that support bounding box (or free-form) information

| Category | Name | Architecture | Method | Dataset / Quantity |
| --- | --- | --- | --- | --- |
| Projection matrix | LLaVA-1.5 | LLaVA | MLP projection layer<br>High resolution (336x336) | LLaVA-Instruct-150K / 150K |
|  | LLaVA-NeXT | LLaVA | supports video input<br>Backbone upgrade (supports up to Qwen1.5 110B) | M4Instruct / 1M |
|  | Shikra | LLaVA | focuses on image grounded tasks | LLaVA-Instruct-150K + Shikra RD 6K / 156K |
|  | Ferret (Apple) | LLaMA + visual encoder | focuses on free-form region image grounded tasks | GRIT / 34K |
|  | Fuyu-8B | Fuyu | handles arbitrary image resolution | Unknown |
| Cross attention | OpenFlamingo (Deepmind) | Flamingo | open-source replica of Flamingo | LAION-2B / 2B<br>Multimodal C4 / 101M<br>Synthetic / 417K |
|  | InstructBLIP (Salesforce) | BLIP | Q-former instruct tuning | COCO caption / 82K<br>Web CapFilt / 14M<br>TextCaps / 21K<br>VQAv2 / 82K<br>OKVQA / 9K<br>A-OKVQA / 17K<br>OCR-VQA / 800K |
|  | MiniGPT-4 | BLIP | Q-former freeze, additional linear layer training | Curated image-description pair / 3.5K |
|  | MiniGPT-v2 (Meta) | BLIP | enhances task-specific performance through 7 task identifier tokens | LLaVA-Instruct / 81K<br>Filcker / 5.5K |
|  | Qwen-VL | BLIP (similar) | 3 stage training | Custom / 350K |
| Adaptation prompt | LLaMA-Adapter V2 | LLaMA-Adapter | 2 stage training<br>separates visual / instruction learning parameters | Text-only instruction / 52K<br>COCO caption / 567K |

### Preference alignment methods

- The alignment methods are all used as ways to reduce hallucination
- All use LLaVA-based models (it's unclear whether the reason for not using other architectures is an engineering issue or because LLaVA's performance is overwhelming)
- LoRA finetuning is commonly used

| Training-method category | Subcategory | Name | Method | Dataset / Quantity |
| --- | --- | --- | --- | --- |
| PPO |  | LLaVA-RLHF (Berkeley) | 10K Human annotated preference dataset ($3000)<br>performs RLHF pipeline | \<SFT\><br>Conversation / 98K<br>VQA-v2 / 83K<br>A-OKVQA / 16K<br>Flicker / 23K<br>\<RLHF\><br>Human annotated / 10K |
| DPO |  | RLHF-V (Tsinghua) | 1.4K Human modified preference dataset<br>directly fixes only the parts where hallucination occurred<br>DDPO (DPO with higher weights on changed tokens) | Human modified / 1.4K |
|  |  | Silkie (CUHK) | 80K GPT-4V annotation → 380K pairs (VLFeedback dataset) | GPT-4V annotated / 80K |
|  |  | LLaVA-Hound-DPO (CMU, Bytedance) | LLaVA-NeXT supporting video<br>80K GPT-4V video captioning<br>240K chatGPT question gen<br>240K chatGPT reward tagging (replace video to caption) → 20$ | chatGPT generated / 17K |
|  |  | mDPO (MS) | solves the problem that existing DPO can't do image conditioning<br>adds a conditioning loss using image / non-image pairs | Silkie / 10K |
|  |  | RLAIF-V (Tsinghua) | shows that using Divide & Conquer for reward tagging makes LLaVA-NeXT sufficient instead of GPT-4V | Diverse set / 4K |
|  | Synthetic pair data | STIC (UCLA) | Good prompt → generates chosen<br>Bad prompt / corrupt image → generates rejected<br>SFT with a normal prompt after DPO | \<DPO\><br>MSCOCO / 6K<br>\<SFT\><br>LLaVA’s SFT / 5K |
|  |  | POVID (Stanford) | Dis-preferred prompt / noisy image → generates rejected<br>Text Hal DPO (3 epochs) → Noisy image Hal DPO (1 epoch) | LLaVA-Instruct / 17K |
|  |  | BPO (HKUST) | Image gaussian noise / text error injection → generates rejected<br>making the negative differ only in the hallucination part like this gives the highest performance | ShareGPT-V / 58K<br><br>LLaVAR / 55K<br>LLaVA-Instruct / 54K |
|  |  | HSA-DPO (Zhejiang Univ.) | hallucination detection and scoring using 6K GPT-4V<br>trains a detection and scoring model<br>performs HSA-DPO (DPO with weights increased when the hallucination score is high) | Visual Genome / 8K |
| Contrastive learning |  | HALVA (Google) | swaps out the object words of the chosen (using an LLM) → generates rejected<br>uses contrastive loss | Visual Genome / 21.5K |
