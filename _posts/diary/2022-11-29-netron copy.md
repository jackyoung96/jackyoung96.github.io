---
layout: post
title: Diary - Netron
tags: archive
---

pre-trained 된 모델이 있다. 그런데 만약 모델의 구조를 모르고 `.pt`, `.h5`, `.onnx`, `.tf` 등의 체크포인트 파일만 있다면 어떻게 할 것인가!! 그게 항상 궁금했었다.  
그 때를 위해서 아주 좋은 프로그램이 있다.  

이름하여 `Netron` !!!  
[![image](https://user-images.githubusercontent.com/57203764/204554505-002cbde3-357d-46a6-86a1-5f19b4f93c6f.png)](https://netron.app/)  

그냥 체크포인트 파일 올리면 모델이 어떻게 생겼는지, input, output shape 은 어떤지 다 알려준다 (사실 체크포인트 파일 안에는 이런 정보가 들어 있으니까 그거 읽어서 뿌려주는 건 어렵진 않다). 유용하게 쓰도록 하자!!