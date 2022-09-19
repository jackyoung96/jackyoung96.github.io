---
layout: post
title: Diary - Curious about GPU memory
tags: archive
---

DRL 학습 도중 신기한 현상을 발견했다. 연구실에는 서버가 총 3개 있는데, 나는 1번 서버를 주로 사용한다. 이번에 3번째 서버를 사용할 수 있게 되면서, 서버 1에서 돌리던 코드를 그대로 서버 3으로 옮겼는데, GPU에서 신기한 일이 발생했다.  

**GPU에 올라간 메모리의 크기가 다르다.**  
<figure>
<img width="556" alt="image" src="https://user-images.githubusercontent.com/57203764/165021666-5300eace-7da5-4d64-b7bc-702841119170.png?style=centerme">
<figcaption>서버1의 GPU memory 현황</figcaption>
</figure>
<figure>
<img width="560" alt="image" src="https://user-images.githubusercontent.com/57203764/165021435-197a295c-5c1c-4443-8820-f49cf58fad92.png?style=centerme">
<figcaption>서버3의 GPU memory 현황</figcaption>
</figure>

일단 첫 번째로는 똑같은 네트워크를 올렸는데도 서버3에서 거의 2배 가까운 메모리를 잡아먹는다.
서버 1은 Titan X로 알고 있고, 서버 3은 최신의 3090 ti로 알고 있다. 
추측으로는 3090은 거의 20GB가 넘다보니까 swap memory같은걸 따로 둬서 연산속도를 올리려는게 아닌가 싶다. 혹시나 아는 분 있으면 알려주세요.  

두 번째로는 갑자기 0번 GPU에 메모리가 잡혔다. 이유는 알 수 없다. pytorch를 사용하는 모든 것들에 device를 적용했었고, 서버 1에서는 이런적이 없었는데, 어떻게 이게 가능한지 모르겠다. 혹은 GYM 환경에서 GPU를 쓰는걸지도 모르겠다. 이것도 아는 분 있으면 알려주세요.
