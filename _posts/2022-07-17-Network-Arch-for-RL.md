---
layout: post
title: Diary - Network architecture for RL
---

강화학습은 **Reward**라는 **Label**을 사용한다는 점에서 Supervised-learning의 일종으로 분류할 수 있다. Supervised learning에서 가장 주의해야 할 것은 overfitting일 것이다. 이를 해결하기 위해서는 적절한 network architecture (Size, depth, hidden dimension, learning rate, regularizer, etc) 를 찾는 것이 가장 중요하다. 그러나 대다수의 RL 연구들은 약속이나 한 듯 고정된 형태의 네트워크를 사용한다. (2-layer feed forward networks) 물론 시뮬레이션 환경이 OpenAI gym이나 Mujoco로 고정되어 있다보니 당연히 그렇겠지만...

UMD에서 하고 있는 연구는 pybullet 엔진으로 구성된 조금 특이한 시뮬레이터를 사용한다. 이 환경을 사용해 강화학습을 적용한 논문이 많으면 좋으련만, 레퍼런스로 삼을만 한 논문이 많지 않아 네트워크 구조를 속단하기가 어려웠다. 무작정 크게 만들어서 학습을 시키다 보니, 뭔가 학습은 잘 되는 것 같아 보이는데 막상 테스트를 해보니 성능이 계속 나오지 않았다. 그래서 갑자기 든 생각이 이거 overfitting 아닐까...??

사실 강화학습은 overfitting을 제대로 막아내는지를 확인하기가 어렵다. 학습시키는 환경과 테스트 환경이 동일한 경우가 많기 때문에 training data와 validation data의 i.i.d. 조건을 만족시킬 수 없기 때문이다. 강제로 i.i.d. 환경을 만듦으로써 강화학습에도 overfitting 문제가 발생한다는 논문이 있다.
[A study on overfitting in deep reinforcement learning](https://arxiv.org/pdf/1804.06893.pdf%C2%A0)

이 연구에서 사용한 환경은, 환경의 시드에 따라서 +1, -1의 reward가 랜덤하게 주어지게 된다. 같은 시드라면 reward 체계도 같다. 즉 training environment seed와 test environment seed를 분리하면 i.i.d. 조건을 만족시킬 수 있다. 아래 그림과 같이 실제로 너무 적은 episode를 통해 학습할 경우 overfitting이 발생한다. 

<figure>
<img width="556" alt="image" src="https://user-images.githubusercontent.com/57203764/179384991-5c9829dd-edbf-4b79-ac3b-3ab1bdf6ecd8.png?style=centerme">{:width="70%"}
<figcaption>Random maze에서 Overfitting 발생</figcaption>
</figure>

재미있는 것은 exploration을 위한 stochasticity가 overfitting를 반드시 해결하지도 못한다는 것이다. 단순히 exploration을 잘 해서 data distribution을 늘리면 될 것 같지만 실험 결과는 그렇지 않았다. 그럼 어떻게 해결하느냐고? 이 논문에 나온 문장을 인용하겠다.
**Those blackbox policies are relatively poorly understood, and they might implicitly acquire certain kind of robustness due to the architectures or the training dynamics**
한마디로 모른다는 것.

그치만 이는 당연하다고 Discussion에서 설명하고 있는데, 꽤나 예시가 마음에 든다. "Backward Brain Bicycle" 는 모든게 거꾸로 된 자전거다. 그니까 오른쪽으로 꺾으면 왼쪽으로 간다. 이건 muscle memory의 일종이라서 사람이 재학습하려면 몇개월이 걸린다고 한다. Biological overfitting이라고 할 수 있다는 것이다. 그런 의미에서 제한된 state distribution만을 경험할 수 있는 강화학습은 필연적으로 overfitting 문제에 직면할 수 밖에 없다. 

그렇다면 최대한 성능을 끌어내기 위해 어떤 network architecture를 사용해야만 할까? Google research team에서 감사하게도 엄청난 양의 GPU를 사용해서 실험을 미리 해 줬다. 안타깝게도 PPO 알고리즘에 대해서만 분석이 되어있지만, Off-policy 알고리즘도 크게 다르지는 않을 것이다 (라는 나의 뇌피셜). 환경에 따라 당연히 다르기 마련이지 직관을 얻는다는데 초점을 두면 될 듯 하다.
[WHAT MATTERS FOR ON-POLICY DEEP ACTORCRITIC METHODS? A LARGE-SCALE STUDY](https://openreview.net/pdf?id=nIAxjsniDzg)

거의 60가지 이상을 비교했으나, 그 중 의미있다고 생각하는 것들만 기록해본다.
- Policy hidden dimension, depth를 마냥 크게 하면 성능이 떨어진다.
- Critic hidden dimension, depth는 크게 해도 성능이 떨어지지 않는다. (골때리는 이름의 논문을 참고하자면, 실제로 Critic 사이즈를 유지하면서 actor 사이즈를 줄였을 때 성능이 크게 변화하지 않는다. [Honey. I Shrunk The Actor: A Case Study on Preserving Performance with Smaller Actors in Actor-Critic RL](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9619008&casa_token=vtwbej0Fo7QAAAAA:jfPsBslj17GfVvt1yYFBOmwrXY-B_cvctfXHFhG8pH9HrjJxSaU7yAHMa5RDkLl1sEcIueZ9HLY&tag=1))
- ReLU 보다 Tanh를 사용했을 때 높은 성능을 보였다.