---
layout: post
title: Diary - Sqaushed Gaussian policy for SAC
---

SAC (Soft Actor-Critic) 은 굉장히 유명한 Model-free, stochasitic policy gradient 방식의 RL 알고리즘이다. Stochastic policy의 특성상 완벽하게 Optimal 한 action을 도출하지는 않지만 더 Robust한 action을 도출할 수는 있다.

나는 이를 Cliff walk problem을 통해서 설명할 수 있다고 생각한다. 

<img width="500" alt="image" src="https://user-images.githubusercontent.com/57203764/184815427-d615ace7-d278-4a70-b560-2b284ad52769.png?style=centerme">
출처 [https://dnddnjs.gitbooks.io/rl/content/q__learning.html](https://dnddnjs.gitbooks.io/rl/content/q__learning.html)

이 그림은 SARSA와 Q-learning을 비교한 것이지만, SARSA 대신에 Stochastic policy를 넣어도 성립한다. Greedy algorithm (deterministic)이 더 optimal 한 것은 자명하지만 절벽에 떨어질 확률을 고려한다면, 더 안전하고 robust 한 action을 선택하게 될 것이니 말이다.

SAC의 핵심은 entropy를 계산하고, 이를 exploration을 위한 regularizer로 사용한다는 것이다. policy entropy를 maximization 함으로써, exploration이 활발하게 일어나도록 하는 것이다. Stochastic policy는 일반적으로 Gaussian policy를 이용한다. Mean 과 std (standard deviation)을 이용해 action을 sampling 하는 것이다. SAC에서 standard deviation은 entropy를 계산하는데 직접적으로 활용되기 때문에, state dependant 해야 한다는 특징이 있다. 다시 말해 back propagation이 가능해야 한다는 것이고, 이는 [reparemterization trick](https://simpling.tistory.com/34)을 사용해야 한다는 것이다.

여기까지는 알고 있었던 사실이었는데, 문제는 tanh를 마지막에 사용하는 방법이었다. 내가 처음에 사용했던 방법은 mean을 구할 때 tanh를 사용하고 std는 linear의 output을 그대로 사용하는 것이었다. 

$$a_\theta(s,z)=tanh(\mu_\theta(s)) + \sigma_\theta(s)\odot z,\ \ \ \ \ z\sim N(0,1)$$

왜 이렇게 했는지는 모르겠는데, 당연히 이렇게 하는 거 아닐까? 라고 생각했었던 듯. 근데 실제 SAC는 조금 다르게 구현되어 있었다. Sqaushed Gaussian policy를 이용한다고 되어 있었고, 아래와 같은 방법으로 action을 생성했다.

$$a_\theta(s,z)=tanh(\mu_\theta(s) + \sigma_\theta(s)\odot z),\ \ \ \ \ z\sim N(0,1)$$

거의 똑같은데, 꽤나 성능차이가 난다. 대신 이렇게 하면 entropy (negative log probability)를 구할 때 트릭을 하나 써야 하는데, [SAC appendix](arxiv.org/pdf/1801.01290.pdf)에 나와 있다.

$$\pi(a|s)=\mu(u|s)\ |det(\frac{da}{du})|^{-1}$$

$$\log\pi(a|s)=\log\mu(u|s)-\sum_{i=1}^D\log(1-tanh^2(u_i))$$

$$where\ \ a=tanh(u)$$

발견은 여기서 함. [Stable-baselines3 SAC document](https://spinningup.openai.com/en/latest/algorithms/sac.html#:~:text=we%20use%20a-,squashed,-Gaussian%20policy%2C%20which)