---
layout: post
title: Diary - Gym 0.26.0 update (truncted, no seed)
tags: archive
---

Jordan Terry (UMD 파견했을 때 잠시 같이 일했었는데)가 gym 0.26.0 update version을 내놓았다. 그런데 문제는 엄청 major change가 있는 것 같다.  

첫 번째로, `env.reset()` 이 return 하는 인자가 2개로 늘었다. 기존에는 obs에서 obs, info로 늘어났다.  

두 번째로,  `env.step()` 이 return 하는 인자가 5개로 늘었다. 기존의 obs, rew, done, info 에서 obs, rew, terminate, `truncated`, info 가 되었다. Truncated 가 뭔가 하면, timeout과 같이 실제로 task가 종료된 것이 아닌 MDP 에서 풀고자 하는 범위를 넘어가서 종료된 것을 의미한다. 이것을 done이라고 보는 순간 truncated 된 지점에서의 Q value가 고정된다. 그렇지만 실제로는 그게 아니기 때문에 Q function estimation에 bias가 생기게 된다. 따라서 terminate와 trunctated를 분리하여 나타내는 것이 중요하다.  

참고 [Issue with MC return estimate when truncating episode before termination.](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjZ5a-i3MH7AhWFfXAKHS-_CYsQFnoECAkQAQ&url=https%3A%2F%2Fwww.reddit.com%2Fr%2Freinforcementlearning%2Fcomments%2F9nalvz%2Fissue_with_mc_return_estimate_when_truncating%2F&usg=AOvVaw39VYKtXEWAK6yjbh3u_U8Y)  

세 번째로, `env.seed()` 가 사라졌다. 사실 이유는 잘 모르겠지만, 단순히 `np.random.seed()`로도 충분히 대응할 수 있기 때문이 아닐까 싶다.  

바뀐 이유들이 이해가 안가는 건 아니지만, 상당히 에러가 많이 발생한다... 특히 multiprocessing의 경우 에러가 나면 그냥 멈추도록 예외처리가 되어 있는 경우가 많은데, 쉽지 않다......

출처 [Gym 0.26.0 release notes](https://github.com/openai/gym/releases?q=truncated&expanded=true#:~:text=Release%20notes%20for%20v0.26.0)