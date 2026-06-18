---
layout: post
title: Diary - Gym 0.26.0 update (truncted, no seed)
tags: archive
lang: en
---

Jordan Terry (I briefly worked with him when I was dispatched to UMD) has released the gym 0.26.0 update version. The problem is that it seems to contain some pretty major changes.

First, the number of arguments that `env.reset()` returns has increased to 2. Previously it was obs; it has increased from obs to obs, info.

Second, the number of arguments that `env.step()` returns has increased to 5. From the previous obs, rew, done, info, it has become obs, rew, terminate, `truncated`, info. As for what truncated is, it means that, like a timeout, the task didn't actually end but terminated because it went beyond the range the MDP is trying to solve. The moment you treat this as done, the Q value at the truncated point gets fixed. But since that's not actually the case, a bias arises in the Q function estimation. Therefore, it's important to represent terminate and truncated separately.

Reference: [Issue with MC return estimate when truncating episode before termination.](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjZ5a-i3MH7AhWFfXAKHS-_CYsQFnoECAkQAQ&url=https%3A%2F%2Fwww.reddit.com%2Fr%2Freinforcementlearning%2Fcomments%2F9nalvz%2Fissue_with_mc_return_estimate_when_truncating%2F&usg=AOvVaw39VYKtXEWAK6yjbh3u_U8Y)

Third, `env.seed()` is gone. I'm not really sure of the reason, but I suspect it's because you can adequately handle it with just `np.random.seed()`.

It's not that I can't understand the reasons for the changes, but a lot of errors come up... Especially in the case of multiprocessing, there are many cases where exception handling is set up to just halt when an error occurs, so it's not easy......

Source: [Gym 0.26.0 release notes](https://github.com/openai/gym/releases?q=truncated&expanded=true#:~:text=Release%20notes%20for%20v0.26.0)
