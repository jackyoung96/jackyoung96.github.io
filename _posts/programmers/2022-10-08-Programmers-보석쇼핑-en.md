---
layout: post
title: Programmers - Gem Shopping
tags: codes
level: silver
star: How to use set
lang: en
---

set(), a very handy function for when you want to count the number of unique items  

```python
def solution(gems):
            
    gem_count = len(set(gems))
    start_, end_ = 0,0
    all_gems_count = {}
    result = [int(1e5),0,0]
    
    for end_, gem in enumerate(gems):
        if gems[end_] in all_gems_count.keys():
            all_gems_count[gems[end_]] += 1
        else:
            all_gems_count[gems[end_]] = 1
        
        if len(all_gems_count) == gem_count:
            while True:
                if all_gems_count[gems[start_]] == 1:
                    if result[0] > end_-start_:
                        result = [end_-start_, start_, end_]
                    del all_gems_count[gems[start_]]
                    start_ += 1
                    break
                else:
                    all_gems_count[gems[start_]] -= 1
                start_ += 1
            if result[0] > end_-start_:
                result = [end_-start_, start_, end_]
    
    
    return [result[1], result[2]+1]
```
