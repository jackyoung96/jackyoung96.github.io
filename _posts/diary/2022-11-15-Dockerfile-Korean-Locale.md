---
layout: post
title: Diary - Dockerfile 한국어 설정
tags: archive
---

Docker image 빌드 과정에서 한국어 설정 (UTF-8) 관련 에러가 자주 발생한다.
그럴 때 Dockerfile에 아래 문장을 넣어주면 잘 해결되었다.

```dockerfile
RUN pip install -r requirements_inference.txt
RUN apt-get update && apt-get install -y locales git
RUN localedef -f UTF-8 -i ko_KR ko_KR.UTF-8
ENV LC_ALL ko_KR.UTF-8
ENV PYTHONIOENCODING=utf-8
```

출처 [dockerfile 한국어 로케일 설정(feat. 우분투와 데비안의 차이)](https://blog.metafor.kr/235)