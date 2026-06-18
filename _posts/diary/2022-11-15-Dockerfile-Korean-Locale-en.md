---
layout: post
title: Diary - Dockerfile Korean Locale Setup
tags: archive
lang: en
---

During the Docker image build process, errors related to Korean locale settings (UTF-8) occur frequently.
When that happens, putting the lines below into the Dockerfile resolved it nicely.

```dockerfile
RUN pip install -r requirements_inference.txt
RUN apt-get update && apt-get install -y locales git
RUN localedef -f UTF-8 -i ko_KR ko_KR.UTF-8
ENV LC_ALL ko_KR.UTF-8
ENV PYTHONIOENCODING=utf-8
```

Or  

```dockerfile
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
```

Source: [Dockerfile Korean locale setup (feat. the difference between Ubuntu and Debian)](https://blog.metafor.kr/235)
