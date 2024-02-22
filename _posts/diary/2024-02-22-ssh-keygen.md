---
layout: post
title: Diary - SSH keygen 이슈
tags: archive
---

SSH keygen 으로 서버에 접속할 때 안되는 경우들이 있다.  
두 시간 삽질로 날렸지만, 또 그럴까봐 기록해둔다.

1. 오타

- `authorized_keys` 오타 내지 말자
- id_rsa.pub 복사 붙여넣기 할 때 전체 복사 잘 하자

2. 권한
- `.ssh` 폴더는 700
- `.ssh/id_rsa` 파일은 644
- `.ssh` 폴더 내부의 다른 파일들은 600