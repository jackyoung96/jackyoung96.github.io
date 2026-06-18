---
layout: post
title: Diary - SSH keygen issue
tags: archive
lang: en
---

There are cases where connecting to a server with SSH keygen doesn't work.  
I wasted two hours flailing around, so I'm recording this in case it happens again.

1. Typos

- Don't make typos in `authorized_keys`
- When copy-pasting id_rsa.pub, make sure you copy the whole thing properly

2. Permissions
- The `.ssh` folder should be 700
- The `.ssh/id_rsa` file should be 644
- Other files inside the `.ssh` folder should be 600
