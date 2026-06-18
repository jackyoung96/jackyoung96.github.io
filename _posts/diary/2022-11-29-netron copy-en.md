---
layout: post
title: Diary - Netron
tags: archive
lang: en
---

Say you have a pre-trained model. But what do you do if you don't know the model's structure and only have a checkpoint file like `.pt`, `.h5`, `.onnx`, `.tf`!! I'd always wondered about that.  
For those moments, there's a really great program.  

Its name is `Netron`!!!  
[![image](https://user-images.githubusercontent.com/57203764/204554505-002cbde3-357d-46a6-86a1-5f19b4f93c6f.png)](https://netron.app/)  

Just upload a checkpoint file and it tells you everything: what the model looks like, what the input and output shapes are (actually, since this information is contained inside the checkpoint file, reading it and displaying it isn't that hard). Let's put it to good use!!
