---
layout: post
title: "Diary - Python shallow-copy 이야기: Top-down 학습의 문제점"
tags: archive
---

난 AI 엔지니어로 일하고 있기는 하지만, 기계공학과를 나왔다. 1학년때 컴퓨터의 기초 수업때 C++ 언어를 배우면서, 대체 이런걸 왜하지 싶었다. (그 때 부터 코딩을 했어야...)

군대 갔다 왔다가 AI 수업 들으면서 너무 재밌어 보인다는 생각을 하게 되었고, 자율 주행을 거쳐서 지금 LLM 필드까지 들어오게 되었지만, 그동안 Python 에 대한 강의를 들어본적은 없다. 전부다 top-down으로 프로젝트를 만들면서 모르는게 있으면 찾아보는 식으로 공부했었다. 그래서 사실 코딩 자체를 못하는건 아닌데, 가끔 기초가 부족하다는 생각이 들 때도 있다. 

그러던 중 얼마전에 문제가 하나 발생했다. 내가 짠 코드가 어떤거였냐면

```python
num_chunk = ...
world_size = ... 
dataset = ... # [data_1, data_2, ... data_n_c]

gathered_data = [[None] * world_size] * num_chunk
for i_c in range(num_chunk):
    torch.distributed.all_gather(gathered_data[i_c], dataset[i_c])
```

뭐 대충 이런거였다. process 마다 서로 다른 dataset을 load 한 상황이고, 이걸 전체를 all-gather를 하려는데, 한번에 하면 너무 양이 많아서 GPU-OOM 이 나니까, chunking을 해서 all-gather를 하려는 것. [비슷한 구현을 다뤘던 포스팅이 있다.](_posts/diary/2025-03-31-all-gather-object.md) 

아마 기본기가 탄탄한 분들은 바로 문제점이 보일지도 모른다. 일단 이 코드는 에러가 안난다. 문제는 뭐냐면 **List의 multiplication 연산은 shallow copy**라는 것이다. 

Shallow-copy 라는 것이 무엇이냐? 값을 복사하는게 아니라 메모리 주소를 복사하는 것. [자세한 설명](https://wikidocs.net/16038) 포인터 개념인 것이다.

그니까 지금 문제는 `[None, None, ..., None]` 이라는 list 를 `num_chunk` 개 shallow-copy 해버리는 바람에, chunk마다 all-gather 연산을 하는게 계속 덮어쓰기 된거다. 결국 가장 마지막 chunk 들로만 num_chunk 개를 복사해서 가지고 있는 꼴이 된 것.

이걸 어쩌다가 알았냐면 이 chunked loading을 구현한 뒤로 뭔가 계속 overfitting이 났기 때문이다. 당연하게도 똑같은 데이터만 N배로 썼으니 overfitting 날 수 밖에. 

해결법은 

```python
gathered_data = [[None] * world_size for _ in range(num_chunk)] 
```

로 해주면 된다. 이건 copy 의 개념이 아니라 `num_chunk` 개를 생성하는 거니까 괜찮다. 혹은 deepcopy 를 사용하던지.

아무튼 기본기의 부족을 다시 한 번 느끼게 된 계기가 되었다.   
**복잡한 AI 모델 학습도 shallow/deep copy 같은 아주 간단한 개념의 혼동에 의해서 망가질 수 있다.** 마음에 새기자.