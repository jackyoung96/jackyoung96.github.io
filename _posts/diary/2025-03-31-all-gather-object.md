---
layout: post
title: Diary - PyTorch 에서 all_gather_object 사용시
tags: archive
---

잘만 돌아가던 코드에서 어느날 갑자기 발생한 OOM. 심지어 학습도 아니고 데이터셋 로딩 단계에서 OOM 이 발생했다.

```python
class GPTDataset(Dataset):
    def __init__(self, *,
                 **kwargs):

# ...Existing code...

temp = []
rank = dist.get_rank() if dist.is_initialized() else -1
world_size = dist.get_world_size() if dist.is_initialized() else -1

if rank >= 0 and world_size > 0:
    total_size = len(self)
    per_rank = (total_size // world_size) + 1
    start_idx = rank * per_rank
    end_idx = start_idx + per_rank if rank != world_size - 1 else total_size
    end_idx = min(end_idx, total_size)
    indices = range(start_idx, end_idx)
else:
    indices = range(len(self))

for i in tqdm(indices, desc=f"Load GPT dataset (rank {rank})", total=len(indices), mininterval=3):
    example = self[i]
    if 'loss_mask' in example and (example['loss_mask'] == 0).all():
        continue
    temp.append(example)

if dist.is_initialized():
    # Gather data from all processes
    all_samples = [[] for _ in range(world_size)]
    dist.all_gather_object(all_samples, temp)

    # Combine all samples
    temp = []
    for samples in all_samples:
        temp.extend(samples)    
        
    print(f"Gathering {len(temp)} data from all processes done")
```

`dist.all_gather_object` 부분에서 CUDA OOM이 발생했는데... dataset 로딩은 cpu 에서 일어나는 것이지, gpu 를 사용하지 않는다. input 데이터를 gpu 로 올리는 것은 학습 시작 후 collator 에서 해주도록 코드가 구현되어 있다. 

이전에 비해서 데이터셋의 사이즈가 늘어나기는 했지만 애초에 cpu memory 에러도 아니고 gpu memory를 사용했다는 사실이 이상하다. all_gather_object 를 뜯어보기로 하자.

```python
@_exception_logger
def all_gather_object(object_list, obj, group=None):
    """
    ... 생략 ...

    .. note:: For NCCL-based processed groups, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user's responsiblity to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    ... 생략 ..
    """
    if _rank_not_in_group(group):
        _warn_not_in_group("all_gather_object")
        return

    current_device = _get_pg_default_device(group)
    input_tensor, local_size = _object_to_tensor(obj, current_device, group)
```

torch distributed 라는게, NCCL-based process 이므로 기본적으로 GPU 를 이용한 통신을 사용한다. obj 가 들어오면 이걸 pickle 로 변환하고, 해당 pickle 을 tensor 로 변환하는 작업부터 시작하는 것이다. 이 때 torch.cuda.current_device() 를 통해 rank 별 GPU 를 설정하고, 각 rank 의 데이터를 각 rank GPU 에 올린 뒤 통신을 시작한다.

그러니까 결국 **CPU 에 있는 데이터를 한데 모으고 싶었던 것 뿐인데, 모든 데이터가 GPU 로 올라가고 all-gather 를 해버린 것.** 데이터가 늘어나면 OOM 이 발생할 수 밖에 없는 구조였다 (지금 발견해서 다행이다)

해결법은 두가지 정도가 생각난다. 
1. 데이터를 일정 chunk 단위로 끊어서 all-gather를 해줌으로써 순간 GPU memory 사용량을 줄여주는 것 
   - 장점: Distributed 세팅에 최적화. 32 노드를 사용한다고 치면, 32 * 8 = 256 GPU 에서 distributed loading 이 가능해진다. (빨라짐)
   - 단점: 근본적으로 GPU 를 사용하지 않아도 되는 것에서 GPU 를 사용한다. GPU 간 통신 속도가 느리다면 데이터 로딩도 느려진다.
2. torch distributed 가 아닌 multiprocessing 을 이용해서 데이터를 로딩하는 것
   - 장점: 모든 데이터가 CPU 에서 로딩된다.
   - 단점: 노드별 distributed loading은 불가능하다. RANK 별로 붙어있는 CPU 개수만큼만 multiprocessing 이 가능 (나의 경우 32)
3. Lazy loading 을 이용하여 학습시 각 rank 에서 필요한 데이터만 loading
   - 장점: 가장 효율적인 해결책
   - 단점: Sorted batch 와 같이 미리 데이터를 로딩해야 사용할 수 있는 테크닉을 쓸 수 없다.

현재 나의 상황은 노드가 매우 많으며 (>32), 노드끼리 전부 infiniband 로 연결되어 GPU간 통신 속도가 매우 빠르고, 데이터가 너무 많아서 최대한 분산을 쳐야 하고, sorted batch 와 같은 테크닉들을 모두 적용해야 한다. 따라서 1번 전략으로 해결해보기로 했다.

```diff
-                all_samples = [[] for _ in range(world_size)]
-                dist.all_gather_object(all_samples, temp)
-
+                num_chunk = 128
+                bsz = len(temp) // num_chunk + 1
+                
+                all_samples = [[None for _ in range(world_size)] for _ in range(num_chunk)]
+                
+                for i in range(num_chunk):
+                    chunk = temp[i * bsz:min((i + 1) * bsz, len(temp))]
+                    dist.all_gather_object(all_samples[i], chunk)
+                            
+                    torch.cuda.empty_cache()
+                    
                 # Combine all samples
                 temp = []
-                for samples in all_samples:
-                    temp.extend(samples)    
+                for j in range(len(all_samples[0])):
+                    for i in range(len(all_samples)):
+                        if all_samples[i][j]:
+                            temp.extend(all_samples[i][j])
```

Chunking 을 통한 all_gather_object 를 통해 데이터 로딩 단계에서의 CUDA OOM 을 해결했다!!