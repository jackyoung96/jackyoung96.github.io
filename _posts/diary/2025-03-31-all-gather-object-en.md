---
layout: post
title: Diary - When using all_gather_object in PyTorch
tags: archive
lang: en
---

An OOM that suddenly occurred one day in code that had been running just fine. And it wasn't even during training—the OOM happened at the dataset loading stage.

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

The CUDA OOM happened at the `dist.all_gather_object` part... but dataset loading happens on the CPU; it doesn't use the GPU. The code is implemented so that moving the input data onto the GPU is done in the collator after training starts.

The dataset size had grown compared to before, but it's strange that it was a GPU memory error rather than a CPU memory error in the first place. Let's dig into all_gather_object.

```python
@_exception_logger
def all_gather_object(object_list, obj, group=None):
    """
    ... omitted ...

    .. note:: For NCCL-based processed groups, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user's responsiblity to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    ... omitted ...
    """
    if _rank_not_in_group(group):
        _warn_not_in_group("all_gather_object")
        return

    current_device = _get_pg_default_device(group)
    input_tensor, local_size = _object_to_tensor(obj, current_device, group)
```

The thing about torch distributed is that, since it's an NCCL-based process, it basically uses GPU-based communication. When an obj comes in, it starts by converting it to a pickle and then converting that pickle into a tensor. At this point it sets the per-rank GPU via torch.cuda.current_device(), puts each rank's data onto its respective rank's GPU, and then begins communication.

So in the end, **I just wanted to gather data that was on the CPU into one place, but all the data got moved onto the GPU and an all-gather was performed.** It was a structure where an OOM was bound to occur once the data grew (lucky I found it now).

I can think of about two—well, three—solutions.
1. Break the data into chunks of a certain size and do all-gather per chunk, thereby reducing the instantaneous GPU memory usage
   - Pros: Optimized for distributed settings. If you use 32 nodes, distributed loading becomes possible across 32 * 8 = 256 GPUs (faster).
   - Cons: It uses the GPU for something that fundamentally doesn't need the GPU. If GPU-to-GPU communication speed is slow, data loading also becomes slow.
2. Load the data using multiprocessing instead of torch distributed
   - Pros: All data is loaded on the CPU.
   - Cons: Per-node distributed loading is impossible. Multiprocessing is only possible up to the number of CPUs attached per RANK (32 in my case).
3. Use lazy loading to load only the data each rank needs during training
   - Pros: The most efficient solution.
   - Cons: You can't use techniques like sorted batch that require loading the data in advance.

My current situation is that there are a great many nodes (>32), the nodes are all connected via InfiniBand so GPU-to-GPU communication speed is very fast, there's so much data that I need to distribute it as much as possible, and I need to apply all the techniques like sorted batch. So I decided to solve it with strategy #1.

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

By doing all_gather_object via chunking, I solved the CUDA OOM at the data loading stage!!
