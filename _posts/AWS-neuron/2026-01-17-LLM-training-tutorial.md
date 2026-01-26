---
layout: post
title: "Tutorial - LLM training with AWS Neuronx-distributed-training"
tags: hidden
---

*2026.01.17 ver*

This is a tutorial to train Llama3 8B with HyperPod cluster consists of [**Trainium chips**](https://aws.amazon.com/ko/ec2/instance-types/trn1/) and **slurm scheduler**. 
The tutorial is basically follow [**Amazon SageMaker HyperPod workshop**](https://catalog.workshops.aws/sagemaker-hyperpod/en-US), so please cross-reference the link and this document.

## Dependency

### Setup the HyperPod cluster

To navigate to the **SageMaker HyperPod Clusters** page and choose **Slurm** orchestration, follow these steps. 

1. Open the Amazon SageMaker AI console at [https://console.aws.amazon.com/sagemaker/](https://console.aws.amazon.com/sagemaker/).
2. Choose **HyperPod Clusters** in the left navigation pane and then **Model training & customization.**
    
    ![image.png](https://github.com/user-attachments/assets/dcf48efb-2611-4c66-8be0-bad8e2f7e86a)
    
3. Highly recommend to set region to **United States (N. Virginia)** which is **`us-east-2`** 
    1. Note that trn1.32xlarge resource is only available in specific **region and AZ**
    2. I recommend to use **us-east-2** for region and **us2-az3** for AZ
4. On the **SageMaker HyperPod Clusters** page, choose **Create HyperPod cluster**.
    1. On the **Create HyperPod cluster** drop-down, choose **Orchestrated by Slurm**.
5. Select **Quick setup** for setup options
6. Set cluster name whatever you want, but note that it will be used for SSH connection.
7. Follow these steps to set **instance groups**. we will add 3 instances; **controller**, **login**, and **worker**.
    1. For **controller** instance
        1. Click Add group
        2. Set **instance group** type as **`Controller`**
        3. Set **name** as **`my-controller-group`** (whatever you want)
        4. Set **instance type** as **`ml.c5.xlarge`**
        5. Set **Target Availability Zone** as **`use2-az3`** 
    2. For **login** instance
        1. Click Add group
        2. Set **instance group** type as **`Login`**
        3. Set **name** as **`my-login-group`** (whatever you want)
        4. Set **instance type** as **`ml.m5.4xlarge`**
        5. Set **Target Availability Zone** as **`use2-az3`** 
    3. For **worker** instance
        1. Click Add group
        2. Set **instance group** type as **`Compute`**
        3. Set **name** as **`worker-group-1`** (whatever you want)
        4. Set **instance type** as **`ml.trn1.32xlarge`**
        5. Set **Target Availability Zone** as **`use2-az3`**
8. Submit

It takes 10~15 minutes to be inService status like below.

![image.png](https://github.com/user-attachments/assets/00664325-d2bf-4f75-bb64-b4bac7be074a)

### Setup permission for IAM

Two permissinos policies for IAM Group should be added as below:

- AmazonSageMakerFullAccess
- AmazonSSMFullAccess

![image.png](https://github.com/user-attachments/assets/ac346700-7017-409c-8fd3-d069c9382954)

### Setup SSH

Configuring settings for the AWS CLI 

```bash
aws configure
>>> AWS Access Key ID: [Account ID]
>>> AWS Secret Access Key: [IAM access key]
>>> Default region name: [HyperPod region] # us-east-2
>>> Default output format: json
```

1. First install the [SSM Session Manager Plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html). Installation commands depends on your OS.

```bash
# MacOS for examle (x86_64) 
curl "https://s3.amazonaws.com/session-manager-downloads/plugin/latest/mac/sessionmanager-bundle.zip" -o "sessionmanager-bundle.zip"
unzip sessionmanager-bundle.zip
sudo ./sessionmanager-bundle/install -i /usr/local/sessionmanagerplugin -b /usr/local/bin/session-manager-plugin
```

1. Generate SSH key pair

```bash
ssh-keygen -t rsa -q -f"$HOME/.ssh/id_rsa" -N""
```

1. run a script [easy-ssh.sh](https://github.com/aws-samples/awsome-distributed-training/blob/main/1.architectures/5.sagemaker-hyperpod/easy-ssh.sh) to login to the cluster and add our keypair. Note that `[controller-group]` is used for SSH connection, not `[login-group-name]`.

```bash
curl -O https://raw.githubusercontent.com/aws-samples/awsome-distributed-training/main/1.architectures/5.sagemaker-hyperpod/easy-ssh.sh
chmod +x easy-ssh.sh
./easy-ssh.sh -c [controller-group-name] [cluster-name]
```

### Test the HyperPod cluster

```bash
ssh [cluster-name]
srun -N —pty bash
neuron-ls
```

we can see the whole resource of `trn1.32xlarge` → 16 x Trainium chips, each with 2 x neuron cores

```bash
instance-type: trn1.32xlarge
instance-id: i-0fc767be0ec5465d7
+--------+--------+----------+--------+---------------+--------------+--------------+------+
| NEURON | NEURON |  NEURON  | NEURON |   CONNECTED   |     PCI      |     CPU      | NUMA |
| DEVICE | CORES  | CORE IDS | MEMORY |    DEVICES    |     BDF      |   AFFINITY   | NODE |
+--------+--------+----------+--------+---------------+--------------+--------------+------+
| 0      | 2      | 0-1      | 32 GB  | 12, 3, 4, 1   | 0000:10:1b.0 | 0-31,64-95   | 0    |
| 1      | 2      | 2-3      | 32 GB  | 13, 0, 5, 2   | 0000:10:1e.0 | 0-31,64-95   | 0    |
| 2      | 2      | 4-5      | 32 GB  | 14, 1, 6, 3   | 0000:a0:1b.0 | 32-63,96-127 | 1    |
| 3      | 2      | 6-7      | 32 GB  | 15, 2, 7, 0   | 0000:a0:1e.0 | 32-63,96-127 | 1    |
| 4      | 2      | 8-9      | 32 GB  | 0, 7, 8, 5    | 0000:20:1e.0 | 0-31,64-95   | 0    |
| 5      | 2      | 10-11    | 32 GB  | 1, 4, 9, 6    | 0000:20:1c.0 | 0-31,64-95   | 0    |
| 6      | 2      | 12-13    | 32 GB  | 2, 5, 10, 7   | 0000:90:1e.0 | 32-63,96-127 | 1    |
| 7      | 2      | 14-15    | 32 GB  | 3, 6, 11, 4   | 0000:90:1c.0 | 32-63,96-127 | 1    |
| 8      | 2      | 16-17    | 32 GB  | 4, 11, 12, 9  | 0000:20:1d.0 | 0-31,64-95   | 0    |
| 9      | 2      | 18-19    | 32 GB  | 5, 8, 13, 10  | 0000:20:1b.0 | 0-31,64-95   | 0    |
| 10     | 2      | 20-21    | 32 GB  | 6, 9, 14, 11  | 0000:90:1d.0 | 32-63,96-127 | 1    |
| 11     | 2      | 22-23    | 32 GB  | 7, 10, 15, 8  | 0000:90:1b.0 | 32-63,96-127 | 1    |
| 12     | 2      | 24-25    | 32 GB  | 8, 15, 0, 13  | 0000:10:1c.0 | 0-31,64-95   | 0    |
| 13     | 2      | 26-27    | 32 GB  | 9, 12, 1, 14  | 0000:10:1d.0 | 0-31,64-95   | 0    |
| 14     | 2      | 28-29    | 32 GB  | 10, 13, 2, 15 | 0000:a0:1c.0 | 32-63,96-127 | 1    |
| 15     | 2      | 30-31    | 32 GB  | 11, 14, 3, 12 | 0000:a0:1d.0 | 32-63,96-127 | 1    |
+--------+--------+----------+--------+---------------+--------------+--------------+------+
```

### (Optional) Shutdown worker node

`trn1.32xlarge` instance is expensive! ($21.5 per hour). We can shut this node down if you want, while doing CPU jobs such as downloading model and dataset.

1. Go to HyperPod cluster management
2. Edit number of worker instance to zero

You can restore the number of worker instance whenever you want. Note that we use worker node only for computing, not for saving data. So adding or removing worker nodes won’t have any effect.

## Training LLM (Llama-8B)

We will train LLama3-8B model on SageMaker HyperPod. The model training script uses Neuronx Distributed Training (NxDT) library, which includes both TP (Tensor Parallelism) and PP (Pipeline Parallelism) techniques. For more information about Neuronx Distributed Training design, see [NxD Training](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-training/index.html).

### Download artifacts

The artifacts contains a precompiled model, and scripts for this tutorial.

```bash
# If you Start from local
ssh [cluster-name]

# HyperPod cluster
cd ~
curl -o training_artifacts.tar.gz 'https://static.us-east-1.prod.workshops.aws/public/d9d47635-8337-41be-8ba7-dd070f51f7a2/assets/training_artifacts.tar.gz'
tar -xvf training_artifacts.tar.gz
cd training_artifacts/
srun ./create_env.sh
```

(Error case) If you got Python3.8 error while setup the virtual env like below:

```
Reading package lists...
Building dependency tree...
Reading state information...
E: Unable to locate package python3.8-venv
E: Couldn't find any package by glob 'python3.8-venv'
E: Couldn't find any package by regex 'python3.8-venv'
srun: error: ip-10-3-61-171: task 0: Exited with exit code 100
```

Use below command before `srun ./create_env.sh`. This error raises because there is no pre-installed Python3.8 in worker node.

```
# Install Python3.8 at worker node
srun -N 1 --pty bash
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.8 python3.8-venv python3.8-dev
exit

# create env at login node
srun ./create_env.sh
```

### Download dataset

We will use AI2’s C4 (Common Crawl’s web Crawl Corpus) dataset which is super famous open-source dataset for pre-training LLM.

```
mkdir ~/training_artifacts/neuronx-distributed-training/examples/examples_datasets/
cd ~/training_artifacts/neuronx-distributed-training/examples
source aws_neuron_venv_pytorch/bin/activate # If you open a new terminal, please activate venv first
python3 get_dataset.py --llama-version 3
```

You can find downloaded dataset in `/fsx/ubuntu/training_artifacts/neuronx-distributed-training/examples/examples_datasets/wikicorpus_llama3_tokenized_8` directory as arrow files

```bash
data-00000-of-00021.arrow  data-00004-of-00021.arrow  data-00008-of-00021.arrow  data-00012-of-00021.arrow  data-00016-of-00021.arrow  data-00020-of-00021.arrow
data-00001-of-00021.arrow  data-00005-of-00021.arrow  data-00009-of-00021.arrow  data-00013-of-00021.arrow  data-00017-of-00021.arrow  dataset_info.json
data-00002-of-00021.arrow  data-00006-of-00021.arrow  data-00010-of-00021.arrow  data-00014-of-00021.arrow  data-00018-of-00021.arrow  state.json
data-00003-of-00021.arrow  data-00007-of-00021.arrow  data-00011-of-00021.arrow  data-00015-of-00021.arrow  data-00019-of-00021.arrow
```

### Submit a training job

Let’s create sbatch file like below:

```bash
cat > submit-llama8b.sh << EOL
#!/bin/bash
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -o llama.out

export OMP_NUM_THREADS=1
export COMPILE=0
export CONF_FILE=hf_llama3_8B_config

srun ./train.sh
EOL
```

It use `/fsx/ubuntu/training_artifacts/neuronx-distributed-training/examples/conf/hf_llama3_8B_config.yaml` for training. 

```bash
# hf_llama3_8B_config.yaml

name: hf_llama
model_source: hf
seed: 1234

trainer:
  devices: 32
  num_nodes: 4
  max_epochs: -1 # PTL default. In practice, max_steps will be reached first.
  max_steps: 10000 # consumed_samples = global_step * micro_batch_size * data_parallel_size * accumulate_grad_batches
  log_every_n_steps: 10

(omitted...)

distributed_strategy:
  tensor_model_parallel_size: 32
  pipeline_model_parallel_size: 1
  virtual_pipeline_model_parallel_size: 1
  zero1: True
  sequence_parallel: True
  kv_replicator: 4

(omitted...)

model:
  # specify micro_batch_size, global_batch_size, and model parallelism
  # gradient accumulation will be done automatically based on data_parallel_size

  # model architecture
  model_config: ./config.json # TODO: Expand this into arguments in this file
  encoder_seq_length: 8192
  max_position_embeddings: 8192
  num_layers: 32
  hidden_size: 4096
  qkv_linear: True
  fuse_qkv: True
  rope_theta: 500000.0
 
(omitted...)
```

The funny thing is that this config file set **num_devices as 32** and **num_nodes as 4**, although we has only single worker node. `trn1.32xlarge` instance contains 16 x Trainium chips, each with 2 x neuron cores. PyTorch sees each core as an individual device, so we'll be training across 32 devices. Typically, one node consists of 8 devices, so **num_nodes is set to 4**. 

We can see the **TP (Tensor Parallel) set to 32** and **PP (Pipeline Parallel) set to 1**. When I set TP to 8, memory issue was occurred. Also I tried to set TP 16 and PP 1 for boosting training speed (Set DP as 2), but it raise error. *It remains for the future work*.

```bash
# Error log when I use DP 2 TP 16 PP 1
2026-Jan-14 11:23:49.987594 95431:108485 ERROR   ENC:enc_init_comm                           [rank 0] failed (2) to init a collective algorithm for provided replica group.
2026-Jan-14 11:23:49.9876022026-Jan-14 11:23:49.987609 95447:105589 ERROR   ENC:enc_init_comm                            95431:108485 ERROR   ENC:enc_init_replica_groups                 [rank 1] failed (2) to init a collective algorithm for provided replica group.[nec_dev 1] failed to init ENC comm

2026-Jan-14 11:23:49.987616 95431:108485 ERROR   ENC:enc_load_operations                     [nec_dev 1] failed to init replica groups
2026-Jan-14 11:23:49.987617 95447:105589 ERROR   ENC:enc_init_replica_groups                 [nec_dev 17] failed to init ENC comm
2026-Jan-14 11:23:49.9876222026-Jan-14 11:23:49.987624 95431:108485 ERROR  TDRV:kbl_exec_build_and_load_cc_resources     95447:105589 ERROR   ENC:enc_load_operations                     [nec_dev 1, gid 1] failed to load operations, model: /tmp/ubuntu/neuroncc_compile_workdir/34c5925f-48be-49e7-bc83-a2309620ca43/model.MODULE_9167294978885178679+e132a0f1.neff[nec_dev 17] failed to init replica groups

2026-Jan-14 11:23:49.9876302026-Jan-14 11:23:49.987631 95431:108485 ERROR  NMGR:dlr_build_and_load_cc_resources          95447:105589 ERROR  TDRV:kbl_exec_build_and_load_cc_resources    Failed to build and load collectives resources for /tmp/ubuntu/neuroncc_compile_workdir/34c5925f-48be-49e7-bc83-a2309620ca43/model.MODULE_9167294978885178679+e132a0f1.neff, exec_id 14[nec_dev 17, gid 17] failed to load operations, model: /tmp/ubuntu/neuroncc_compile_workdir/34c5925f-48be-49e7-bc83-a2309620ca43/model.MODULE_9167294978885178679+e132a0f1.neff

2026-Jan-14 11:23:49.987638 95447:105589 ERROR  NMGR:dlr_build_and_load_cc_resources         Failed to build and load collectives resources for /tmp/ubuntu/neuroncc_compile_workdir/34c5925f-48be-49e7-bc83-a2309620ca43/model.MODULE_9167294978885178679+e132a0f1.neff, exec_id 14
2026-Jan-14 11:23:49.9876692026-Jan-14 11:23:49.987669 95447:105589 ERROR   NRT:nrt_infodump                             95431:108485 ERROR   NRT:nrt_infodump                            Neuron runtime information - please include in any support request:Neuron runtime information - please include in any support request:

2026-Jan-14 11:23:49.9876762026-Jan-14 11:23:49.987676 95447:105589 ERROR   NRT:nrt_infodump
```

### Monitor the job

We can simply check the neuron status by commands below.

```bash
ssh $(sinfo -h | awk '{ print $6 }' | head -n 1)
neuron-top
```

![image.png](https://github.com/user-attachments/assets/17e7e0a1-9491-4846-91ff-9b15b6ce282b)

Training progress can be checked by log file `llama.out`

```bash
 Epoch 0:   1%|          | 1/91 [10:35<15:52:47, 635.19s/it]
 Epoch 0:   2%|▏         | 2/91 [21:04<15:37:48, 632.23s/it]
 ...
 Epoch 0:  11%|█         | 10/91 [1:44:15<14:04:32, 625.59s/it]
```

Also we can check training curves by tensorboard

```bash
# HyperPod Cluster Side
cd nemo_experiments/hf_llama
tensorboard --logdir ./ --port 8000

# Port-forwarding from Local
ssh -L 8000:localhost:8000 -N [cluster-name] # localhost:8000
```

![image.png](https://github.com/user-attachments/assets/bb223564-2fea-4789-9c0b-75e383d329bf)

## Analyze

### Training speed

Training **FLOPS** can be approximated as **634TFLOPS** (`TFLOPS ~ 6*N*D`). According to the [AWS neuron documents of Trainium performance](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium.html), each Trainium chip delivering 190TFLOPS for FP16/BF16 and 47.5 TFLOPS for FL32. A Trn1 instance consists of 16 Trainium chips and we used mixed precision (almost FP16 compute) for training. So **the expected FLOPS we should achieve is 3000TFLOPS**. If we use p4d instance (Nvidia H100 GPUs * 8), we can ideally achieve 2496TFLOPS (ref: [Nvidia A100 spec](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf)).

This training code **only uses 1/5 of the computation resource**, but it’s reasonable number considering network bottlenecks. It can be boosted by different parallelism setting such as smaller TP and larger DP.

### Cost

HyperPod cluster we setup consumed `$22` per hours. We know that Llama3-8B model is trained by 15T tokens and it means the entire computation is **720ZFLOPs** (7.2+E23 FLOPs). With a single Trn1 instance, we need **36 years** and **$7M budget**.
