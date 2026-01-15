---
layout: post
title: "Diary - EFA와 InfiniBand, LLM 아래 숨겨진 인프라 이야기"
tags: archive
---

# EFA와 InfiniBand, LLM 아래 숨겨진 인프라 이야기

LLM 학습시킬 때 아키텍처 디자인이나 학습 코드쪽 경험을 많이 쌓아왔는데, 사실 그 아래에는 GPU 클러스터를 비롯한 인프라가 있다. 
SKT에 있을 때에는 인프라를 담당해주시던 전문가 분들이 계셔서 해당 부분을 자세히 보지 않았다. 근데 AWS는 클라우드 회사이기도 하고, 내가 풀 파이프라인을 풀스택으로 다 봐야 할 것 같다. 
그래서 이참에 인프라에 대해서 공부를 좀 해보려고 한다.

![image.png](https://github.com/user-attachments/assets/e8a42e0f-3102-4799-8e0d-997a54ecb257)

그리고 사실 LLM 아키텍쳐나 학습 코드들도, 사실은 인프라의 구성이나 상황을 고려한 Infra-wise 로 설계해야 한다. 
마음으로는 미리미리 공부해야한다고 생각했는데도, 항상 학습 코드 속도가 안나오거나 에러가 나는 문제가 생기고 나서야 살짝 공부하면서 해결하기 급급했던 것 같다. 
그리고 사실 핵심은, **어떤 모델이 되었든 모델은 앞으로 점점 커질꺼지만 VRAM의 크기가 폭발적으로 증가하기는 어려울 것**이기 ****때문에 인프라 관련 지식 기반을 더 다져야 될 것 같다. 
그래야 새로운 아키텍쳐가 나왔을 때 이런 기초지식들을 잘 갖추고 있어야 더 빨리 적용할 수 있는, 더 뛰어나고 희소성 있는 AI 엔지니어가 될 수 있지 않을까. 

처음 공부해 볼 내용은 대규모 GPU cluster의 핵심인 InfiniBand와, 그리고 AWS가 InfiniBand 대신 사용하는 EFA다. 
알아보는 과정에서 진짜 모르는게 산더미같이 있었다. 
“이것도 몰라?” 싶은 약간 부끄러운 것들이 많았지만, 암튼 이런 탑다운 방식이 기억에 잘 남으니까…

### 일단 InfiniBand와 EFA가 왜 궁금한가?

DeepSeek가 MoE모델을 가속화할 때 사용했던 DeepEP라는 라이브러리가 있다. 
근데 이걸 AWS 클러스터에서 사용하려고 했더니 문제가 생겼다. 
왜냐면 [DeepEP 라이브러리는 InfiniBand 사용을 가정](https://github.com/deepseek-ai/DeepEP/issues/369)하고 있기 때문이다! 
문의를 해보니 놀랍게도 AWS는 100노드가 넘어가는 대형 클러스터였음에도 InfiniBand를 사용하고 있지 않고, Ethernet으로 연결하고 있다고 했다. 
이때까지만해도 InfiniBand가 GPU 클러스터를 구성하는 필수품이라고 생각했었는데, 알고보니 AWS는 자체 기술인 EFA를 활용해 InfiniBand와 유사하게 노드간 네트워크 병목을 줄이고 있었다. 
AWS 입사했는데 이것도 몰라서 되겠어? 싶은 마음에 공부 시작!

### Inter-node network

GPU 서버를 구성할 때 전력 문제 때문에 랙 하나에는 GPU를 8장밖에 못넣는다. 
보통 랙 단위를 노드라고도 부르는데, 노드 내의 통신, 즉 intra-node 네트워크는 다 연결되어 있으니 엄청 빠르다. 
그런데 노드와 노드 사이의 통신, 즉 inter-node 네트워크는 비교적으로 좀 느릴 수 밖에 없다. 
그 이유가 뭐냐면 1) 데이터 복사로 인한 지연이 생기고 2) 대역폭이 비교적 작아서 한번에 이동 가능한 데이터가 작기 때문이다.

데이터 복사와 대역폭에 대해 무진장 설명이 잘 되어 있는 Toss 테크블로그([고성능 GPU 클러스터 도입기 #2: 이주하는 데이터](https://toss.tech/article/30767))를 많이 참고했다. 
일단 AI 학습시 연산은 거의다 GPU에서 일어난다. 
즉 GPU의 VRAM에 있는 데이터를 읽고 쓰는게 연산의 거의 전부고, CPU는 코드 돌리고 데이터 처리 정도만 해주면 된다. 
그런데, 노드 1번의 GPU VRAM에서 노드 2번의 GPU VRAM으로 데이터를 옮길 때 그냥 옮기면 해당 메모리를 CPU에 복사했다가 CPU 끼리 통신을 하고, 다시 GPU에 복사하는 번거로운 과정을 해야 한다. 
이게 데이터 복사로 인한 지연이다.  

![image.png](https://github.com/user-attachments/assets/04433c2b-3166-405e-978c-c709fdd64dd2)
> Original GPU-to-GPU communication

![image.png](https://github.com/user-attachments/assets/2142681b-75b2-47f6-b572-33b4f050b589)
> OS-bypass GPU-to-GPU communication

대역폭의 경우에는 좀 더 간단한 문제인데, Intra-node 네트워크의 경우에는 메인보드에 칩들이 다 붙어있으니 괜찮지만, Inter-node는 물리적으로 떨어져 있다보니 광케이블 같은 선으로 연결해줘야 한다. 
그러니까 신호 왜곡도 있을 수 있고, 누락도 되니까 이런거 처리 과정을 거치느라 속도를 좀 떨어지게 된다. 

따라서 노드간 GPU 통신 병목을 줄이기 위한 핵심은 1) 불필요한 GPU-to-CPU 데이터 복사를 없애고 2) 대역폭이 큰 네트워크를 사용해야 하는 것이다.

### InfiniBand

놀랍게도 InfiniBand는 역사가 꽤 오래됐다. 
2000년에 최초로 InfiniBand 아키텍처가 출시됐는데, 저지연성이면서도 고대역폭의 네트워트가 필요할거라고 생각했던 IBTA에 의해 제안되었고 **HP, IBM, Intel** 등으로 구성된 운영 위원회에 의해 그 스펙이 결정되었다. 
대표적인 제조사로는 Mellanox가 있는데, 2019년에 Nvidia에 69억달러 가치를 인정받고 인수되었다.

InfiniBand의 중요한 점중 하나는 RDMA (Remote Direct Memory Access) 전용 네트워크라는 것이다. 
RDMA라는걸 쉽게 말하면 **CPU나 운영체제의 개입 없이 메모리를 직접 읽고 쓰는 기술**이다. 
GPU를 예로들면, GPU의 메모리인 VRAM에 데이터를 읽거나 쓰려면 원래는 CPU나 OS를 거쳐야 한다. 
그게 무슨말이냐 하면 CPU로 일단 데이터 복사를 해와야 읽거나 쓸 수 있다는 말이다. 
근데 RDMA라는건 그렇게 안하고 GPU에서 GPU로 바로 데이터를 옮길 수 있는 네트워크라고 보면 된다. 

그러면 RDMA라는게 InfiniBand만 있느냐는 질문을 할 수 있는데, 그렇지 않다. 
Nvidia에서 RoCE (RDMA over Converged Ethernet) 라는 솔루션을 개발했는데, Ethernet을 가지고 GPU RDMA를 하는것이다. 
사실 속도는 고속 이더넷 케이블을 사용할 경우 800Gbps 정도, InfiniBand 는 1.6Tbps 정도 나오니 2배정도 차이가 난다. 
옛날에는 격차가 더 컸는데, 고속 이더넷 기술이 많이 발전되어서 격차가 많이 줄어들었다. 

Nvidia는 **RDMA통신을 위해 정의된 InfiniBand의 패킷을 Ethernet으로 전송**하는 방식으로 RoCE를 구현했다. 
원래는 Microsoft가 데이터센터에서 쓰는 이더넷 통신 기술로 개발한 것이라고 하는데, Nvidia에서 GPU RDMA 통신을 위해 정착시켰다고 한다. 
RoCE가 중요한 이유는 Ethernet의 높은 호환성과 낮은 가격 때문이다. 
옛날에 듣기로는 InfiniBand가 미터당 100달러가 넘는다고 들었는데, 이더넷은 이것보다 훨씬 싸고 제조사도 많으니 싸질 가능성도 훨씬 높다. 

### EFA

그렇다면 EFA는 무엇일까? 
EFA (Elastic Fabric Adapter)는 AWS 자체개발 네트워크 프로토콜인 **SRD (Scalable Reliable Datagram)** 을 활용한 네트워크 인터페이스다. 
SRD라는건 TCP만큼 신뢰성을 보장하면서도 UDP처럼 단일경로가 아닌 다양한 네트워크로 정보를 주고 받기 위해 개발된 AWS 인프라에서만 동작하는 방식이다 ([TCP/UDP 설명 참고](https://mangkyu.tistory.com/15)). 
SRD는 AWS의 [Annapurna Labs 에서 게재한 논문](https://ieeexplore.ieee.org/document/9167399)을 기반으로 하는 기술인데, 신기한 히스토리가 있다. 

Annapurna Labs는 2015년 AWS가 3.7억달러에 인수한 이스라엘 반도체 회사로 현재 AWS 내에서 자체 반도체를 개발하고 있는 조직으로, 인수된 후에는 Graviton, Trainium 등 다양한 칩들을 개발하여 AWS의 컴퓨팅 비용 효율화에 기여하고 있다. 
Annapurna Labs의 창업자는 Galileo Tech라는 반도체 칩 회사를 창업했던 이력이 있는데, InfiniBand의 제조사이자 Nvidia에 인수된 Mellanox의 창업자들 또한 Galileo Tech 임원진 출신이다. 
뿌리가 비슷해서 그런지, 하는 생각도 비슷한가보다.

아무튼 EFA는 SRD를 써서 네트워크 대역폭을 확 늘렸고, **libfabric이라는 오픈소스를 활용해 RDMA를 가능케**했다. 
Application → MPI/NCCL → **Libfabric** → EFA 로 연결되는 구조라고 보면 되는데, OS-bypass communication 을 통해서 RDMA를 가능하게 만들어주는 라이브러리라고 보면 된다. 
2018년부터 AWS EC2 인스턴스에서 EFA를 지원하기 시작했는데, CPU 자원은 c5n 부터 가능하다는것만 주의하면 되고 GPU 자원은 다 된다고 보면 된다. ([EFA 지원 instance 참고](https://docs.aws.amazon.com/ko_kr/AWSEC2/latest/UserGuide/efa.html))

**AWS는 결국 InfiniBand가 아니라 EFA를 사용함으로써 범용성을 선택**했다고 볼 수 있다. 
GPU instance를 멀티노드로 원하는 사람이 많으면 모르겠는데, 싱글노드를 빌려버리면 inter-node network는 쓸모가 없다. 
또 GPU가 아닌 instance 들도 다양하게 취급하니까 Ethernet 기반의 범용 네트워크를 깔아둘 수 밖에 없다. 
다만 초대형 모델 개발자들이 있는 회사는 보통 On-prem cluster를 구축해서 쓰니까 보니까 아직까지는 InfiniBand에 dependency가 걸린 코드들이 많은 것 같다 (DeepEP도 그 중 하나…). 
앞으로 멀티노드 학습하는 오픈소스 컨트리뷰터들이 생기면 자연스레 해결될 문제일 것 같다.

### NVLink

사실 InfiniBand나 EFA는 물리적으로 떨어져 있는 서버를 연결하려는 솔루션이기 때문에 하나의 랙에 GPU를 다 넣어놓는것보다 빠를수가 없다. 
같은 랙에 GPU가 있기만 하면 극강의 속도로 통신이 가능한데, 그 기술이 바로 NVLink다. NVLink 란 Nvidia에서 만들어낸 **GPU 간 초고속 데이터 통신 기술**이다. 
CPU를 거치지 않고 GPU끼리 다이렉트로 데이터를 송수신하는 것으로 Blackwell GPU에서는 거의 14Tbps 정도 나오니까 InfiniBand랑은 비교불가로 무진장 빠르다고 보면 된다. 

원래는 이게 point-to-point 방식으로 GPU 두 개를 서로 양방향으로 직접 연결하는 네트워크인데, NVLink Switch 라는 걸 사용해서 여러개의 GPU를 묶어줄 수 있다. 
실제로 NVL72라는 특수 랙이 있는데, 이건 B200 72개를 하나의 랙에다가 NVLink Switch로 묶어놓은 무지막지한 녀석이다. 
총 대역폭이 1040Tbps라고 하니…말 다했다. 
그러나 아주 큰 전력을 요구하고, 냉각처리도 수냉식으로만 동작하므로 아직까지는 제한적으로 사용할 수 밖에 없다.

### 몰랐던/헷갈리는 용어들 정리

마지막으로, 공부하면서 보게 된 단어들인데 제대로 모르고 있었거나 헷갈리는 개념이었던 것들은 간단히 정리해봤다.

- NIC (Network Interface Controller)
    - 컴퓨터를 네트워크에 연결해주는 하드웨어 장치
    - 네트워크 카드, 랜카드, 이더넷 카드 등 다양한 카드들을 통칭
    - MAC 주소를 통한 네트워크 식별 / 하드웨어 기반 암호화 등
    - 1~100Gbps 정도 속도
- ENI (Elastic Network Interface)
    - EC2 인스턴스간 네트워크 통신을 위한 가상 네트워크 인터페이스
    - NIC 가 물리서버를 연결하는 네트워크 인터페이스라면 ENI는 가상서버인 EC2를 연결
    - 10Gbps 이하 (10Gbps 이상은 ENA: Elastic Network Adapter)
        - ENA는 **SR-IOV** (Single Root I/O Virtualization) 을 통해 네트워크 속도를 올림 → **하나의 PCIe 장치를 여러개의 가상 PCIe 장치로 보이게 만드는 기술** (PCIe 는 간섭없는 병렬방식이므로 개수가 늘어나면 속도를 올릴 수 있음)
- HCA (Host Channel Adapter)
    - InfiniBand 네트워크 연결을 위한 특수 인터페이스 카드
    - Latency 1~2 µs / 200Gbps / RDMA
- HBA (Host Bus Adapter)
    - 서버와 스토리지 장치를 연결하는 인터페이스 카드
    - 서버 CPU 부하를 줄이기 위한 하드웨어 기반 오프로딩 (CPU 부담 없이 독립적으로 I/O)
    - 8~64Gbps
- **PCIe (Peripheral Component Interconnect express)**
    - **범용 인터페이스 → CPU, GPU, 스토리지, 이더넷, IB 등 다양한 장비의 연결을 지원하는 스위치**
    - 레인별로 간섭없는 송수신이 이뤄지는 병렬 방식 (각 레인은 직렬)
    - PCIe x16 이라고 하면 16 레인 할당을 의미
    - PCIe 슬롯이 있어서, 해당 슬롯에 카드들이 장착되는 것
- **NCCL (Nvidia Collective Communication Library)**
    - 여러 프로세스간의 통신: collective communication
        - Broadcast, scatter, gather, all-gather, all-to-all, reduce, all-reduce
    - 모든 collective communication은 Ring topology 로 생각하면 거의 최적임이 증명되었음
- MPI (Message Passing Interface) / GLOO
    - CPU 기반 병렬 컴퓨팅에 쓰이는 라이브러리
    - GLOO 는 Meta에서 만든거라 torch distributed 의 기본 백엔드로 지정되어 있음
    - OpenMPI는 다양한 언어를 지원하고 기능이 풍부하다는 장점
- Storage (SSD) ↔ CPU 통신
    - SATA (Serial ATA)
        - AHCI 통신 프로토콜을 사용하는 직렬 인터페이스
        - 600MB/s 정도의 속도
    - **NVMe (Non-Volatile Memory express)**
        - NAND (SSD storage) 속도를 극대화 하기 위해서 PCIe bus를 직접 사용하는 고속 프로토콜
        - 6GB/s 정도의 속도

### Reference

- [https://computing-jhson.tistory.com/81](https://computing-jhson.tistory.com/81)
- [https://toss.tech/article/30767](https://toss.tech/article/30767)
- [https://aws.amazon.com/ko/blogs/tech/aws-efaelastic-fabric-adaptor/](https://aws.amazon.com/ko/blogs/tech/aws-efaelastic-fabric-adaptor/)
- 코드 쪽에서는 진짜 유용하게 쓰일지도 모르겠음

