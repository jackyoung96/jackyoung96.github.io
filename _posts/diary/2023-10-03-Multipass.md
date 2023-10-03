---
layout: post
title: Diary - Mac OS (Apple silicon M1) 에서 Ubuntu 사용하기
tags: archive
---

연구실을 나와서 가장 불편한 점을 하나만 꼽으라면 나만 사용할 수 있는 Linux 컴퓨터가 사라진 점이다. 개인 GPU 도 없고...Ubuntu 를 기반으로 동작하는 open-source 들이 너무 많아서 이것저것 튜토리얼을 돌려보고 싶어도 안될 때가 많다.  
그렇다고 내 피같은 맥북을 듀얼부팅 할수는 없다고 생각해왔는데, 너무 불편해서 못참겠다. VM 을 사용하는 방법도 있지만 조금 더 간편한 방법을 찾았다. 바로 `Multipass` 다.  

## Multipass 란?

> Multipass란 Ubuntu 개발사인 Canonical 에서 제공하는 가상머신 솔루션이다. Canonical에서 2019년에 만든 가상머신 솔루션으로 Terminal 기반의 인터페이스를 지원하는 대신 매우 적은 자원 소비를 장점으로 한다.

**매우 적은 자원 소비?** 라는 말에 바로 혹해서 설치를 결심한다.

설치 방법은 간단하다.

```bash
brew install --cask multipass
```
그래도 linux 이미지들을 설치해야하기 때문에 시간이 꽤 걸린다.  
설치가 완료되면 이미지 리스트 찾아볼 수 있다.
```bash
multipass find
```
<img width="771" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/cf64cc5b-6ea3-4caa-83b4-be570693e735">
재밌는건 ROS 를 제공한다는 것. 로보틱스가 이정도까지 올라왔나 싶기도 하다. (괜히 버리고 도망쳤나?)

내가 원하는 이미지는 18.04 였는데 여기에서는 보이지 않는다. 그래도 추가 설치가 가능하기 때문에 걱정하지 말자.

```bash
# multipass launch [Image] -n [VM 이름] -c [CPU 코어] -m [메모리] -d [디스크 공간]
multipass launch 18.04 -n linux -c 4 -m 4G -d 20G
```
위 명령어를 통해 Ubuntu 18.04 를 linux 라는 가상머신 이름으로 설치한다. 메모리와 디스크 공간, CPU 개수는 알아서 적절하게 선택해주면 된다. (많이 쓰면 그만큼 켜뒀을 때 맥북 성능이 떨어진다.)

이제 linux 쉘에 접근하여 Ubuntu 18.04 를 사용하면 된다.

```bash
# multipass shell [VM 이름]
multipass shell linux
```

<img width="431" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/092eb326-9479-4cf1-bb95-15a6ad5c2bcc">

Oh yeeahhh~~

## Multipass vscode 에서 사용하기

나는 VScode 가 없으면 코딩을 못하기 때문에, multipass 환경을 vscode 를 통해 작업하는 방법을 알아보겠다.

우선 ssh 접속을 위한 공개키를 만들어준다.
```
ssh-keygen -t rsa
```
나오는 문구들 (키 생성 위치, passphrase) 은 그냥 다 enter 로 넘겨준다. 그러면 `~/.ssh/id_rsa.pub` 파일이 생긴 것을 확인할 수 있다. 열어보면 공개키가 생성된 것을 확인할 수 있다.  
이제 공개키를 이용해 yaml 파일을 하나 만들어준다. yaml 파일의 내용은 아래와 같다.

```yaml
groups:
  - vscode
runcmd:
  - adduser ubuntu vscode
ssh_authorized_keys:
  - ssh-rsa <public key>
```
public key 에 생성된 공개키를 복사해주면 된다. 이 yaml 파일을 이용해 VM 을 다시 launch 해준다.
```bash
multipass launch 18.04 -n linux -c 4 -m 2G -d 20GB --cloud-init vscode.yaml
```
vscode 를 통한 ssh 접근이 가능하도록 설정되었다!  
그러면 가상머신의 IP 를 아래 커맨드로 확인해준다.
```bash
multipass info linux
```

이제 로컬의 위치로 돌아가서 ssh config 를 설정해준다.  
`~/.ssh/config` 파일을 열고 아래 내용을 추가한다.

<img width="159" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/696fc1d0-a7a7-473c-ad8b-95620cb29ba2">

이러면 ssh 를 통한 ubuntu 가상머신에 접근이 가능하다!  
마지막으로 local 의 저장공간을 mount 해준다.

```bash
multipass mount [원하는 local 경로] [VM 이름]:[VM 경로]
```

Mount 성공 이후 권한이 없다는 오류가 발생할 수 있는데, 이는 디스크 접근에 대한 맥북의 권한 문제 때문이다.  
`설정 -> 개인정보 및 보안 -> 전체 디스크 접근 권한` 으로 가서 mutlpassd 에 권한을 부여해주면 해결할 수 있다.  

이제 맥북에서 vscode 를 통해 ubuntu 환경을 사용할 수 있다!

## Uninstall 

multipass 환경을 계속 켜두면 메모리를 잡아먹으니 안쓸때에는 꺼두는게 좋다.
```bash
multipass stop linux # 가상 머신을 멈춘다
multipass delete linux # 가상 머신을 관리 대상에서 삭제한다
multipass purge # 가상 머신을 완전히 삭제한다
```
purge 를 해버리면 복구할 수가 없다. 나머지는 복구가 가능하다.
```bash
multipass start linux # stop 에 대한 복구
multipass recover linux # delete 에 대한 복구
```

stop 과 delete 의 차이는 다음과 같다.
- stop: primary instance 를 삭제
- delete: 모든 instance 를 삭제

## References

https://elsainmac.tistory.com/870  
https://discourse.ubuntu.com/t/using-multipass-with-vscode/34905  
https://github.com/canonical/multipass/issues/1389   
https://multipass.run/docs/delete-command  
https://multipass.run/docs/stop-command