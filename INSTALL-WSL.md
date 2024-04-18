# Polyglot-ko-1.3b-lite (on WSL2)

[EleutherAI/polyglot-ko-1.3b](https://huggingface.co/EleutherAI/polyglot-ko-1.3b/)를 기반으로, PEFT 기법 중에 하나인 QLoRA로 미세조정한 모델입니다.<br/>
본 프로젝트의 기반인 Polyglot 모델에 대한, 자세한 정보는 [여기](https://github.com/EleutherAI/polyglot)를 참조하세요.

본 문서는 WSL2가 설치 완료된 상태를 전제 하에, 본 프로젝트의 개발환경을 구축하고, 사용하는 절차를 설명합니다.


## Windows WSL2 시스템에, AI 개발환경 구축
Windows 환경에서 AI 프로젝트를 개발하려면, 여러가지 제약을 마주치게 됩니다.<br/>
중요 패키지가 아직은 Windows 환경에 지원되지 않아서, 기다려야 하는 것들입니다.

빠르게 발전하는 AI 환경에서, 높은 수준의 AI 개발자를 희망한다면, WSL2 환경 안에 구축하는 것을 권장합니다.


### WSL2 설정
```
- [WSL2]에서 su 혹은 sudo 명령어를 사용하기 위해, [/etc/wsl.conf] 파일을 열고, 아래와 같이 설정
  [boot]
  systemd=true
- [WSL2] 재부팅
  > wsl --shutdown
```

### WSL2에, NVIDIA 그래픽 드라이버 설치
Windows 시스템에 설치된 NVIDIA 드라이버가 있다면, WSL2에 자동으로 적용되기에 건너뛰어도 된다.

```
- 기본 드라이버 설치
  > sudo apt-get update
  > sudo apt install -y ubuntu-drivers-common
- Cuda 재설치 == 전체 삭제
  *만약 재설치가 아니라면, 이 부분은 건너뛰고 다음 항목 진행*
  > sudo apt-get purge nvidia*
  > sudo apt-get autoremove
  > sudo apt-get autoclean
  > sudo rm -rf /usr/local/cuda*
- 삭제 확인
  > nvcc -V
- Cuda 가능한 GPU인지 확인
  > lspci | grep -i nvidia
- Ubuntu 시스템 확인
  > uname -m && cat /etc/*release
- gcc 컴파일러 확인/설치
  > gcc --version
  - 없다면 development tools 설치
  > sudo apt update
  > sudo apt install build-essential
- 그래픽 카드 확인
  > sudo lshw -numeric -C display
- 호환 가능한 버전 확인
  https://docs.nvidia.com/deploy/cuda-compatibility/
- 원하는 버전 설치(다른 버전 설치 시 525 변경)
  > sudo apt install nvidia-driver-525
- 재부팅 후, 아래의 명령이 제대로 수행되면 성공
  > nvidia-smi
```

### CUDA 설치
```
- CUDA 12.1 설치 페이지 접속
  https://developer.nvidia.com/cuda-12-1-1-download-archive
- 적합한 것들 선택
  Linux > x86_64  > WSL-Ubuntu > 2.0 deb(local)
- 나열되는 명령들 수행
  > wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
  > sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
  > wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-wsl-ubuntu-12-1-local_12.1.1-1_amd64.deb
  > sudo dpkg -i cuda-repo-wsl-ubuntu-12-1-local_12.1.1-1_amd64.deb
  > sudo cp /var/cuda-repo-wsl-ubuntu-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
  > sudo apt-get update
  > sudo apt-get -y install cuda
- 환경변수 설정/적용
  > sudo vim ~/.bashrc
    export PATH=/usr/local/cuda-12.1/bin:${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
  > source ~/.bashrc
- 확인
  > nvcc -V
```

### CuDNN 설치
```
- CuDNN 12.1 설치 페이지 접속
  https://developer.nvidia.com/rdp/cudnn-download
- 적합한 것들 선택
  Linux > x86_64  > Ubuntu > 22.04 > deb(local)
- 나열되는 명령들 수행
  # 네트웍 설치
    > wget https://developer.download.nvidia.com/compute/cudnn/9.0.0/local_installers/cudnn-local-repo-ubuntu2204-9.0.0_1.0-1_amd64.deb
    > sudo dpkg -i cudnn-local-repo-ubuntu2204-9.0.0_1.0-1_amd64.deb
    > sudo cp /var/cudnn-local-repo-ubuntu2204-9.0.0/cudnn-*-keyring.gpg /usr/share/keyrings/
      sudo cp /var/cudnn-local-repo-ubuntu2204-9.0.0/cudnn-local-960825AE-keyring.gpg /usr/share/keyrings/
    > sudo apt-get update
    > sudo apt-get -y install cudnn
  # 로컬 다운로드 설치
    # mv /mnt/c/Users/${계정명}/Downloads/${파일명} ./
    > mv /mnt/c/Users/user/Downloads/cudnn-local-repo-ubuntu2204-9.0.0_1.0-1_amd64.deb ./
    > sudo dpkg -i cudnn-local-repo-ubuntu2204-9.0.0_1.0-1_amd64.deb
    > sudo cp /var/cudnn-local-repo-ubuntu2204-9.0.0/cudnn-*-keyring.gpg /usr/share/keyrings/
    > sudo apt-get update
    > sudo apt-get -y install cudnn
      or
      sudo apt-get install libcudnn9=9.0.0-1+cuda12.1
      sudo apt-get install libcudnn9-dev=9.0.0-1+cuda12.1
      sudo apt-get install libcudnn9-samples=9.0.0-1+cuda12.1
- Cuda 관련 패키지 설치
  # CUDA 11용으로 설치하려면 위 구성을 수행하되 CUDA 11 관련 패키지를 설치하십시오.
    > sudo apt-get -y install cudnn-cuda-11
  # CUDA 12용으로 설치하려면 위 구성을 수행하되 CUDA 12 관련 패키지를 설치하십시오.
    > sudo apt-get -y install cudnn9-cuda-12
- 확인
  > cat /usr/include/x86_64-linux-gnu/cudnn_version_v9.h | grep CUDNN_MAJOR
```

### Anaconda 설치 및 가상환경(pgko) 생성
```
- Anconda-Archive(https://repo.anaconda.com/archive/)에서 필요한 버전 다운로드
  > wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
    or
    cd <로컬다운로드경로>   # ex: /mnt/d/CD/WSL2-Ubuntu/
- 다운받은 경로에서 bash로 실행
  > bash Anaconda3-2024.02-1-Linux-x86_64.sh
- 설치 진행
  엔터를 누르다 보면, 동의할지 물어보는데, "yes" 입력
- 루트 경로를 어떻게 할지도 물어본다.
  기본 경로는 /home/{user}/anaconda3 이나 바꾸고 싶으면, 원하는 경로를 입력해주면 된다.
- "conda init" 수행할지 여부를 물으면, "yes" 입력
- 위에서 자동으로 경로 설정을 해주었지만, .bashrc를 최신화 해주어야 적용된다.
  > source ~/.bashrc
  (base) master@LAPTOP:/mnt/d/CD/WSL2-Ubuntu$ cd /home/master/anaconda3
  (base) master@LAPTOP:~/anaconda3$ 
- 경로를 바꿔야 하거나, 제대로 추가되지 않았다면, 직접 경로 설정
  > sudo vim ~/.bashrc
    export PATH="/home/{user}/anaconda3/bin:$PATH"
  > source ~/.bashrc
- 확인
  > conda env list
  > conda list
- 가상환경(pgko) 생성 : python 3.10
  > conda create -n pgko python=3.10
  > conda env list
  > conda activate pgko
  (pgko) > python -V
  (pgko) > exit
```

### Python 가상환경(pgko)에, Pytorch 설치
```
[Pytorch](https://pytorch.org/) 사이트에서, 자신의 환경에 적합한 명령을 제작합니다.
  Stable (2.2.2)  > Linux > Conda > Python > CUDA 12.1
Pytorch를 개발에 직접 사용하지 않더라도 설치하는 이유는, Python 가상환경에서 Cuda가 정상적으로 연동되는지 알아보기 위함이다.
이때, Pytorch와 연동시킬 Cuda 버전을 [12.1]로 설치하였기 때문에, 앞으로 CudaToolkit(Cuda + CudNN)도 동일하게 맞추어 설치해야만 된다.
- Pytorch 설치
  > conda activate pgko
  (pgko) > conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
- Python에서 확인
  (pgko) > python
  >>> import torch
  >>> print(torch.cuda.is_available())
  True
  >>> exit()
```

### 본 프로젝트의 Python 가상환경(pgko)에, 종속성 설치
```
> conda activate pgko
(pgko) > pip install -r requirements.txt
```


## 라이센스

[Apache 2.0](./LICENSE) 라이센스를 따릅니다.<br/>
라이센스에 따라, 주의사항을 지켜주시기 바랍니다.
