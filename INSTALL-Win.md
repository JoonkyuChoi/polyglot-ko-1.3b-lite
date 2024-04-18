# Polyglot-ko-1.3b-lite (on Windows)

[EleutherAI/polyglot-ko-1.3b](https://huggingface.co/EleutherAI/polyglot-ko-1.3b/)를 기반으로, PEFT 기법 중에 하나인 QLoRA로 미세조정한 모델입니다.<br/>
본 프로젝트의 기반인 Polyglot 모델에 대한, 자세한 정보는 [여기](https://github.com/EleutherAI/polyglot)를 참조하세요.

본 문서는 WSL2를 사용하지 않은, 순수한 Windows 환경에서, 본 프로젝트의 개발환경을 구축하고, 사용하는 절차를 설명합니다.


## Windows 시스템에, AI 개발환경 구축
Windows 환경에서 AI 프로젝트를 개발하려면, 여러가지 제약을 마주치게 됩니다.<br/>
중요 패키지가 아직은 Windows 환경에 지원되지 않아서, 기다려야 하는 것들입니다.

빠르게 발전하는 AI 환경에서, 높은 수준의 AI 개발자를 희망한다면, WSL2 환경 안에 구축하는 것을 권장합니다.<br/>
다만, 모델을 직접 제작(모델제작>미세조정(훈련)>병합>저장)하지 않고, 사전훈련된 모델의 추론(생성) 기능만을 사용하는, 프롬프트 엔지니어링을 원한다면, Windows 환경에서도 충분합니다.

다행히, 본 프로젝트는 가벼운 과거 모델에 해당하기에, [미세조정(훈련)>병합>저장>추론(생성)]의 모든 절차를 수행시킬 수 있습니다.


### 운영시스템에, NVIDIA 드라이버 설치

- 먼저 GPU 카드에 액세스하려면, 어떤 드라이버가 필요한지 파악해야 합니다.<br/>
  장치 관리자를 검색하면, 디스플레이 어댑터 아래에서 볼 수 있습니다.
- 자신의 PC에서, 모든 Visual Studio 프로그램 종료
- [NVIDIA](https://www.nvidia.co.kr/Download/index.aspx?lang=kr) 사이트에서, 자신에게 알맞는 드라이버를 선택하고, 다운로드하여 설치<br/>
  설치가 시작된 후, Windows 업데이트가 시작되면, 설치가 실패할 수 있습니다.<br/>
  Windows 업데이트가 완료될 때까지, 기다린 후 설치를 다시 시도하세요.

### 운영시스템에, Python 가상환경을 사용하기 위한, [Anaconda] 설치
- [Anaconda](#https://www.anaconda.com/download)를 다운로드하여 설치
- [Anaconda] 설치 후, 최초 단 한번만, 콘솔창을 열고 "conda init" 명령 수행
- Path 환경변수에, Anaconda 스크립트 경로 등록(ex: D:\ProgramData\anaconda3\Scripts)<br/>
  모든 콘솔창을 닫고, 관리자 모드로 다시 열어야만 적용된다.

### 본 프로젝트를 위한, Python 가상환경(pgko) 생성 및 활성화
```
> conda create -n pgko python=3.10
> conda env list
> conda activate pgko
```

### 본 프로젝트의 Python 가상환경(pgko)에, Pytorch 설치
[Pytorch](https://pytorch.org/) 사이트에서, 자신의 환경에 적합한 명령을 제작합니다.<br/>
Pytorch를 개발에 직접 사용하지 않더라도 설치하는 이유는, Python 가상환경에서 Cuda가 정상적으로 연동되는지 알아보기 위함이다.<br/>
이때, Pytorch와 연동시킬 Cuda 버전을 [12.1]로 설치하였기 때문에, 앞으로 CudaToolkit(Cuda + CudNN)도 동일하게 맞추어 설치해야만 된다.
```
(pgko) > conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 운영시스템에, CudaToolkit 설치
<font color="#FF5733">이미 운영시스템에, CudaToolkit이 설치되었다면, 본 섹션은 무시해야 합니다.</font>

[cuda-12-1-1-download-archive](https://developer.nvidia.com/cuda-12-1-1-download-archive)에서, 다운로드하여 시스템 계정에 설치하자.<br/>
반드시, NVIDIA Driver가 설치되어야만, [nvidia-smi.exe]를 사용할 수 있으며, 이를 pytorch가 사용합니다.<br/>
Windows 시스템에 Cuda는 여러버전을 [설치](https://tw0226.tistory.com/79)할 수 있다.<br/>
Windows 환경에서 TensorFlow를 사용하려면, 더 낮은 버전의 Cuda를 사용해야 합니다.<br/>
설치가 완료되었다면, 환경변수들을 설정해야 합니다.
```
- 환경변수 설정
  > SETX CUDA_PATH        "D:\NVIDIA_GPU_Computing_Toolkit\CUDA\v12.1" /M
  > SETX CUDA_PATH_V12_1  "D:\NVIDIA_GPU_Computing_Toolkit\CUDA\v12.1" /M
- Path 환경변수에 경로 추가
  > [Windows + S] > 시스템 환경 변수 편집 > 환경 변수 > 시스템 변수 > Path
    D:\NVIDIA_GPU_Computing_Toolkit\CUDA\v12.1\bin
    D:\NVIDIA_GPU_Computing_Toolkit\CUDA\v12.1\libnvvp
```

### 운영시스템에, CudNN 설치 및 테스트
<font color="#FF5733">이미 운영시스템에, CudNN을 다운받아 구성하였다면, 본 섹션은 무시해야 합니다.</font>

[CudNN]을 다운로드하려면, 회원가입이 필요하다.<br/>
[cudnn-archive](https://developer.nvidia.com/rdp/cudnn-archive)에서, 다운로드하여 압축을 해제하여, CUDA_PATH 경로에 복사하면 된다.<br/>
새로운 콘솔창을 관리자 모드로 열고, 아래의 콘솔명령을 수행하여, Cuda가 정상적으로 동작하는지 확인하자.

Python 코드를 수행시켜, True 결과를 얻어야 합니다.
```
(pgko) > python
>>> import torch
>>> print(torch.cuda.is_available())
True
>>> exit()

> python -m torch.utils.collect_env
```

### 운영시스템에, Cuda를 MSVC와 연동
<font color="#FF5733">이미 운영시스템에, 구성한 상태라면, 본 섹션은 무시해야 합니다.</font>

아래와 같이, Cuda 폴더에 존재하는 파일들을 MSVC 폴더에 복사시켜야 합니다.
```
> Cuda: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\extras\visual_studio_integration\MSBuildExtensions
  to
> MSVC: C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\BuildCustomizations
```

### 본 프로젝트의 Python 가상환경(pgko)에, 종속성 설치
```
> conda activate pgko
(pgko) > pip install -r requirements.txt
```


## 라이센스

[Apache 2.0](./LICENSE) 라이센스를 따릅니다.<br/>
라이센스에 따라, 주의사항을 지켜주시기 바랍니다.
