"""
> 이전 훈련가중치를 병합하지 않을 것이라면, output_dir 폴더를 삭제한 후, 훈련시켜야 한다.
  병합 후에 훈련을 시켜야만, 새로운 가중치가 적용됩니다.
> 병합모델 훈련 후에, 상태사전에 존재하는 lora 모델의 가중치를 제거시켜야, 저장용량을 줄일 수 있습니다.
> 저장은 병합모델이 아닌, 사전훈련모델(기반모델)을 사용해야만 합니다.
"""
import os
import sys

import fire
import torch

from peft import (
    PeftModel,
    PeftConfig,
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# 훈련가중치 최종 누적 경로를 얻기 위해
from transformers.trainer_utils import get_last_checkpoint

def merge(
    # model/peft params
    base_model: str = None,         # 기반 모델 경로
    peft_root: str = "train/lora-e5/b16",                   # PEFT 훈련가중치 루트경로
    output_dir: str = "resources/polyglot-ko-1.3b-e5b16",   # 병합된 사전훈련모델 저장경로
    max_shard_size : str = "10GB",  # 모델 파일 단위 사이즈
):
    # --------------------------------------
    # 환경변수 설정
    # --------------------------------------
    # GPU 수량에 따른, 환경변수 설정
    GPUs = torch.cuda.device_count()
    if GPUs <= 1:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # OS에 따른, 환경변수 설정
    if sys.platform == "win32":
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"     # cpu 기반 분산훈련
        #os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "mpi"     # cpu 기반 분산훈련 : 선택적 백엔드로, PyTorch를 소스에서 빌드할 때만 포함
    else:
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "nccl"     # gpu 기반 분산훈련
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    # --------------------------------------
    # 정보 화면출력
    # --------------------------------------
    print("----------------------------------------")
    print(f"platform: {sys.platform}, torch: {torch.__version__}, GPU: {GPUs}")
    print("----------------------------------------")
    # [LOCAL_RANK] 환경변수가 정의되지 않았거나, [0]이면 파라미터 출력
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Merge model and peft with params:\n"
            f"base_model: {base_model or False}\n"
            f"peft_root: {peft_root}\n"
            f"output_dir: {output_dir}\n"
            f"max_shard_size: {max_shard_size}"
        )
        print("----------------------------------------")
    # --------------------------------------
    # 파라미터 체크
    # --------------------------------------
    # [peft_root] 파라미터 체크
    assert(peft_root), "Please specify a --peft_root, e.g. --peft_root='train/lora-e5/b16'"
    assert(os.path.isdir(peft_root)), "Invalid Path: peft_root"
    # [output_dir] 파라미터 체크
    assert(output_dir), "Please specify a --output_dir, e.g. --output_dir='resources/polyglot-ko-1.3b-e5b16'"
    # --------------------------------------
    # [WORLD_SIZE] 환경변수에 따른, [device_map] 설정
    # --------------------------------------
    """
    device_map = "auto"
    # [WORLD_SIZE] 환경변수 얻기(정의하지 않았으면 1)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # [WORLD_SIZE != 1]라면, [device_map] 재설정
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    """
    # --------------------------------------
    # 0. 준비 작업
    # --------------------------------------
    # 0-1. 기반모델 경로(base_path) 얻기
    # ------------------
    base_path = None
    if base_model and os.path.isdir(base_model):
        base_path = base_model
    if not base_path:
        # PEFT 가중치 경로로부터, 구성정보(adapter_config.json) 로드
        peft_config = PeftConfig.from_pretrained(peft_root)
        base_path = peft_config.base_model_name_or_path
    # [base_path] 파라미터 체크
    assert(os.path.isdir(base_path)), "Found not a path for base model. Please specify a --base_model, e.g. --base_model='resources/polyglot-ko'"
    # ------------------
    # 0-2. PEFT 루트경로(peft_root)로부터, 최종 체크포인트(훈련가중치) 경로(last_peft_path) 얻기
    # ------------------
    last_peft_path = get_last_checkpoint(peft_root)
    if last_peft_path is None:
        if len(os.listdir(peft_root)) > 0:
            raise ValueError(f"PEFT directory({peft_root}) already exists and is not empty.")
        else:
            raise ValueError(f"Found not checkpoint for PEFT in {peft_root}.")
    else:
        print(
            f"Checkpoint detected: {last_peft_path}\n"
            "----------------------------------------"
        )
    # --------------------------------------
    # 1. 기반 모델+토크나이저, PEFT 가중치 로드
    # --------------------------------------
    # 1-1. 사전훈련모델(komodel) 로드
    # ------------------
    print(
        f"Loading model: {base_path}\n"
        "----------------------------------------"
    )
    komodel = AutoModelForCausalLM.from_pretrained(
        base_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )
    """
    komodel = AutoModelForCausalLM.from_pretrained(
        base_path,
        # 4비트 로드 양자화 적용
        use_safetensors=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            bnb_4bit_use_double_Quant=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type='nf4'
        ),
        torch_dtype=torch.float16,
        device_map=device_map,
        #trust_remote_code=True,        # HuggingFace 원격저장소에서 코드를 다운로드할 때, 코드를 신뢰할지 여부
    )
    """
    # ------------------
    # 1-2. 사전훈련된 토크나이저(tokenizer) 로드
    # ------------------
    print(
        f"Loading tokenizer: {base_path}\n"
        "----------------------------------------"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    # ------------------
    # 1-3. 사전훈련모델에 PEFT 가중치 로드
    # ------------------
    print(
        f"Loading PEFT weights: {last_peft_path}\n"
        "----------------------------------------"
    )
    lora_model = PeftModel.from_pretrained(
        komodel,
        last_peft_path,
        device_map={"": "cpu"},
        torch_dtype=torch.float16,
    )
    # --------------------------------------
    # 2. 사전훈련모델에 PEFT 가중치를 병합한 후, 기존 PEFT 제거
    # --------------------------------------
    print(
        "Merging and Unload...\n"
        "----------------------------------------"
    )
    merged_model = lora_model.merge_and_unload(progressbar=True, safe_merge=True)
    # --------------------------------------
    # 3. 병합모델  훈련 시작
    # --------------------------------------
    print(
        "Training...\n"
        "----------------------------------------"
    )
    merged_model.train(False)
    # --------------------------------------
    # 4. 병합모델 상태사전 교정
    # --------------------------------------
    print(
        "Correcting state_dict...\n"
        "----------------------------------------"
    )
    # 병합모델 상태사전 얻기
    merged_model_sd = merged_model.state_dict()
    # 병합모델 상태사전에서, lora 모델의 가중치를 제거시킨 상태사전을 만든다.
    deloreanized_sd = {
        k.replace("base_model.model.", ""): v
        for k, v in merged_model_sd.items()
        if "lora" not in k
    }
    # --------------------------------------
    # 5. [병합모델 & 토크나이저] 저장
    # --------------------------------------
    # 5-1. 모델 저장
    # ------------------
    # 미세조정 가중치 저장
    print(
        f"Saving model: {output_dir}\n"
        "----------------------------------------"
    )
    komodel.save_pretrained(
        output_dir,
        state_dict=deloreanized_sd,
        max_shard_size=max_shard_size,
    )
    # ------------------
    # 5-2. 토크나이저 저장
    # ------------------
    print(
        f"Saving tokenizer: {output_dir}\n"
        "----------------------------------------"
    )
    tokenizer.save_pretrained(output_dir)
    # --------------------------------------
    
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
def _mp_fn(index):
    # For xla_spawn (TPUs)
    fire.Fire(merge)


if __name__ == "__main__":
    fire.Fire(merge)
