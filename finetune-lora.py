"""
> 윈도우 환경에서는 torch.compile(model) 함수가 작동하지 않습니다.
  이 함수가 작동하지 않으면, 훈련시킨 가중치를 기존 모델에 병합시킬 수 없습니다.
  이 것은 자신이 학습시킨 모델을 제작할 수 없음을 의미합니다.
  윈도우에서는 사전학습된 모델을 추론하거나 평가만 할 수 있습니다.
> 이전 훈련가중치를 병합하지 않을 것이라면, output_dir 폴더를 삭제한 후, 훈련시켜야 한다.
"""
import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    #get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
# [JKC:20240408-1420] 4비트 로드 양자화 적용(BitsAndBytesConfig)
from transformers import (
    Trainer,
    BitsAndBytesConfig,
    default_data_collator
)
# [JKC:20240411-1456] 훈련가중치 누적을 위해 추가
from transformers.trainer_utils import get_last_checkpoint
# [JKC:20240416-2239] gpt2 훈련 데이터셋 로드를 위해
from transformers.testing_utils import CaptureLogger
from itertools import chain

from utils import Prompter

def train(
    # model/data params
    base_model: str = "",  # 유일한 필수 인수
    data_path: str = "ko-datasets/KoAlpaca_v1.1a_textonly.json",
    output_dir: str = "train/lora-e5/b16",
    # past params
    gpt2_datasets: bool = True,             # 과거 데이터셋 포맷(GPT2) 여부
    keep_linebreaks: bool = True,           # 줄바꿈 유지 여부
    block_size: int = None,                 # 토큰화 후 선택적 입력 시퀀스 길이 : 훈련 데이터셋은 훈련을 위해, 이 크기의 블록으로 잘립니다.
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_train_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 5,                  # 검증 분할이 없는 경우, 검증 데이터셋처럼 사용되어질 훈련 데이터셋의 퍼센트
    cache_dir: str = None,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [ "query_key_value", ],
    # llm hyperparams
    train_on_inputs: bool = True,  # False인 경우 손실된 입력을 마스킹합니다.
    add_eos_token:  bool = False,
    group_by_length: bool = False,  # 더 빠르지만 이상한 훈련 손실 곡선이 생성됩니다.
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,     # 누적된 훈련가중치 체크포인트 또는 최종 어댑터
    prompt_template: str = "polyglot-ko",   # 사용할 프롬프트 템플릿은 기본적으로 한국어로 설정됩니다.
):
    # [JKC:20240411-1412] GPU 수량에 따른, 환경변수 설정
    GPUs = torch.cuda.device_count()
    if GPUs <= 1:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # [JKC:20240412-0934] OS에 따른, 환경변수 설정
    if sys.platform == "win32":
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"     # cpu 기반 분산훈련
        #os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "mpi"     # cpu 기반 분산훈련 : 선택적 백엔드로, PyTorch를 소스에서 빌드할 때만 포함
    else:
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "nccl"     # gpu 기반 분산훈련
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    # [JKC:20240408-1420] 정보 로그 추가
    print("----------------------------------------")
    print(f"platform: {sys.platform}, torch: {torch.__version__}, GPU: {GPUs}")
    print("----------------------------------------")
    # [JKC:20240411-1600] [batch_size] 보정 추가
    if (batch_size < micro_batch_size):
        print(f"Mismatched batch_size/micro_batch_size: {batch_size}/{micro_batch_size}")
        micro_batch_size = batch_size
        print(f"Changed micro_batch_size: {micro_batch_size}")
        print("----------------------------------------")
    # [LOCAL_RANK] 환경변수가 정의되지 않았거나, [0]이면 파라미터 출력
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            "Training polyglot-ko-1.3b model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            
            f"gpt2_datasets: {gpt2_datasets}\n"
            f"keep_linebreaks: {keep_linebreaks}\n"
            f"block_size: {block_size}\n"

            f"num_train_epochs: {num_train_epochs}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template}\n"
            "----------------------------------------"
        )
    # --------------------------------------
    # 파라미터 체크
    # --------------------------------------
    # [base_model] 파라미터 체크
    assert(base_model), "Please specify a --base_model, e.g. --base_model='resources/polyglot-ko-1.3b'"
    # [JKC:20240411-1600] [batch_size] 파라미터 체크
    assert(batch_size > 0), f"Mismatched batch_size: {batch_size}"
    # --------------------------------------
    # [그라디언트 누적 스탭] 산출 & [프롬프트] 객체생성
    # --------------------------------------
    # [gradient_accumulation_steps] 산출/설정
    gradient_accumulation_steps = batch_size // micro_batch_size
    prompter = Prompter(prompt_template)
    # --------------------------------------
    # [WORLD_SIZE] 환경변수에 따른, [device_map/gradient_accumulation_steps] 재산출
    # --------------------------------------
    device_map = "auto"
    # [WORLD_SIZE] 환경변수 얻기(정의하지 않았으면 1)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # [WORLD_SIZE != 1]라면, [device_map/gradient_accumulation_steps] 재산출
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    # ------------------
    # 데이터셋 로드
    # ------------------
    if gpt2_datasets:
        data_files = {}     # [훈련/검증] 데이터셋 파일 목록
        dataset_args = {}
        # 훈련 데이터셋 파일 로드
        if data_path is not None:
            data_files["train"] = data_path
        # 검증 데이터셋 파일 로드
        #if validation_file is not None:
        #   data_files["validation"] = validation_file
        # 데이터셋 파일확장자 추출
        extension = (
            data_path.split(".")[-1]
            #if data_path is not None
            #else validation_file.split(".")[-1]
        )
        # 데이터셋이 txt 파일이면, "keep_linebreaks" 속성을 적용하여, 로드
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = keep_linebreaks
        # 데이터셋 로드
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=cache_dir,
            **dataset_args,
        )
        #print(f"gpt2 datasets: {raw_datasets}")
        train_samples = raw_datasets["train"].num_rows
        # 검증 데이터가 없으면, val_set_size를 사용하여, 데이터셋을 나눕니다.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{val_set_size}%]",
                cache_dir=cache_dir,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{val_set_size}%:]",
                cache_dir=cache_dir,
                **dataset_args,
            )
        #print(f"gpt2 datasets: {raw_datasets}")
    else:
        if data_path.endswith(".json") or data_path.endswith(".jsonl"):
            raw_datasets = load_dataset("json", data_files=data_path)
        else:
            raw_datasets = load_dataset(data_path)
        train_samples = raw_datasets["train"].num_rows
    # ------------------
    # 정보 출력
    # ------------------
    print(
        "Internal variables:\n"
        "----------------------------------------\n"
        f"world_size: {world_size}\n"
        f"device_map: {device_map}\n"
        f"gradient_accumulation_steps: {gradient_accumulation_steps}\n"
        f"train_samples: {train_samples}\n"
        "----------------------------------------"
    )
    # --------------------------------------
    # [wandb] 환경변수 설정
    # --------------------------------------
    # 매개변수가 전달되었는지 또는 Environ 내에 설정되었는지 확인하세요.
    use_wandb = len(wandb_project) > 0 or ("WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0)
    # wandb 매개변수가 전달된 경우에만 환경을 덮어씁니다.
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    # --------------------------------------
    # 사전훈련모델(komodel) 및 토크나이저(tokenizer) 준비
    # --------------------------------------
    komodel = transformers.AutoModelForCausalLM.from_pretrained(
        base_model,
        # [JKC:20240408-1420] 4비트 로드 양자화 적용
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
        cache_dir=cache_dir,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model,
        cache_dir=cache_dir,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token_id = (
        0  # 엉. 우리는 이것이 EOS 토큰과 다르기를 원합니다.
    )
    # --------------------------------------
    # [신형] 해당 프롬프트를 토크나이즈하는 함수
    # --------------------------------------
    def tokenize(prompt, add_eos_token=True):
        # tokenizer 설정을 사용하여 이를 수행할 수 있는 방법이 있을 수 있지만, 다시 한 번 말씀드리지만 빠르게 움직여야 합니다.
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (result["input_ids"][-1] != tokenizer.eos_token_id and len(result["input_ids"]) < cutoff_len and add_eos_token):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result
    # --------------------------------------
    # [신형] 해당 데이터에 대하여, 프롬프트를 생성하고, 토그나이즈하는 함수
    # --------------------------------------
    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt       = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt             = prompter.generate_prompt(data_point["instruction"], data_point["input"])
            tokenized_user_prompt   = tokenize(user_prompt, add_eos_token=add_eos_token)
            user_prompt_len         = len(tokenized_user_prompt["input_ids"])
            if add_eos_token:
                user_prompt_len     -= 1
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]  # 속도가 빨라질 수도 있겠네요
        return tokenized_full_prompt
    # --------------------------------------
    # 사전훈련모델(komodel)에 PEFT 어댑터 부착
    # --------------------------------------
    # kbit 훈련용 모델 준비
    komodel = prepare_model_for_kbit_training(komodel)
    # LoRA 구성 설정
    config = LoraConfig(
        r               = lora_r,
        lora_alpha      = lora_alpha,
        target_modules  = lora_target_modules,
        lora_dropout    = lora_dropout,
        bias            = "none",
        task_type       = "CAUSAL_LM",
    )
    # PEFT 어댑터 모델로 전환
    komodel = get_peft_model(komodel, config)
    # --------------------------------------
    # 누적된 훈련가중치 체크포인트 경로(resume_from_checkpoint)로부터, 가중치 파일을 로드하여, PEFT 모델에 설정
    # --------------------------------------
    # [누적된 훈련가중치 체크포인트 재개 경로]가 존재하면, [pytorch/adapter] 가중치 로드
    last_checkpoint = None      # [JKC:20240411-1456] 훈련가중치 누적을 위해 추가
    if resume_from_checkpoint:
        # 유용한 가중치 파일을 확인하고 로드
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")     # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(resume_from_checkpoint, "adapter_model.bin") # LoRA 모델만 해당 - 위의 LoRA 구성이 맞아야 함
            resume_from_checkpoint = (False)    # 따라서 trainer는 상태를 로드하려고 시도하지 않습니다.
        # 위 두 파일은 저장 방법에 따라, 이름이 다르지만 실제로는 동일합니다.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)          # 해당 경로의 가중치 로드
            set_peft_model_state_dict(komodel, adapters_weights)    # PEFT 모델에 가중치 설정
        else:
            print(f"Checkpoint {checkpoint_name} not found")
    # [JKC:20240411-1456] 훈련가중치 누적을 위해 추가
    # --------------------------------------
    # 누적된 훈련가중치 체크포인트 경로(last_checkpoint) 얻기
    # --------------------------------------
    else:
        # 이전 가중치 출력경로(output_dir)로부터, last_checkpoint 감지하기 : 사전훈련모델에 이전훈련 가중치 누적에 사용
        if os.path.isdir(output_dir):
            last_checkpoint = get_last_checkpoint(output_dir)
            if last_checkpoint is None and len(os.listdir(output_dir)) > 0:
                raise ValueError(f"Output directory ({output_dir}) already exists and is not empty.")
            elif last_checkpoint is not None and resume_from_checkpoint is None:
                print(f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change the `--output_dir` to train from scratch.")
    # --------------------------------------
    # 훈련/검증 데이터셋 맵핑 : 검증셋은 훈련셋을 조금 떼어서 만들며, 보통 훈련셋:검증셋 = 8:2 비율로 제작
    # --------------------------------------
    komodel.print_trainable_parameters()  # 훈련 가능한 매개변수의 %에 대해, 더 투명해집니다.

    if gpt2_datasets:
        val_data = None
        for i in range(2):
            if i==0:
                column_names = list(raw_datasets["train"].features)
            else:
                if val_set_size <= 0:
                    break
                column_names = list(raw_datasets["validation"].features)
            # 컬럼 체크
            text_column_name = "text" if "text" in column_names else column_names[0]
            # tokenize_function 전에, Hasher 강제 로거 로딩 시, _LazyModule 오류를 방지하기 위해, 피클되어질 것이다.
            tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
            def tokenize_function(examples):
                with CaptureLogger(tok_logger) as cl:
                    output = tokenizer(examples[text_column_name])
                # clm 입력은 block_size보다 훨씬 길 수 있습니다.
                if "Token indices sequence length is longer than the" in cl.out:
                    tok_logger.warning(
                        "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                        " before being passed to the model."
                    )
                return output
            # 데이터셋 map 토큰화
            tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=column_names,)

            if block_size is None:
                block_size = tokenizer.model_max_length
                if block_size > 1024:
                    print(
                        "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                        " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                        " override this default with `--block_size xxx`."
                    )
                    block_size = 1024
            else:
                if block_size > tokenizer.model_max_length:
                    print(
                        f"The block_size passed ({block_size}) is larger than the maximum length for the model"
                        f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
                    )
                block_size = min(block_size, tokenizer.model_max_length)
            # 데이터셋의 모든 텍스트를 연결하고, block_size 청크를 생성하는, 주요 데이터 처리 기능입니다.
            def group_texts(examples):
                # 모든 텍스트를 연결합니다.
                concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # 작은 나머지 부분을 삭제하고, 모델이 지원하는 경우, 이 드랍 대신 패딩을 추가할 수 있으며, 필요에 따라 이 부분을 사용자 정의할 수 있습니다.
                if total_length >= block_size:
                    total_length = (total_length // block_size) * block_size
                # max_len 청크로 분할
                result = {
                    k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                    for k, t in concatenated_examples.items()
                }
                result["labels"] = result["input_ids"].copy()
                return result
            # 텍스트들 그룹화
            lm_datasets = tokenized_datasets.map(group_texts, batched=True,)
            # 체크/추출
            if i==0:
                if "train" not in tokenized_datasets:
                    raise ValueError("requires a train dataset")
                train_data  = lm_datasets["train"]
                print(f"train_data: {train_data}")
            else:
                if "validation" not in tokenized_datasets:
                    raise ValueError("requires a validation dataset")
                val_data    = lm_datasets["validation"]
                print(f"val_data: {val_data}")
    else:
        if val_set_size > 0:
            train_val   = raw_datasets["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
            train_data  = (train_val["train"].shuffle().map(generate_and_tokenize_prompt))
            val_data    = (train_val["test"].shuffle().map(generate_and_tokenize_prompt))
        else:
            train_data  = raw_datasets["train"].shuffle().map(generate_and_tokenize_prompt)
            val_data    = None
    # --------------------------------------
    # 다중 GPU를 사용할 수 있는 경우, Trainer가 자체 DataParallelism을 시도하지 못하도록 합니다.
    # Trainer는 여러 GPU를 자동으로 처리하므로 DataParallel에서 모델을 래핑할 필요가 없다.
    if not ddp and GPUs > 1:
        print("Set to GPU parallel in model.")
        komodel.is_parallelizable = True
        komodel.model_parallel = True
    # --------------------------------------
    # 훈련기 생성
    # --------------------------------------
    if gpt2_datasets:
        trainer = Trainer(
            model=komodel,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps        = 10,   #100,
                num_train_epochs    = num_train_epochs,
                learning_rate       = learning_rate,

                fp16                = True,
                logging_steps       = 10,
                optim               = "adamw_torch",
                
                evaluation_strategy = "steps" if val_set_size > 0 else "no",
                eval_steps          = 20 if val_set_size > 0 else None,     #200
                
                save_strategy       = "steps",
                save_steps          = 20,   #200,
                save_total_limit    = 3,
                
                output_dir          = output_dir,
                load_best_model_at_end      = True if val_set_size > 0 else False,
                ddp_find_unused_parameters  = False if ddp else None,
                group_by_length     = group_by_length,
                report_to           = "wandb" if use_wandb else None,
                run_name            = wandb_run_name if use_wandb else None,
            ),
            tokenizer=tokenizer,
            # 데이터 조합기는 기본적으로 DataCollatorWithPadding으로 설정되므로 이를 변경합니다.
            data_collator=default_data_collator,
            compute_metrics=None,
            preprocess_logits_for_metrics=None,
        )
    else:
        trainer = Trainer(
            model=komodel,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps        = 10,   #100,
                num_train_epochs    = num_train_epochs,
                learning_rate       = learning_rate,

                fp16                = True,
                logging_steps       = 10,
                optim               = "adamw_torch",
                
                evaluation_strategy = "steps" if val_set_size > 0 else "no",
                eval_steps          = 20 if val_set_size > 0 else None,     #200
                
                save_strategy       = "steps",
                save_steps          = 20,   #200,
                save_total_limit    = 3,
                
                output_dir          = output_dir,
                load_best_model_at_end      = True if val_set_size > 0 else False,
                ddp_find_unused_parameters  = False if ddp else None,
                group_by_length     = group_by_length,
                report_to           = "wandb" if use_wandb else None,
                run_name            = wandb_run_name if use_wandb else None,
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        )
    print("----------------------------------------")
    # --------------------------------------
    # 사전훈련모델에, 지금까지 누적된 훈련가중치와 함께 훈련시킨 후, (이전~현재) 누적된 훈련가중치를 경로(output_dir)에 저장
    # --------------------------------------
    komodel.config.use_cache = False
    # ------------------
    # 사전훈련모델의 사전상태를 PEFT 사전상태로 적용
    # ------------------
    # [JKC:20240415-1107] 주석 처리 : 현재 PyTorch와 PEFT 라이브러리 간에, 비호환성이 있기 때문
    """
    old_state_dict = komodel.state_dict
    komodel.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(komodel, type(komodel))
    """
    # ------------------
    # 사전훈련모델 컴파일
    # ------------------
    # [JKC:20240415-1017] 주석 처리 : torch.compile()을 사용하면, model.state_dict()가 완전히 비어 버려, 헤더 역직렬화 문제 발생
    """
    if torch.__version__ >= "2" and sys.platform != "win32":
        print(
            "Starting torch compile...\n"
            "----------------------------------------"
        )
        #  PyTorch 코드를 JIT 컴파일하여, 최적화된 커널로 실행
        komodel = torch.compile(komodel)
    """
    # ------------------
    # 훈련 : 지금까지 누적된 훈련가중치(checkpoint)와 함께...
    # ------------------
    # [JKC:20240411-1456] 훈련가중치 누적을 위해 추가
    checkpoint = None
    if resume_from_checkpoint is not None:
        checkpoint = resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    # 사전훈련모델에, 누적된 훈련 가중치 경로(checkpoint)를 추가하여 훈련 시작
    print(
        f"Starting train(resume_from_checkpoint={checkpoint})\n"
        "----------------------------------------"
    )
    trainer.train(resume_from_checkpoint=checkpoint)
    # ------------------
    # 저장 : 사전훈련모델에서, 추가적으로 훈련된 (이전~현재) LoRA 가중치만을 지정한 경로(output_dir)에 저장한다.
    # ------------------
    # 해당 경로(output_dir)는 나중에 재훈련할 때, [checkpoint] 변수에 적용되어, 자동으로 다음 훈련에 추가된다.
    print(
        "----------------------------------------\n"
        f"Saving model: {output_dir}\n"
        "----------------------------------------"
    )
    komodel.save_pretrained(output_dir)
    print("\n If there's a warning about missing keys above, please disregard :)")
    # --------------------------------------

# --------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    fire.Fire(train)
