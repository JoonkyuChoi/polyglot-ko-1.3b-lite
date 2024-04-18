import os
import sys

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel

from transformers import pipeline, BitsAndBytesConfig

# 훈련가중치 최종 누적 경로를 얻기 위해
from transformers.trainer_utils import get_last_checkpoint

from utils import Iteratorize
from utils import Prompter


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    base_model: str = "",
    peft_root: str = None,
    cache_dir: str = None,
    load_4bit: bool = False,
    load_8bit: bool = False,
    prompt_template: str = "polyglot-ko",  
    server_name: str = "0.0.0.0",
    share_gradio: bool = True,
    use_pipe: bool = False
):
    # --------------------------------------
    # 초기 작업
    # --------------------------------------
    GPUs = torch.cuda.device_count()
    # 기반모델 확인
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert(base_model), "Please specify a --base_model, e.g. --base_model='resources/polyglot-ko-1.3b'"
    assert(os.path.isdir(base_model)), f"Found not a path for base_model: {base_model}"
    # 프롬프터 생성
    prompter = Prompter(prompt_template)
    # 파라미터 체크
    if load_4bit == load_8bit:
        raise ValueError(f"Please specify either 4bit({load_4bit}) or 8bit({load_8bit}) model.")
    # --------------------------------------
    # 정보 화면출력
    # --------------------------------------
    print("----------------------------------------")
    print(f"platform: {sys.platform}, torch: {torch.__version__}, GPU: {GPUs}, device: {device}")
    print("----------------------------------------")
    # [LOCAL_RANK] 환경변수가 정의되지 않았거나, [0]이면 파라미터 출력
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Generate params:\n"
            f"base_model: {base_model}\n"
            f"peft_root: {peft_root}\n"
            f"cache_dir: {cache_dir}\n"
            f"load_4bit: {load_4bit}\n"
            f"load_8bit: {load_8bit}\n"
            f"prompt_template: {prompt_template}\n"
            f"server_name: {server_name}\n"
            f"share_gradio: {share_gradio}"
        )
        print("----------------------------------------")
    # --------------------------------------
    # 0. 준비 작업
    # --------------------------------------
    # 0-1. PEFT 루트경로(peft_root)로부터, 최종 체크포인트(훈련가중치) 경로(last_peft_path) 얻기 : 없으면 사용안함
    # ------------------
    last_peft_path = None
    if peft_root:
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
    # ------------------
    # 0-2. 토크나이저(tokenizer) 로드
    # ------------------
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model,
        device_map={"": device},
        cache_dir=cache_dir,
        padding_side="right",
        use_fast=False,
    )
    # --------------------------------------
    # 1. device 타입에 따른, 사전훈련모델(komodel) & PEFT 가중치 로드
    # --------------------------------------
    # 1-1. cuda
    # ------------------
    if device == "cuda":
        komodel = transformers.AutoModelForCausalLM.from_pretrained(
            base_model,
            #load_in_8bit=load_8bit,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=load_4bit,
                load_in_8bit=load_8bit,
                load_in_8bit_fp32_cpu_offload=load_8bit,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=load_8bit,
                bnb_4bit_quant_type='nf4'
            ),
            torch_dtype=torch.float16,
            device_map={"": device},
            cache_dir=cache_dir,
        )
        if last_peft_path:
            komodel = PeftModel.from_pretrained(
                komodel,
                last_peft_path,
                torch_dtype=torch.float16,
            )
    # ------------------
    # 1-2. mps
    # ------------------
    elif device == "mps":
        komodel = transformers.AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map={"": device},
            cache_dir=cache_dir,
        )
        if last_peft_path:
            komodel = PeftModel.from_pretrained(
                komodel,
                last_peft_path,
                device_map={"": device},
                torch_dtype=torch.float16
            )
    # ------------------
    # 1-3. cpu
    # ------------------
    else:
        komodel = transformers.AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
        )
        if last_peft_path:
            komodel = PeftModel.from_pretrained(
                komodel,
                last_peft_path,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
    # --------------------------------------
    #
    # --------------------------------------
    """
    push_to_hub()는 자신이 훈련시킨 모델을 HuggingFace에 푸시하는 함수다.
    아래의 push_to_hub()함수가 작동하려면, 자신의 HuggingFace 계정에, repo_id를 구축하고, push_to_hub()함수에 제공해야 한다.
        1. 사용자 공간에 기본적으로 할 때,
            tokenizer.push_to_hub("my-tokenizer")
        2. 사용자 명칭 명시할 때,
            tokenizer.push_to_hub("username/my-tokenizer")
        3. 조직 명칭을 명시할 때
            tokenizer.push_to_hub("my-organization/my-tokenizer")
    이래도 안된다면, 자신이 repo_id를 구축하지 않은 것입니다.
    """
    #komodel.push_to_hub('EleutherAI/polyglot-ko-1.3b')
    # --------------------------------------
    # 양자화 모델은 half()를 지원하지 않는다.
    """
    if not load_8bit:
        komodel.half()
    """
    # --------------------------------------
    # 사전훈련모델 평가 진행
    # --------------------------------------
    komodel.eval()
    # --------------------------------------
    # 사전훈련모델 컴파일
    # --------------------------------------
    if torch.__version__ >= "2" and sys.platform != "win32":
        #  PyTorch 코드를 JIT 컴파일하여, 최적화된 커널로 실행
        komodel = torch.compile(komodel)
    # --------------------------------------
    # 프롬프트 띄우기
    # --------------------------------------
    pipe = None
    examples = [
        ["양파는 식물의 어떤 부위인가요? 그리고 고구마는 뿌리인가요?"],
        ["스웨터의 유래는 어디에서 시작되었나요?"],
        ["토성의 고리가 빛의 띠로 보이는 이유는 무엇인가요?"]
    ]
    bad_words = []
    if use_pipe:
        pipe = pipeline(
            "text-generation",
            model=komodel,
            tokenizer=tokenizer,
            eos_token_id=tokenizer.eos_token_id
        )
    else:
        bad_words = [
            '....',
            'http://'
        ]
    # 추론(질의 > 응답)
    def inference(
        instruction,
        input=None,
        temperature=0.8,
        top_p=0.8,
        top_k=50,
        max_new_tokens=64,
        **kwargs,
    ):
        prompt = prompter.generate_ask(instruction, input)
        print(f"prompt: {prompt}")
        # 훈련데이터셋 형식으로 변환 : 여러 문장을 포함시킬 수 있다.
        ask_msg = [{"role": "질문", "content": prompt}]
        question = "\n".join([f"### {msg['role']}: {msg['content']}" for msg in ask_msg])
        question += "\n\n### 답변:"
        print(f"question: {question}")
        # 사전훈련모델에 질의/응답(생성)
        if use_pipe and pipe:
            answer = pipe(
                question,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                #return_full_text=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )[0]['generated_text']
        else:
            # [tokenizer/komodel]을 직접 사용하여, 질답 처리
            with torch.no_grad():
                # 토큰화
                ask_tokens = tokenizer(question, return_tensors="pt").input_ids.to(device)
                # 생성토록 하고, question 내에 <|endoftext|>가 없다면, 생성을 종료한다.
                gen_tokens = komodel.generate(
                    ask_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_new_tokens=max_new_tokens,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                    bad_words_ids=[
                        tokenizer.encode(bad_word) for bad_word in bad_words
                    ],
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )
                # 토큰을 문장화
                answer = tokenizer.batch_decode(gen_tokens)[0]
        # ------------------
        # 응답메세지 보정
        # ------------------
        print(f"answer: {answer}")
        # 답변 리스트로 분할
        answer = answer.split("### ")
        has_ans = False
        # 답변만 추출
        for one in answer:
            fidx = one.find("답변: ")
            if fidx >= 0 and fidx < 2:
                answer = one.lstrip("답변: ")
                has_ans = True
                break
        # 답변이 없으면
        if not has_ans:
            answer = "..."
        # 특수 토큰 제거
        answer = answer.rstrip(tokenizer.special_tokens_map['eos_token'])
        # ------------------
        return answer

    gr.Interface(
        fn=inference,
        inputs=[
            gr.components.Textbox(lines=2, label="Instruction", placeholder="Tell me what."),
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(minimum=0, maximum=1, value=0.8, label="Temperature"),
            gr.components.Slider(minimum=0, maximum=1, value=0.8, label="Top p"),
            gr.components.Slider(minimum=0, maximum=100, step=1, value=50, label="Top k"),
            gr.components.Slider(minimum=1, maximum=2000, step=1, value=256, label="Max tokens")
        ],
        outputs=[
            gr.Textbox(lines=5, label="Output")
        ],
        title="🤗 [polyglot-ko-1.3b-lite] 가벼운 한국어 언어모델 미세조정(QLoRA) 🤗",
        examples=examples,
        description="""
<p align="center" width="100%">
<img src="https://aeiljuispo.cloudimg.io/v7/https://cdn-uploads.huggingface.co/production/uploads/noauth/jpn8a_aJ5etAJwFUd_nno.png?w=200&h=200&f=face" alt="Joonkyu icon" style="width: 200px; display: block; margin: auto; border-radius: 20%;">
한국어 언어모델 프롬프트
</p><br>

""",
        thumbnail="https://aeiljuispo.cloudimg.io/v7/https://cdn-uploads.huggingface.co/production/uploads/noauth/jpn8a_aJ5etAJwFUd_nno.png?w=200&h=200&f=face",        
    ).queue().launch(server_name="0.0.0.0", share=share_gradio)
    # --------------------------------------
    
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    fire.Fire(main)