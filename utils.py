import dataclasses
import logging
import math
import os
import io
import sys
import time
import json
import re

from typing import Optional, Sequence, Union
import os.path as osp

import openai
import tqdm
from openai import openai_object
import copy

import traceback
from queue import Queue
from threading import Thread

StrOrOpenAIObject = Union[str, openai_object.OpenAIObject]

openai_org = os.getenv("OPENAI_ORG")
if openai_org is not None:
    openai.organization = openai_org
    logging.warning(f"Switching to organization: {openai_org} for OAI API key.")


@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 1800
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    suffix: Optional[str] = None
    logprobs: Optional[int] = None
    echo: bool = False


def openai_completion(
    prompts: Union[str, Sequence[str], Sequence[dict[str, str]], dict[str, str]],
    decoding_args: OpenAIDecodingArguments,
    model_name="text-davinci-003",
    sleep_time=2,
    batch_size=1,
    max_instances=sys.maxsize,
    max_batches=sys.maxsize,
    return_text=False,
    **decoding_kwargs,
) -> Union[Union[StrOrOpenAIObject], Sequence[StrOrOpenAIObject], Sequence[Sequence[StrOrOpenAIObject]],]:
    """
    OpenAI API로 디코딩합니다.

    Args:
        prompts:
            완성할 문자열 또는 문자열 목록입니다.
            채팅 모델인 경우, 여기에 설명된 대로 문자열 형식을 지정해야 합니다.: https://github.com/openai/openai-python/blob/main/chatml.md.
            채팅 모델인 경우 여기에 설명된 대로 사전(또는 해당 목록)이 될 수도 있습니다.: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
        decoding_args:  인수를 디코딩합니다.
        model_name:     모델명. "org/model" 형식이거나 "model" 형식일 수 있습니다.
        sleep_time:     rate-limit 에 도달하면 잠잘 시간입니다.
        batch_size:     단일 요청으로 보낼 프롬프트 수입니다. 비채팅 모델에만 해당됩니다.
        max_instances:  디코딩할 최대 프롬프트 수입니다.
        max_batches:    디코딩할 최대 배치 수입니다. 이 인수는 앞으로 더 이상 사용되지 않습니다.
        return_text:    True인 경우 전체 완료 개체(logprob와 같은 항목 포함) 대신 텍스트를 반환합니다.
        decoding_kwargs:추가 디코딩 인수입니다. 필요한 경우 'best_of' 및 'logit_bias'를 전달하세요.

    Returns:
        완료 또는 완료 목록입니다.
        [return_text, return_openai_object, decodering_args.n]에 따라 완료 유형은 다음 중 하나일 수 있습니다.
            - 문자열 (return_text=True인 경우)
            - openai_object.OpenAIObject 객체 (return_text=False인 경우)
            - 위 타입들의 객체 목록 (decoding_args.n > 1 경우)
    """
    is_single_prompt = isinstance(prompts, (str, dict))
    if is_single_prompt:
        prompts = [prompts]

    if max_batches < sys.maxsize:
        logging.warning(
            "`max_batches` will be deprecated in the future, please use `max_instances` instead."
            "Setting `max_instances` to `max_batches * batch_size` for now."
        )
        max_instances = max_batches * batch_size

    prompts = prompts[:max_instances]
    num_prompts = len(prompts)
    prompt_batches = [
        prompts[batch_id * batch_size : (batch_id + 1) * batch_size]
        for batch_id in range(int(math.ceil(num_prompts / batch_size)))
    ]

    completions = []
    for batch_id, prompt_batch in tqdm.tqdm(
        enumerate(prompt_batches),
        desc="prompt_batches",
        total=len(prompt_batches),
    ):
        batch_decoding_args = copy.deepcopy(decoding_args)  # cloning the decoding_args

        while True:
            try:
                shared_kwargs = dict(
                    model=model_name,
                    **batch_decoding_args.__dict__,
                    **decoding_kwargs,
                )
                completion_batch = openai.Completion.create(prompt=prompt_batch, **shared_kwargs)
                choices = completion_batch.choices

                for choice in choices:
                    choice["total_tokens"] = completion_batch.usage.total_tokens
                completions.extend(choices)
                break
            except openai.error.OpenAIError as e:
                logging.warning(f"OpenAIError: {e}.")
                if "Please reduce your prompt" in str(e):
                    batch_decoding_args.max_tokens = int(batch_decoding_args.max_tokens * 0.8)
                    logging.warning(f"Reducing target length to {batch_decoding_args.max_tokens}, Retrying...")
                else:
                    logging.warning("Hit request rate limit; retrying...")
                    time.sleep(sleep_time)  # 요청에 대한 성가신 비율 제한.

    if return_text:
        completions = [completion.text for completion in completions]
    if decoding_args.n > 1:
        # 완성을 중첩 목록으로 만듭니다. 여기서 각 항목은 원래 항목의 연속적인 decoding_args.n입니다.
        completions = [completions[i : i + decoding_args.n] for i in range(0, len(completions), decoding_args.n)]
    if is_single_prompt:
        # 입력이 1개이고 생성이 1개인 경우, 튜플이 아닌 것을 반환합니다.
        (completions,) = completions
    return completions

# [JKC:20240408-1354] 한글 지원을 위한, 파일 인코딩 적용
def _make_w_io_base(f, mode: str, encoding: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode, encoding=encoding)
    return f


def _make_r_io_base(f, mode: str, encoding: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode, encoding=encoding)
    return f


def jdump(obj, f, mode="w", encoding="utf-8", indent=4, default=str):
    """
    문자열 또는 사전을 json 형식의 파일로 덤프합니다.

    Args:
        obj: 쓰여질 객체
        f: 디스크 위치에 대한 문자열 경로입니다.
        mode: 파일을 여는 모드입니다.
        indent: json 사전을 저장하기 위한, 들여쓰기 사이즈
        default: 직렬화할 수 없는 항목을 처리하는 함수입니다. 기본값은 'str'입니다.
    """
    f = _make_w_io_base(f, mode, encoding)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r", encoding="utf-8"):
    """
    .json 파일을 사전에 로드합니다.
    """
    f = _make_r_io_base(f, mode, encoding)
    jdict = json.load(f)
    f.close()
    return jdict

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            template_name = "polyglot-ko"   # 여기에 기본값을 적용하면, 생성자가 ''로 호출될 수 있는데, 중단되지 않습니다.
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        # [JKC:20240406-1107] 한글 지원을 위한, 파일 인코딩 적용
        with _make_r_io_base(file_name, "r", "utf-8") as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_ask(
        self,
        instruction: str,
        input: Union[None, str] = None
    ) -> str:
        pattern = r"\{[^}]*\}"
        if instruction:
            if input:
                return re.sub(pattern=pattern, repl=input, string=instruction)
            return instruction
        else:
            return input

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # 명령 및 선택적 입력에서 전체 프롬프트를 반환합니다.
        # 레이블(=response, =output)이 제공되면, 레이블도 추가됩니다.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
    

class Iteratorize:
    """
    콜백을 게으른 반복자(생성기)로 변환하는 함수를 변환합니다.
    """

    def __init__(self, func, kwargs={}, callback=None):
        self.mfunc = func
        self.c_callback = callback
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs
        self.stop_now = False

        def _callback(val):
            if self.stop_now:
                raise ValueError
            self.q.put(val)

        def gentask():
            try:
                ret = self.mfunc(callback=_callback, **self.kwargs)
            except ValueError:
                pass
            except:
                traceback.print_exc()
                pass

            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        self.thread = Thread(target=gentask)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True, None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True
