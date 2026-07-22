# lmms_gen.py
# lmms-eval의 llava_vid.py (lmms_eval/models/simple/llava_vid.py) 재현 유틸.
#
# core/data_pipeline.py 경로와의 차이 (전부 의도된 것 — lmms-eval과 bit-faithful):
#   * 프롬프트에 " \nAnswer the question using a single word or phrase." suffix가 없다.
#   * 모델은 torch_dtype="bfloat16", attn_implementation="sdpa", device_map="cuda:0"로 로드.
#   * 비디오 프레임도 .bfloat16() 캐스팅.
#   * 평가는 prefill-logit argmax가 아니라 model.generate(...) + 문자 추출.
#
# 이 모듈로 뽑은 hidden/축과 core/data_pipeline로 뽑은 것은 분포가 달라
# 서로 섞어 쓰면 안 된다 (suffix 유무만으로도 last-token hidden이 달라짐).

import os

import numpy as np
import torch

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path,
                            tokenizer_image_token)
from llava.model.builder import load_pretrained_model

QWEN_PAD_TOKEN_ID = 151643  # llava_vid.py: qwen 계열 pad 미설정 시 하드코딩


def load_model_lmms(pretrained, lora_pretrained=None, torch_dtype="bfloat16",
                    attn_implementation="sdpa", device_map="cuda:0"):
    """llava_vid.LlavaVid.__init__의 overwrite=False 로딩 경로를 그대로 재현."""
    hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
    if lora_pretrained is not None:
        model_name = get_model_name_from_path(lora_pretrained)
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            lora_pretrained, pretrained, model_name, device_map=device_map,
            cache_dir=hf_home, torch_dtype=torch_dtype,
            attn_implementation=attn_implementation)
    else:
        model_name = get_model_name_from_path(pretrained)
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            pretrained, None, model_name, device_map=device_map,
            cache_dir=hf_home, torch_dtype=torch_dtype,
            attn_implementation=attn_implementation)

    if tokenizer.pad_token_id is None and "qwen" in tokenizer.name_or_path.lower():
        tokenizer.pad_token_id = QWEN_PAD_TOKEN_ID
    model.eval()
    model.tie_weights()
    return tokenizer, model, image_processor, max_length, model_name


def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    """llava_vid.load_video 그대로: fps 샘플링 → 초과/강제 시 uniform max_frames_num."""
    from decord import VideoReader, cpu
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3)), "", 0
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames, frame_time, video_time


def preprocess_video(image_processor, frames, torch_dtype="bfloat16"):
    video = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].cuda()
    return video.bfloat16() if torch_dtype == "bfloat16" else video.half()


def build_prompt_inputs(tokenizer, question, conv_template):
    """<image>\\n + question → conv 템플릿 → input_ids/attention_mask/stop_str."""
    qs = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv = conv_templates[conv_template].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    pad_token_ids = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                     else tokenizer.eos_token_id)
    attention_mask = input_ids.ne(pad_token_ids).long().cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    return input_ids, attention_mask, stop_str


@torch.inference_mode()
def generate_answer(model, tokenizer, input_ids, attention_mask, videos, stop_str,
                    max_new_tokens=256):
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
    output_ids = model.generate(
        inputs=input_ids,
        images=videos,
        attention_mask=attention_mask,
        modalities="video",
        use_cache=True,
        stopping_criteria=[stopping_criteria],
        do_sample=False,
        temperature=0,
        top_p=None,
        num_beams=1,
        max_new_tokens=max_new_tokens,
    )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


@torch.inference_mode()
def prefill_hidden_states(model, input_ids, attention_mask, videos):
    """생성과 동일한 prefill의 per-layer last-token hidden (tuple of (B, T, D))."""
    out = model(input_ids=input_ids, attention_mask=attention_mask, images=videos,
                modalities="video", output_hidden_states=True, return_dict=True)
    return out.hidden_states
