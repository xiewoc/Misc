import sys
import os
sys.path.append('./third_party/Matcha-TTS')  # 确保路径正确指向你的Matcha-TTS位置
sys.path.insert(0, 'E:/CosyVoice/third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import requests
import json
from pydub import AudioSegment
import simpleaudio as sa
import re

API_URL = "http://127.0.0.1:11434/api/generate"
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=True, fp16=False)
prompt_speech_16k = load_wav('zero_shot_prompt_米雪儿李.wav', 16000)
global on_init,context
on_init = True

def remove_think_tag(text):
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text

def get_response(prompt, context=None):
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "qwen2.5:7b",
        "prompt": prompt,
        "stream": False
    }
    if context is not None:
        data["context"] = context

    response = requests.post(API_URL, headers=headers, json=data)
    return response.json()

def chat(input_):
    global on_init,context
    if on_init == True:
        context = None
        on_init = False
    response_data = get_response(input_ ,context)
    context = response_data.get("context")
    resp = remove_think_tag(response_data['response'])
    return resp

def playsoundfromfile(file):
    sys.stdout = sys.__stdout__
    try: # 加载音频并检查路径
        audio = AudioSegment.from_file(file, format="wav")
    except Exception as e:
        print("文件加载失败:", e)
        exit()
    print(f"时长: {len(audio)}ms, 声道: {audio.channels}, 位深: {audio.sample_width*8}bit, 采样率: {audio.frame_rate}Hz")# 打印音频信息
    if audio.sample_width != 2: # 转换为支持的格式（如16位）
        audio = audio.set_sample_width(2)
    raw_data = audio.raw_data# 获取原始数据
    try:# 播放音频
        play_obj = sa.play_buffer(
            raw_data,
            num_channels=audio.channels,
            bytes_per_sample=audio.sample_width,
            sample_rate=audio.frame_rate
        )
        play_obj.wait_done()
    except Exception as e:
        print("播放错误:", e)
    sys.stdout = open(os.devnull, 'w')

while True:
    sys.stdout = sys.__stdout__
    inpt = input("输入文字以对话\n")
    sys.stdout = open(os.devnull, 'w')
    if inpt in ["q", "quit", "离开"]:
        break
    text = chat(inpt)
    text_no_newlines = text.replace('\n', ' ').replace('\r', ' ')
    for i, j in enumerate(cosyvoice.inference_instruct2(text_no_newlines, '用四川话说这句话', prompt_speech_16k, stream=False)):
        filename = f"instruct_{i}.wav"  # 动态生成文件名
        torchaudio.save(filename, j['tts_speech'], cosyvoice.sample_rate)
        playsoundfromfile(filename)  # 使用变量传递文件名
