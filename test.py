import sys
import os
sys.path.append('./third_party/Matcha-TTS')
sys.path.insert(0, './third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import requests
import json
from pydub import AudioSegment
import simpleaudio as sa
import re
from bs4 import BeautifulSoup

API_URL = "http://127.0.0.1:11434/api/generate"
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=True, fp16=True)
prompt_speech_16k = load_wav('zero_shot_prompt_明.wav', 16000)
global on_init,context
on_init = True
context = None

def remove_think_tag(text):
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text

def extract_urls(text):
    # 匹配以http(s)://开头的URL
    http_url_pattern = r'https?://[a-zA-Z0-9\-.:%/?&#=]+'
    # 匹配以www.开头但不包含http(s)://的URL
    www_url_pattern = r'www\.[a-zA-Z0-9\-.%/?&#=]+'

    # 查找所有匹配项
    http_urls = re.findall(http_url_pattern, text)
    www_urls = re.findall(www_url_pattern, text)

    # 为以www.开头的URL添加http://前缀
    www_urls = ['http://' + url for url in www_urls]

    # 合并两种类型的URL并去重
    urls = list(set(http_urls + www_urls))
    
    return urls

def fetch_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        # 设置正确的编码
        response.encoding = 'utf-8'
        
        soup = BeautifulSoup(response.text, 'lxml')  # 使用'lxml'作为解析器
        
        # 定义通常不会显示的内容的标签
        ignored_tags = ['script', 'style', 'head', 'title', 'meta', '[document]']
        
        # 提取页面上所有可能的文本和图片alt属性
        texts_and_alts = []
        for element in soup.find_all(text=True):
            if element.parent.name not in ignored_tags:
                # 添加非空文本节点
                if element.strip():
                    texts_and_alts.append(element.strip())
            if element.parent.name == 'img':
                # 如果是图片标签，添加alt属性值（如果存在）
                alt = element.parent.get('alt')
                if alt:
                    texts_and_alts.append(f"[Image Alt Text]: {alt}")
                    
        # 将所有收集到的信息组合成一个字符串
        all_visible_content = '\n'.join(texts_and_alts)
        
        resp = f"#website_info_for{url}:\n{all_visible_content}"
        return resp
    else:
        print(f"Failed to retrieve data from {url}. Status code: {response.status_code}")
        return None

def prompt_fetch_data(prompt):
    new_prompt = ''
    final_prompt = None
    urls = extract_urls(prompt)
    for url in urls:
        data = fetch_data(url)
        if data:  # 检查data是否为None
            new_prompt = new_prompt + prompt + data
            new_prompt = new_prompt
            #如果需要最终结果为最后一个更新的prompt
        
    if new_prompt != '':
        final_prompt = new_prompt
    else:
            final_prompt = prompt
    return final_prompt

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
    prompt = prompt_fetch_data(input_)
    response_data = get_response(prompt,context)
    context = response_data.get("context")
    resp = remove_think_tag(response_data['response'])
    return resp

def playsoundfromfile(file):
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

while True:
    inpt = input("输入文字以对话\n")
    if inpt in ["q", "quit", "离开"]:
        break
    text = chat(inpt)
    print(text)
    text_ = text.replace('\n', ' ').replace('\r', ' ').replace('#','').replace('*','').replace('-','')
    for i, j in enumerate(cosyvoice.inference_instruct2(text_, '用普通话说这句话', prompt_speech_16k, stream=False)):
        filename = f"instruct_{i}.wav"  
        torchaudio.save(filename, j['tts_speech'], cosyvoice.sample_rate)
        playsoundfromfile(filename)  
