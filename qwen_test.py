import json
import os
import base64
import re
import edge_tts
import asyncio
import pygame
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

async def text_to_speech(text, voice='zh-CN-YunxiNeural'):
    """使用edge-tts将文字转换为语音"""
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save("temp.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("temp.mp3")
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
        os.remove("temp.mp3")
    except Exception as e:
        print(f"语音播报失败: {e}")

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def plot_bounding_boxes(im, bounding_boxes):
    """在图片上绘制边界框"""
    img = im
    draw = ImageDraw.Draw(img)
    colors = ['red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple']
    
    try:
        font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)
    except:
        font = ImageFont.load_default()

    try:
        bboxes = json.loads(bounding_boxes)
        if not bboxes:
            return False
            
        for i, bbox in enumerate(bboxes):
            color = colors[i % len(colors)]
            x1, y1, x2, y2 = map(int, bbox["bbox_2d"])
            label = bbox.get("label", f"Object {i+1}")
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            draw.text((x1 + 8, y1 + 6), label, fill=color, font=font)
        
        img.show()
        return True
    except Exception as e:
        print(f"绘图失败: {e}")
        return False

def extract_json(text):
    """从文本中提取JSON内容"""
    try:
        # 尝试直接解析整个文本
        return json.loads(text)
    except:
        pass
    
    # 尝试提取代码块中的JSON
    try:
        json_str = re.search(r'```(?:json)?\n([\s\S]*?)\n```', text)
        if json_str:
            return json.loads(json_str.group(1))
    except:
        pass
    
    # 尝试提取类似JSON的结构
    try:
        json_str = re.search(r'\[[\s\S]*\{[\s\S]*\}[\s\S]*\]', text)
        if json_str:
            fixed = json_str.group(0).replace("'", '"').replace("True", "true").replace("False", "false")
            return json.loads(fixed)
    except:
        pass
    
    return None

def encode_image(image_path):
    """将图片编码为base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def generate_response_with_bboxes(image_path, user_input):
    """生成智能响应，由AI决定是否需要展示图片"""
    base64_image = encode_image(image_path)
    
    prompt = f"""用户询问: {user_input}

请根据以下规则响应：
1. 首先理解用户意图，判断是否需要分析图片内容
2. 如果问题明确涉及图片内容，找出相关物品并标注
3. 如果是普通对话或感谢，给出自然回应
4. 如果是结束对话的表示，礼貌结束

请用以下JSON格式返回：
```json
{{
    "response": "你的回答内容",
    "need_show_image": true/false,  // 是否需要展示图片
    "bounding_boxes": [             // 如果需要展示图片
        {{
            "bbox_2d": [x1, y1, x2, y2],
            "label": "物品名称",
            "description": "物品描述"
        }}
    ]
}}
```"""
    
    response = client.chat.completions.create(
        model="qwen2.5-vl-72b-instruct",
        messages=[
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": "你是一个智能视觉助手，能自然对话也能分析图片。请自行判断何时需要展示图片内容。"
                }]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    )
    
    return response.choices[0].message.content


def process_user_input(image_path, user_input, messages):
    """处理用户输入并生成响应"""
    assistant_response = generate_response_with_bboxes(image_path, user_input)
    
    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": user_input}]
    })
    
    response_data = extract_json(assistant_response)
    
    if response_data:
        response_text = response_data.get('response', assistant_response)
        text_for_speech = response_text  # 原始回复用于语音
        
        # 为所有关键物体追加坐标信息（仅文本显示）
        if "bounding_boxes" in response_data and response_data["bounding_boxes"]:
            coordinates_info = []
            for bbox in response_data["bounding_boxes"]:
                label = bbox.get("label", "物体")
                cx, cy = bbox.get("center_point", [
                    (bbox["bbox_2d"][0] + bbox["bbox_2d"][2]) // 2,
                    (bbox["bbox_2d"][1] + bbox["bbox_2d"][3]) // 2
                ])
                coordinates_info.append(f"{label}({cx},{cy})")
            
            if coordinates_info:
                response_text += f" [位置: {' | '.join(coordinates_info)}]"
        
        print(f"\nAI回复: {response_text}")
        
        try:
            # 使用不包含坐标的原始文本进行语音播报
            asyncio.run(text_to_speech(text_for_speech))
        except Exception as e:
            print(f"语音播报失败: {e}")
        
        if response_data.get("need_show_image", False) and "bounding_boxes" in response_data:
            try:
                image = Image.open(image_path)
                if plot_bounding_boxes(image, json.dumps(response_data["bounding_boxes"])):
                    print("已标记相关物品")
            except Exception as e:
                print(f"图片处理失败: {e}")
        
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response_text}]
        })
    else:
        print(f"\nAI回复: {assistant_response}")
        asyncio.run(text_to_speech(assistant_response))
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_response}]
        })

def chat_with_image(image_path, initial_prompt=None):
    """与图片进行连续对话"""
    if not os.path.exists(image_path):
        print("错误: 文件不存在")
        return
    elif not image_path.lower().endswith(('.jpg', '.jpeg')):
        print("错误: 请提供JPG格式的图片")
        return
    
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "你是一个有帮助的视觉助手，能够分析图片并回答用户问题。"}]
        }
    ]
    
    if initial_prompt:
        print(f"\n用户问题: {initial_prompt}")
        process_user_input(image_path, initial_prompt, messages)
    
    while True:
        try:
            user_input = input("\n请输入您的问题(输入'退出'结束对话): ").strip()
            if user_input.lower() in ['退出', 'exit', 'quit']:
                break
                
            process_user_input(image_path, user_input, messages)
            
        except Exception as e:
            print(f"发生错误: {e}")
            break

if __name__ == "__main__":
    image_path = input("请输入图片路径(JPG格式): ").strip()
    initial_prompt = input("请输入初始提示词(直接回车跳过): ").strip()
    chat_with_image(image_path, initial_prompt if initial_prompt else None)