#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import pyaudio
import wave
import os
from datetime import datetime
import requests
import threading
from pynput import keyboard

class BaiduVoiceRecognitionNode(Node):
    def __init__(self):
        super().__init__('baidu_voice_recognition_node')
        
        # 百度API配置（需自行申请）
        self.baidu_api_key = "xx"     # 替换为你的API Key
        self.baidu_secret_key = "xx" # 替换为你的Secret Key
        self.baidu_token_url = "https://openapi.baidu.com/oauth/2.0/token"
        self.baidu_asr_url = "http://vop.baidu.com/server_api"
        
        # 音频参数
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.record_seconds = 5
        
        # 创建发布者
        self.text_publisher = self.create_publisher(String, '/voice_command', 10)
        
        # 录音控制
        self.is_recording = False
        self.stop_recording = False
        self.audio_interface = None
        self.audio_stream = None
        
        # 键盘监听
        self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press)
        self.keyboard_listener.start()
        
        # 创建临时目录
        os.makedirs('tmp', exist_ok=True)
        
        # 列出可用麦克风
        self.list_microphones()
        
        # 状态提示
        self.get_logger().info("百度语音识别节点已启动，按F2开始/停止录音")
    
    def list_microphones(self):
        """列出所有麦克风设备"""
        p = pyaudio.PyAudio()
        self.get_logger().info("\n可用的音频输入设备:")
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev['maxInputChannels'] > 0:
                self.get_logger().info(f"{i}: {dev['name']}")
        p.terminate()
    
    def on_key_press(self, key):
        try:
            if key == keyboard.Key.f2:
                if not self.is_recording:
                    self.start_recording()
                else:
                    self.stop_recording = True
        except Exception as e:
            self.get_logger().error(f"键盘监听错误: {str(e)}")
    
    def start_recording(self):
        if self.is_recording:
            return
            
        self.is_recording = True
        self.stop_recording = False
        self.get_logger().info("开始录音...")
        
        # 启动录音线程
        threading.Thread(target=self.record_audio).start()
    
    def record_audio(self):
        try:
            self.audio_interface = pyaudio.PyAudio()
            self.audio_stream = self.audio_interface.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            
            self.get_logger().info(f"正在录音，请说话...（{self.record_seconds}秒）")
            frames = []
            
            while self.is_recording and not self.stop_recording:
                data = self.audio_stream.read(self.chunk)
                frames.append(data)
                
                # 检查是否达到录音时长
                if len(frames) >= int(self.rate / self.chunk * self.record_seconds):
                    break
            
            # 停止录音
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_interface.terminate()
            
            # 保存录音文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_file = f"tmp/recording_{timestamp}.wav"
            
            with wave.open(audio_file, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio_interface.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(frames))
            
            self.get_logger().info(f"录音已保存: {audio_file}")
            
            # 语音识别
            self.get_logger().info("使用百度API识别中...")
            text = self.recognize_with_baidu(audio_file)
            
            if text:
                self.get_logger().info(f"识别结果: {text}")
                
                # 发布识别结果
                msg = String()
                msg.data = text
                self.text_publisher.publish(msg)
                
                # 保存识别结果
                with open(f"tmp/result_{timestamp}.txt", 'w', encoding='utf-8') as f:
                    f.write(text)
            else:
                self.get_logger().error("识别失败，请检查音频文件或API配置")
                
        except Exception as e:
            self.get_logger().error(f"录音错误: {str(e)}")
        finally:
            self.is_recording = False
            self.stop_recording = False
            self.get_logger().info("录音已停止")
    
    def get_baidu_token(self):
        """获取百度语音识别的Access Token"""
        params = {
            'grant_type': 'client_credentials',
            'client_id': self.baidu_api_key,
            'client_secret': self.baidu_secret_key
        }
        response = requests.get(self.baidu_token_url, params=params)
        return response.json().get('access_token')
    
    def recognize_with_baidu(self, audio_file):
        """使用百度API识别音频文件"""
        token = self.get_baidu_token()
        if not token:
            self.get_logger().error("错误: 无法获取百度API Token")
            return None

        with open(audio_file, 'rb') as f:
            speech_data = f.read()

        headers = {'Content-Type': 'audio/wav; rate=16000'}
        params = {
            'cuid': 'ros2-client',  # 用户标识（可自定义）
            'token': token,
            'dev_pid': 1537  # 1537表示普通话(纯中文识别)
        }

        try:
            response = requests.post(
                self.baidu_asr_url,
                headers=headers,
                params=params,
                data=speech_data
            )
            result = response.json()
            if 'result' in result:
                return ''.join(result['result'])
            else:
                self.get_logger().error(f"识别错误: {result.get('err_msg', '未知错误')}")
                return None
        except Exception as e:
            self.get_logger().error(f"请求百度API失败: {e}")
            return None

def main(args=None):
    rclpy.init(args=args)
    node = BaiduVoiceRecognitionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()