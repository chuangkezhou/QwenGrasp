#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import tf2_ros
import numpy as np
import cv2
import yaml
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import String
from scipy.spatial.transform import Rotation as R
import threading
import sys
import select
from PIL import Image, ImageDraw, ImageFont
import json
import os
import base64
import re
import edge_tts
import asyncio
import pygame
import pyrealsense2 as rs
import io
import time
from openai import OpenAI
import torch
from ultralytics import SAM
from ultralytics.models.sam import Predictor as SAMPredictor
import math

class QwenVision:
    def __init__(self, api_key=None):
        """初始化视觉助手"""
        self.client = OpenAI(
            api_key=api_key if api_key else os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.pipeline = None
        pygame.mixer.init()

    def initialize_realsense(self):
        """初始化RealSense相机"""
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
            profile = self.pipeline.start(config)

            # 设置自动曝光
            color_sensor = profile.get_device().first_color_sensor()
            if color_sensor.supports(rs.option.auto_exposure_priority):
                color_sensor.set_option(rs.option.auto_exposure_priority, 1.0)
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 1)
            
            # 相机预热
            for _ in range(5):
                self.pipeline.wait_for_frames()
                time.sleep(0.1)
            return True
        except Exception as e:
            print(f"初始化RealSense失败: {e}")
            return False

    def capture_frame(self):
        """捕获一帧图像"""
        try:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                print("警告: 未能获取彩色图像帧")
                return None
            return cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"捕获图像失败: {e}")
            return None

    async def text_to_speech(self, text, voice='zh-CN-YunxiNeural'):
        """文字转语音"""
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save("temp.mp3")
            pygame.mixer.music.load("temp.mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            os.remove("temp.mp3")
        except Exception as e:
            print(f"语音播报失败: {e}")

    @staticmethod
    def encode_image(image_array):
        """编码图像为base64"""
        try:
            img = Image.fromarray(image_array)
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"图像编码失败: {e}")
            return None

    @staticmethod
    def extract_json(text):
        """从文本提取JSON"""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                # 尝试提取代码块中的JSON
                json_str = re.search(r'```(?:json)?\n([\s\S]*?)\n```', text)
                if json_str:
                    return json.loads(json_str.group(1))
            except:
                try:
                    # 尝试提取其他格式的JSON
                    json_str = re.search(r'\{[\s\S]*\}', text)
                    if json_str:
                        fixed = json_str.group[0].replace("'", '"').replace("True", "true").replace("False", "false")
                        return json.loads(fixed)
                except Exception as e:
                    print(f"JSON解析失败: {e}")
        return None

    def generate_response(self, image_array, user_input):
        """生成AI响应"""
        base64_image = self.encode_image(image_array)
        if not base64_image:
            return None
        
        prompt = f"""用户指令: {user_input}

请根据以下规则响应：
1. 首先理解用户意图，判断是否需要分析图片内容
2. 如果问题明确涉及图片内容，找出目标物品和容器（盒子等）并标注,如果用户指令不包含盒子等容器，则只标注目标物体
3. 如果指令中包含明确的容器（如"放到盒子里"），标注容器位置
4. 如果没有明确容器，只标注目标物体
5. 如果是普通对话或感谢，给出自然回应
6. 如果是结束对话的表示，礼貌结束
7. 如果图片中的物体不符合用户意图，不要标注
8. 如果标注物体就要进行显示图片
9. 如果对话明显没有逻辑，可能是语音识别错误，只给出自然语言回复，不进行标注

响应格式要求：
```json
{{
    "response": "自然语言回复",
    "target_objects": [
        {{
            "bbox_2d": [x1,y1,x2,y2],
            "label": "物体名称",
            "description": "物体描述"
        }},
        // 更多物体...
    ],
    "container": {{
        "bbox_2d": [x1,y1,x2,y2],
        "label": "容器名称",
        "description": "容器描述"
    }},
    "need_show_image": true/false
}}```"""
        
        try:
            response = self.client.chat.completions.create(
                model="qwen2.5-vl-72b-instruct",
                messages=[
                    {
                        "role": "system",
                        "content": [{
                            "type": "text",
                            "text": "你是智能视觉助手，能识别多个物体和容器位置"
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
        except Exception as e:
            print(f"AI请求失败: {e}")
            return None

    @staticmethod
    def plot_bounding_boxes(image_array, bounding_boxes):
        """绘制边界框（支持多物体）"""
        try:
            img = Image.fromarray(image_array)
            draw = ImageDraw.Draw(img)
            colors = ['red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple']
            
            try:
                font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)
            except:
                font = ImageFont.load_default()

            # 统一处理输入格式
            if isinstance(bounding_boxes, str):
                try:
                    bounding_boxes = json.loads(bounding_boxes)
                except json.JSONDecodeError:
                    print("错误: 无法解析边界框JSON字符串")
                    return False
                    
            if isinstance(bounding_boxes, dict):
                bounding_boxes = [bounding_boxes]
                
            if not isinstance(bounding_boxes, list):
                print("错误: 边界框数据格式不正确")
                return False
                
            for i, bbox in enumerate(bounding_boxes):
                if not isinstance(bbox, dict):
                    continue
                    
                color = colors[i % len(colors)]
                try:
                    # 确保bbox_2d存在且是列表
                    if "bbox_2d" not in bbox or not isinstance(bbox["bbox_2d"], (list, tuple)) or len(bbox["bbox_2d"]) != 4:
                        continue
                        
                    x1, y1, x2, y2 = map(int, bbox["bbox_2d"])
                    label = bbox.get("label", f"Object {i+1}")
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                    draw.text((x1 + 8, y1 + 6), label, fill=color, font=font)
                except Exception as e:
                    print(f"绘制边界框{i}失败: {e}")
                    continue
            
            img.show()
            return True
        except Exception as e:
            print(f"绘图失败: {e}")
            return False

class HandEyeControl(Node):
    def __init__(self):
        super().__init__('handeye_control_node')
        
        # 加载配置文件
        try:
            with open('src/xarm_ros2/handeye_calibration_ros2/handeye_realsense/config.yaml', 'r') as file:
                config = yaml.safe_load(file)
            
            self.handeye_result_file_name = config["handeye_result_file_name"]
            self.base_link = config["base_link"]
            self.ee_link = config["ee_link"]
            self.calculated_camera_optical_frame_name = config["calculated_camera_optical_frame_name"]
            self.camera_calibration_parameters_filename = config["camera_calibration_parameters_filename"]
        except Exception as e:
            self.get_logger().error(f"加载配置文件失败: {e}")
            raise

        # 初始化手眼标定矩阵
        self.load_handeye_calibration()

        # 初始化相机内参
        try:
            cv_file = cv2.FileStorage(self.camera_calibration_parameters_filename, cv2.FILE_STORAGE_READ)
            self.intrinsics = cv_file.getNode('K').mat()
            cv_file.release()
        except Exception as e:
            self.get_logger().error(f"加载相机内参失败: {e}")
            raise

        # TF相关初始化
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 初始化Qwen视觉模块
        self.qwen = QwenVision()
        if not self.qwen.initialize_realsense():
            raise RuntimeError("RealSense初始化失败")

        # 初始化SAM分割模型
        self.sam_predictor = self.init_sam()

        # ROS接口
        self.camera_point_pub = self.create_publisher(PointStamped, '/camera_clicked_point', 10)
        self.robot_pose_pub = self.create_publisher(PoseStamped, '/robot_target_pose', 10)
        self.container_pose_pub = self.create_publisher(PoseStamped, '/container_pose', 10)
        
        # 语音命令订阅
        self.voice_command_sub = self.create_subscription(
            String,
            '/voice_command',
            self.voice_command_callback,
            10
        )

        # 交互状态
        self.current_target = None
        self.running = True

        # 启动终端输入线程
        self.input_thread = threading.Thread(target=self.terminal_input_loop)
        self.input_thread.daemon = True
        self.input_thread.start()

        # 主定时器
        self.create_timer(0.1, self.main_loop)

    def voice_command_callback(self, msg):
        """处理语音命令"""
        self.get_logger().info(f"收到语音命令: {msg.data}")
        frame = self.qwen.capture_frame()
        if frame is not None:
            self.process_user_input(frame, msg.data)

    def init_sam(self):
        """初始化SAM分割模型"""
        try:
            overrides = dict(
                task='segment',
                mode='predict',
                model='sam_b.pt',
                conf=0.01,
                save=False
            )
            sam = SAMPredictor(overrides=overrides)
            self.get_logger().info("SAM模型初始化成功")
            return sam
        except Exception as e:
            self.get_logger().error(f"SAM初始化失败: {str(e)}")
            return None

    def load_handeye_calibration(self):
        """加载手眼标定结果"""
        try:
            with open(self.handeye_result_file_name, 'r') as file:
                hand_eye_data = yaml.safe_load(file)
                if isinstance(hand_eye_data, list):
                    hand_eye_data = hand_eye_data[-1] if len(hand_eye_data) > 1 else hand_eye_data[0]

            self.T_camera_to_ee = np.eye(4)
            self.T_camera_to_ee[:3, :3] = np.array(hand_eye_data['rotation']).reshape((3, 3))
            self.T_camera_to_ee[:3, 3] = np.array(hand_eye_data['translation']).reshape((3,))

            # 验证旋转矩阵
            det = np.linalg.det(self.T_camera_to_ee[:3, :3])
            if not np.isclose(det, 1.0, atol=1e-3):
                self.get_logger().warn(f"旋转矩阵行列式为 {det:.3f} (应接近1.0)")
        except Exception as e:
            self.get_logger().error(f"加载手眼标定失败: {e}")
            raise

    def terminal_input_loop(self):
        """终端输入循环"""
        print("\n交互系统已启动，请输入您的问题(输入'退出'结束对话):")
        while self.running:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                user_input = sys.stdin.readline().strip()
                if not user_input:
                    continue
                    
                if user_input.lower() in ['退出', 'exit', 'quit']:
                    self.running = False
                    break
                
                frame = self.qwen.capture_frame()
                if frame is None:
                    print("警告: 未能获取图像")
                    continue
                
                self.process_user_input(frame, user_input)

    def process_user_input(self, image_array, user_input):
        """处理用户输入（支持多物体）"""
        response = self.qwen.generate_response(image_array, user_input)
        if not response:
            print("警告: 未能获取AI响应")
            return
            
        response_data = self.qwen.extract_json(response)
        
        if not response_data:
            print(f"\nAI回复: {response}")
            asyncio.run(self.qwen.text_to_speech(response))
            return

        response_text = response_data.get('response', response)
        print(f"\nAI回复: {response_text}")
        asyncio.run(self.qwen.text_to_speech(response_text))

        # 获取深度图像（所有目标共享同一帧）
        frames = self.qwen.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            print("警告: 未能获取深度或彩色图像")
            return
        
        # 准备RGB图像
        color_image = np.asanyarray(color_frame.get_data())
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        # 创建主输出目录
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        main_save_dir = os.path.join(os.getcwd(), "out", timestamp)
        os.makedirs(main_save_dir, exist_ok=True)
        
        # 保存原始彩色和深度图像
        cv2.imwrite(os.path.join(main_save_dir, "master_color.png"), color_image)
        depth_data = np.asanyarray(depth_frame.get_data()).astype(np.float32)/1000.0
        cv2.imwrite(os.path.join(main_save_dir, "master_depth.tiff"), depth_data)

        # 处理容器（如果有）
        container_info = None
        if "container" in response_data and response_data["container"] and "bbox_2d" in response_data["container"]:
            try:
                container_info = self.process_container(response_data["container"], rgb_image)
                if container_info:
                    self.publish_container_pose(container_info)
            except Exception as e:
                print(f"处理容器时出错: {e}")

        # 处理目标物体（支持多物体）
        targets = []
        if "target_objects" in response_data:  # 多物体格式
            targets = response_data["target_objects"] if isinstance(response_data["target_objects"], list) else [response_data["target_objects"]]
        elif "target_object" in response_data:  # 兼容单物体格式
            targets = [response_data["target_object"]]

        all_target_info = []
        for i, target_data in enumerate(targets):
            if not target_data or "bbox_2d" not in target_data:
                continue
                
            try:
                target_dir = os.path.join(main_save_dir, f"target_{i}")
                target_info = self.process_target(target_data, rgb_image, target_dir)
                if target_info:
                    all_target_info.append(target_info)
                    self.publish_target_pose(target_info["point_3d"], target_info["angle"])
            except Exception as e:
                print(f"处理目标物体{i}时出错: {e}")

        # 显示标记图像（如果需要）
        if response_data.get("need_show_image", False):
            boxes_to_show = []
            if "target_objects" in response_data:
                boxes_to_show.extend(response_data["target_objects"])
            elif "target_object" in response_data:
                boxes_to_show.append(response_data["target_object"])
            if "container" in response_data and response_data["container"]:
                boxes_to_show.append(response_data["container"])
            
            if boxes_to_show:
                try:
                    # 确保boxes_to_show是字典列表
                    if isinstance(boxes_to_show, dict):
                        boxes_to_show = [boxes_to_show]
                    self.qwen.plot_bounding_boxes(image_array, boxes_to_show)
                except Exception as e:
                    print(f"显示标注图像失败: {e}")

    def process_container(self, container_data, image_array):
        """处理容器位置"""
        try:
            x1, y1, x2, y2 = container_data["bbox_2d"]
            center = [(x1 + x2) // 2, (y1 + y2) // 2]
            
            sam_mask = None
            if self.sam_predictor is not None:
                try:
                    torch.cuda.empty_cache()
                    
                    # 设置当前图像（限制尺寸）
                    max_size = 1280
                    h, w = image_array.shape[:2]
                    if max(h, w) > max_size:
                        scale = max_size / max(h, w)
                        image_resized = cv2.resize(image_array, (int(w*scale), int(h*scale)))
                    else:
                        image_resized = image_array
                    
                    self.sam_predictor.set_image(image_resized)
                    
                    # 使用检测框提示
                    input_box = np.array([x1, y1, x2, y2])
                    results = self.sam_predictor(bboxes=[input_box])
                    
                    if results and results[0].masks:
                        sam_mask = results[0].masks.data[0].cpu().numpy()
                        sam_mask = (sam_mask > 0).astype(np.uint8) * 255
                        
                        M = cv2.moments(sam_mask)
                        if M["m00"] > 0:
                            center = [int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])]
                            print(f"使用SAM计算容器精确中心: {center}")
                except Exception as e:
                    print(f"SAM处理容器失败: {e}")
                finally:
                    if hasattr(self.sam_predictor, 'reset_image'):
                        self.sam_predictor.reset_image()
                    torch.cuda.empty_cache()
            
            point_3d = self.pixel_to_3d(center[0], center[1])
            if point_3d is None:
                print(f"警告: 无法计算容器 {container_data.get('label', '')} 的3D坐标")
                return None
            
            print(f"容器精确位置: {point_3d}")
            return {
                "point_3d": point_3d,
                "label": container_data.get("label", "container"),
                "description": container_data.get("description", "")
            }
        except Exception as e:
            print(f"处理容器时发生错误: {e}")
            return None

    def process_target(self, target_data, rgb_image, save_dir):
        """处理目标物体"""
        try:
            os.makedirs(save_dir, exist_ok=True)
            x1, y1, x2, y2 = target_data["bbox_2d"]
            center = [(x1 + x2) // 2, (y1 + y2) // 2]
            target = {
                "center": center,
                "label": target_data.get("label", "object"),
                "description": target_data.get("description", ""),
                "bbox": [x1, y1, x2, y2]
            }
            print(f"处理目标: {target['label']} - {target['description']}")
            
            qwen_mask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)
            cv2.rectangle(qwen_mask, (x1, y1), (x2, y2), 1, -1)
            
            sam_mask = None
            if self.sam_predictor is not None:
                try:
                    torch.cuda.empty_cache()
                    max_size = 1280
                    h, w = rgb_image.shape[:2]
                    if max(h, w) > max_size:
                        scale = max_size / max(h, w)
                        rgb_image_resized = cv2.resize(rgb_image, (int(w*scale), int(h*scale)))
                    else:
                        rgb_image_resized = rgb_image
                    
                    self.sam_predictor.set_image(rgb_image_resized)
                    input_box = np.array([x1, y1, x2, y2])
                    results = self.sam_predictor(bboxes=[input_box])
                    
                    if results and results[0].masks:
                        sam_mask = results[0].masks.data[0].cpu().numpy()
                        sam_mask = (sam_mask > 0).astype(np.uint8) * 255
                        print(f"{target['label']} SAM掩码生成成功")
                except Exception as e:
                    print(f"SAM处理失败: {e}")
                finally:
                    if hasattr(self.sam_predictor, 'reset_image'):
                        self.sam_predictor.reset_image()
                    torch.cuda.empty_cache()
            
            # 保存目标相关图像
            cv2.imwrite(os.path.join(save_dir, "color.png"), rgb_image)
            if sam_mask is not None:
                cv2.imwrite(os.path.join(save_dir, "sam_mask.png"), sam_mask)
            
            # 可视化最终结果 - 使用rgb_image
            vis_img = rgb_image.copy()
            if sam_mask is not None:
                # 绘制掩码边界
                contours, _ = cv2.findContours(sam_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis_img, contours, -1, (0, 0, 255), 2)
            
            # 计算抓取点和角度
            grasp_point = None
            grasp_angle = 0
            if sam_mask is not None:
                M = cv2.moments(sam_mask)
                if M["m00"] > 0:
                    grasp_point = [int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])]
                    
                    # 提取掩码点并归一化
                    pts = np.column_stack(np.where(sam_mask > 0)).astype(np.float32)
                    pts[:, 0] -= grasp_point[0]
                    pts[:, 1] -= grasp_point[1]
                    
                    # PCA计算主轴方向
                    _, eigenvectors = cv2.PCACompute(pts, mean=None)
                    v1 = eigenvectors[0]
                    
                    grasp_angle = (math.atan2(v1[1], v1[0]) + math.pi) % math.pi
                    print(f"抓取点: {grasp_point}, 主轴角度: {math.degrees(grasp_angle):.1f}°")

                    # 在可视化图像上绘制抓取点和方向
                    cv2.circle(vis_img, (grasp_point[0], grasp_point[1]), 8, (0, 255, 0), -1)
                    length = 30
                    end_x = int(grasp_point[0] + length * math.cos(grasp_angle))
                    end_y = int(grasp_point[1] - length * math.sin(grasp_angle))
                    cv2.line(vis_img, (grasp_point[0], grasp_point[1]), (end_x, end_y), (0, 255, 0), 2)
            
            # 保存可视化结果
            cv2.imwrite(os.path.join(save_dir, "grasp_vis.png"), vis_img)
            
            # 角度调整
            if grasp_angle > math.pi/2:
                grasp_angle = math.pi - grasp_angle
            else:
                grasp_angle = - grasp_angle
            print(f"{target['label']} 夹爪抓取角度: {math.degrees(grasp_angle):.1f}°")

            # 确定目标点并转换3D坐标
            target_point = grasp_point if grasp_point is not None else target["center"]
            print(f"{target['label']} 最终目标点: {target_point}")
            
            point_3d = self.pixel_to_3d(target_point[0], target_point[1])
            if point_3d is None:
                print(f"警告: 无法计算目标 {target['label']} 的3D坐标，使用默认值")
                point_3d = [0, 0, 0.3]
            
            print(f"{target['label']} 3D坐标: {point_3d}")
            return {
                "label": target["label"],
                "point_2d": target_point,
                "point_3d": point_3d,
                "angle": grasp_angle,
                "description": target["description"]
            }
        except Exception as e:
            print(f"处理目标物体时发生错误: {e}")
            return None

    def publish_container_pose(self, container_info):
        """发布容器位姿"""
        try:
            tf_ee = self.tf_buffer.lookup_transform(
                self.base_link, 
                self.ee_link, 
                rclpy.time.Time()
            )

            # 转换为齐次变换矩阵
            homogeneous_matrix = np.eye(4)
            homogeneous_matrix[0, 3] = tf_ee.transform.translation.x
            homogeneous_matrix[1, 3] = tf_ee.transform.translation.y
            homogeneous_matrix[2, 3] = tf_ee.transform.translation.z

            rotation = R.from_quat([
                tf_ee.transform.rotation.x,
                tf_ee.transform.rotation.y,
                tf_ee.transform.rotation.z,
                tf_ee.transform.rotation.w
            ])
            homogeneous_matrix[:3, :3] = rotation.as_matrix()
            
            # 计算目标位置
            point_ee = homogeneous_matrix @ self.T_camera_to_ee @ np.append(container_info["point_3d"], 1.0)
            
            # 发布容器位姿
            container_msg = PoseStamped()
            container_msg.header.stamp = self.get_clock().now().to_msg()
            container_msg.header.frame_id = self.base_link
            container_msg.pose.position.x = point_ee[0]
            container_msg.pose.position.y = point_ee[1]
            container_msg.pose.position.z = 0.35  # 固定高度

            q = R.from_euler('xyz', [-np.pi, 0, 0]).as_quat()
            container_msg.pose.orientation.x = q[0]
            container_msg.pose.orientation.y = q[1]
            container_msg.pose.orientation.z = q[2]
            container_msg.pose.orientation.w = q[3]
                    
            self.container_pose_pub.publish(container_msg)

            self.get_logger().info(
                f"发布容器位姿: [{point_ee[0]:.3f}, {point_ee[1]:.3f}, 0.350]"
            )
            print(f"发布容器位姿: [{point_ee[0]:.3f}, {point_ee[1]:.3f}, 0.350]")
            
        except (tf2_ros.LookupException, 
               tf2_ros.ConnectivityException, 
               tf2_ros.ExtrapolationException) as e:
            self.get_logger.error(f"TF错误: {str(e)}")

    def publish_target_pose(self, point_camera, angle):
        """发布目标位姿"""
        try:
            # 获取当前末端执行器位姿
            tf_ee = self.tf_buffer.lookup_transform(
                self.base_link, 
                self.ee_link, 
                rclpy.time.Time()
            )

            # 转换为齐次变换矩阵
            homogeneous_matrix = np.eye(4)
            homogeneous_matrix[0, 3] = tf_ee.transform.translation.x
            homogeneous_matrix[1, 3] = tf_ee.transform.translation.y
            homogeneous_matrix[2, 3] = tf_ee.transform.translation.z

            rotation = R.from_quat([
                tf_ee.transform.rotation.x,
                tf_ee.transform.rotation.y,
                tf_ee.transform.rotation.z,
                tf_ee.transform.rotation.w
            ])
            homogeneous_matrix[:3, :3] = rotation.as_matrix()
            
            # 计算目标位置
            point_ee = homogeneous_matrix @ self.T_camera_to_ee @ np.append(point_camera, 1.0)
            
            # 发布目标位姿
            robot_msg = PoseStamped()
            robot_msg.header.stamp = self.get_clock().now().to_msg()
            robot_msg.header.frame_id = self.base_link
            robot_msg.pose.position.x = point_ee[0]
            robot_msg.pose.position.y = point_ee[1]
            robot_msg.pose.position.z = point_ee[2]

            q = R.from_euler('xyz', [-np.pi, 0, 0]).as_quat()
            robot_msg.pose.orientation.x = q[0]
            robot_msg.pose.orientation.y = q[1]
            robot_msg.pose.orientation.z = angle
            robot_msg.pose.orientation.w = q[3]
                    
            self.robot_pose_pub.publish(robot_msg)

            self.get_logger().info(
                f"发布目标位姿: [{point_ee[0]:.3f}, {point_ee[1]:.3f}, {point_ee[2]:.3f}]"
            )
            print(f"发布目标位姿: [{point_ee[0]:.3f}, {point_ee[1]:.3f}, {point_ee[2]:.3f}]")
            
        except (tf2_ros.LookupException, 
               tf2_ros.ConnectivityException, 
               tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f"TF错误: {str(e)}")

    def pixel_to_3d(self, x, y):
        """像素坐标转3D坐标"""
        try:
            frames = self.qwen.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                print("警告: 未能获取深度图像")
                return None

            depth = depth_frame.get_distance(x, y)
            print(f"获取像素({x}, {y})的深度: {depth:.3f}米")
            
            # 深度值有效性检查
            if depth <= 0 or math.isnan(depth):
                print("警告: 获取到无效深度值，使用默认值0.36米")
                depth = 0.36

            return np.array([
                (x - self.intrinsics[0, 2]) * (depth / self.intrinsics[0, 0]),
                (y - self.intrinsics[1, 2]) * (depth / self.intrinsics[1, 1]),
                depth
            ])
        except Exception as e:
            print(f"3D坐标转换失败: {e}")
            return None

    def main_loop(self):
        """主循环"""
        if not self.running:
            self.destroy_node()
            rclpy.shutdown()
            return
            
        # 捕获并显示当前帧
        try:
            frames = self.qwen.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame:
                color_image = np.asanyarray(color_frame.get_data())
                cv2.imshow('Camera View', color_image)
                cv2.waitKey(1)
        except Exception as e:
            print(f"显示图像失败: {e}")

    def destroy_node(self):
        """清理资源"""
        self.running = False
        if hasattr(self, 'qwen') and hasattr(self.qwen, 'pipeline'):
            self.qwen.pipeline.stop()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    try:
        node = HandEyeControl()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"程序运行出错: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()