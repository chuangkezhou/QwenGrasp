#!/usr/bin/env python3
import os
import time
import struct
import logging
import serial
import serial.tools.list_ports
import glob
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
from datetime import datetime
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading

# 配置日志系统
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('emm42_controller.log')
    ]
)

class Emm42Controller(Node):
    def __init__(self, port=None, baudrate=115200, address=0x01):
        """
        EMM42 串口控制器
        :param port: 串口设备路径(如/dev/ttyUSB0)
        :param baudrate: 波特率(默认115200)
        :param address: 设备地址(1-255, 0为广播地址)
        """
        super().__init__('emm42_controller')
        
        # 参数声明
        self.declare_parameters(namespace='',
            parameters=[
                ('port', '/dev/ttyUSB0'),
                ('baudrate', 115200),
                ('address', 1),
                ('speed_rpm', 120),
                ('accel_rpm_s', 0),
                ('home_position', 0),
                ('target_position', 17000),
                ('current_threshold', 0.7)
            ])
        
        # 获取参数值
        self.port = self.get_parameter('port').value
        self.baudrate = self.get_parameter('baudrate').value
        self.address = self.get_parameter('address').value
        self.speed_rpm = self.get_parameter('speed_rpm').value
        self.accel_rpm_s = self.get_parameter('accel_rpm_s').value
        self.home_position = self.get_parameter('home_position').value
        self.target_position = self.get_parameter('target_position').value
        self.current_threshold = self.get_parameter('current_threshold').value

        # 串口连接
        self.ser = None
        self.serial_lock = threading.Lock()
        
        # 绘图相关
        self.plot_lock = threading.Lock()
        self.current_history = []
        self.time_history = []
        self.plot_start_time = None
        self.plot_active = False
        self.fig = None
        self.ax = None
        self.line = None
        self.threshold_line = None

        # ROS2订阅器
        self.subscription = self.create_subscription(
            String,
            'gripper_commands',
            self.command_callback,
            10)
        
        # 延迟初始化串口
        self.connect_timer = self.create_timer(1.0, self._delayed_connect)

    def _delayed_connect(self):
        """延迟串口连接，确保节点完全初始化"""
        self.connect_timer.cancel()
        if not self.connect():
            self.get_logger().error("串口连接失败")

    def command_callback(self, msg):
        """命令回调函数"""
        command = msg.data.strip().lower()
        self.get_logger().info(f'收到命令: {command}')
        
        if command == 'g':
            self.get_logger().info('执行抓取操作...')
            threading.Thread(target=self._run_grasp_operation, daemon=True).start()
        elif command == 'r':
            self.get_logger().info('执行回零操作...')
            threading.Thread(target=self.return_to_home, daemon=True).start()
        else:
            self.get_logger().warn(f'未知命令: {command}')

    def _run_grasp_operation(self):
        """线程安全的抓取操作"""
        # 初始化绘图
        self._init_plot()
        
        # 执行抓取
        result = self.grip_operation()
        
        # 显示并销毁绘图
        self._show_and_close_plot()
        
        if result:
            self.get_logger().info('抓取成功！')
        else:
            self.get_logger().error('抓取失败！')

    def _init_plot(self):
        """初始化绘图"""
        with self.plot_lock:
            self.current_history = []
            self.time_history = []
            self.plot_start_time = time.time()
            self.plot_active = True
            
            # 在创建图形前设置全局参数
            plt.rcParams.update({
                'xtick.labelsize': 18,  # x轴刻度
                'ytick.labelsize': 18,  # y轴刻度
                # 'axes.labelsize': 14,   # 坐标轴标签
                # 'axes.titlesize': 16     # 标题
            })
            
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
                
            self.line, = self.ax.plot([], [], 'b-', linewidth=2)
            self.threshold_line = self.ax.axhline(
                y=self.current_threshold, 
                color='r', 
                linestyle='--', 
                linewidth=2,  # 增加线宽
                label='Threshold'
                #设置label字体大小

            )
            
            # 设置标签和标题（使用更大的字体）
            self.ax.set_xlabel('Time (s)', fontsize=16 )
            self.ax.set_ylabel('Current (A)', fontsize=16)
            self.ax.set_title('EMM42 Motor Current Monitoring', fontsize=18)
            
            # 网格和图例
            self.ax.grid(True, linestyle='--', alpha=0.6)
            self.ax.legend(fontsize=18)
            
            # 强制立即应用刻度标签设置
            self.ax.tick_params(axis='both', which='major', labelsize=18)
            plt.tight_layout()

    def _update_plot(self, current, timestamp):
        """更新绘图数据"""
        with self.plot_lock:
            if not self.plot_active:
                return
                
            self.current_history.append(current)
            self.time_history.append(timestamp)
            
            self.line.set_data(self.time_history, self.current_history)
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)

    def _show_and_close_plot(self):
        """显示并关闭绘图"""
        with self.plot_lock:
            if not self.plot_active:
                return
                
            # 确保图形更新
            if self.current_history:
                plt.pause(0.5)  # 短暂显示
            
            plt.close(self.fig)
            plt.ioff()
            self.plot_active = False

    def connect(self):
        """连接串口设备"""
        with self.serial_lock:
            try:
                if not self.port:
                    self.port = self._find_serial_port()
                    if not self.port:
                        return False
                
                if not self._check_serial_permission():
                    return False
                
                self.ser = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    bytesize=8,
                    parity='N',
                    stopbits=1,
                    timeout=0.5,
                    write_timeout=0.5
                )
                
                # Linux特殊设置
                self.ser.dtr = False
                self.ser.rts = False
                
                logging.info(f"已连接 {self.port} 波特率 {self.baudrate}")
                return True
            except Exception as e:
                logging.error(f"连接失败: {str(e)}")
                return False

    def _find_serial_port(self):
        """自动检测串口设备"""
        ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
        if not ports:
            logging.error("未找到可用串口设备")
            return None
        
        logging.info("可用串口设备:")
        for i, port in enumerate(ports):
            logging.info(f"{i+1}. {port}")
        return ports[0]

    def _check_serial_permission(self):
        """检查串口权限"""
        if not os.access(self.port, os.R_OK | os.W_OK):
            logging.error(f"无权限访问 {self.port}")
            logging.error("请执行: sudo chmod 666 {self.port}")
            logging.error("或将自己加入dialout组: sudo usermod -a -G dialout $USER")
            return False
        return True

    def _build_frame(self, cmd_code, data=None):
        """
        构建指令帧
        格式: [地址][功能码][数据...][结束字节0x6B]
        """
        frame = bytearray()
        frame.append(self.address)  # 设备地址
        frame.append(cmd_code)      # 功能码
        if data is not None:
            data = bytearray(data) 
            for byte in data:
                frame.append(byte)
        
        frame.append(0x6B)  # 固定结束字节
        logging.debug(f"构建帧: {frame.hex(' ')}")
        return frame

    def _send_command(self, cmd_code, data=None, retry=3):
        """发送指令并读取响应"""
        with self.serial_lock:
            frame = self._build_frame(cmd_code, data)
            
            for attempt in range(retry):
                try:
                    if not self.ser or not self.ser.is_open:
                        if not self.connect():
                            continue
                    
                    self.ser.flushInput()
                    self.ser.flushOutput()
                    
                    self.ser.write(frame)
                    time.sleep(0.05)  # 指令间隔
                    
                    # 根据功能码确定响应长度
                    if cmd_code == 0xFD:  # 位置控制
                        response_len = 4
                    elif cmd_code == 0x27:  # 读取相电流
                        response_len = 5
                    elif cmd_code == 0xF3:  # 电机使能
                        response_len = 4
                    elif cmd_code == 0xFE:  # 急停
                        response_len = 4
                    else:
                        response_len = 5  # 默认
                    
                    response = self.ser.read(response_len)
                    
                    if response:
                        logging.debug(f"收到响应: {response.hex(' ')}")
                        if response[-1] == 0x6B:  # 校验最后一个字节
                            return response
                        else:
                            logging.warning("响应校验失败")
                    
                except serial.SerialTimeoutException:
                    logging.warning(f"超时 (尝试 {attempt+1}/{retry})")
                except Exception as e:
                    logging.error(f"通信错误: {str(e)}")
                
                time.sleep(0.1)
            
            return None

    def enable_motor(self, enable=True):
        """电机使能控制"""
        frame = bytearray()
        frame.append(self.address)  # 地址
        frame.append(0xF3)         # 功能码
        frame.append(0xAB)         # 固定值
        frame.append(0x01 if enable else 0x00)  # 使能状态
        frame.append(0x00)         # 多机同步标志(0=不启用)
        frame.append(0x6B)         # 结束字节
        
        logging.debug(f"电机使能帧: {frame.hex(' ')}")
        return self._send_command(0xF3, [0xAB, 0x01 if enable else 0x00, 0x00]) is not None

    def move_to_position(self, position, speed=None, accel=None):
        """位置控制"""
        speed = speed or self.speed_rpm
        accel = accel or self.accel_rpm_s
        speed_low = speed & 0xFF
        speed_high = (speed >> 8) & 0xFF
        
        # 32位位置拆分为4个8位寄存器
        pos_0 = position & 0xFF
        pos_1 = (position >> 8) & 0xFF
        pos_2 = (position >> 16) & 0xFF
        pos_3 = (position >> 24) & 0xFF
        
        data = [
            0x01,       # 方向(0=CW, 1=CCW)
            speed_high, # 速度(RPM)
            speed_low,  # 速度(RPM)
            0x00,       # 加速度档位
            pos_3,      # 位置低字节
            pos_2,      # 位置次低字节
            pos_1,      # 位置次高字节
            pos_0,      # 位置高字节
            0x01,       # 绝对位置模式(0=相对,1=绝对)
            0x00        # 多机同步标志(0=不启用,1=启用)
        ]
        return self._send_command(0xFD, data) is not None

    def read_current(self):
        """读取相电流"""
        response = self._send_command(0x27)
        if response and len(response) >= 4:
            current_ma = struct.unpack('>H', response[2:4])[0]
            return current_ma / 1000.0  # mA转A
        return None

    def stop_motor(self):
        """急停指令"""
        return self._send_command(0xFE, [0x98, 0x00]) is not None

    def return_to_home(self):
        """回零操作"""
        if not self.connect():
            return False
        
        try:
            # 1. 使能电机
            if not self.enable_motor(True):
                return False
            time.sleep(0.1)  # 等待使能稳定

            # 2. 归零
            logging.info("正在归零...")
            if not self.move_to_position(000):
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"回零操作异常: {str(e)}")
            return False
        finally:
            if self.ser and self.ser.is_open:
                self.ser.close()

    def grip_operation(self):
        """完整的抓取控制流程"""
        if not self.connect():
            return False
        
        try:
            # 1. 使能电机
            if not self.enable_motor(True):
                return False
            time.sleep(0.1)

            # 2. 归零
            logging.info("正在归零...")
            if not self.move_to_position(0):
                return False
            
            # 3. 移动到目标位置
            logging.info(f"移动到目标位置: {self.target_position}脉冲")
            if not self.move_to_position(self.target_position):
                return False
            
            # 4. 监测堵转
            logging.info("监测电流中...")
            start_time = time.time()
            while time.time() - start_time < 10:  # 超时10秒
                current = self.read_current()
                if current is None:
                    logging.warning("电流读取失败")
                    time.sleep(0.1)
                    continue
                
                current_time = time.time() - self.plot_start_time
                logging.debug(f"时间: {current_time:.2f}s, 电流: {current:.3f}A")
                
                # 更新曲线
                self._update_plot(current, current_time)
                
                if current >= self.current_threshold:
                    logging.info(f"达到堵转电流({current:.3f}A)，停止电机")
                    self.stop_motor()
                    return True
                
                time.sleep(0.05)
            
            logging.warning("移动超时未完成")
            return False
            
        except Exception as e:
            logging.error(f"抓取操作异常: {str(e)}")
            return False
        finally:
            if self.ser and self.ser.is_open:
                self.ser.close()

    def destroy_node(self):
        """重写销毁方法确保资源释放"""
        self._show_and_close_plot()
        if self.ser and self.ser.is_open:
            self.ser.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    controller = Emm42Controller()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()
        logging.info("程序结束")

if __name__ == "__main__":
    main()
