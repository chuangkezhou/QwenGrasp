# QwenGrasp
项目使用xarm5机械臂，使用ros2进行通信，由于是5自由度机械臂不适合6d抓取，最终没有采用Graspnet进行预测抓取，使用平面方法

# qwen_test
为使用qwen测试图片标注能力

# publish_goalpose_qwen_sam_voice_generality(核心代码)
接收用户语音/文本指令，调用qwen对目标物体进行标注，使用sam分割出目标物体，计算形心与pca主轴，坐标转换，发布目标物体的抓取点以及角度到 test_xarm_grasp 节点

# voice_recognition
调用百度api语音识别，发布指令到 publish_goalpose_qwen_sam_voice_generality 代码节点

# test_xarm_grasp
受限于无自由度，将运动分解为平面直线运动，队列处理多个目标物体抓取
