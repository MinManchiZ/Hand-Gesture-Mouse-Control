# 🚀 **飞鼠工具** (Hand Gesture Mouse Control) 

![Python](https://img.shields.io/badge/Language-Python-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![OpenCV](https://img.shields.io/badge/OpenCV-4.5.1-red) ![PyAutoGUI](https://img.shields.io/badge/PyAutoGUI-v0.9-blueviolet)

---

## 📝 项目简介

飞鼠工具（Hand Gesture Mouse Control）是一个基于 **Python**、**MediaPipe** 和 **OpenCV** 的隔空手势识别系统，能够通过用户的手势控制计算机鼠标。它通过摄像头捕捉手部动作并将其转换为鼠标操作，完全解放你的双手，摆脱传统鼠标和触控板的束缚，带来前所未有的交互体验。无触摸、无物理接触，让你体验未来科技的便捷。


💡 **核心优势**:
- **精准识别**：高效的手势识别和追踪，极低的延迟。
- **极简操作**：通过简单手势即可实现鼠标操作，手势越少，效率越高。
- **完全可定制**：只需少量代码修改，快速扩展更多手势和功能。

### 调整手势灵敏度：
- `sensitivity_factor = 3.5`，该值越大，手势识别越灵敏，可能会出现误判。
- `threshold = 0.5`，该值越小，识别越敏感，越容易误判。

---

## 🖱️ 手势操作
通过以下手势，你可以控制虚拟鼠标的各种操作：

- **鼠标左键单击**：拇指与食指捏合。
- **鼠标滚轮向下**：拇指与中指捏合。
- **鼠标滚轮向上**：拇指与无名指捏合。
- **鼠标左键双击**：拇指与小指捏合。

手掌的中心点（手腕位置）会实时映射为鼠标位置，虚拟鼠标会跟随手部的动作移动。

---

## 🔧 核心技术栈

我们使用了以下视觉技术和工具：

- **Python 3.x**：强大且易于扩展的编程语言。
- **OpenCV**：经典的计算机视觉库，负责图像捕获和处理。
- **MediaPipe**：由 Google 推出的高效手部追踪框架，提供精准的关键点检测。
- **PyAutoGUI**：流行的自动化控制库，模拟鼠标点击、滚动等操作。
- **Numpy**：基础的数值计算库，处理数据转换和数学运算。

📚 **依赖**：

- `opencv-python` : 4.5.1
- `mediapipe` : 0.8.3
- `pyautogui` : 0.9.50
- `numpy` : 1.21.0

---

## 🛠️ 使用说明
### 可以拓展的功能如下：

1.截屏功能：通过特定手势触发 pyautogui.screenshot() 进行屏幕截图。

2.音量控制：通过手势触发 pyautogui.hotkey() 来增加或减少音量。

3.窗口切换：通过手势触发 pyautogui.hotkey('alt', 'tab') 实现窗口切换。

4.手势操作灵敏度：通过调整 gesture_distance_threshold 和 sensitivity_factor 来控制手势识别灵敏度和鼠标移动速度。

5.增加新的手势：在 hand_gesture.py 中添加新的手势识别条件（例如拇指与中指捏合触发右键操作）。

6.自定义手势：修改 move_mouse() 方法中的坐标映射来调整鼠标的移动范围。

7.摄像头选择：修改 cv2.VideoCapture() 参数来选择不同的摄像头（默认为 0）。

### 运行，将项目下载部署在本地————按照requirements.txt文件搭建python运行环境，运行hand_gesture.py.py文件。

---

## 联系我们
如有问题或建议，欢迎通过 GitHub 提交 Issues 或者直接联系我们。
我们非常欢迎开发者和用户提供反馈，帮助我们做得更好！
