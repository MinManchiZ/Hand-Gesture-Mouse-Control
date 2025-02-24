import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
# 在导入部分添加 CUDA 支持
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import torch
import numba
from numba import cuda

# 检查是否支持CUDA
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    print(f"GPU 加速已启用: {torch.cuda.get_device_name(0)}")

# 使用 Numba 优化距离计算
@numba.jit(nopython=True, fastmath=True)
def calculate_distance(x1, y1, x2, y2):
    """使用 Numba 优化的距离计算"""
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

class MouseControl:
    def __init__(self):
        self.last_smooth_x = screen_width / 2
        self.last_smooth_y = screen_height / 2
        self.last_move_time = time.time()
        # 添加坐标缓存
        self.coord_buffer_x = np.zeros(5)
        self.coord_buffer_y = np.zeros(5)
        self.buffer_index = 0
        
    @numba.jit(forceobj=True)
    def _apply_nonlinear_mapping(self, y, max_height):
        """使用 Numba 优化的非线性映射"""
        normalized_y = y / max_height
        mapped_y = normalized_y ** 1.2
        return mapped_y * max_height

    def move_mouse(self, x, y, frame_width, frame_height):
        move_interval = 0.05

        if time.time() - self.last_move_time >= move_interval:
            # 使用坐标缓冲进行平滑处理
            self.coord_buffer_x[self.buffer_index] = x
            self.coord_buffer_y[self.buffer_index] = y
            self.buffer_index = (self.buffer_index + 1) % 5

            # 计算平滑后的坐标
            x = np.median(self.coord_buffer_x)
            y = np.median(self.coord_buffer_y)

            vertical_scale = 2.5  # 增加垂直缩放
            horizontal_scale = 1.2
            y_offset = frame_height * 0.25  # 增加偏移量
            
            # 使用 Numba 优化的计算
            adjusted_y = max(0, y - y_offset)
            
            target_mouse_x = np.clip((screen_width - (x / frame_width) * screen_width * horizontal_scale), 0, screen_width)
            target_mouse_y = np.clip((adjusted_y / frame_height) * screen_height * vertical_scale, 0, screen_height)
            
            # 应用优化后的非线性映射
            target_mouse_y = self._apply_nonlinear_mapping(target_mouse_y, screen_height)
            
            # 使用自适应平滑因子
            velocity = np.sqrt((target_mouse_x - self.last_smooth_x)**2 + 
                             (target_mouse_y - self.last_smooth_y)**2)
            adaptive_smoothing = min(0.8, max(0.3, 1.0 - velocity / 1000))
            
            smooth_x = self.last_smooth_x * (1 - adaptive_smoothing) + target_mouse_x * adaptive_smoothing
            smooth_y = self.last_smooth_y * (1 - adaptive_smoothing) + target_mouse_y * adaptive_smoothing

            # 确保坐标在屏幕范围内
            smooth_x = np.clip(smooth_x, 0, screen_width)
            smooth_y = np.clip(smooth_y, 0, screen_height)

            self.last_smooth_x = smooth_x
            self.last_smooth_y = smooth_y
            pyautogui.moveTo(smooth_x, smooth_y)
            self.last_move_time = time.time()
            
# 设置 MediaPipe 手部检测模块
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)  # 不需要 use_gpu 参数
mp_drawing = mp.solutions.drawing_utils

# 设置屏幕大小
screen_width, screen_height = pyautogui.size()

# 设置点击和移动的灵敏度
gesture_distance_threshold = 20  # 点击的灵敏度
click_delay = 0.3  # 点击延迟
last_gesture_time = time.time()  # 上次手势执行时间

# 定义灵敏度系数，用来增强手势与屏幕之间的映射
sensitivity_factor = 10  # 这个系数可以根据需要进行调整，值越大鼠标移动越灵敏

# 添加平滑的移动频率控制
last_move_time = time.time()

# 插值平滑的参数
interpolation_factor = 0.1  # 插值因子，越小越平滑

# 平滑（EMA）参数
smoothing_factor = 0.5  # 越小越平滑，越大越快响应

class MouseControl:
    def __init__(self):
        self.last_smooth_x = screen_width / 2
        self.last_smooth_y = screen_height / 2
        self.last_move_time = time.time()

    def move_mouse(self, x, y, frame_width, frame_height):
        move_interval = 0.05

        if time.time() - self.last_move_time >= move_interval:
            current_mouse_x, current_mouse_y = pyautogui.position()

            # 增加垂直缩放系数
            vertical_scale = 2.0  # 从1.5增加到2.0
            horizontal_scale = 1.2

            # 修改Y轴映射方式，添加偏移量使手势范围向上延伸
            y_offset = frame_height * 0.2  # 添加20%的向上偏移
            adjusted_y = max(0, y - y_offset)
            
            target_mouse_x = np.clip((screen_width - (x / frame_width) * screen_width * horizontal_scale), 0, screen_width)
            target_mouse_y = np.clip((adjusted_y / frame_height) * screen_height * vertical_scale, 0, screen_height)

            # 使用更激进的非线性映射
            target_mouse_y = self._apply_nonlinear_mapping(target_mouse_y, screen_height)

            # 使用平滑（EMA）
            smooth_x = self.last_smooth_x * (1 - smoothing_factor) + target_mouse_x * smoothing_factor
            smooth_y = self.last_smooth_y * (1 - smoothing_factor) + target_mouse_y * smoothing_factor

            # 确保坐标在屏幕范围内
            smooth_x = np.clip(smooth_x, 0, screen_width)
            smooth_y = np.clip(smooth_y, 0, screen_height)

            self.last_smooth_x = smooth_x
            self.last_smooth_y = smooth_y
            pyautogui.moveTo(smooth_x, smooth_y)
            self.last_move_time = time.time()

    def _apply_nonlinear_mapping(self, y, max_height):
        """使用更激进的非线性映射"""
        normalized_y = y / max_height
        # 使用更小的指数来增强上部区域的精确度
        mapped_y = normalized_y ** 1.2  # 从1.5减小到1.2
        return mapped_y * max_height
        
        
# 设置点击动作
def left_click():
    pyautogui.click(button='left')

def right_click():
    pyautogui.click(button='right')

def press_key(key):
    """模拟按下键盘按键"""
    pyautogui.press(key)

def double_click():
    """模拟双击鼠标左键"""
    pyautogui.doubleClick(button='left')

def scroll_up():
    """模拟鼠标滚轮向上滚动"""
    pyautogui.scroll(10)  # 滚动向上

def scroll_down():
    """模拟鼠标滚轮向下滚动"""
    pyautogui.scroll(-10)  # 滚动向下

# 主程序，进行手势控制
def start_hand_gesture_control():
    global last_gesture_time

    # 创建 MouseControl 类的实例
    mouse_control = MouseControl()

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera!")
        return

    # 设置视频窗口
    cv2.namedWindow("Hand Gesture Control")

    while True:
        try:
            # 读取每一帧
            ret, frame = cap.read()

            if not ret:
                print("Failed to capture image from camera.")
                break

            # 获取视频帧的宽度和高度
            frame_height, frame_width, _ = frame.shape

            # 使用 OpenCV 转换为 RGB 图像
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 使用 MediaPipe 进行手部检测
            results = hands.process(rgb_frame)

            # 如果检测到手部
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 获取手的关键点
                    landmarks = hand_landmarks.landmark
                    # 判断当前手是左手还是右手
                    hand_x = landmarks[mp_hands.HandLandmark.WRIST].x

                    # 左手通常x坐标较小，右手较大
                    hand_label = "Left Hand" if hand_x < 0.5 else "Right Hand"

                    # 绘制手部关键点
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # 获取五个手指尖的坐标
                    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
                    index_finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_finger_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    ring_finger_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
                    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

                    # 获取屏幕上的坐标
                    thumb_x, thumb_y = int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height)
                    index_x, index_y = int(index_finger_tip.x * frame_width), int(index_finger_tip.y * frame_height)
                    middle_x, middle_y = int(middle_finger_tip.x * frame_width), int(middle_finger_tip.y * frame_height)
                    ring_x, ring_y = int(ring_finger_tip.x * frame_width), int(ring_finger_tip.y * frame_height)
                    pinky_x, pinky_y = int(pinky_tip.x * frame_width), int(pinky_tip.y * frame_height)

                    # 获取手腕的位置（作为手掌的中心）
                    wrist = landmarks[mp_hands.HandLandmark.WRIST]
                    hand_center = (int(wrist.x * frame_width), int(wrist.y * frame_height))

                    # 将手掌中心映射到鼠标位置
                    mouse_control.move_mouse(hand_center[0], hand_center[1], frame_width, frame_height)

                    # 判断拇指与食指是否对碰，模拟左键单击
                    thumb_index_distance = np.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)
                    if thumb_index_distance < gesture_distance_threshold:
                        if time.time() - last_gesture_time > click_delay:
                            left_click()
                            last_gesture_time = time.time()

                    # 判断食指与中指是否对碰，模拟右键单击
                    index_middle_distance = np.sqrt((index_x - middle_x) ** 2 + (index_y - middle_y) ** 2)
                    if index_middle_distance < gesture_distance_threshold:
                        if time.time() - last_gesture_time > click_delay:
                            right_click()
                            last_gesture_time = time.time()

                    # 判断大拇指与中指是否对碰，模拟滑轮向下滚动
                    thumb_middle_distance = np.sqrt((thumb_x - middle_x) ** 2 + (thumb_y - middle_y) ** 2)
                    if thumb_middle_distance < gesture_distance_threshold:
                        # 手指保持叠加，持续向下滚动
                        scroll_down()  # 模拟滑轮向下滚动
                        time.sleep(0.0)  # 滚动的间隔时间
                        last_gesture_time = time.time()
                    else:
                        # 如果手指分开，停止滚动
                        pass

                    # 判断大拇指与无名指是否对碰，模拟滑轮向上滚动
                    thumb_ring_distance = np.sqrt((thumb_x - ring_x) ** 2 + (thumb_y - ring_y) ** 2)
                    if thumb_ring_distance < gesture_distance_threshold:
                        # 手指保持叠加，持续向上滚动
                        scroll_up()  # 模拟滑轮向上滚动
                        time.sleep(0)  # 滚动的间隔时间
                        last_gesture_time = time.time()
                    else:
                        # 如果手指分开，停止滚动
                        pass

                    # 判断大拇指与小指是否对碰，模拟双击鼠标左键
                    thumb_pinky_distance = np.sqrt((thumb_x - pinky_x) ** 2 + 2 * (thumb_y - pinky_y) ** 2)
                    if thumb_pinky_distance < gesture_distance_threshold:
                        if time.time() - last_gesture_time > click_delay:
                            double_click()  # 模拟双击
                            last_gesture_time = time.time()

            # 显示带有手部标记的图像
            cv2.imshow("Hand Gesture Control", frame)

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Error occurred: {e}")
            break

    # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_hand_gesture_control()
