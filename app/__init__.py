#Gpu模式测试页面，目前正在测试中……暂时无法使用。
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"TensorFlow is using the following GPU(s): {physical_devices}")
else:
    print("TensorFlow is using CPU. Please ensure that the correct GPU version is installed.")
