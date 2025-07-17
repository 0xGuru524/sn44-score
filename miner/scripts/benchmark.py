from ultralytics import YOLO
import time
import cv2
import numpy as np
import tensorrt as trt

model = YOLO("../data/football-player-detection-b4.engine", task="detect")
# model.export(format="engine", device=0, half=True, int8=False)
# model.export(
#     format="onnx",
#     dynamic=True,           # Enables dynamic input
#     opset=14,
#     simplify=True,
#     imgsz=640,
#     device=0
# )

img = cv2.imread("frame.jpg")
img = cv2.resize(img, (640, 640))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

start = time.time()
for _ in range(100):
    predict_result = model.predict(source=img, device=0, verbose=False)
    predict_result[0].save('result.jpg')
end = time.time()

print(f"FPS: {100 / (end - start):.2f}")