from ultralytics import YOLO

# 加载之前的 checkpoint
model = YOLO("/workspace/yolov11bee/experiments/0729L2norm-lr0.00003/01train/weights/last.pt")

# 继续训练
model.train(resume=True)