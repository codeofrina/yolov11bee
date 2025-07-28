from ultralytics import YOLO
model = YOLO("/workspace/yolov11bee/experiments/0728trainbee24/01train4/")
results = model.train(resume=True)