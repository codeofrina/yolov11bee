from ultralytics import YOLO

def main():
    # 1. 路径
    model_path = "/workspace/yolov11bee/experiments/yolo11n.pt"
    data_yaml  = "/workspace/dataset/BEE24-yolo/data.yaml"

    # 2. 载入模型
    model = YOLO(model_path)          # 自动下载/载入官方预训练权重

    # 3. 训练参数
    # https://docs.ultralytics.com/zh/usage/cfg/#train-settings
    train_args = dict(
        data=data_yaml,
        epochs=100,
        patience=10,
        batch=4,
        imgsz=800,
        device=0,
        workers=8,                # dataloader 线程
        project="/workspace/yolov11bee/experiments/0728trainbee24",
        name="01train",           # 日志目录 runs/train_bee_yolo11s

        # ---- 训练设置 ----
        optimizer="AdamW",
        seed=667788,
        single_cls=True,
        rect=True,                # 缩放不改变尺寸
        multi_scale=True,         # 多尺度训练
        cos_lr=False,             # 余弦学习率
        close_mosaic=10,          # 最后几个epoch关闭mosaic
        resume=False,
        amp=True,                 # 混合精度训练
        fraction=0.4,             # 只训练40%
        freeze=None,              # 冻结模型的前 N 层或按索引指定的层
        lr0=0.00005,              # 初始学习率
        lrf=0.01,                 # 最终学习率占初始学习率的百分比
        momentum=0.937,           # 用于Adam 优化器的 beta1
        weight_decay=0.0005,      # L2正则化项
        warmup_epochs=3,          # 学习率预热的epoch数，学习率从低值逐渐增加到初始学习率
        warmup_momentum=0.8,      # 热身阶段的初始动力，在热身期间逐渐调整到设定动力。
        warmup_bias_lr=0.1,       # 热身阶段的偏置参数学习率，有助于稳定初始历元的模型训练
        box=7.5,                  # 损失函数中边框损失部分的权重
        cls=0.5,                  # 分类损失在总损失函数中的权重
        dfl=1.5,                  # 分布焦点损失权重，在某些YOLO 版本中用于精细分类
        nbs=64,                   # 用于损耗正常化的标称批量大小
        dropout=0.0,
        val=Ture,                 # 在训练过程中进行验证
        plots=True,               # 生成并保存训练和验证指标图以及预测示例图


        # ---- 数据增强 ----
        hsv_h=0.015,              # 调整图像的色调
        hsv_s=0.7,                # 改变图像饱和度
        hsv_v=0.4,                # 改变图像的亮度
        degrees=15,               # 在指定的度数范围内随机旋转图像
        translate=0.1,            # 将图像进行水平和垂直平移，平移幅度为图像大小的一小部分，有助于学习检测部分可见的物体
        scale=0.5,            #   # 通过增益因子缩放图像，模拟物体与摄像机的不同距离。
        shear=0.0,                # 按指定角度剪切图像，模拟从不同角度观察物体的效果。
        perspective=0.0,          # 对图像进行随机透视变换，增强模型理解三维空间中物体的能力。
        flipud=0.0,               # 以指定的概率将图像翻转过来，在不影响物体特征的情况下增加数据的可变性。
        fliplr=0.5,               # 以指定概率从左到右翻转图像，这对学习对称物体和增加数据集多样性很有用。
        mosaic=1.0,           #   # 将四幅训练图像合成一幅，模拟不同的场景构成和物体互动。对复杂场景的理解非常有效
        mixup=0.0,                # 混合两幅图像及其标签，创建合成图像。通过引入标签噪声和视觉变化，增强模型的泛化能力
        cutmix=0.0               # 合并两幅图像的部分内容，创建部分混合图像，同时保留不同的区域。通过创建遮挡场景来增强模型的鲁棒性。
    )

    # 4. 开始训练
    model.train(**train_args)

    # 5. 训练完在 val 上评估
    metrics = model.val()
    print("mAP@0.5 :", metrics.box.map50)
    print("mAP@0.5:0.95 :", metrics.box.map)

if __name__ == "__main__":
    main()