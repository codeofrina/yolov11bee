"""
YOLOv11 K-Fold Cross-Validation Training Script
Author: ?
"""

import os
import yaml
import json
from pathlib import Path
from ultralytics import YOLO
from kfold_yolo import prepare_fold                                                           # 切分函数

# --------------------------------------------------
# 2. 训练主函数
# --------------------------------------------------
def run_kfold():
    k = 5                                                                                  #* # 折数
    metrics_all = []                                                                          # 存放每一折的 val metrics dict

    for fold in range(k):
        print(f"\n{'='*60}")
        print(f"🔥 Starting Fold {fold + 1}/{k}")
        print('='*60)

        # 2.1 切分数据（把对应折移动到 val_*）
        prepare_fold(k=fold, n=k)

        # 2.2 创建本次折的 project 子目录
        model_path   = "/workspace/yolov11bee/experiments/yolo11n.pt",
        data_yaml    = "/workspace/dataset/BEE24-yolo/data.yaml",
        project_name = "para01",                                                           #* # 本次参数描述
        project_root = "/workspace/yolov11bee/experiments/kfold_runs/"+project_name,          # 所有折日志的根目录

        # 2.3 载入模型
        model = YOLO(model_path)

        # 2.4 组成训练参数字典
        train_args = dict(
            project     = str(project_root),
            name        = "train-fold-"+str(fold),
            data        = data_yaml,

            epochs=500,               #  ##################################
            patience=2000000,            # # early stop ##########################
            batch=24,               # # batch size
            imgsz=800,
            device=0,
            workers=8,                # dataloader 线程 train 为 8 核

            # ---- 训练设置 ----
            optimizer="AdamW",
            seed=667788,
            single_cls=True,
            rect=True,                # 缩放不改变尺寸
            multi_scale=True,     #   # 多尺度训练
            cos_lr=False,         #*  # 余弦学习率
            close_mosaic=10,          # 最后几个epoch关闭mosaic
            resume=False,
            amp=True,                 # 混合精度训练
            fraction=1.0,         #*  # 只训练40%
            freeze=None,              # 冻结模型的前 N 层或按索引指定的层
            lr0=0.00002,          #*  # 初始学习率
            lrf=0.01,             #   # 最终学习率占初始学习率的百分比
            momentum=0.937,           # 用于Adam 优化器的 beta1
            weight_decay=0.01,    #*  # L2正则化项
            warmup_epochs=3,      #   # 学习率预热的epoch数，学习率从低值逐渐增加到初始学习率
            warmup_momentum=0.8,      # 热身阶段的初始动力，在热身期间逐渐调整到设定动力。
            warmup_bias_lr=0.1,       # 热身阶段的偏置参数学习率，有助于稳定初始历元的模型训练
            box=7.5,                  # 损失函数中边框损失部分的权重
            cls=0.5,                  # 分类损失在总损失函数中的权重
            dfl=1.5,                  # 分布焦点损失权重，在某些YOLO 版本中用于精细分类
            nbs=64,                   # 用于损耗正常化的标称批量大小
            dropout=0.0,
            val=True,                 # 在训练过程中进行验证
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
            flipud=0.0,           #*  # 以指定的概率将图像翻转过来，在不影响物体特征的情况下增加数据的可变性。
            fliplr=0.5,               # 以指定概率从左到右翻转图像，这对学习对称物体和增加数据集多样性很有用。
            mosaic=1.0,           #   # 将四幅训练图像合成一幅，模拟不同的场景构成和物体互动。对复杂场景的理解非常有效
            mixup=0.0,            #*  # 混合两幅图像及其标签，创建合成图像。通过引入标签噪声和视觉变化，增强模型的泛化能力
            cutmix=0.0            #*  # 合并两幅图像的部分内容，创建部分混合图像，同时保留不同的区域。通过创建遮挡场景来增强模型的鲁棒性。
        )

        # 2.5 训练
        model.train(**train_args)

        # 2.6 验证并收集指标
        val_res = model.val()
        metric_dict = dict(
            fold       = fold + 1,
            map50      = float(val_res.box.map50),
            map5095    = float(val_res.box.map),
            precision  = float(val_res.box.mp),
            recall     = float(val_res.box.mr),
            f1         = float(val_res.box.f1),
        )
        metrics_all.append(metric_dict)

        # 可选：把单折结果也写进折目录
        with open(project_root / "val_metrics.json", "w") as f:
            json.dump(metric_dict, f, indent=2)

    # --------------------------------------------------
    # 3. 汇总 & 写 log
    # --------------------------------------------------
    log_file = Path(project_root) / "kfold_log.txt"
    with open(log_file, "w") as f:
        f.write("YOLOv11 K-Fold Summary\n")
        f.write("="*40 + "\n")
        headers = ["Fold", "mAP50", "mAP50-95", "Precision", "Recall", "F1"]
        f.write("\t".join(headers) + "\n")
        for m in metrics_all:
            f.write("\t".join([f"{m[h]:.4f}" for h in headers]) + "\n")

        # 计算并写平均
        avg = {k: sum(d[k] for d in metrics_all) / len(metrics_all)
               for k in ["map50", "map5095", "precision", "recall", "f1"]}
        f.write("="*40 + "\n")
        f.write("Average:\n")
        for k, v in avg.items():
            f.write(f"{k:<12}: {v:.4f}\n")

    print("\n🎉 K-Fold training finished!")
    print("📊 Summary saved to:", log_file)

# --------------------------------------------------
# 4. 入口
# --------------------------------------------------
if __name__ == "__main__":
    run_kfold()