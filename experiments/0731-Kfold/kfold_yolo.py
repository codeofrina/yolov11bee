import os
import shutil
import random
from pathlib import Path
from collections import defaultdict, Counter

# ====== 用户只需改这里 ======
TRAIN_ROOT = "/workspace/dataset/custom-v1/train"   # 绝对路径
# ===========================

IMG_DIR   = Path(TRAIN_ROOT) / "images"
LBL_DIR   = Path(TRAIN_ROOT) / "labels"
VAL_IMG   = Path("/workspace/dataset/custom-v1/val/images")
VAL_LBL   = Path("/workspace/dataset/custom-v1/val/labels")

# 确保目录存在
for p in (VAL_IMG, VAL_LBL):
    p.mkdir(exist_ok=True)

# 固定的随机种子，保证重现
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ---------------- 工具函数 ----------------
def parse_label(lbl_path):
    """返回该标注的类别集合 {0,1}，空文件返回 set()"""
    if not lbl_path.exists():
        return set()
    with open(lbl_path) as f:
        lines = f.read().strip().splitlines()
    classes = {int(line.split()[0]) for line in lines if line.strip()}
    return classes

def bucket_key(img_stem):
    """
    根据文件名前缀和类别情况生成 8 个桶的 key
    格式: (prefix, label_type)
    label_type: 0_only, 1_only, both, none
    """
    prefix = "0721" if img_stem.startswith("0721-") else "custom"
    lbl_path = LBL_DIR / f"{img_stem}.txt"
    cats = parse_label(lbl_path)
    if cats == {0}:
        lt = "0_only"
    elif cats == {1}:
        lt = "1_only"
    elif cats == {0, 1}:
        lt = "both"
    else:
        lt = "none"
    return (prefix, lt)

def prepare_fold(k: int, n: int = 5):
    """
    把第 k 折（0-based）作为验证集，其余为训练集。
    k 必须 0 <= k < n
    """
    if not (0 <= k < n):
        raise ValueError(f"k must be in [0, {n-1}]")

    # 1. 扫描所有图片
    imgs = [p.stem for p in IMG_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    # 2. 分桶
    buckets = defaultdict(list)
    for stem in imgs:
        buckets[bucket_key(stem)].append(stem)

    # 3. 每桶内部固定随机顺序
    for b in buckets.values():
        random.shuffle(b)

    # 4. 计算每个桶内第 k 折的切片
    val_stems = []
    for b in buckets.values():
        size = len(b)
        fold_size = max(1, size // n)
        start = k * fold_size
        end   = (k + 1) * fold_size if k != n - 1 else size
        val_stems.extend(b[start:end])

    # 5. 清空前一次 val 目录
    for p in VAL_IMG.iterdir():
        p.unlink()
    for p in VAL_LBL.iterdir():
        p.unlink()

    # 6. 移动文件
    def move_to_val(stem):
        # 图片
        src_img = next(IMG_DIR.glob(f"{stem}.*"))   # 支持任意后缀
        dst_img = VAL_IMG / src_img.name
        shutil.move(str(src_img), str(dst_img))
        # 标签
        src_lbl = LBL_DIR / f"{stem}.txt"
        dst_lbl = VAL_LBL / f"{stem}.txt"
        if src_lbl.exists():
            shutil.move(str(src_lbl), str(dst_lbl))

    def move_to_train(stem):
        # 图片
        src_img = next(VAL_IMG.glob(f"{stem}.*"), None)
        if src_img:
            dst_img = IMG_DIR / src_img.name
            shutil.move(str(src_img), str(dst_img))
        # 标签
        src_lbl = VAL_LBL / f"{stem}.txt"
        if src_lbl.exists():
            dst_lbl = LBL_DIR / f"{stem}.txt"
            shutil.move(str(src_lbl), str(dst_lbl))

    # 先把所有可能存在于 val 的文件搬回 train
    for stem in [p.stem for p in VAL_IMG.glob("*")]:
        move_to_train(stem)

    # 再把当前折的验证集搬过去
    for stem in val_stems:
        move_to_val(stem)

    print(f"Fold {k+1}/{n} ready: 验证集 {len(val_stems)} 张，其余为训练集")


# ---------------- 示例调用 ----------------
if __name__ == "__main__":
    # 演示：直接运行脚本会把第 0 折作为验证集
    prepare_fold(k=0, n=5)