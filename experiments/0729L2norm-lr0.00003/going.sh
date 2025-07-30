#!/bin/bash
# wait_and_run.sh
# 当 T4 显存连续 5 秒低于 1 GiB 后，执行 python 01train.py 并退出

GPU_INDEX=$(nvidia-smi --query-gpu=index,name --format=csv,noheader,nounits | awk -F',' 'tolower($2) ~ /t4/ {print $1; exit}')
[[ -z "$GPU_INDEX" ]] && { echo "未找到 Tesla T4 显卡"; exit 1; }

THRESHOLD=1024   # 1 GiB
COUNT=0
NEED=5

while true; do
    USED=$(nvidia-smi --id="$GPU_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if (( USED < THRESHOLD )); then
        ((COUNT++))
        echo "显存 ${USED}MiB < 1 GiB，已连续 ${COUNT} 秒"
    else
        COUNT=0
    fi

    if (( COUNT >= NEED )); then
        echo "条件满足，启动训练..."
        exec python 01train.py   # 替换当前进程，执行完自动退出
    fi

    sleep 1
done