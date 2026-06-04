#!/bin/bash
# Sequential scaling sweep: CRNN+CTC and ViT-AR across {1k, 5k, 20k, 50k}.
# 100k already exists for both models, skipped here.
# Logs land in logs/sweep_<model>_<size>.log.
set -u

cd /workspace
mkdir -p logs/sweep

run_one() {
    local model="$1" size="$2" batch="$3" workers="$4" lr="$5" base_warmup="$6"
    local steps_per_epoch=$(( size / batch ))
    local total_steps=$(( steps_per_epoch * 30 ))
    local eval_every=$(( steps_per_epoch > 0 ? steps_per_epoch : 1 ))
    # warmup: 10% of total, capped at base_warmup, min 100 (or all steps if total < 100)
    local warmup=$(( total_steps / 10 ))
    [ "$warmup" -gt "$base_warmup" ] && warmup="$base_warmup"
    [ "$warmup" -lt 100 ] && [ "$total_steps" -gt 100 ] && warmup=100
    [ "$warmup" -ge "$total_steps" ] && warmup=$(( total_steps / 4 ))

    local log="logs/sweep/${model}_${size}.log"
    echo "[$(date +%H:%M:%S)] === $model size=$size  steps/epoch=$steps_per_epoch  total=$total_steps  eval_every=$eval_every  warmup=$warmup ==="
    python -u -m src.model.train \
        model="$model" \
        data.train_size="$size" data.batch_size="$batch" data.num_workers="$workers" \
        train.epochs=30 train.eval_every_steps="$eval_every" train.log_every_steps=20 \
        train.gen_max_length=64 train.mixed_precision=bf16 \
        optim.optimizer.lr="$lr" optim.scheduler.num_warmup_steps="$warmup" \
        logging=tensorboard \
        > "$log" 2>&1
    local rc=$?
    echo "[$(date +%H:%M:%S)] === $model size=$size finished (rc=$rc) ==="
    return 0  # never stop the sweep on a single failure
}

# ViT-only sweep, workers=8 (after restart sweep — workers=4 path no longer
# matches v1's fast trajectory but is also broken in a different way; workers=8
# converges to ~SER 0.026 like the June-2 vit-20260602-122323 run).
# CRNN runs collected on pre-revert main are preserved in /workspace/runs/.
run_one vit   1000   32  8 5e-4 2000
run_one vit   5000   32  8 5e-4 2000
run_one vit   20000  32  8 5e-4 2000
run_one vit   50000  32  8 5e-4 2000

echo "[$(date +%H:%M:%S)] === sweep complete ==="
