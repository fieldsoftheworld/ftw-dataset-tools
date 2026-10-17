#!/usr/bin/env bash
#SBATCH --job-name=planet-scenes
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/planet-scenes-%A_%a.log
set -euo pipefail

stage="${1:?Pass search, activate, extract, or quality}"
: "${WORK_DIR:?Set WORK_DIR to an absolute output directory}"
ftwd="${FTWD:-$PWD/.venv/bin/ftwd}"
workers="${WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
if [[ "$stage" == quality ]]; then
    exec "$ftwd" planet-scenes quality --work-dir "$WORK_DIR" --workers "$workers"
fi

case "$stage" in search|activate|extract) ;; *) exit 2 ;; esac
args=("$stage")
if [[ "$stage" == search ]]; then
    args+=("${MANIFEST:?Set MANIFEST to the AOI JSONL file}")
fi
exec "$ftwd" planet-scenes "${args[@]}" --work-dir "$WORK_DIR" \
    --workers "$workers" --shard-id "${SLURM_ARRAY_TASK_ID:-0}" \
    --num-shards "${NUM_SHARDS:-1}"
