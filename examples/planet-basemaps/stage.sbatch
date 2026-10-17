#!/usr/bin/env bash
#SBATCH --job-name=planet-basemaps
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/planet-basemaps-%A_%a.log
set -euo pipefail

stage="${1:?Pass plan, download, or assemble}"
: "${WORK_DIR:?Set WORK_DIR to an absolute output directory}"
case "$stage" in plan|download|assemble) ;; *) exit 2 ;; esac
ftwd="${FTWD:-$PWD/.venv/bin/ftwd}"
args=("$stage")
if [[ "$stage" == plan ]]; then
    args+=("${MANIFEST:?Set MANIFEST to JSONL containing per-AOI mosaic names}")
fi
exec "$ftwd" planet-basemaps "${args[@]}" --work-dir "$WORK_DIR" \
    --workers "${WORKERS:-${SLURM_CPUS_PER_TASK:-4}}" \
    --shard-id "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "${NUM_SHARDS:-1}"
