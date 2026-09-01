#!/usr/bin/env bash
set -Eeuo pipefail

# Future server usage:
#   DATA_ROOT=/absolute/path/to/PointCT/data \
#     nohup bash run_m4_osseous_strength_selection_clean10.sh > m4_selection.nohup.log 2>&1 &
# Contract-only local audit (does not require data or a GPU and writes no outputs):
#   DRY_RUN=1 bash run_m4_osseous_strength_selection_clean10.sh

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

print_command() {
  local label="$1"
  shift
  printf '%s' "${label}"
  printf ' %q' "$@"
  printf '\n'
}

verify_canonical_json_hash() {
  local manifest="$1"
  local expected_hash="$2"
  local embedded_hash_field="$3"

  "${PYTHON_BIN}" - "${manifest}" "${expected_hash}" "${embedded_hash_field}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key!r}')
        result[key] = value
    return result


path = Path(sys.argv[1])
expected = sys.argv[2]
hash_field = sys.argv[3]
with path.open('r', encoding='utf-8') as handle:
    payload = json.load(handle, object_pairs_hook=reject_duplicate_keys)
embedded = payload.pop(hash_field, None)
if embedded != expected:
    raise SystemExit(
        f'{path}: embedded {hash_field} {embedded!r} does not equal frozen {expected!r}'
    )
canonical = json.dumps(
    payload,
    sort_keys=True,
    separators=(',', ':'),
    ensure_ascii=True,
    allow_nan=False,
).encode('utf-8')
actual = hashlib.sha256(canonical).hexdigest()
if actual != expected:
    raise SystemExit(
        f'{path}: canonical SHA-256 {actual} does not equal frozen {expected}'
    )
PY
}

readonly DRY_RUN="${DRY_RUN:-0}"
case "${DRY_RUN}" in
  0|1) ;;
  *) die 'DRY_RUN must be exactly 0 or 1.' ;;
esac

readonly PYTHON_BIN="${PYTHON_BIN:-python}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 \
  || die "Python executable not found: ${PYTHON_BIN}"
export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SCRIPT_DIR
readonly REPO_ROOT="${SCRIPT_DIR}"
readonly EXPERIMENT_DIR="${REPO_ROOT}/experiments/geotransformer.pointct.baseline_v1"
readonly SELECTION_PROTOCOL="${EXPERIMENT_DIR}/protocols/m4_osseous_strength_selection_clean10_v1.json"
readonly SELECTION_PROTOCOL_SIDECAR="${EXPERIMENT_DIR}/protocols/m4_osseous_strength_selection_clean10_v1.sha256"
readonly TRAINING_PROTOCOL="${EXPERIMENT_DIR}/protocols/m3_6b_5fold_clean10_v2.json"
readonly TRAIN_ENTRYPOINT="${EXPERIMENT_DIR}/train_m3_defect.py"
readonly VALIDATION_PRODUCER="${EXPERIMENT_DIR}/evaluate_m4_osseous_strength_validation.py"
readonly OUTPUT_ROOT="${REPO_ROOT}/checkpoints/m4_osseous_strength_selection_v1"

readonly EXPECTED_SELECTION_SHA256='a03a3cd030830c58844d10163a34d10a7db635181a2da5ed1681d09aa802ca38'
readonly EXPECTED_TRAINING_PROTOCOL_SHA256='34866ebc5c7e3c7b18ecb1c4010217d8b2de9b64fae0d7406dabbce2d86d4a3c'
readonly EXPECTED_BRANCH='m4_osseous_strength_selection_v1'
readonly EXPECTED_BASE_COMMIT='d6a10cf092089426307b5f93766b8c5a5a2636f9'

for required_file in \
  "${SELECTION_PROTOCOL}" \
  "${SELECTION_PROTOCOL_SIDECAR}" \
  "${TRAINING_PROTOCOL}" \
  "${TRAIN_ENTRYPOINT}" \
  "${VALIDATION_PRODUCER}"; do
  [[ -f "${required_file}" ]] || die "Required file is missing: ${required_file}"
done

command -v git >/dev/null 2>&1 || die 'Git is required to verify the frozen code base.'
actual_branch="$(git -C "${REPO_ROOT}" branch --show-current)"
actual_commit="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
[[ "${actual_branch}" == "${EXPECTED_BRANCH}" ]] \
  || die "Branch differs from frozen protocol: expected=${EXPECTED_BRANCH}, actual=${actual_branch}"
[[ "${actual_commit}" == "${EXPECTED_BASE_COMMIT}" ]] \
  || die "HEAD differs from frozen protocol: expected=${EXPECTED_BASE_COMMIT}, actual=${actual_commit}"
git -C "${REPO_ROOT}" diff --quiet --exit-code -- \
  || die 'Tracked worktree differs from the frozen base commit.'
git -C "${REPO_ROOT}" diff --cached --quiet --exit-code -- \
  || die 'Staged tracked content differs from the frozen base commit.'

verify_canonical_json_hash \
  "${SELECTION_PROTOCOL}" \
  "${EXPECTED_SELECTION_SHA256}" \
  'protocol_sha256'
verify_canonical_json_hash \
  "${TRAINING_PROTOCOL}" \
  "${EXPECTED_TRAINING_PROTOCOL_SHA256}" \
  'protocol_hash'

IFS=' ' read -r sidecar_hash _ < "${SELECTION_PROTOCOL_SIDECAR}"
[[ "${sidecar_hash}" == "${EXPECTED_SELECTION_SHA256}" ]] \
  || die "Protocol sidecar does not contain the frozen SHA-256: ${SELECTION_PROTOCOL_SIDECAR}"

# These ordered arrays are the complete frozen grid. Directory arrays are
# positional mappings, not additional candidate definitions.
readonly -a LAMBDAS=('0.25' '0.5' '1.0' '2.0')
readonly -a LAMBDA_DIRS=('lambda_0p25' 'lambda_0p50' 'lambda_1p00' 'lambda_2p00')
readonly -a FOLDS=('Fold1' 'Fold2' 'Fold3' 'Fold4' 'Fold5')
readonly -a FOLD_DIRS=('fold1' 'fold2' 'fold3' 'fold4' 'fold5')

[[ "${#LAMBDAS[@]}" -eq 4 ]] || die 'Internal contract error: expected exactly four lambdas.'
[[ "${#LAMBDAS[@]}" -eq "${#LAMBDA_DIRS[@]}" ]] \
  || die 'Internal contract error: lambda directory mapping is incomplete.'
[[ "${#FOLDS[@]}" -eq 5 ]] || die 'Internal contract error: expected exactly five folds.'
[[ "${#FOLDS[@]}" -eq "${#FOLD_DIRS[@]}" ]] \
  || die 'Internal contract error: fold directory mapping is incomplete.'

if [[ -L "${OUTPUT_ROOT}" ]]; then
  die "Output root must not be a symbolic link: ${OUTPUT_ROOT}"
fi
if [[ -e "${OUTPUT_ROOT}" && ! -d "${OUTPUT_ROOT}" ]]; then
  die "Output root exists but is not a directory: ${OUTPUT_ROOT}"
fi

if [[ "${DRY_RUN}" == '1' ]]; then
  DATA_ROOT_ARG="${DATA_ROOT:-${REPO_ROOT}/__DRY_RUN_DATA_ROOT_NOT_ACCESSED__}"
else
  [[ -n "${DATA_ROOT:-}" ]] \
    || die 'DATA_ROOT must name the formal PointCT data directory when DRY_RUN=0.'
  [[ -d "${DATA_ROOT}" ]] || die "DATA_ROOT is not a directory: ${DATA_ROOT}"
  DATA_ROOT_ARG="$(cd -- "${DATA_ROOT}" && pwd -P)"
fi
readonly DATA_ROOT_ARG

# Audit all 20 fold/candidate contracts before printing or executing any work.
# The producer's contract-audit mode is CPU-only and must not touch fold output.
combination_count=0
for lambda_index in "${!LAMBDAS[@]}"; do
  lambda_oss="${LAMBDAS[${lambda_index}]}"
  lambda_directory="${LAMBDA_DIRS[${lambda_index}]}"
  for fold_index in "${!FOLDS[@]}"; do
    fold_id="${FOLDS[${fold_index}]}"
    fold_directory="${FOLD_DIRS[${fold_index}]}"
    fold_dir="${OUTPUT_ROOT}/${lambda_directory}/${fold_directory}"
    contract_audit_command=(
      "${PYTHON_BIN}"
      "${VALIDATION_PRODUCER}"
      --protocol-manifest "${SELECTION_PROTOCOL}"
      --fold-id "${fold_id}"
      --lambda-oss "${lambda_oss}"
      --fold-dir "${fold_dir}"
      --contract-audit
    )
    CUDA_VISIBLE_DEVICES='' "${contract_audit_command[@]}" >/dev/null
    combination_count=$((combination_count + 1))
  done
done
[[ "${combination_count}" -eq 20 ]] \
  || die "Internal contract error: audited ${combination_count} combinations, expected 20."

if [[ "${DRY_RUN}" == '1' ]]; then
  combination_count=0
  for lambda_index in "${!LAMBDAS[@]}"; do
    lambda_oss="${LAMBDAS[${lambda_index}]}"
    lambda_directory="${LAMBDA_DIRS[${lambda_index}]}"
    for fold_index in "${!FOLDS[@]}"; do
      fold_id="${FOLDS[${fold_index}]}"
      fold_directory="${FOLD_DIRS[${fold_index}]}"
      fold_dir="${OUTPUT_ROOT}/${lambda_directory}/${fold_directory}"
      checkpoint_dir="${fold_dir}/checkpoints"
      checkpoint="${checkpoint_dir}/best_val_loss.pt"
      json_log="${fold_dir}/train.jsonl"

      train_command=(
        "${PYTHON_BIN}"
        "${TRAIN_ENTRYPOINT}"
        --data-root "${DATA_ROOT_ARG}"
        --protocol-manifest "${TRAINING_PROTOCOL}"
        --fold-id "${fold_id}"
        --checkpoint-dir "${checkpoint_dir}"
        --json-log "${json_log}"
        --device cuda
        --epochs 20
        --seed 20260815
        --learning-rate 0.0003
        --weight-decay 0.0001
        --batch-size 1
        --precision fp32
        --temperature 0.1
        --sinkhorn-iterations 20
        --alpha-init 1.0
        --enable-m4-defect-mapping
        --enable-m4-soft-modulation
        --m4-soft-sigma-mm 60
        --m4-soft-strength 2
        --m4-osseous-prior
        --m4-osseous-strength "${lambda_oss}"
      )
      validation_command=(
        "${PYTHON_BIN}"
        "${VALIDATION_PRODUCER}"
        --protocol-manifest "${SELECTION_PROTOCOL}"
        --fold-id "${fold_id}"
        --lambda-oss "${lambda_oss}"
        --fold-dir "${fold_dir}"
        --execute-validation
        --data-root "${DATA_ROOT_ARG}"
        --checkpoint "${checkpoint}"
      )

      combination_count=$((combination_count + 1))
      printf '[DRY-RUN %02d/20] lambda_oss=%s fold=%s\n' \
        "${combination_count}" "${lambda_oss}" "${fold_id}"
      print_command '  TRAIN:' "${train_command[@]}"
      print_command '  VALIDATION:' "${validation_command[@]}"
    done
  done
  [[ "${combination_count}" -eq 20 ]] \
    || die "Internal contract error: printed ${combination_count} combinations, expected 20."
  printf 'DRY_RUN_COMPLETE: protocol verified; 20 train+validation command pairs printed; no outputs written.\n'
  exit 0
fi

# Fail closed across the entire grid before starting the first training job.
# A completion marker is trusted only after the producer revalidates it on CPU.
declare -A COMPLETED_COMBINATIONS=()
for lambda_index in "${!LAMBDAS[@]}"; do
  lambda_oss="${LAMBDAS[${lambda_index}]}"
  lambda_directory="${LAMBDA_DIRS[${lambda_index}]}"
  lambda_dir="${OUTPUT_ROOT}/${lambda_directory}"
  if [[ -L "${lambda_dir}" ]]; then
    die "Lambda output path must not be a symbolic link: ${lambda_dir}"
  fi
  if [[ -e "${lambda_dir}" && ! -d "${lambda_dir}" ]]; then
    die "Lambda output path exists but is not a directory: ${lambda_dir}"
  fi
  for fold_index in "${!FOLDS[@]}"; do
    fold_id="${FOLDS[${fold_index}]}"
    fold_directory="${FOLD_DIRS[${fold_index}]}"
    fold_dir="${lambda_dir}/${fold_directory}"
    checkpoint="${fold_dir}/checkpoints/best_val_loss.pt"
    completion_marker="${fold_dir}/SELECTION_FOLD_COMPLETE.json"
    combination_key="${lambda_oss}|${fold_id}"

    if [[ -L "${fold_dir}" ]]; then
      die "Fold output path must not be a symbolic link: ${fold_dir}"
    fi
    if [[ -e "${fold_dir}" && ! -d "${fold_dir}" ]]; then
      die "Fold output path exists but is not a directory: ${fold_dir}"
    fi
    if [[ -f "${completion_marker}" ]]; then
      verify_command=(
        "${PYTHON_BIN}"
        "${VALIDATION_PRODUCER}"
        --protocol-manifest "${SELECTION_PROTOCOL}"
        --fold-id "${fold_id}"
        --lambda-oss "${lambda_oss}"
        --fold-dir "${fold_dir}"
        --checkpoint "${checkpoint}"
        --verify-completion
      )
      if ! CUDA_VISIBLE_DEVICES='' "${verify_command[@]}"; then
        die "Completion marker failed CPU verification; refusing overwrite: ${completion_marker}"
      fi
      COMPLETED_COMBINATIONS["${combination_key}"]=1
    elif [[ -d "${fold_dir}" ]] \
      && [[ -n "$(find "${fold_dir}" -mindepth 1 -print -quit)" ]]; then
      die "Non-empty incomplete fold directory; refusing overwrite: ${fold_dir}"
    else
      COMPLETED_COMBINATIONS["${combination_key}"]=0
    fi
  done
done

for lambda_index in "${!LAMBDAS[@]}"; do
  lambda_oss="${LAMBDAS[${lambda_index}]}"
  lambda_directory="${LAMBDA_DIRS[${lambda_index}]}"
  for fold_index in "${!FOLDS[@]}"; do
    fold_id="${FOLDS[${fold_index}]}"
    fold_directory="${FOLD_DIRS[${fold_index}]}"
    fold_dir="${OUTPUT_ROOT}/${lambda_directory}/${fold_directory}"
    checkpoint_dir="${fold_dir}/checkpoints"
    validation_dir="${fold_dir}/validation"
    checkpoint="${checkpoint_dir}/best_val_loss.pt"
    json_log="${fold_dir}/train.jsonl"
    train_log="${fold_dir}/train.log"
    combination_key="${lambda_oss}|${fold_id}"

    if [[ "${COMPLETED_COMBINATIONS[${combination_key}]:-0}" == '1' ]]; then
      printf 'SKIP verified completion: lambda_oss=%s fold=%s directory=%s\n' \
        "${lambda_oss}" "${fold_id}" "${fold_dir}"
      continue
    fi

    mkdir -p -- "${checkpoint_dir}" "${validation_dir}"
    train_command=(
      "${PYTHON_BIN}"
      "${TRAIN_ENTRYPOINT}"
      --data-root "${DATA_ROOT_ARG}"
      --protocol-manifest "${TRAINING_PROTOCOL}"
      --fold-id "${fold_id}"
      --checkpoint-dir "${checkpoint_dir}"
      --json-log "${json_log}"
      --device cuda
      --epochs 20
      --seed 20260815
      --learning-rate 0.0003
      --weight-decay 0.0001
      --batch-size 1
      --precision fp32
      --temperature 0.1
      --sinkhorn-iterations 20
      --alpha-init 1.0
      --enable-m4-defect-mapping
      --enable-m4-soft-modulation
      --m4-soft-sigma-mm 60
      --m4-soft-strength 2
      --m4-osseous-prior
      --m4-osseous-strength "${lambda_oss}"
    )
    validation_command=(
      "${PYTHON_BIN}"
      "${VALIDATION_PRODUCER}"
      --protocol-manifest "${SELECTION_PROTOCOL}"
      --fold-id "${fold_id}"
      --lambda-oss "${lambda_oss}"
      --fold-dir "${fold_dir}"
      --execute-validation
      --data-root "${DATA_ROOT_ARG}"
      --checkpoint "${checkpoint}"
    )
    verify_command=(
      "${PYTHON_BIN}"
      "${VALIDATION_PRODUCER}"
      --protocol-manifest "${SELECTION_PROTOCOL}"
      --fold-id "${fold_id}"
      --lambda-oss "${lambda_oss}"
      --fold-dir "${fold_dir}"
      --checkpoint "${checkpoint}"
      --verify-completion
    )

    printf 'START lambda_oss=%s fold=%s\n' "${lambda_oss}" "${fold_id}" \
      | tee "${train_log}"
    print_command 'TRAIN:' "${train_command[@]}" | tee -a "${train_log}"
    "${train_command[@]}" 2>&1 | tee -a "${train_log}"
    print_command 'VALIDATION:' "${validation_command[@]}" | tee -a "${train_log}"
    "${validation_command[@]}" 2>&1 | tee -a "${train_log}"

    CUDA_VISIBLE_DEVICES='' "${verify_command[@]}"
    printf 'COMPLETE and CPU-verified: lambda_oss=%s fold=%s\n' \
      "${lambda_oss}" "${fold_id}" | tee -a "${train_log}"
  done
done

printf 'All 20 train/validation combinations are complete and CPU-verified.\n'
