#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [--force] /path/to/OpenGait" >&2
}

force=0
if [[ "${1:-}" == "--force" ]]; then
  force=1
  shift
fi
if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_root="$(cd "$1" && pwd)"

if [[ ! -f "$target_root/opengait/main.py" || ! -d "$target_root/opengait/modeling/models" ]]; then
  echo "Not an OpenGait checkout: $target_root" >&2
  exit 2
fi

existing=()
while IFS= read -r source_file; do
  relative="${source_file#"$source_root/"}"
  if [[ -e "$target_root/$relative" ]]; then
    existing+=("$relative")
  fi
done < <(find "$source_root/opengait" -type f | sort)

for source_file in "$source_root/configs/star_cvi_ag_vpreid.yaml" \
                   "$source_root/configs/baselines/deepgaitv2_ag_vpreid.yaml"; do
  relative="${source_file#"$source_root/"}"
  if [[ -e "$target_root/$relative" ]]; then
    existing+=("$relative")
  fi
done

if [[ ${#existing[@]} -gt 0 && $force -ne 1 ]]; then
  echo "STAR-CVI files already exist in the target. Re-run with --force to replace them:" >&2
  printf '  %s\n' "${existing[@]}" >&2
  exit 3
fi

cp -R "$source_root/opengait/." "$target_root/opengait/"
mkdir -p "$target_root/configs/baselines" "$target_root/datasets/AG-VPReID"
cp "$source_root/configs/star_cvi_ag_vpreid.yaml" "$target_root/configs/"
cp "$source_root/configs/baselines/deepgaitv2_ag_vpreid.yaml" "$target_root/configs/baselines/"
cp "$source_root/datasets/AG-VPReID/README.md" "$target_root/datasets/AG-VPReID/README.md"

echo "STAR-CVI extension installed into: $target_root"
echo "Next: prepare AG-VPReID, edit dataset_root/class_num, and start training from the OpenGait root."

