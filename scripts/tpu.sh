#!/usr/bin/env bash
# Minimal spot TPU VM helper.
#   scripts/tpu.sh up              create spot VM (tries zones in order)
#   scripts/tpu.sh sync            rsync repo to VM and install env
#   scripts/tpu.sh run "<cmd>"     run a command in the repo on the VM
#   scripts/tpu.sh fetch           copy results/ back
#   scripts/tpu.sh down            delete the VM (stops billing)
set -euo pipefail

NAME="${TPU_NAME:-quokka}"
ACCEL="${TPU_ACCEL:-v6e-1}"
VERSION="${TPU_VERSION:-v2-alpha-tpuv6e}"
# Every zone that offered v6e-1 on 2026-11-15. Spot quota is per zone and some
# are 0 (us-east4-a and us-east5-c were); those fail fast and are reported.
ZONES="${TPU_ZONES:-us-east1-d us-east4-a us-east5-a us-east5-b us-east5-c us-central1-a
  us-central1-b us-central1-c us-south1-a us-south1-c us-west1-c europe-west4-a
  asia-northeast1-b asia-southeast1-b southamerica-east1-c}"
# How long to wait for any zone to reach READY before giving up and cleaning up.
WAIT_S="${TPU_WAIT_S:-900}"
ZONE_FILE=".tpu_zone"
REPO_DIR="quokka"

zone() { cat "$ZONE_FILE"; }
ssh_vm() { gcloud compute tpus tpu-vm ssh "$NAME" --zone "$(zone)" --command "$1"; }
state() {
  gcloud compute tpus tpu-vm describe "$NAME" --zone "$1" --format="value(state)" 2>/dev/null || true
}
# Deletes return at once: a VM stuck in CREATING otherwise blocks until GCP aborts it.
delete_async() { gcloud compute tpus tpu-vm delete "$NAME" --zone "$1" --quiet --async >/dev/null 2>&1 || true; }

case "${1:-}" in
  up)
    # A spot create with no capacity sits in CREATING instead of failing, so
    # trying zones one at a time can stall for a long time. Request every zone
    # at once, keep the first VM that reaches READY, and delete the others.
    pending=()
    for z in $ZONES; do
      out="$(gcloud compute tpus tpu-vm create "$NAME" --zone "$z" --accelerator-type "$ACCEL" \
               --version "$VERSION" --spot --async --quiet 2>&1)" && pending+=("$z") ||
        echo "  $z: $(grep -m1 -oE "Quota limit '[A-Za-z]+' has been exceeded[^\"]*|\"message\": \"[^\"]*" <<<"$out")"
    done
    echo "requested in: ${pending[*]:-none}"
    winner=""
    for ((t = 0; t < WAIT_S && ${#pending[@]} > 0; t += 15)); do
      for z in "${pending[@]}"; do
        [[ "$(state "$z")" == READY ]] && { winner="$z"; break 2; }
      done
      sleep 15
    done
    for z in "${pending[@]}"; do [[ "$z" == "$winner" ]] || delete_async "$z"; done
    if [[ -n "$winner" ]]; then echo "$winner" > "$ZONE_FILE"; echo "up in $winner"; exit 0; fi
    echo "no VM reached READY in ${WAIT_S}s; deleting all requests" >&2; exit 1 ;;
  sync)
    gcloud compute tpus tpu-vm scp --zone "$(zone)" --recurse \
      pyproject.toml uv.lock LICENSE quokka bench tests "$NAME:~/$REPO_DIR/"
    ssh_vm "command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh;
            cd ~/$REPO_DIR && ~/.local/bin/uv sync -q --extra tpu && ~/.local/bin/uv pip install -q 'jax[tpu]==0.11.2'" ;;
  run)
    ssh_vm "cd ~/$REPO_DIR && ~/.local/bin/uv run --no-sync ${2:?command}" ;;
  fetch)
    mkdir -p results
    gcloud compute tpus tpu-vm scp --zone "$(zone)" --recurse "$NAME:~/$REPO_DIR/results/*" results/ ;;
  down)
    gcloud compute tpus tpu-vm delete "$NAME" --zone "$(zone)" --quiet && rm -f "$ZONE_FILE" ;;
  *)
    sed -n '2,8p' "$0"; exit 1 ;;
esac
