#!/usr/bin/env bash
# 一鍵啟動 AETHER cluster：確保 ssh-agent 起著、金鑰已加入，再跑 chia up，
# log 導到 out/chia-up-<timestamp>.log。
#
# 用法：
#   bash cluster/up.sh [yaml]   # default yaml: cluster/cluster.yaml
#
# 背景：build node 的 docker run_options 用 -v ${SSH_AUTH_SOCK:-/dev/null}:/ssh-agent
# 把 ssh-agent socket 掛進 chisel-build container 給 git@github.com:ucb-bar/chipyard.git
# 這類 ssh remote 用。這個值是在「chia 透過 ssh -A 連到 head node後，遠端那個
# bash --login shell」裡展開的，所以本機（也就是這裡）呼叫 chia up 時的 shell
# 必須已經有 ssh-agent 在跑、且金鑰已 ssh-add，agent forwarding 才有東西可轉發。
# 若沒有，遠端 SSH_AUTH_SOCK 會是空的，此腳本會退回 /dev/null（不會讓 docker run 崩潰），
# 但容器內走 ssh 的 git 操作就會失敗——所以這裡確保金鑰一定先加進 agent。

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
AETHER_DIR="$(dirname -- "$SCRIPT_DIR")"
YAML="${1:-$AETHER_DIR/cluster/cluster.yaml}"

# --- required env vars -----------------------------------------------------
# cluster.yaml is written against chia's ${VAR} substitution, and chia passes
# an UNSET ${VAR} through literally instead of failing. Check them here so a
# missing value is a loud error rather than a container with a "${...}" path.
missing=()
for v in AETHER_HEAD_IP AETHER_RAY_TMPDIR AETHER_WORKDIR; do
    [ -n "${!v:-}" ] || missing+=("$v")
done
if [ ${#missing[@]} -ne 0 ]; then
    echo "[up.sh] missing required env var(s): ${missing[*]}" >&2
    echo "[up.sh] see examples/aether/requirements.md; e.g." >&2
    echo "        export AETHER_HEAD_IP=\$(hostname -I | awk '{print \$1}')" >&2
    echo "        export AETHER_RAY_TMPDIR=/big/disk/aether_ray" >&2
    echo "        export AETHER_WORKDIR=$AETHER_DIR/out" >&2
    exit 2
fi

# Host-side dirs the cluster.yaml bind-mounts must exist before docker run.
mkdir -p "$AETHER_WORKDIR"/work/{kernel,sim,build} \
         "$AETHER_RAY_TMPDIR" \
         "$AETHER_RAY_TMPDIR"_ct_{llm,build,cosim,riscv}

# --- conda env ---
CONDA_SH="/usr/local/anaconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate chia_env
elif [ -x "$HOME/.conda/envs/chia_env/bin/chia" ]; then
    export PATH="$HOME/.conda/envs/chia_env/bin:$PATH"
else
    echo "[up.sh] 找不到 conda.sh 也找不到 ~/.conda/envs/chia_env，請確認 chia_env 環境位置" >&2
fi

# --- ssh-agent：沒有就啟動，金鑰沒加就加 ---
if [ -z "${SSH_AUTH_SOCK:-}" ] || ! ssh-add -l >/dev/null 2>&1; then
    if [ -z "${SSH_AUTH_SOCK:-}" ]; then
        echo "[up.sh] 沒有偵測到 ssh-agent，啟動一個新的"
        eval "$(ssh-agent -s)"
    fi
    if ! ssh-add -l >/dev/null 2>&1; then
        echo "[up.sh] ssh-agent 內沒有金鑰，ssh-add ~/.ssh/id_rsa"
        ssh-add ~/.ssh/id_rsa
    fi
fi
ssh-add -l || true

# --- chia up ---
mkdir -p "$AETHER_DIR/out"
TS="$(date +%Y%m%d-%H%M%S)"
LOG="$AETHER_DIR/out/chia-up-${TS}.log"

echo "[up.sh] chia up -y $YAML"
echo "[up.sh] log -> $LOG"
chia up -y "$YAML" 2>&1 | tee "$LOG"
