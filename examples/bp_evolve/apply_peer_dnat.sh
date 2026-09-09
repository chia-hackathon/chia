#!/bin/bash
# Make peer-to-peer Ray object pulls work on the GCP cbp-ng cluster.
#
# chia addresses every node by a head-relative loopback alias, because those are
# the addresses the head's `ssh -L` tunnels bind. Those aliases only mean
# anything on the head. Ray's PullManager dials them from the workers too, where
# 127.0.0.x resolves to the worker's OWN raylet (it listens on the IPv6
# wildcard, so it accepts on every 127.x). A node then asks itself for an object
# it does not have and the pull hangs silently.
#
# Fix: on each VM, DNAT the peer aliases to the peers' VPC internal addresses,
# and SNAT so the packet does not egress with a 127.0.0.1 source. Each VM
# deliberately skips its own aliases, which must stay on loopback.
#
# Host state, not cluster state: re-run after any VM re-provision.
#
# ADDRESSES ARE DERIVED, NOT HARDCODED. An earlier version pinned
# 10.138.0.37-.40 and the external IPs of one particular provision; the next
# `chia up` landed on .41-.44 and the script applied DNAT rules pointing at
# four addresses that were no longer the cluster. It fails loudly now if the
# instance list does not have exactly four running VMs.
set -euo pipefail

PROJECT=${GCP_PROJECT:-project-65f9e385-584e-47a2-906}
ZONE=${GCP_ZONE:-us-west1-c}
PREFIX=${VM_PREFIX:-chia-bp-evolve-gcp-gcp-bp-}
KEY=${SSH_KEY:-$HOME/.ssh/id_ed25519}
USER=${SSH_USER:-$(id -un)}

# index -> external IP, internal IP.  Alias N+2 (cbp-ng) and N+6 (champsim).
# Sorted by instance name so index i is always gcp_bp:i, matching the order
# chia assigns the loopback aliases in.
mapfile -t rows < <(gcloud compute instances list \
    --project="$PROJECT" --zones="$ZONE" \
    --filter="name~^${PREFIX}[0-9]+$ AND status=RUNNING" \
    --format="csv[no-heading](name,networkInterfaces[0].accessConfigs[0].natIP,networkInterfaces[0].networkIP)" \
    | sort)

if [ "${#rows[@]}" -ne 4 ]; then
    echo "ERROR: expected 4 running ${PREFIX}* instances, found ${#rows[@]}" >&2
    printf '  %s\n' "${rows[@]:-<none>}" >&2
    exit 1
fi

EXT=(); INT=()
for r in "${rows[@]}"; do
    IFS=, read -r _name ext int <<<"$r"
    [ -n "$ext" ] && [ -n "$int" ] || { echo "ERROR: missing IP in row: $r" >&2; exit 1; }
    EXT+=("$ext"); INT+=("$int")
done

# The /20 the VMs actually sit in, rather than a pinned 10.138.0.0/20.
SUBNET="$(IFS=. read -r a b _ _ <<<"${INT[0]}"; echo "$a.$b.0.0/20")"

echo "peers: ${INT[*]}  (subnet $SUBNET)"

for i in 0 1 2 3; do
    cmds=""
    for j in 0 1 2 3; do
        [ "$i" = "$j" ] && continue          # own aliases stay on loopback
        for base in 2 6; do                  # 2..5 = cbp-ng, 6..9 = champsim
            alias="127.0.0.$((base + j))"
            cmds="$cmds sudo iptables -w 15 -t nat -C OUTPUT -d $alias/32 -p tcp -j DNAT --to-destination ${INT[$j]} 2>/dev/null || sudo iptables -w 15 -t nat -A OUTPUT -d $alias/32 -p tcp -j DNAT --to-destination ${INT[$j]};"
        done
    done
    cmds="$cmds sudo iptables -w 15 -t nat -C POSTROUTING -s 127.0.0.0/8 -d $SUBNET -j SNAT --to-source ${INT[$i]} 2>/dev/null || sudo iptables -w 15 -t nat -A POSTROUTING -s 127.0.0.0/8 -d $SUBNET -j SNAT --to-source ${INT[$i]};"
    cmds="$cmds echo \"VM$i (${INT[$i]}): \$(sudo iptables -t nat -S OUTPUT | grep -c '127\\.0\\.0\\.[2-9]/32') dnat, \$(sudo iptables -t nat -S POSTROUTING | grep -c 'SNAT --to-source ') snat\""
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 \
        -i "$KEY" "$USER@${EXT[$i]}" "$cmds" 2>&1 | grep -v "Warning: Permanently"
done
