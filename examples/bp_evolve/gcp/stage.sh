#!/bin/bash
# One-time: put everything the GCP workers need into a GCS bucket, so that
# `chia up` is a fast in-region copy rather than a long upload from your house.
#
# Three objects go in:
#
#   gs://$BUCKET/cbp_traces/               168 .gz,  9.5 GB   Tier 0
#   gs://$BUCKET/champsim_traces/          168 .gz,   10 GB   Tier 1
#   gs://$BUCKET/build_context.tar.gz      ~6 MB              image builds
#
# WHY A THROWAWAY VM RATHER THAN `gcloud storage cp` FROM HERE.
#   The traces are already on this machine, so uploading them looks obvious.
#   It is 19.5 GB up a domestic uplink. cbp_official/download.log records gdown
#   pulling the source tarball at 2.05 MB/s sustained -- 1 h 11 m for 8.85 GB
#   DOWN -- and home links are asymmetric, so up is the slower direction. A GCP
#   VM fetches the same tarball over Google's network and writes into a bucket
#   in the region the workers run in, where the write is free and their six
#   reads are free. The VM costs about thirty cents and is deleted at the end.
#
#   STAGE_FROM=local uploads the local directories instead. Slower, but it is
#   byte-for-byte what the local sweep ran on -- use it if you have modified
#   the set, or if the Drive link ever rots.
#
# WHY champsim_traces IS BUILT AND NOT COPIED.
#   It is derived from cbp_traces by tools/cbp2champsim.cpp. Tiers 0 and 1 must
#   see the same branches or the port self-check measures the workload rather
#   than the predictor, so the conversion is pinned to this repo's converter and
#   run with no instruction cap, exactly as convert_all.sh does it.
#
# WHY build_context.tar.gz.
#   The workers build their own images (see cluster_gcp.yaml), and the build
#   context is this repo plus the cbp-ng checkout. Neither can be cloned on the
#   VM: examples/bp_evolve/ is untracked, ChampSimDockerfile is locally
#   modified, and cbp-ng is a private checkout whose traces are not
#   redistributable. Shipping it through the bucket rather than through
#   file_mounts also makes worker setup independent of when CHIA runs its rsync,
#   which is not ordered against gcp_nodes setup_commands.
#
# Storage is about $0.39/month for the lot. Leave it staged; re-staging costs
# more than keeping it.
#
# MIND THE GLOBAL vCPU QUOTA. This project's CPUS_ALL_REGIONS is 32 and
# cluster_gcp.yaml uses all 32, so the e2-standard-8 below must be gone before
# `chia up`. It is deleted at the end of a successful run; if this script fails
# part way, `gcloud compute instances delete bp-evolve-stage --zone=<z>` by
# hand, and check `gcloud compute instances list` before bringing the cluster
# up.
#
# Usage:
#   ./stage.sh                        # everything, fetching on a throwaway VM
#   STAGE_FROM=local ./stage.sh       # everything, uploading from here
#   ONLY=context ./stage.sh           # just refresh the 6 MB build context
set -euo pipefail

PROJECT=${BPE_GCP_PROJECT:-project-65f9e385-584e-47a2-906}
REGION=${BPE_GCP_REGION:-us-west1}
ZONE=${BPE_GCP_ZONE:-us-west1-a}
BUCKET=${BPE_GCS_BUCKET:-bp-evolve-traces-${PROJECT##*-}}
STAGE_FROM=${STAGE_FROM:-vm}
ONLY=${ONLY:-all}
VM=bp-evolve-stage
DRIVE_ID=${BPE_DRIVE_ID:-1kLKn_iKVBP-YxRpC4WiCy-ca-agU0BFG}   # cbp_official/download.log
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
CBP_NG=${BPE_CBP_NG_SRC:-$HOME/cbp-ng}

say() { echo "[stage] $*"; }

say "project=$PROJECT region=$REGION bucket=gs://$BUCKET only=$ONLY from=$STAGE_FROM"

if ! gcloud storage buckets describe "gs://$BUCKET" --project="$PROJECT" >/dev/null 2>&1; then
    say "creating bucket in $REGION (same region as the workers: their reads are free)"
    gcloud storage buckets create "gs://$BUCKET" \
        --project="$PROJECT" --location="$REGION" --uniform-bucket-level-access
fi

# The VMs authenticate to this bucket as the default compute service account,
# not as you. With uniform bucket-level access there is no ACL fallback, and
# project-level Editor does NOT imply object access, so without this binding
# every VM -- the staging one below and all four cluster workers, which share
# this service account -- fails its first read with
#
#   HTTPError 403: ... does not have storage.objects.get access
#
# objectAdmin rather than objectViewer because the staging VM also writes: it
# rsyncs both trace directories up. Scoped to this bucket, not the project.
SA=${BPE_GCP_SA:-985927362397-compute@developer.gserviceaccount.com}
say "granting objectAdmin on gs://$BUCKET to $SA"
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
    --project="$PROJECT" \
    --member="serviceAccount:$SA" \
    --role=roles/storage.objectAdmin >/dev/null

# ---------------------------------------------------------------------------
# The build context. Small, and the only part that goes stale, so it is its own
# step and ONLY=context re-runs just this.
# ---------------------------------------------------------------------------
stage_context() {
    [ -d "$CBP_NG" ] || { echo "ERROR: no cbp-ng checkout at $CBP_NG" >&2; exit 1; }
    tmp=$(mktemp -d)
    say "packing build context (chia repo + cbp-ng)"
    tar czf "$tmp/build_context.tar.gz" \
        --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
        --exclude='examples/reveng/out' \
        -C "$(dirname "$REPO")" "$(basename "$REPO")" \
        -C "$(dirname "$CBP_NG")" "$(basename "$CBP_NG")"
    say "  $(du -h "$tmp/build_context.tar.gz" | cut -f1)"
    gcloud storage cp "$tmp/build_context.tar.gz" "gs://$BUCKET/build_context.tar.gz" \
        --project="$PROJECT"
    rm -rf "$tmp"
}

stage_context
if [ "$ONLY" = context ]; then say "done (context only)"; exit 0; fi

# ---------------------------------------------------------------------------
# The traces.
# ---------------------------------------------------------------------------
if [ "$STAGE_FROM" = local ]; then
    DATA=${BPE_DATA_DIR:-$HOME/bp_evolve_data}
    for d in cbp_traces champsim_traces; do
        n=$(ls "$DATA/$d"/*.gz 2>/dev/null | wc -l)
        [ "$n" -eq 168 ] || { echo "ERROR: $DATA/$d has $n traces, expected 168" >&2; exit 1; }
        say "uploading $d ($n traces) -- the slow path, see the header"
        gcloud storage rsync -r "$DATA/$d" "gs://$BUCKET/$d" --project="$PROJECT"
    done
    say "done"
    exit 0
fi

# e2-standard-8: a download, a gunzip and 168 short conversions, all I/O bound.
# 200 GB because the tarball (8.85), the extraction (9.5) and the conversion
# output (10) all coexist on the disk before the upload.
if ! gcloud compute instances describe "$VM" --zone="$ZONE" --project="$PROJECT" >/dev/null 2>&1; then
    say "creating $VM ($ZONE)"
    gcloud compute instances create "$VM" \
        --project="$PROJECT" --zone="$ZONE" \
        --machine-type=e2-standard-8 \
        --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
        --boot-disk-size=200GB --boot-disk-type=pd-balanced \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --labels=purpose=bp-evolve-staging
    say "waiting for ssh"
    for _ in $(seq 1 40); do
        gcloud compute ssh "$VM" --zone="$ZONE" --project="$PROJECT" \
            --command=true --quiet >/dev/null 2>&1 && break
        sleep 15
    done
fi

say "fetch + extract + convert + upload (expect 30-50 min)"
gcloud compute ssh "$VM" --zone="$ZONE" --project="$PROJECT" --quiet --command "
set -euo pipefail
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip g++ zlib1g-dev >/dev/null
pip3 install --quiet --upgrade gdown

mkdir -p ~/stage/cbp_traces ~/stage/champsim_traces
cd ~/stage

if [ ! -f traces.tar.gz ]; then
    # --continue so a throttled or dropped Drive transfer resumes rather than
    # restarting 8.85 GB from zero.
    # gdown >=5 removed --id; the ID is positional, and the /uc?id= URL form
    # is what makes it take the confirm-token path for a file this large.
    python3 -m gdown --continue -O traces.tar.gz \
        \"https://drive.google.com/uc?id=$DRIVE_ID\"
fi
tar xzf traces.tar.gz
SRC=\$(find . -type d -name cbp-ng_training_traces | head -1)
n=\$(ls \"\$SRC\"/*.gz | wc -l)
echo \"extracted \$n traces from \$SRC\"
[ \"\$n\" -eq 168 ] || { echo 'ERROR: expected 168 traces' >&2; exit 1; }
cp \"\$SRC\"/*.gz cbp_traces/

# The converter comes out of the staged build context, so it is the same source
# the local set was built with rather than whatever happens to be on the VM.
gcloud storage cp gs://$BUCKET/build_context.tar.gz .
tar xzf build_context.tar.gz
g++ -O2 -std=c++17 -o cbp2champsim chia/examples/bp_evolve/tools/cbp2champsim.cpp -lz

conv() {
    b=\$(basename \"\$1\" .gz)
    t=~/stage/champsim_traces/\$b.champsimtrace.gz.partial
    k=\$(~/stage/cbp2champsim \"\$1\" \"\$t\" 999999999 2>/dev/null | awk -F, '\$1==\"instructions\"{print \$2}')
    if [ -n \"\$k\" ] && [ \"\$k\" -gt 0 ] 2>/dev/null; then
        mv \"\$t\" ~/stage/champsim_traces/\$b.champsimtrace.gz; echo \"ok \$b \$k\"
    else
        rm -f \"\$t\"; echo \"FAIL \$b\"; fi
}
export -f conv
ls cbp_traces/*.gz | xargs -P 8 -I{} bash -c 'conv \"\$1\"' _ {}

c=\$(ls champsim_traces/*.champsimtrace.gz | wc -l)
echo \"converted \$c\"
[ \"\$c\" -eq 168 ] || { echo 'ERROR: conversion incomplete' >&2; exit 1; }

gcloud storage rsync -r cbp_traces      gs://$BUCKET/cbp_traces
gcloud storage rsync -r champsim_traces gs://$BUCKET/champsim_traces
echo staged
"

say "verifying the bucket"
for d in cbp_traces champsim_traces; do
    n=$(gcloud storage ls "gs://$BUCKET/$d/**" --project="$PROJECT" | wc -l)
    say "  gs://$BUCKET/$d: $n objects"
    [ "$n" -eq 168 ] || { echo "ERROR: expected 168 in $d" >&2; exit 1; }
done

say "deleting $VM (the bucket is what we wanted; the VM bills by the second)"
gcloud compute instances delete "$VM" --zone="$ZONE" --project="$PROJECT" --quiet

say "done -- export BPE_GCS_BUCKET=$BUCKET before \`chia up\`"
