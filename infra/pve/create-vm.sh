#!/bin/bash
# Run as root on pve, once, after install-nat.sh. Creates VM 1094 "nomos" from the Ubuntu 24.04
# cloud image: no installer; cloud-init sets user vlad (key vlad.pub next to this script, sudo
# without a password), the IP and DNS.
#
# What it is for: the searches and exams of the tissue line (CPU PyTorch, many small processes
# side by side), so the desktop's 16 GB stop being the limit. Nothing depends on it: stop it any
# time (`qm stop 1094`), it does not start by itself (`onboot 0`), it is in no backup job (the
# code is in git and results are copied back).
#
# Sizing, from what the host carried on 2026-11-13:
#   - 20 vCPU `host`, pinned to socket 1 (host cpus 10-19,30-39: the whole half the stopped llm VM
#     used, and no more, so socket 0 stays with production). The host idles at ~16 % CPU and its
#     `pve.cpu` alert warns at 70 %; 20 busy threads of 40 put it at about 66 %, on the edge. It was
#     made with 16 for that reason; the user raised it the same day and accepts the warnings.
#   - 8 GB, no balloon. The host was at 80 % RAM (38 GB available of 187) and its `pve.ram` alert
#     warns at 90 %, clears under 85 %; 8 GB keeps it under 85. A search process takes about half a
#     gigabyte, so this is room for a dozen at once. Raise with `qm set 1094 --memory N` (and the
#     numa0 line) only after freeing RAM on the host.
#   - 40 GB system disk, thin on local-lvm, with discard.
# Network: stb's isolated bridge vmbr1; internet through nomos-nat (by IP + the MAC pinned here);
# no way into any private range, reached only from pve (ssh -J).
set -euo pipefail
cd "$(dirname "$0")"
ID=1094 NAME=nomos IP=10.99.0.70 MAC=BC:24:11:5E:5E:70 MEM=8192
IMG=/var/lib/vz/template/iso/noble-server-cloudimg-amd64.img
if qm status $ID >/dev/null 2>&1; then echo "VM $ID already exists"; exit 1; fi
if ! iptables -L NOMOS_OUT -n >/dev/null 2>&1; then echo "run install-nat.sh first"; exit 1; fi
[ -f $IMG ] || wget -q -O $IMG https://cloud-images.ubuntu.com/noble/current/noble-server-cloudimg-amd64.img

qm create $ID --name $NAME --ostype l26 --machine q35 \
    --cpu host --sockets 1 --cores 20 --affinity 10-19,30-39 \
    --numa 1 --numa0 cpus=0-19,hostnodes=1,memory=$MEM,policy=preferred \
    --memory $MEM --balloon 0 \
    --scsihw virtio-scsi-single \
    --scsi0 local-lvm:0,import-from=$IMG,discard=on,iothread=1,ssd=1 \
    --ide2 local-lvm:cloudinit --boot order=scsi0 --serial0 socket --vga serial0 \
    --net0 virtio=$MAC,bridge=vmbr1 \
    --ciuser vlad --sshkeys vlad.pub --ipconfig0 ip=$IP/24,gw=10.99.0.1 --nameserver "1.1.1.1 8.8.8.8" \
    --onboot 0 \
    --description "nomos (research compute for the nomos-engine repo: CPU searches and exams). Safe to stop any time, nothing depends on it. Isolated: vmbr1, own NAT (nomos-nat), no access to private ranges. Repo nomos-engine, docs/operations.md."
qm disk resize $ID scsi0 40G
qm start $ID
echo "started; ssh -J root@pve.pgs.dom vlad@$IP in ~1 min"
