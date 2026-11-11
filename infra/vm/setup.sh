#!/bin/bash
# Run on the nomos VM as vlad, once, after infra/pve/create-vm.sh:
#   - ufw: incoming closed, ssh only from pve (10.99.0.1)
#   - Python venv in ~/nomos-engine/.venv with the same PyTorch (CPU) as the desktop, numpy, pytest
# The code itself arrives from the desktop (infra/dev/vm.sh push): the VM has no way into any
# private range and needs no credentials for the repository.
set -euo pipefail
sudo DEBIAN_FRONTEND=noninteractive apt-get -qq update
sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install -y python3-venv python3-pip

sudo ufw default deny incoming && sudo ufw default allow outgoing
sudo ufw allow from 10.99.0.1 to any port 22 proto tcp
echo y | sudo ufw enable

mkdir -p ~/nomos-engine
[ -x ~/nomos-engine/.venv/bin/python ] || python3 -m venv ~/nomos-engine/.venv
~/nomos-engine/.venv/bin/pip install -q --upgrade pip
~/nomos-engine/.venv/bin/pip install -q --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH:-2.14.0}"
~/nomos-engine/.venv/bin/pip install -q numpy pytest
~/nomos-engine/.venv/bin/python -c "import torch, os; print('torch', torch.__version__, '| cpus', os.cpu_count())"
free -g | sed -n 2p
