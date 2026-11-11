# Operations — where the heavy runs go

The desktop has 16 GB of RAM shared with everything else on it. On 2026-11-11 five parallel
searches in sparse brains were paged out and crawled, the harness reaped a waiting shell for low
memory, and the runs were stopped at a third of their steps. Since then the searches and exams of
the `tissue` line run on their own VM on the user's Proxmox host.

```
desktop ──ssh nomos (ProxyJump root@pve.pgs.dom)──▶ nomos VM 10.99.0.70
nomos 10.99.0.70 ──vmbr1──▶ pve 10.99.0.1 ──NAT via vmbr0──▶ internet (private ranges dropped)
```

## VM 1094 `nomos` on pve

- Ubuntu 24.04 cloud image, **20 vCPU `host` pinned to socket 1** (host cpus `10-19,30-39`, NUMA
  node 1 preferred: the whole half the stopped `llm` VM used; 16 vCPU and 8 GB for its first run), **16 GB no
  balloon**, no swap, 40 GB (thin,
  `local-lvm`, `discard`), **`onboot 0`**, MAC pinned `BC:24:11:5E:5E:70`, IP `10.99.0.70` on stb's
  isolated bridge `vmbr1`. No GPU. Created 2026-11-11 by `infra/pve/create-vm.sh`.
- **Nothing depends on it.** Stop it any time: `ssh root@pve.pgs.dom qm stop 1094`. It does not
  start by itself after a host reboot: `qm start 1094`. It is in no backup job: the code is in git
  and results are copied back to the desktop.
- Sizing against the host's own alerts. `pve.cpu` warns at 70 % and the host idles near 16 %, so
  20 busy threads of 40 put it at about 66 %, on the edge: the user chose that on 2026-11-11 and
  accepts the warnings. The VM stays inside socket 1 so that socket 0 is left to production; more
  than 20 would take CPU from the platform, not just ring a bell. `pve.ram` warns at 90 % and
  clears under 85 %; the host was
  at 80 % (38 GB available of 187) before this VM. A search in sparse brains of up to 256 cells
  takes 0.8-1.2 GB (measured), an exam about half that, so the first 8 GB held six searches and
  left the threads idle; the user raised it to 16 GB the same day, knowing the host then sits at
  82-88 %. That is room for a dozen searches or twenty exams. More RAM
  (`qm set 1094 --memory N --numa0 cpus=0-19,hostnodes=1,memory=N,policy=preferred`) only after
  something is freed on the host. Memory, not threads, is what limits the VM.
- Own NAT `/usr/local/sbin/nomos-nat` on pve (chains `NOMOS_*`, internet only, every private range
  dropped, matched on IP + MAC), hooked into the `vmbr1` stanza after the last `*-nat` lines
  (`infra/pve/install-nat.sh`, backup `/root/interfaces.bak-before-nomos`). The same recipe as
  blogme's. A manual `stb-nat up` puts stb's jumps above: run `nomos-nat up` after it.
- On the VM (`infra/vm/setup.sh`): ufw with incoming closed and ssh from 10.99.0.1 only; venv
  `~/nomos-engine/.venv` with `torch==2.14.0` (CPU, the desktop's version), numpy, pytest.
  cloud-init's passwordless sudo for `vlad` is left in place.
- ssh alias `nomos` in `~/.ssh/config`: `vlad@10.99.0.70` via `ProxyJump root@pve.pgs.dom`.
- Undo, all of it: `qm stop 1094 && qm destroy 1094 --purge`; `nomos-nat down`; delete the two
  `nomos-nat` lines from the `vmbr1` stanza of `/etc/network/interfaces`;
  `rm /usr/local/sbin/nomos-nat`; drop `Host nomos` from `~/.ssh/config`.

## Working with it (`infra/dev/vm.sh`)

```sh
infra/dev/vm.sh push                       # the committed tree (HEAD) -> ~/nomos-engine on the VM
infra/dev/vm.sh run few-s0 ".venv/bin/python -m tissue.train --arm stream ... --out out/tissue/v2/few-s0.pt"
infra/dev/vm.sh jobs                       # what is running, memory left, load
infra/dev/vm.sh tail few-s0                # the last lines of that job's log
infra/dev/vm.sh pull out/tissue/v2         # copy results back to the desktop
```

`push` sends exactly what is committed, so a run can be named by a commit; the VM holds no
credentials. Rules of the house: at most twelve searches or twenty exams at once, and no more threads in
total than the VM has (memory was the limit at 8 GB; more threads per process buy little: 50 steps took 32 s at one thread and 26 s at three);
start them a few seconds apart; a run is still a pure function of its command line.
Numbers from the VM and from the desktop agree only as far as two CPUs' floating point does:
compare within one machine.
