#!/bin/bash
# The desktop's side of the nomos VM (ssh alias `nomos`, see docs/operations.md).
#
#   infra/dev/vm.sh push            the committed tree (HEAD) -> ~/nomos-engine on the VM
#   infra/dev/vm.sh run NAME CMD    start CMD there detached, output in ~/nomos-engine/out/jobs/NAME.log
#   infra/dev/vm.sh jobs            what is running there, and the memory left
#   infra/dev/vm.sh tail NAME       the last lines of a job's log
#   infra/dev/vm.sh pull DIR        copy ~/nomos-engine/DIR (e.g. out/tissue/v2) back here
#
# `push` sends exactly what is committed, so a run on the VM can be named by a commit. The VM
# holds no credentials and reaches no private range; everything goes through `ssh nomos`.
set -euo pipefail
cd "$(dirname "$0")/../.."
VM=nomos HOME_DIR=nomos-engine
case "${1:-}" in
push)
    git archive --format=tar HEAD | ssh $VM "mkdir -p $HOME_DIR && tar -x -C $HOME_DIR"
    ssh $VM "cd $HOME_DIR && echo $(git rev-parse --short HEAD) > .pushed && cat .pushed" ;;
run)
    name=$2; shift 2
    # the subshell is what detaches: without it the remote shell waits on the job and ssh never returns
    ssh -n $VM "cd $HOME_DIR && mkdir -p out/jobs && (nohup setsid bash -c '$*' > out/jobs/$name.log 2>&1 < /dev/null &) && echo started $name" ;;
jobs)
    ssh $VM "pgrep -af 'python -m tissue' | cut -c1-150 || echo 'nothing running'; free -g | sed -n 2p; uptime" ;;
tail)
    ssh $VM "tail -n ${3:-5} $HOME_DIR/out/jobs/$2.log" ;;
pull)
    mkdir -p "$2"
    ssh $VM "cd $HOME_DIR && tar -c $2" | tar -x ;;
*)
    sed -n '2,12p' "$0"; exit 1 ;;
esac
