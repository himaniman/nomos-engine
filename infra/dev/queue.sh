#!/usr/bin/env bash
# A queue of runs on the VM: each line of JOBS is `name|command`; a run starts when fewer than
# CAP runs of the tissue are going (default 20: the VM's threads, one each). Logs go to
# out/jobs/NAME.log, the queue's own to out/jobs/QUEUE.log. Run it detached on the VM:
#
#   scp jobs.txt nomos:nomos-engine/out/jobs.txt
#   infra/dev/vm.sh run queue3 "bash infra/dev/queue.sh out/jobs.txt"
#
# It counts python processes running tissue.* (so runs started by hand count too) and never
# kills anything. Lines starting with # are skipped.
set -u
jobs_file=$1; cap=${2:-20}
running() { ps -C python -o args= | grep -c "tissue\." ; }
while IFS= read -r job || [ -n "$job" ]; do
    [ -z "$job" ] || [ "${job:0:1}" = "#" ] && continue
    name=${job%%|*}; cmd=${job#*|}
    while [ "$(running)" -ge "$cap" ]; do sleep 60; done
    (nohup setsid bash -c "$cmd" > "out/jobs/$name.log" 2>&1 < /dev/null &)
    echo "$(date +%H:%M) started $name"
    sleep 5
done < "$jobs_file"
echo "$(date +%H:%M) queue done"
