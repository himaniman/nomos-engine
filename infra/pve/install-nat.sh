#!/bin/bash
# Run as root on pve, once, before create-vm.sh, from a directory holding nomos-nat. Idempotent.
# Needs stb's bridge vmbr1 (10.99.0.0/24, repo simply-the-best, infra/pve/install-network.sh).
# Installs /usr/local/sbin/nomos-nat and hooks it into the vmbr1 stanza of /etc/network/interfaces
# (two lines after the last "pre-down /usr/local/sbin/*-nat down" of the stanza; backup first), so it
# comes up with the bridge after a reboot. Same recipe as prime-time's infra/pve/install-nat.sh.
# Undo: nomos-nat down; delete the two nomos-nat lines from /etc/network/interfaces;
#       rm /usr/local/sbin/nomos-nat
set -euo pipefail
cd "$(dirname "$0")"
IFACES=/etc/network/interfaces

if ! grep -q '^iface vmbr1 ' $IFACES; then
    echo "vmbr1 is not in $IFACES: set up stb's bridge first (simply-the-best infra/pve/install-network.sh)"
    exit 1
fi
install -m 755 nomos-nat /usr/local/sbin/nomos-nat

if ! grep -q 'nomos-nat' $IFACES; then
    cp -a $IFACES /root/interfaces.bak-before-nomos
    # the line number of the last NAT pre-down inside the vmbr1 stanza
    last=$(awk '/^iface vmbr1 /{s=1; next} s && /^(iface|auto|#|$)/{s=0}
                s && /pre-down \/usr\/local\/sbin\/[a-z]+-nat down/{n=NR} END{print n}' $IFACES)
    if [ -z "$last" ]; then
        echo "no '*-nat down' line in the vmbr1 stanza; add these two lines to it by hand, then run: nomos-nat up"
        printf '\tpost-up /usr/local/sbin/nomos-nat up\n\tpre-down /usr/local/sbin/nomos-nat down\n'
        exit 1
    fi
    awk -v last="$last" '{ print }
         NR == last {
             print "\tpost-up /usr/local/sbin/nomos-nat up"
             print "\tpre-down /usr/local/sbin/nomos-nat down"
         }' /root/interfaces.bak-before-nomos > $IFACES.nomos-new
    [ "$(grep -c 'nomos-nat' $IFACES.nomos-new)" = 2 ] || { rm -f $IFACES.nomos-new; echo "insert failed"; exit 1; }
    cat $IFACES.nomos-new > $IFACES   # keep the file's inode, owner and mode
    rm -f $IFACES.nomos-new
fi
sed -n '/^iface vmbr1 /,/^#/p' $IFACES

nomos-nat up
nomos-nat status
