#!/bin/bash
# memguard.sh: run a heavy job under a memory watchdog so it can never drag the machine into a hang.
# usage: memguard.sh <log-prefix> <command...>
#   pause (SIGSTOP the job's process group) when: kernel memory pressure >= warn, or available memory < PAUSE_AVAIL_GB
#   resume (SIGCONT) when: pressure normal and available >= RESUME_AVAIL_GB
#   kill (TERM, then KILL after 10 s) when: pressure critical, or paused longer than MAX_PAUSE_S,
#        or swap grew more than SWAP_GROW_GB since the job started, or available < KILL_AVAIL_GB
# Writes <log-prefix>.memguard.log (one line per state change + a status line every 60 s) and, on a kill,
# <log-prefix>.memguard.killed with the reason. Exit code: the job's, or 137 when memguard killed it.
set -u
PREFIX=$1; shift
PAUSE_AVAIL_GB=${PAUSE_AVAIL_GB:-4}
RESUME_AVAIL_GB=${RESUME_AVAIL_GB:-6}
KILL_AVAIL_GB=${KILL_AVAIL_GB:-1.5}
SWAP_GROW_GB=${SWAP_GROW_GB:-6}
MAX_PAUSE_S=${MAX_PAUSE_S:-120}
WARN_OK_AVAIL_GB=${WARN_OK_AVAIL_GB:-999}   # opt-in (2026-11-24): pressure "warn" alone does not pause while available >= this; default = old behaviour
INTERVAL=${INTERVAL:-5}
LOG=$PREFIX.memguard.log
rm -f "$PREFIX.memguard.killed"

pressure() { sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null || echo 1; }   # 1 normal, 2 warn, 4 critical
swap_gb() { sysctl -n vm.swapusage | awk '{for(i=1;i<=NF;i++) if($i=="used"){v=$(i+2); u=substr(v,length(v)); v=v+0; if(u=="G") v*=1024; printf "%.2f", v/1024}}'; }
avail_gb() { [ "${AVAIL_MODE:-pages}" = level ] && { avail_level; return; }; avail_pages; }
avail_level() {  # opt-in AVAIL_MODE=level (2026-11-24): kernel free percentage x RAM; counts reclaimable file cache (active file-backed pages too)
  awk -v l="$(sysctl -n kern.memorystatus_level)" -v m="$(sysctl -n hw.memsize)" 'BEGIN{printf "%.2f", l/100*m/1073741824}'
}
avail_pages() {  # free + inactive + speculative + purgeable pages: what can be had without swapping
  vm_stat | awk '/page size of/ {ps=$8} /Pages free/ {f=$3} /Pages inactive/ {i=$3} /Pages speculative/ {s=$3} /Pages purgeable/ {p=$3}
                 END {printf "%.2f", (f+i+s+p)*ps/1073741824}'
}
lt() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a<b)}'; }
gt() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a>b)}'; }
note() { echo "$(date '+%F %T') $*" >> "$LOG"; }

SWAP0=$(swap_gb)
set -m                                   # job gets its own process group, so signals reach its children too
"$@" &
PID=$!
PGID=$PID
set +m
note "start pid $PID: $* | avail mode ${AVAIL_MODE:-pages} | thresholds pause<${PAUSE_AVAIL_GB}G resume>=${RESUME_AVAIL_GB}G kill<${KILL_AVAIL_GB}G swap+${SWAP_GROW_GB}G maxpause ${MAX_PAUSE_S}s | swap0 ${SWAP0}G avail $(avail_gb)G"

kill_job() {
  note "KILL: $1"
  echo "$(date '+%F %T') $1" > "$PREFIX.memguard.killed"
  kill -CONT -"$PGID" 2>/dev/null; kill -TERM -"$PGID" 2>/dev/null
  for _ in $(seq 10); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
  kill -KILL -"$PGID" 2>/dev/null
  wait "$PID" 2>/dev/null
  exit 137
}
trap 'kill_job "memguard itself was signalled"' INT TERM HUP

paused=0; paused_at=0; tick=0
while kill -0 "$PID" 2>/dev/null; do
  p=$(pressure); a=$(avail_gb); s=$(swap_gb); grow=$(awk -v s="$s" -v z="$SWAP0" 'BEGIN{printf "%.2f", s-z}')
  [ "$p" -ge 4 ] && kill_job "memory pressure critical (avail ${a}G, swap ${s}G)"
  lt "$a" "$KILL_AVAIL_GB" && kill_job "available ${a}G < ${KILL_AVAIL_GB}G (pressure $p, swap ${s}G)"
  gt "$grow" "$SWAP_GROW_GB" && kill_job "swap grew ${grow}G > ${SWAP_GROW_GB}G (now ${s}G, avail ${a}G)"
  if [ $paused = 0 ]; then
    if { [ "$p" -ge 2 ] && lt "$a" "$WARN_OK_AVAIL_GB"; } || lt "$a" "$PAUSE_AVAIL_GB"; then
      kill -STOP -"$PGID" 2>/dev/null; paused=1; paused_at=$(date +%s)
      note "PAUSE: pressure $p avail ${a}G swap ${s}G"
    fi
  else
    if { [ "$p" -eq 1 ] || ! lt "$a" "$WARN_OK_AVAIL_GB"; } && ! lt "$a" "$RESUME_AVAIL_GB"; then
      kill -CONT -"$PGID" 2>/dev/null; paused=0
      note "RESUME: pressure $p avail ${a}G swap ${s}G (paused $(( $(date +%s) - paused_at ))s)"
    elif [ $(( $(date +%s) - paused_at )) -gt "$MAX_PAUSE_S" ]; then
      kill_job "paused ${MAX_PAUSE_S}s without recovery (pressure $p, avail ${a}G, swap ${s}G)"
    fi
  fi
  tick=$((tick + 1)); [ $((tick % (60 / INTERVAL))) = 0 ] && note "ok: pressure $p avail ${a}G swap ${s}G paused $paused"
  sleep "$INTERVAL"
done
wait "$PID"; rc=$?
note "job exited rc $rc"
exit $rc
