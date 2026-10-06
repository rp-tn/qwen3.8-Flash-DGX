#!/usr/bin/env bash
# Profile a vLLM boot with py-spy in fixed time slices, from container start to
# "Application startup complete". Every process in the container (APIServer = pid 1, EngineCore)
# and every thread, idle ones included, so the slices can be split by process/thread afterwards.
#
# The container must run with ptrace allowed and py-spy mounted: add
#   --cap-add=SYS_PTRACE -v $HOME/run/qwen-load-profile:/prof
# to the docker run line in scripts/serve.sh, with py-spy installed there once:
#   docker run --rm -v ~/run/qwen-load-profile:/out --entrypoint pip qwen38-flash-dgx:v0.30 install --target /out/pyspy py-spy
# Start this script first, then start the server: it waits for a container started after it.
set -u
NAME="${NAME:-qwen38-flash}"
PYSPY="${PYSPY:-/prof/pyspy/bin/py-spy}"  # paths inside the container
OUT_IN="${OUT_IN:-/prof/boot}"
CHUNK="${CHUNK:-10}" RATE="${RATE:-100}" MAX="${MAX:-90}"

t0=$(date -u +%s)
echo "waiting for a $NAME container started after $(date -u -d "@$t0" +%T) UTC"
while :; do
  read -r running started < <(docker inspect -f '{{.State.Running}} {{.State.StartedAt}}' "$NAME" 2>/dev/null || echo "false 1970-01-01T00:00:00Z")
  [ "$running" = true ] && [ "$(date -u -d "$started" +%s)" -ge "$t0" ] && break
  sleep 0.2
done
docker exec "$NAME" mkdir -p "$OUT_IN"
echo "container started $started, recording ${CHUNK} s slices into $OUT_IN"
for i in $(seq -w 0 "$MAX"); do
  docker exec "$NAME" "$PYSPY" record --pid 1 --subprocesses --threads --idle --nonblocking \
    -r "$RATE" -d "$CHUNK" -f raw -o "$OUT_IN/$i-$(date -u +%H%M%S).txt" >/dev/null 2>&1
  if docker logs "$NAME" 2>&1 | grep -q "Application startup complete"; then
    echo "ready at $(date -u +%T) UTC after $((10#$i + 1)) slices"; exit 0
  fi
done
echo "gave up after $MAX slices"
