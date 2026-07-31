#!/bin/bash
# Restarts main.py (under caffeinate) if it dies. Meant to run overnight
# unattended so a crash we haven't caught yet doesn't silently kill monitoring.
cd "$(dirname "$0")"
source .venv/bin/activate

while true; do
    if ! pgrep -f "caffeinate -is python3 main.py" > /dev/null; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') watchdog: main.py not running, starting it" >> data/watchdog.log
        nohup caffeinate -is python3 main.py >> data/main.log 2>&1 &
        disown
    fi
    sleep 30
done
