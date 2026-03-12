#!/bin/bash
cd /Users/yusuf/Documents/falcon/AI/attendance/backend
rm -f /tmp/cctv_backend.log
PYTHONUNBUFFERED=1 nohup /Users/yusuf/Documents/falcon/AI/attendance/backend/venv/bin/python app.py > /tmp/cctv_backend.log 2>&1 &
PID=$!
disown $PID
echo "Backend PID: $PID"
echo $PID > /tmp/cctv_backend.pid
echo "Waiting for startup..."
sleep 35
echo ""
echo "=== Last 25 lines of log ==="
tail -25 /tmp/cctv_backend.log
echo ""
echo "=== API Check ==="
curl -s http://127.0.0.1:5000/api/cameras 2>/dev/null | head -5
echo ""
curl -s http://127.0.0.1:5000/api/faces/status 2>/dev/null
