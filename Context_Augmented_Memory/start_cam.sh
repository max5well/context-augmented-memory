#!/bin/bash

# === Context-Augmented Memory startup script ===
CHROMA_PATH="./CAM_project/chroma_db"
CHROMA_PORT=8000
PROXY_PORT=8080

echo "🚀 Starting Context-Augmented Memory (CAM) system..."

# --- Step 1: Check if Chroma is already running ---
if lsof -i :$CHROMA_PORT | grep -q LISTEN; then
  echo "✅ Chroma already running on port $CHROMA_PORT"
else
  echo "🧠 Starting Chroma..."
  nohup chroma run --path $CHROMA_PATH --port $CHROMA_PORT > chroma.log 2>&1 &
  sleep 3
fi

# --- Step 2: Wait for Chroma to respond ---
echo "⏳ Waiting for Chroma heartbeat..."
for i in {1..10}; do
  if curl -s http://localhost:$CHROMA_PORT/api/v2/heartbeat > /dev/null; then
    echo "✅ Chroma is alive!"
    break
  fi
  echo "...retrying ($i/10)"
  sleep 1
done

# --- Step 3: Start Proxy API ---
if lsof -i :$PROXY_PORT | grep -q LISTEN; then
  echo "⚠️ Proxy already running on port $PROXY_PORT"
else
  echo "🧩 Starting CAM Proxy API..."
  nohup uvicorn proxy_api.app:app --port $PROXY_PORT --reload > proxy.log 2>&1 &
  echo "✅ Proxy running on http://127.0.0.1:$PROXY_PORT"
fi

echo "✨ CAM system ready — memory and proxy active!"
