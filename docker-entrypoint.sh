#!/bin/sh
set -eu

mkdir -p /app/data
for file in pairs.jsonl stickers.json stats.json; do
  if [ ! -f "/app/data/$file" ] && [ -f "/app/seed/$file" ]; then
    cp "/app/seed/$file" "/app/data/$file"
  fi
done

exec "$@"
