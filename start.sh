#!/usr/bin/env bash
set -e
echo "正在启动服务..."
docker compose up -d
echo "已启动，访问 http://localhost:7860"
