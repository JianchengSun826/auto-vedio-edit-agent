#!/usr/bin/env bash
set -e
echo "正在停止服务..."
docker compose down
echo "已停止"
