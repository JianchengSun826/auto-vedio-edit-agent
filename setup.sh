#!/usr/bin/env bash
set -e

echo "=== 视频自动剪辑 Agent — 安装向导 ==="
echo ""

# 1. Check Docker installed
if ! command -v docker &>/dev/null; then
    echo "错误：未检测到 Docker，请先安装 Docker Desktop"
    echo "下载地址：https://www.docker.com/products/docker-desktop/"
    exit 1
fi

# 2. Check Docker daemon running
if ! docker info &>/dev/null 2>&1; then
    echo "错误：Docker 未运行，请先启动 Docker Desktop，再重新运行此脚本"
    exit 1
fi

# 3. Initialise api_keys.env from example if missing
if [ ! -f api_keys.env ]; then
    cp api_keys.env.example api_keys.env
    echo "已创建 api_keys.env（从模板复制）"
fi

# 4. Prompt for Anthropic API Key if placeholder
current_anthropic=$(grep "^ANTHROPIC_API_KEY=" api_keys.env | cut -d'=' -f2-)
if [ -z "$current_anthropic" ] || [ "$current_anthropic" = "your_key_here" ]; then
    echo "需要填入 Anthropic API Key"
    echo "获取地址：https://console.anthropic.com → API Keys → Create Key"
    echo ""
    read -rp "请粘贴 Anthropic API Key: " anthropic_key
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s|^ANTHROPIC_API_KEY=.*|ANTHROPIC_API_KEY=${anthropic_key}|" api_keys.env
    else
        sed -i "s|^ANTHROPIC_API_KEY=.*|ANTHROPIC_API_KEY=${anthropic_key}|" api_keys.env
    fi
    echo "Anthropic API Key 已保存"
fi

echo ""
# 5. Prompt for HF Token (optional)
current_hf=$(grep "^HF_TOKEN=" api_keys.env | cut -d'=' -f2-)
if [ -z "$current_hf" ]; then
    echo "[可选] 说话人分离功能需要 HuggingFace Token"
    echo "获取地址：https://huggingface.co/settings/tokens（创建 Read token）"
    echo "注意：还需在 HuggingFace 接受 pyannote 模型授权（详见 api_keys.env）"
    echo ""
    read -rp "请粘贴 HF Token（直接回车跳过，说话人分离将不可用）: " hf_key
    if [ -n "$hf_key" ]; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s|^HF_TOKEN=.*|HF_TOKEN=${hf_key}|" api_keys.env
        else
            sed -i "s|^HF_TOKEN=.*|HF_TOKEN=${hf_key}|" api_keys.env
        fi
        echo "HF Token 已保存"
    else
        echo "已跳过，说话人分离功能不可用（可随时编辑 api_keys.env 补填）"
    fi
fi

echo ""
echo "正在构建并启动服务（首次构建约需 5–10 分钟，请耐心等待）..."
docker compose up --build -d

echo ""
echo "==================================="
echo "  安装完成！"
echo "  打开浏览器访问：http://localhost:7860"
echo ""
echo "  停止服务：./stop.sh"
echo "  再次启动：./start.sh"
echo "==================================="
