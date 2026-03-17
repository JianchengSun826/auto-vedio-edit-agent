# 视频自动剪辑 Agent

AI 驱动的视频自动剪辑工具，支持自然语言需求输入，经人工审核后导出多平台视频。

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

确保系统已安装 FFmpeg：
```bash
# macOS
brew install ffmpeg
# Ubuntu
sudo apt install ffmpeg
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 ANTHROPIC_API_KEY
```

### 3. 启动 Valkey（Redis）和 Celery worker

```bash
docker compose up -d valkey
celery -A tasks.celery_app worker --loglevel=info
```

### 4. 启动 UI

```bash
python app/main.py
```

打开 http://127.0.0.1:7860

## 使用方法

1. 上传本地视频文件
2. 用自然语言描述剪辑需求（如：`提取所有提到竞品的片段，每段保留前后5秒`）
3. 点击"开始分析"，等待转录和 AI 解析
4. 在候选片段列表中勾选要保留的片段
5. 选择导出平台，点击"批准并导出"

## 三种剪辑模式

| 模式 | 示例指令 |
|------|---------|
| 精华提取 | `提取所有提到竞品价格的片段` |
| 素材拼接 | `按脚本拼接：开场30秒用第一个视频，产品演示用第二个视频2-4分钟` |
| 社媒生产 | `把采访剪成3条抖音短视频，每条不超过60秒` |

## 未来扩展

- S3 云存储：实现 `storage/s3.py`，修改 `.env` 中 `STORAGE_BACKEND=s3`
- 视觉 AI：在转录与规则引擎之间插入场景检测模块
