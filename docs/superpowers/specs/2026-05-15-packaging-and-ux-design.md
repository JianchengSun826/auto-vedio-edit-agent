# 设计文档：打包与用户体验改进

**日期**：2026-05-15  
**状态**：待实现

---

## 目标

让无编程经验的用户能在任何平台（macOS / Windows / Linux）通过 Docker 一键安装并使用本项目，同时将 API Key 和用户剪辑偏好分离到独立的、易于修改的文件中。

---

## 核心变更概览

| 变更 | 原状态 | 新状态 |
|------|--------|--------|
| 运行方式 | 手动建 venv + 分别启动 worker 和 UI | Docker Compose 一键全起 |
| API Key 位置 | 混在 `.env` 里 | 独立 `api_keys.env` 文件 |
| 非敏感配置 | 混在 `.env` 里 | 独立 `docker.env` 文件 |
| 用户偏好 | 无 | `USER_PREFERENCES.md`，注入 LLM 提示词 |
| 安装引导 | 无 | `setup.sh` / `setup.bat` 交互式向导 |
| Python 兼容性 | 需要 3.10+（`str \| None` 语法） | 本地支持 3.9+（`from __future__ import annotations`） |

---

## 一、文件结构

### 新增文件

```
api_keys.env.example          ← git 提交，密钥模板
api_keys.env                  ← git 忽略，用户填写
docker.env                    ← git 提交，非敏感 Docker 配置
USER_PREFERENCES.example.md  ← git 提交，偏好文件模板
USER_PREFERENCES.md           ← git 忽略，用户个人偏好
setup.sh                      ← macOS/Linux 安装向导
setup.bat                     ← Windows 安装向导
start.sh / start.bat          ← 重新启动服务
stop.sh / stop.bat            ← 停止服务
```

### 更新 .gitignore

新增忽略项：
```
api_keys.env
USER_PREFERENCES.md
output/
data/
tmp/
```

---

## 二、配置文件设计

### api_keys.env.example

```dotenv
# ============================================================
# API 密钥配置 — 这里是唯一需要填写的地方
# 修改后运行 ./start.sh 重启服务生效
# ============================================================

# Anthropic API Key（必填）
# 获取地址：https://console.anthropic.com → API Keys → Create Key
ANTHROPIC_API_KEY=your_key_here
```

### docker.env

```dotenv
WHISPER_MODEL=medium
WHISPER_DEVICE=auto
STORAGE_BACKEND=local
LOCAL_STORAGE_ROOT=./data
TEMP_DIR=./tmp
OUTPUT_DIR=./output
CELERY_BROKER_URL=redis://valkey:6379/0
CELERY_RESULT_BACKEND=redis://valkey:6379/0
```

注意：`CELERY_BROKER_URL` 在 Docker 内使用服务名 `valkey`，不是 `localhost`。

---

## 三、Docker Compose 重构

当前 docker-compose 只包含 `valkey` + `worker`，UI 需手动启动。新版将 UI 也纳入编排。

### 三个服务

**valkey**：消息队列，仅内部访问，不暴露端口到宿主机。

**worker**：Celery 异步任务，读取 `api_keys.env` + `docker.env`，挂载 `data/`、`output/`、`tmp/` 目录。

**ui**：Gradio 应用，暴露 `7860:7860`，读取相同 env 文件，依赖 `valkey` 启动。所有服务设置 `restart: unless-stopped`。

### Celery Broker URL

docker-compose 内的服务间通信使用服务名，而非 `localhost`。`docker.env` 中配置为 `redis://valkey:6379/0`。

### Dockerfile 补充

当前 Dockerfile 缺少端口声明，补充：
```dockerfile
EXPOSE 7860
```

---

## 四、Setup 脚本逻辑

### setup.sh（macOS / Linux）

执行顺序：

1. 检测 `docker` 命令是否存在，否则打印下载链接并退出
2. 检测 Docker daemon 是否运行（`docker info`），否则提示启动 Docker Desktop 并退出
3. 如果 `api_keys.env` 不存在，从 `api_keys.env.example` 复制
4. 读取当前 `ANTHROPIC_API_KEY` 值，若为空或等于 `your_key_here`，交互提示用户粘贴 Key，用 `sed` 写入 `api_keys.env`
5. 运行 `docker compose up --build -d`
6. 打印成功信息，告知访问地址和后续命令

### setup.bat（Windows）

逻辑与 `setup.sh` 一致，使用 `where docker`、`docker info`、`findstr`、`powershell -Command "(Get-Content ...) -replace ..."` 实现相同步骤。

### start.sh / start.bat

```bash
docker compose up -d
```

不重新构建，适合日常启动。

### stop.sh / stop.bat

```bash
docker compose down
```

---

## 五、USER_PREFERENCES.md 注入机制

### 读取时机

`IntentParser.__init__` 时读取，作为实例变量缓存（避免每次 `parse()` 重复 IO）。

### 读取逻辑

```python
def _load_preferences(self) -> str:
    path = Path("USER_PREFERENCES.md")
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""
```

### 注入位置

拼接到 `SYSTEM_PROMPT` 开头，格式：

```
[用户剪辑偏好]
{preferences_content}

---

{原 SYSTEM_PROMPT}
```

若 `preferences_content` 为空，则直接使用原 `SYSTEM_PROMPT`，不附加任何内容。

### 不影响范围

偏好文件只影响 `IntentParser` 的 LLM 调用。规则引擎、转录、导出均不读取该文件。

---

## 六、Python 3.9 兼容性修复

在以下使用 `X | Y` union 语法的文件顶部添加 `from __future__ import annotations`：

- `processing/transcriber.py`
- `processing/exporter.py`
- `agent/rule_engine.py`
- `app/main.py`

此修改使类型注解在运行时作为字符串处理，Pydantic v2 可正确解析，测试可在本地 Python 3.9 运行。Docker 镜像继续使用 Python 3.11，不受影响。

---

## 七、不在本次范围内

- 字幕烧录（FFmpeg subtitles filter）
- S3 云存储实现
- 视觉 AI 场景检测
- USER_PREFERENCES.md 中结构化字段的程序化解析（当前全部作为自然语言注入 LLM）
