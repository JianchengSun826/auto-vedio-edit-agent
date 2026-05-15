# 视频自动剪辑 Agent

面向新闻视频剪辑师的 AI 辅助粗剪工具。将原始素材的转录、字幕对齐、关键片段提取、多平台分发自动化，让剪辑师把时间花在判断与叙事上，而非反复拖拽时间线。

---

## 功能概览

### 基于字幕的粗剪

工具在本地运行 faster-whisper 对素材完成语音转录，生成带时间码的逐句字幕。剪辑师输入主题词或采访问题，系统在字幕中定位所有相关声效（soundbite），自动生成带 in/out 点的粗剪列表，供人工审核确认后直接导出。

10 分钟采访素材的转录约需 1–3 分钟，全程本地运行，素材不离开内网。

### 关键词声效提取

从长时录像（发布会、采访、会议）中按关键词批量提取声效段落，支持多关键词组合、指定时间范围缩小搜索区间、自动设置入点前留白和出点后留白。适合快速归档声效素材库或为多条稿件同时备料。

### 静音与冗余段落清理

检测并移除采访录像中的长时间停顿、无效口头禅片段（"那个"、"嗯"、"然后呢"），将整段对话压缩为有效说话时间，可直接作为音频精剪的参考轨。

### 多平台分发流

同一素材一次操作，自动输出适配不同平台的版本：

- **播出/网络版**（16:9，无损流复制，B 站 / YouTube）
- **竖屏社媒版**（9:16，中心裁切重编码，抖音 / 微信视频号）
- 抖音单条超 60 秒自动按时长分割，视频号超 10 分钟自动分割

### 人工审核把关

AI 提案，人工决定。粗剪候选列表以表格形式呈现，剪辑师逐条审阅时间码与字幕内容，勾选保留或剔除，确认后再导出，不存在黑盒自动裁决。

### 个人剪辑偏好持久化

通过 `USER_PREFERENCES.md` 文件记录剪辑师的工作习惯（默认留白时长、最短片段阈值、常用关键词、目标受众描述等），AI 在解析每条指令时自动参考，无需每次重复说明背景。

### 多说话人识别与按人提取

系统使用 WhisperX 完成转录后，自动调用 pyannote 对音轨做说话人分离，输出带说话人标签的逐句字幕：

```
[12.3s – 18.5s] SPEAKER_00: 我们这款产品比竞品便宜 30%
[19.1s – 24.8s] SPEAKER_01: 那在续航方面呢？
[25.0s – 31.2s] SPEAKER_00: 续航实测是同类最长的
```

剪辑师可直接按说话人过滤片段，无需逐字对照时间线：

```
提取 SPEAKER_00 所有关于产品价格的回答
只保留记者的提问部分（SPEAKER_01）
```

模型本地运行，素材不经过任何外部服务。需配置 HuggingFace Token（一次性免费申请）。未配置时自动降级为无说话人标签模式，其余功能不受影响。

### 个人剪辑偏好持久化

通过 `USER_PREFERENCES.md` 文件记录剪辑师的工作习惯（默认留白时长、最短片段阈值、常用关键词、目标受众描述等），AI 在解析每条指令时自动参考，无需每次重复说明背景。

### 批量处理

提供 Python API，可对整个素材目录循环处理，适合同一事件多机位素材的统一整理或日常归档流程集成。

---

## 目录

- [快速开始](#快速开始)
- [安装步骤](#安装步骤)
- [API Key 配置](#api-key-配置)
  - [Anthropic API Key](#anthropic-api-key)
  - [HuggingFace Token（说话人分离）](#huggingface-token说话人分离)
- [个人剪辑偏好配置](#个人剪辑偏好配置)
- [界面使用流程](#界面使用流程)
- [功能详解](#功能详解)
  - [三种剪辑模式](#三种剪辑模式)
  - [四个导出平台](#四个导出平台)
  - [指令写法参考](#指令写法参考)
- [批量处理（Python API）](#批量处理python-api)
- [常见问题](#常见问题)
- [停止与重启](#停止与重启)
- [项目结构](#项目结构)

---

## 快速开始

**三步完成安装：**

```
第一步：安装 Docker Desktop
第二步：运行 setup.sh（macOS/Linux）或 setup.bat（Windows）
第三步：打开浏览器访问 http://localhost:7860
```

**系统要求：**

| 项目 | 要求 |
|------|------|
| 操作系统 | macOS 12+、Windows 10/11、Ubuntu 20.04+ |
| Docker Desktop | 最新版，[点此下载](https://www.docker.com/products/docker-desktop/) |
| 内存 | 建议 8 GB 以上（Whisper medium 模型约需 5 GB） |
| 磁盘空间 | 约 6 GB（首次构建镜像） |
| 其他 | 无需安装 Python、ffmpeg 或任何其他软件 |

---

## 安装步骤

### macOS / Linux

**1. 安装 Docker Desktop**

前往 [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/) 下载安装，安装后启动 Docker Desktop，确认菜单栏出现鲸鱼图标。

**2. 下载项目**

```bash
git clone <仓库地址>
cd auto-vedio-edit-agent
```

或直接下载 ZIP 压缩包解压。

**3. 运行安装向导**

```bash
chmod +x setup.sh
./setup.sh
```

脚本会自动完成以下操作：
- 检测 Docker 是否就绪
- 引导你填入 Anthropic API Key（一次性操作）
- 构建并启动所有服务（首次约需 3–5 分钟）

看到以下提示即安装成功：

```
=== 启动完成 ===
打开浏览器访问：http://localhost:7860
```

---

### Windows

**1. 安装 Docker Desktop**

前往 [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/) 下载 Windows 版，安装后重启电脑，启动 Docker Desktop，确认任务栏出现鲸鱼图标。

**2. 下载项目**

下载 ZIP 压缩包，解压到任意目录（路径中不要包含中文或空格）。

**3. 运行安装向导**

双击 `setup.bat`，或在项目目录打开命令提示符运行：

```bat
setup.bat
```

脚本引导流程与 macOS 相同。

---

## API Key 配置

所有密钥统一存放在项目根目录的 **`api_keys.env`** 文件，这是唯一需要填写的配置文件。

```
auto-vedio-edit-agent/
├── api_keys.env        ← 在这里填写所有密钥
└── ...
```

> **安全提示**：`api_keys.env` 已被 `.gitignore` 排除，不会意外上传到代码仓库。

---

### Anthropic API Key

用于 AI 意图解析（必填）。

**获取步骤：**

1. 访问 [console.anthropic.com](https://console.anthropic.com)
2. 注册或登录账号
3. 进入 **API Keys** 页面，点击 **Create Key**，复制生成的 Key
4. 填入 `api_keys.env`：

```dotenv
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
```

---

### HuggingFace Token（说话人分离）

用于下载 pyannote 说话人分离模型（可选，不填则跳过说话人识别）。

**获取步骤：**

1. 注册 [huggingface.co](https://huggingface.co) 账号（免费）
2. 进入 [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)，点击 **New token**，权限选 **Read**，复制 Token
3. 访问以下两个页面，分别点击 **Agree and access repository** 接受模型授权：
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
4. 填入 `api_keys.env`：

```dotenv
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

> **首次启动说明**：填入 HF_TOKEN 后第一次启动时，系统会自动下载 pyannote 模型（约 1.5 GB），完成后后续启动无需重新下载。

**不需要说话人分离时**，`HF_TOKEN` 留空即可，转录功能完整可用：

```dotenv
HF_TOKEN=
```

**更换或更新密钥：** 直接编辑 `api_keys.env`，保存后运行 `./start.sh` 重启服务。

---

## 个人剪辑偏好配置

项目根目录的 **`USER_PREFERENCES.md`** 是你的个人剪辑习惯文件，类似于给 AI 的一份"了解我"说明书。

AI 在理解你的每一条剪辑指令时，都会先读取这份文件，从而：
- 自动应用你习惯的片段时长、留白时间
- 理解你的内容方向和目标受众
- 在你没有特别说明时，按你的偏好做默认选择

**第一次使用：** 复制模板文件并按需修改：

```bash
cp USER_PREFERENCES.example.md USER_PREFERENCES.md
```

**文件示例：**

```markdown
# 我的剪辑偏好

## 剪辑习惯
- 每段片段前后默认保留：5 秒
- 最短保留片段时长：8 秒
- 节奏风格：紧凑，去掉停顿和废话
- 尽量保留完整句子，不在句子中间切断
- 相邻片段若时间间隔不足 3 秒则自动合并

## 字幕偏好（未来版本生效）
- 字体：思源黑体
- 颜色：白色，黑色描边
- 大小：适中
- 位置：底部居中

## 默认导出平台
- 抖音

## 内容背景（帮助 AI 更准确理解）
- 内容方向：科技产品测评
- 目标受众：18–35 岁年轻用户
- 常用关键词：评测、上手、体验、对比、推荐

## 额外偏好
- 优先提取观点鲜明、有信息量的片段
- 避免保留纯粹的过渡性废话（"那个"、"嗯"、"然后呢"）
- 产品名称尽量完整保留，不要切断
```

> **说明**：偏好文件是可选的，不创建也不影响正常使用。AI 会完全按你的实时指令工作，偏好文件只是补充默认值。

---

## 界面使用流程

启动后打开浏览器访问 **http://localhost:7860**。

### 第一步：上传视频

点击界面左侧"上传视频"区域，选择本地视频文件。

支持格式：`mp4`、`mov`、`avi`、`mkv`、`webm` 等常见格式。

### 第二步：输入剪辑需求

在"剪辑需求"文本框中，用自然语言描述你想要什么。越具体越好。

**简单示例：**
```
提取所有提到竞品的片段
```

**加条件示例：**
```
提取所有提到竞品和价格的片段，每段前后各保留 5 秒
```

**指定时间示例：**
```
截取视频第 2 分钟到第 8 分钟，导出到 YouTube
```

**社媒生产示例：**
```
把这个采访剪成 3 条抖音短视频，每条不超过 60 秒，聚焦产品优势部分
```

### 第三步：点击「开始分析」

系统依次执行三个步骤，状态栏实时显示进度：

| 步骤 | 说明 | 参考耗时 |
|------|------|---------|
| 转录 | faster-whisper 将视频音频转为带时间戳的文字 | 10 分钟视频约 1–3 分钟 |
| AI 解析 | Claude Haiku 将你的需求转为结构化剪辑计划 | 约 5–10 秒 |
| 规则匹配 | 根据剪辑计划在转录结果中找出候选片段 | 约 1 秒 |

完成后状态栏显示：
```
找到 5 个候选片段 | 模式: highlight_extraction
```

### 第四步：审核候选片段

界面出现候选片段表格：

| 序号 | 时间范围 | 内容预览 | 置信度 | 包含 |
|------|---------|---------|------|------|
| 1 | 12.3s – 28.5s | 竞品的价格策略非常激进，我们... | 1.00 | ☑ |
| 2 | 45.0s – 62.1s | 跟竞品相比，我们在续航方面... | 1.00 | ☑ |
| 3 | 120.5s – 135.0s | 说到价格，用户反馈最多的是... | 1.00 | ☑ |

- **取消勾选**不想要的片段
- 右侧可**预览原始视频**，对照时间戳确认内容
- 时间范围和内容预览可帮助你快速判断是否保留

### 第五步：选择导出平台并导出

勾选目标平台（可多选），点击「批准并导出」。

导出完成后，界面显示所有输出文件的完整路径，文件保存在项目 `output/` 目录。

---

## 功能详解

### 三种剪辑模式

AI 会根据你的指令自动判断使用哪种模式，无需手动选择。

---

#### 模式一：精华提取（highlight_extraction）

从长视频中找出符合条件的关键片段。

**适用场景：**
- 从采访中提取特定话题片段
- 找出提到关键词的所有段落
- 截取指定时间段

**指令示例：**

```
提取所有提到"竞品"和"价格"的片段，每段前后各保留 5 秒
```
```
找出视频里所有提到退款、投诉、售后的片段
```
```
截取第 5 分钟到第 15 分钟的内容
```
```
提取所有时长超过 30 秒的完整段落
```
```
把所有停顿超过 2 秒的地方切掉，保留说话内容
```

---

#### 模式二：素材拼接（material_assembly）

按顺序提取多个指定片段，拼接成一个完整视频。

**适用场景：**
- 按脚本顺序组合不同片段
- 制作有开场、正文、结尾结构的视频

**指令示例：**

```
截取第 0–30 秒作为开场，第 5–8 分钟作为产品演示，最后 30 秒作为结尾，按顺序拼接
```
```
提取第 1 分钟的产品介绍 + 第 10 分钟的用户评价 + 第 20 分钟的总结部分
```
```
按顺序拼接：第 2 分钟到第 4 分钟，第 8 分钟到第 10 分钟，第 18 分钟到第 20 分钟
```

---

#### 模式三：社媒生产（social_media）

将长视频自动切割成多条适合发布的短视频，每条满足平台时长限制。

**适用场景：**
- 一键生成多条抖音/视频号短视频
- 将长会议录像切成多集发布

**指令示例：**

```
把这个采访剪成 3 条抖音短视频，每条不超过 60 秒，聚焦产品优势部分
```
```
把 1 小时的直播切成多条 10 分钟以内的视频号内容
```
```
生成 5 条 YouTube Shorts，每条 60 秒，提取最精彩的观点
```

---

### 四个导出平台

| 平台 | 画面比例 | 时长限制 | 处理方式 | 备注 |
|------|---------|---------|---------|------|
| **抖音** | 9:16 竖屏 | 60 秒/条 | 重新编码 + 中心裁切 | 超长片段自动分割为多条 |
| **微信视频号** | 9:16 竖屏 | 10 分钟/条 | 重新编码 + 中心裁切 | 超长片段自动分割 |
| **B 站** | 16:9 横屏 | 无限制 | 流复制（无损） | 导出速度极快 |
| **YouTube** | 16:9 横屏 | 无限制 | 流复制（无损） | 导出速度极快 |

**关于竖屏裁切：** 横屏原视频转为 9:16 时，系统做**中心裁切**，保留画面中央区域。如果画面主体不在中央（如双人对话靠两侧），建议源视频直接使用竖屏拍摄。

**关于自动分割：** 导出到抖音的片段若超过 60 秒，系统自动按时间切割为多个文件：
```
output/
  interview_douyin_seg1.mp4         # 0–60s
  interview_douyin_seg1_part2.mp4   # 60–120s
  interview_douyin_seg1_part3.mp4   # 120–135s
```

---

### 指令写法参考

以下是常见场景的指令写法，可以直接复制修改使用。

**按关键词提取：**
```
提取所有提到"价格"、"优惠"、"折扣"的片段，每段前 3 秒后 5 秒
```

**指定时间范围：**
```
截取视频第 2 分钟到第 8 分钟
```
```
只要前 30 分钟的内容
```

**去除停顿：**
```
删除所有超过 1.5 秒的静音停顿，保留说话内容
```

**限定最短时长：**
```
只保留时长超过 20 秒的片段，提取提到竞品的内容
```

**多条件组合：**
```
提取第 5 到第 30 分钟内提到"用户体验"或"交互设计"的片段，
每段保留前 3 秒后 8 秒，过滤掉短于 15 秒的片段，导出到 B 站和 YouTube
```

**社媒快剪：**
```
从这个 1 小时的产品发布会视频里，提取最重要的 5 个亮点片段，
剪成 5 条抖音，每条控制在 45 秒以内
```

**采访整理：**
```
这是一个 40 分钟采访，帮我提取所有讲到创业经历的片段，
去掉废话和停顿，拼成一条完整的 B 站视频
```

---

## 批量处理（Python API）

如果需要处理大量视频，可以绕过 UI 直接调用 Python API（需要本地安装 Python 环境）：

```python
from pathlib import Path
from agent.orchestrator import Orchestrator
from processing.exporter import Exporter
from models.edit_plan import OutputFormat, Platform

orch = Orchestrator()
exporter = Exporter(output_dir="./output")

videos = list(Path("./input").glob("*.mp4"))
for video in videos:
    result = orch.run(
        video_path=video,
        user_instruction="提取所有提到竞品的片段，前后各保留 5 秒"
    )
    output_files = exporter.export(
        src=video,
        candidates=result.candidates,
        formats=[
            OutputFormat(platform=Platform.DOUYIN),
            OutputFormat(platform=Platform.YOUTUBE),
        ]
    )
    print(f"{video.name}: 找到 {len(result.candidates)} 个片段，导出 {len(output_files)} 个文件")
```

---

## 常见问题

**Q：转录速度很慢怎么办？**

在 `docker.env` 中将模型换小：
```
WHISPER_MODEL=small   # 从 medium 改为 small，速度提升约 2 倍，准确度略降
```
修改后运行 `./start.sh` 重启生效。

各模型参考：

| 模型 | 内存占用 | 速度 | 准确度 |
|------|---------|------|--------|
| tiny | ~1 GB | 极快 | 一般 |
| small | ~2 GB | 快 | 较好 |
| medium | ~5 GB | 中等 | 好（默认）|
| large-v3 | ~10 GB | 慢 | 最佳 |

---

**Q：找到的片段不准确怎么办？**

在指令里加入更具体的关键词，或同时指定时间范围辅助定位：
```
提取第 5 到第 30 分钟内提到"价格"、"优惠"、"折扣"的片段，每段保留前 3 秒后 5 秒
```

也可以在 `USER_PREFERENCES.md` 里写明你的常用关键词，AI 会优先考虑。

---

**Q：API Key 填错了或 Key 失效了怎么办？**

编辑项目根目录的 `api_keys.env`，替换 `ANTHROPIC_API_KEY` 的值，然后：
```bash
./start.sh
```

---

**Q：视频超过 2 小时怎么办？**

系统会自动按 30 分钟一段分块处理，处理完成后时间戳自动合并回原始时间轴，无需手动切割。

---

**Q：导出的竖屏视频主体被裁切掉了？**

这是中心裁切的局限。如果原始视频中说话人不在画面正中央（如双人对谈各在两侧），转竖屏后可能会裁掉一人。建议：
- 拍摄时使用竖屏，直接导出原比例
- 或在导出时不选竖屏平台，改选 B 站 / YouTube 保留横屏

---

**Q：Docker 容器一直重启怎么办？**

查看日志：
```bash
docker compose logs ui
docker compose logs worker
```

最常见原因是 API Key 未填写或格式错误，参考 [API Key 配置](#api-key-配置) 重新填写。

---

**Q：可以在没有网络的情况下使用吗？**

转录（faster-whisper）完全本地运行，不需要网络。  
AI 解析（Claude Haiku）需要访问 Anthropic API，必须联网。  
视频导出不需要网络。

---

## 停止与重启

```bash
# 停止所有服务
./stop.sh

# 再次启动（不重新构建）
./start.sh

# 更新代码后重新构建并启动
docker compose up --build -d
```

查看运行状态：
```bash
docker compose ps
```

查看实时日志：
```bash
docker compose logs -f
```

---

## 项目结构

```
auto-vedio-edit-agent/
│
├── api_keys.env             ← API 密钥（你填写的，不上传 git）
├── api_keys.env.example     ← 密钥模板
├── docker.env               ← 非敏感配置（Whisper 型号等）
├── USER_PREFERENCES.md      ← 你的剪辑偏好（可选，不上传 git）
├── USER_PREFERENCES.example.md  ← 偏好文件模板
│
├── setup.sh / setup.bat     ← 首次安装向导
├── start.sh / start.bat     ← 启动服务
├── stop.sh / stop.bat       ← 停止服务
│
├── app/
│   └── main.py              ← Gradio UI 入口
├── agent/
│   ├── intent_parser.py     ← AI 意图解析（读取 USER_PREFERENCES.md）
│   ├── orchestrator.py      ← 流水线编排
│   └── rule_engine.py       ← 规则执行引擎
├── processing/
│   ├── transcriber.py       ← 语音转录（faster-whisper）
│   ├── ffmpeg_utils.py      ← FFmpeg 工具函数
│   └── exporter.py          ← 多平台导出
├── storage/                 ← 存储抽象层（本地 / S3 扩展）
├── models/                  ← Pydantic 数据模型
├── tasks/                   ← Celery 异步任务
├── config/
│   └── settings.py          ← 配置管理
│
├── output/                  ← 导出视频保存位置
├── docker-compose.yml
└── Dockerfile
```

---

## 未来计划

- **字幕烧录**：将转录文字嵌入视频，支持 `USER_PREFERENCES.md` 中设置的字体和样式
- **S3 云存储**：将输出文件自动上传到云端
- **视觉 AI**：根据画面内容（非仅文字）筛选片段
- **多语言**：faster-whisper 原生支持 99 种语言，修改转录参数即可启用
