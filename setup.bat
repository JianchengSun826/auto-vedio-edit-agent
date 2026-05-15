@echo off
setlocal enabledelayedexpansion
echo === 视频自动剪辑 Agent — 安装向导 ===
echo.

where docker >nul 2>&1
if errorlevel 1 (
    echo 错误：未检测到 Docker，请先安装 Docker Desktop
    echo 下载地址：https://www.docker.com/products/docker-desktop/
    pause & exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
    echo 错误：Docker 未运行，请先启动 Docker Desktop，再重新运行此脚本
    pause & exit /b 1
)

if not exist api_keys.env (
    copy api_keys.env.example api_keys.env >nul
    echo 已创建 api_keys.env（从模板复制）
)

for /f "tokens=1,* delims==" %%a in ('findstr "^ANTHROPIC_API_KEY=" api_keys.env') do set CURRENT_KEY=%%b
if "!CURRENT_KEY!"=="" set CURRENT_KEY=your_key_here
if "!CURRENT_KEY!"=="your_key_here" (
    echo 需要填入 Anthropic API Key
    echo 获取地址：https://console.anthropic.com 右上角 API Keys
    echo.
    set /p ANTHROPIC_KEY="请粘贴 Anthropic API Key: "
    powershell -Command "(Get-Content api_keys.env) -replace '^ANTHROPIC_API_KEY=.*', 'ANTHROPIC_API_KEY=!ANTHROPIC_KEY!' | Set-Content api_keys.env"
    echo Anthropic API Key 已保存
)

echo.
for /f "tokens=1,* delims==" %%a in ('findstr "^HF_TOKEN=" api_keys.env') do set CURRENT_HF=%%b
if "!CURRENT_HF!"=="" (
    echo [可选] 说话人分离功能需要 HuggingFace Token
    echo 获取地址：https://huggingface.co/settings/tokens
    echo.
    set /p HF_KEY="请粘贴 HF Token（直接回车跳过）: "
    if not "!HF_KEY!"=="" (
        powershell -Command "(Get-Content api_keys.env) -replace '^HF_TOKEN=.*', 'HF_TOKEN=!HF_KEY!' | Set-Content api_keys.env"
        echo HF Token 已保存
    ) else (
        echo 已跳过说话人分离功能
    )
)

echo.
echo 正在构建并启动服务（首次构建约需 5-10 分钟）...
docker compose up --build -d

echo.
echo ===================================
echo   安装完成！
echo   打开浏览器访问：http://localhost:7860
echo.
echo   停止服务：stop.bat
echo   再次启动：start.bat
echo ===================================
pause
