@echo off
echo ========================================
echo 模型加载、评估和再训练以提高 F1 Score
echo 快速启动 Jupyter (使用现有镜像)
echo ========================================
echo.

echo 启动容器...
docker-compose up -d

if errorlevel 1 (
    echo 错误: 容器启动失败
    pause
    exit /b 1
)

echo.
echo ========================================
echo 容器启动成功！
echo ========================================
echo.
echo Jupyter Lab 地址: http://localhost:8891
echo.
echo 注意: 
echo 1. 如果 models/distilbert-base-uncased-finetuned-squad/ 不存在
echo    会先进行初始训练（需要一些时间）
echo 2. 再训练过程也需要一定时间
echo.
echo 查看日志: docker-compose logs -f
echo 停止容器: docker-compose down
echo.
timeout /t 5
start http://localhost:8891
pause

