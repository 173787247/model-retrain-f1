@echo off
echo ========================================
echo 检查训练进度
echo ========================================
echo.
docker exec model-retrain-f1-jupyter tail -n 50 /app/training_log.txt
echo.
pause

