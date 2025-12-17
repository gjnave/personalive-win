@echo off
cd frontend
call npm install
call npm run build
if %errorlevel% equ 0 (
    echo frontend build success
) else (
    echo frontend build failed
    exit /b 1
)
cd ..
call python ../inference_online.py --host 0.0.0.0 --port 7860
