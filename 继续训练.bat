@echo off
chcp 65001 >nul
title VisDrone YOLOv8 Training - keep this window open!

echo ============================================
echo  VisDrone YOLOv8 training (resume from last.pt)
echo  DO NOT CLOSE THIS WINDOW
echo ============================================
echo.

cd /d "D:\WorkBuddyWorkspace\2026-11-03-21-15-46\uav_starter"
if errorlevel 1 goto :nofolder

call "D:\WorkBuddyWorkspace\2026-11-03-21-15-46\.testvenv\Scripts\activate.bat"

echo [1/2] Checking checkpoint ...
if not exist "runs\train\full\weights\last.pt" (
    echo   last.pt not found, starting a fresh 50-epoch run instead.
    python -m uavfusion.train_visdrone --epochs 50 --imgsz 640 --batch 16 --name full --workers 0
    goto :done
)

echo   last.pt found, resuming ...
echo [2/2] Training started. Close this window ONLY if you want to stop.
echo.
python -m uavfusion.resume_train

:done
echo.
echo ============================================
echo  Finished (or stopped). Scroll up for metrics.
echo  Press any key to close this window.
echo ============================================
pause >nul
exit /b

:nofolder
echo ERROR: project folder not found. Edit the path inside this .bat file.
pause >nul
