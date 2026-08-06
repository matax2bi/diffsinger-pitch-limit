@echo off
REM Drag & drop your acoustic.onnx onto this file. It asks for the limit note and patches.
if "%~1"=="" (
    echo Usage: drag and drop an acoustic.onnx file onto this bat file.
    pause
    exit /b 1
)
set /p LIMIT=Enter limit note (e.g. D5, F#4):
python "%~dp0patch_acoustic_limit.py" "%~1" --limit %LIMIT%
echo.
pause
