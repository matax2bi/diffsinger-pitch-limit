@echo off
setlocal EnableDelayedExpansion
REM Drag & drop your acoustic.onnx onto this file.
REM Asks for high/low limit notes (Enter to skip) and optional unvoiced mel-marker.
if "%~1"=="" (
    echo Usage: drag and drop an acoustic.onnx file onto this bat file.
    pause
    exit /b 1
)
set "SRC=%~1"
set "FINAL=%~dpn1.patched.onnx"
set "TMP=%~dpn1.__tmp__.onnx"

set /p HI=High limit note (e.g. D5, Enter to skip):
set /p LO=Low limit note (e.g. G3, Enter to skip):
set "LIMARGS="
if not "!HI!"=="" set "LIMARGS=--limit-high !HI!"
if not "!LO!"=="" set "LIMARGS=!LIMARGS! --limit-low !LO!"

echo.
echo Unvoiced mel-marker: deterministic F0 gate for unvoiced consonants.
echo   *** ONLY for marker-aware vocoders! Standard vocoders (nsf_hifigan etc.)
echo   *** will produce a loud 16kHz beep on every unvoiced consonant.
set /p UV=Apply unvoiced mel-marker? [y/N]:

set "CUR=%SRC%"
if not "!LIMARGS!"=="" (
    echo.
    echo [1/2] pitch limit patch...
    python "%~dp0patch_acoustic_limit.py" "!CUR!" !LIMARGS! --out "%TMP%"
    if errorlevel 1 goto :fail
    set "CUR=%TMP%"
)
if /i "!UV!"=="y" (
    set "DSDICT=%~dp1..\dsvariance\dsdict.yaml"
    if not exist "!DSDICT!" set /p DSDICT=Path to dsdict.yaml:
    echo.
    echo [2/2] unvoiced mel-marker patch...
    python "%~dp0patch_acoustic_uvmark.py" "!CUR!" --dsdict "!DSDICT!" --out "%FINAL%"
    if errorlevel 1 goto :fail
) else (
    if "!LIMARGS!"=="" (
        echo Nothing selected - no patch applied.
        goto :end
    )
    move /y "%TMP%" "%FINAL%" >nul
)
if exist "%TMP%" del "%TMP%"
echo.
echo Done: %FINAL%
echo Point the 'acoustic:' entry of dsconfig.yaml to this file.
goto :end
:fail
if exist "%TMP%" del "%TMP%"
echo.
echo PATCH FAILED - see messages above.
:end
echo.
pause
