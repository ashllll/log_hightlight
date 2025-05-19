@echo off
setlocal enabledelayedexpansion

REM Change to the directory where the batch script is located
cd /d "%~dp0"

echo ================================================
echo         Log Highlighter - Unified Launcher
echo ================================================

REM Check if Python is available
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python is not found. Please install Python.
    pause
    exit /b 1
)
echo Python found.

REM Check if requirements.txt exists
set "requirements_file=requirements.txt"
if not exist "!requirements_file!" (
    echo ERROR: !requirements_file! is missing. Cannot proceed.
    pause
    exit /b 1
)
echo !requirements_file! found.

REM Check if libs directory exists and is not empty
set "libs_dir=libs"
if not exist "!libs_dir!" (
    echo ERROR: The '!libs_dir!' directory is missing.
    echo Please ensure it exists and contains the required library packages.
    pause
    exit /b 1
)

dir /b "!libs_dir!\*.whl" >nul 2>&1 & set "libs_empty=1"
dir /b "!libs_dir!\*.tar.gz" >nul 2>&1 & set "libs_empty=1"
dir /b "!libs_dir!\*.zip" >nul 2>&1 & set "libs_empty=1"
for /F %%i in ('dir /b "!libs_dir!"') do set "libs_empty=0" & goto :libs_not_empty
:libs_not_empty

if "!libs_empty!"=="1" (
    echo ERROR: The '!libs_dir!' directory is empty.
    echo Please ensure it contains the required library packages.
    pause
    exit /b 1
)
echo '!libs_dir!' directory found and contains packages.

REM Install dependencies from the local libs directory
echo Installing dependencies from local './!libs_dir!' directory...
python -m pip install --no-index --find-links="./!libs_dir!" -r "!requirements_file!"

if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install dependencies from the './!libs_dir!' directory.
    echo Please check the console output for more details and ensure all required packages are present in './!libs_dir!'.
    pause
    exit /b 1
)
echo Dependencies installed successfully.

REM Run platform check
echo Running platform compatibility check...
python platform_check.py
if %ERRORLEVEL% neq 0 (
    echo WARNING: Platform check failed. The application might not run correctly.
    echo Do you want to continue? (Y/N)
    set /p continue_choice=
    if /i "!continue_choice!" neq "Y" (
        echo User cancelled execution.
        pause
        exit /b 1
    )
)

REM Start the application
echo Starting Log Highlighter application...
python main.py
if %ERRORLEVEL% neq 0 (
    echo ERROR: Application exited abnormally with error code: %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo Application exited normally.
exit /b 0 