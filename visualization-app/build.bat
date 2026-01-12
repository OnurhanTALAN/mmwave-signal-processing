@echo off
echo ========================================
echo mmWave Visualization App - Setup
echo ========================================
echo.

:: Install dependencies
echo Installing dependencies...
pip install PyQt6 pyqtgraph numpy pyinstaller

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Failed to install dependencies.
    echo Please make sure Python and pip are installed.
    pause
    exit /b 1
)

echo.
echo Dependencies installed successfully!
echo.
echo ========================================
echo Building executable...
echo ========================================

:: Build the executable
pyinstaller --onefile --windowed --name mmwave_visualizer --icon=NONE main.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Failed to build executable.
    pause
    exit /b 1
)

echo.
echo ========================================
echo BUILD COMPLETE!
echo ========================================
echo.
echo The executable is located at:
echo   dist\mmwave_visualizer.exe
echo.
echo You can now run the app by double-clicking the exe file.
echo.
pause
