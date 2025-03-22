@echo off
echo Installing TrainIT - Advanced Object Detection and Tracking System
echo ==============================================================

:: Check Python version
python -c "import sys; assert sys.version_info >= (3, 8), 'Python 3.8+ required'" || (
    echo Error: Python 3.8 or higher is required
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

:: Create necessary directories
echo Creating project directories...
if not exist data\models mkdir data\models
if not exist data\output\videos mkdir data\output\videos
if not exist data\output\analytics mkdir data\output\analytics
if not exist data\output\logs mkdir data\output\logs

:: Download YOLOv8 weights if they don't exist
if not exist data\models\yolov8n.pt (
    echo Downloading YOLOv8 weights...
    powershell -Command "Invoke-WebRequest -Uri https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -OutFile data\models\yolov8n.pt"
)

:: Check CUDA availability
echo Checking CUDA availability...
python -c "import torch; print('CUDA is available' if torch.cuda.is_available() else 'CUDA is not available')"

:: Create default config if it doesn't exist
if not exist configs\default.yaml (
    echo Creating default configuration...
    copy configs\default.yaml.example configs\default.yaml
)

echo ==============================================================
echo Installation complete!
echo To start using TrainIT:
echo 1. Activate the virtual environment: venv\Scripts\activate
echo 2. Run the application: python -m src.main
echo ==============================================================

pause 