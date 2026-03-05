@echo off
echo ============================================
echo  Krishi Kavach - ML Server (Local)
echo  Crops: Banana, Chilli, Radish, Groundnut, Cauliflower
echo ============================================
echo.

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo No .venv found - using system Python
)

echo Installing / checking dependencies...
pip install -r requirements.txt --quiet

echo.
echo Starting ML server on http://localhost:8000
echo Press Ctrl+C to stop
echo.
python app.py
