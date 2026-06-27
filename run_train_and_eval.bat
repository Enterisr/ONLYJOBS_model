@echo off
cd /d C:\Users\drunk\ONLYJOBS
call .venv\Scripts\activate.bat
echo === Training model ===
python train.py
echo.
echo === Evaluating on test set ===
python evaluate.py
echo.
echo === Done ===
pause
