@echo off
python -m src.train_model
if errorlevel 1 pause
streamlit run app.py
pause
