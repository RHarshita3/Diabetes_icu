@echo off
REM One-click setup + launch for Windows (PowerShell or CMD):  .\run_app.bat
cd /d %~dp0
if not exist .venv (
  py -3 -m venv .venv 2>nul || python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || goto :error
if not exist data\sample_icu_synthetic.csv if not exist data\TrainingWiDS2021.csv python -m diabetes_icu.synthetic
if not exist models\model.joblib python -m diabetes_icu.train --fast || goto :error
streamlit run app\dashboard.py
goto :eof
:error
echo.
echo Setup failed - see the message above.
exit /b 1
