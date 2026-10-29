@echo off
rem Paper-only incubation run for tt_moc (Amendment 6). Never places an order.
rem Retries while the network is not up yet (for example right after logon or wake). The
rem script checks the New York decision window itself, so a retry that lands after 15:58 ET
rem only reconciles and cannot decide late.
cd /d "%~dp0.."
set PYTHONPATH=src
set /a tries=0
:again
set /a tries+=1
.venv\Scripts\python.exe research\06_incubate_tt_moc.py >> state\incubation_run.log 2>&1
if %errorlevel%==0 goto :eof
if %tries% geq 6 (
    echo [%date% %time%] gave up after %tries% attempts, see errors above >> state\incubation_run.log
    exit /b 1
)
timeout /t 60 /nobreak > nul
goto again
