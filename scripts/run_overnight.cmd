@echo off
rem Resumable overnight run of everything still pending.
rem Start it each night (double-click, or: scripts\run_overnight.cmd), close the
rem window in the morning. The CNN checkpoints every epoch, so nothing finished
rem is lost; the next start continues where it stopped. When the CNN curve is
rem complete, the evaluation and the reduction/CV experiments run automatically.
rem Progress: logs\progress.txt (one line per step) and logs\cnn.log (per epoch).
cd /d "%~dp0.."
if not exist logs mkdir logs
set PY=.venv\Scripts\python.exe
title Modi Lipi XAI - overnight run (safe to close)
echo %date% %time% START session >> logs\progress.txt
%PY% -u scripts\run_cnn_baseline.py >> logs\cnn.log 2>&1
if errorlevel 1 goto failed
echo %date% %time% CNN curve complete >> logs\progress.txt
if exist logs\reduction_cv.done goto eval
%PY% -u scripts\run_evaluation.py --skip-shap > logs\eval_after_cnn.log 2>&1
%PY% -u scripts\run_training.py --only reduction cv > logs\reduction_cv.log 2>&1
if errorlevel 1 goto failed
echo done > logs\reduction_cv.done
:eval
%PY% -u scripts\run_evaluation.py --skip-shap > logs\final_eval.log 2>&1
echo %date% %time% ALL REMAINING DONE >> logs\progress.txt
goto end
:failed
echo %date% %time% STOPPED WITH AN ERROR - see logs >> logs\progress.txt
:end
