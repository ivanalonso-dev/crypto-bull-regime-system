@echo off
REM Actualiza velas, y anota las senales de v2 y h2. Programar DESPUES del 006 (p.ej. 10:15).
cd /d "%~dp0"
python fetch_data.py || exit /b 1
python fetch_data.py --symbols LTC BCH TRX DOT ATOM ETC XLM NEAR FIL UNI --out data_val || exit /b 1
python signals.py
