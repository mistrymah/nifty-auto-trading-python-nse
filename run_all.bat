@echo off
title Ultimate Strategy and Adjustment Engine (All-In-One)
set "VENV=C:\Users\admin\Desktop\Trading_Workspace\Env antigravity\.venv\Scripts"
call "%VENV%\activate.bat"
cd /d "E:\Trading_Agents\nifty_mcp_upstox"
python ultimate_execution_manager.py
pause
