@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

rem ============================================================
rem  KoeDub launcher - bilingual voice-drama workbench
rem
rem  Double-click this file to start.
rem  Extra arguments are passed straight to app.py:
rem      start.cmd --port 9000
rem      start.cmd --cli input.mp3
rem      start.cmd --install faster-whisper --force
rem      start.cmd --app-window      open in a standalone app window
rem
rem  NOTE: keep this file ASCII-only. cmd.exe reads .cmd in the OEM
rem  codepage, so non-ASCII comments can be mangled into commands.
rem ============================================================

cd /d "%~dp0"

set "TEMP=%~dp0.tmp"
set "TMP=%~dp0.tmp"
if not exist "%TEMP%" mkdir "%TEMP%" >nul 2>nul

rem Force UTF-8 so the Chinese output in this console is readable
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo.
echo   KoeDub is starting, please wait...
echo.

rem ---- step 1: ffmpeg ----
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo   [WARN] ffmpeg not found. It is required for audio processing.
  echo          Install with:  winget install Gyan.FFmpeg
  echo          Then reopen this window.
  echo.
)

rem ---- step 2: pick a Python that this software can actually run ----
rem  The decision is delegated to "app.py --pick-python", which checks for
rem  numpy AND that the Python version matches the optional dependencies
rem  already installed in vendor/ (C extensions are version-bound).
set "PY="
set "TRIED="

for %%C in (
  "py -3.12"
  "py -3.11"
  "py -3.13"
  "py -3"
  "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
  "C:\Python312\python.exe"
  "C:\Python311\python.exe"
  "C:\Python313\python.exe"
  "python"
  "python3"
  "C:\Users\%USERNAME%\anaconda3\python.exe"
  "C:\Users\%USERNAME%\miniconda3\python.exe"
) do (
  if not defined PY (
    set "CAND=%%~C"
    if not "!CAND!"=="" call :trypython "!CAND!"
  )
)

if not defined PY (
  rem No immediately usable Python. The most common case is NOT "Python is
  rem missing" but "Python is installed, numpy is missing". Telling the user
  rem to reinstall from python.org is useless there. numpy is the only hard
  rem dependency and it is a tiny pure wheel, so just install it for them.
  if defined NEED_NUMPY_PY (
    echo   [INFO] Found Python at: !NEED_NUMPY_PY!
    echo          It is missing numpy ^(the only hard dependency^). Installing...
    "!NEED_NUMPY_PY!" -m pip install --user --disable-pip-version-check -q numpy
    if not errorlevel 1 (
      "!NEED_NUMPY_PY!" "%~dp0app.py" --pick-python >nul 2>nul
      if not errorlevel 1 set "PY=!NEED_NUMPY_PY!"
    )
    if defined PY (
      echo          numpy installed. Continuing.
      echo.
    ) else (
      echo   [ERROR] numpy could not be installed automatically.
      echo           Run this by hand, then start again:
      echo               "!NEED_NUMPY_PY!" -m pip install numpy
      echo.
    )
  )
)

if not defined PY (
  echo   [ERROR] No usable Python found.
  echo.
  echo           This tool needs Python 3.9+ with numpy installed.
  echo           Install from https://www.python.org/downloads/
  echo           and tick "Add Python to PATH", then reopen this window.
  echo.
  echo           If you installed the optional local-ASR packages with a
  echo           different Python, either use that same Python here, or
  echo           reinstall them for the one you want:
  echo               python app.py --install faster-whisper --force
  echo.
  echo           Tried:
  echo !TRIED!
  echo.
  pause
  exit /b 1
)

echo   Python: !PY!
echo.

rem ---- step 3: start ----
"!PY!" "%~dp0app.py" %*
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
  echo.
  echo   Startup failed ^(exit code %RC%^). Diagnostics:
  echo       "!PY!" "%~dp0app.py" --selftest
  echo       "!PY!" "%~dp0app.py" --deps
  echo.
  pause
)
endlocal
exit /b %RC%

rem ------------------------------------------------------------------
rem  Ask app.py whether this interpreter is usable. It prints the
rem  matching python tag and returns 0 when OK.
rem  The Microsoft Store stub in WindowsApps is skipped outright, since
rem  running it just pops up the Store.
rem ------------------------------------------------------------------
:trypython
set "CAND=%~1"
set "TRIED=!TRIED!    !CAND!&echo."
echo !CAND! | findstr /i "WindowsApps" >nul
if not errorlevel 1 exit /b 0
"%CAND%" "%~dp0app.py" --pick-python >nul 2>nul
rem "if errorlevel 3" means ">= 3", which would also swallow exit code 4
rem (vendor built by a different Python version). Compare exactly instead.
set "RC=!errorlevel!"
if "!RC!"=="3" (
  rem Exit code 3 = interpreter works, only numpy is missing. Remember it and
  rem handle it after all candidates fail, because a later candidate may still
  rem be fully usable and there is no reason to install anything for this one.
  set "NEED_NUMPY_PY=!CAND!"
  exit /b 0
)
if not "!RC!"=="0" exit /b 0
set "PY=!CAND!"
exit /b 0
