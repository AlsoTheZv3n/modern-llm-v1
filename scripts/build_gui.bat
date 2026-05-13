@echo off
setlocal enabledelayedexpansion

REM The user has a stale MSBuildSDKsPath env var pointing to a .NET SDK that
REM is not installed (9.0.203). Unset it so dotnet picks up the correct path.
set "MSBuildSDKsPath="

set "ROOT=%~dp0.."
pushd "%ROOT%\gui\Monitor"

if "%~1"=="clean" (
    echo Cleaning bin\ and obj\...
    if exist bin  rmdir /s /q bin
    if exist obj  rmdir /s /q obj
)

echo === Restore ===
dotnet restore
if errorlevel 1 goto :fail

echo.
echo === Build (Release) ===
dotnet build -c Release --no-restore
if errorlevel 1 goto :fail

echo.
echo Run:
echo   gui\Monitor\bin\Release\net8.0-windows\ModernLLM.Monitor.exe
echo.

popd
exit /b 0

:fail
popd
exit /b 1
