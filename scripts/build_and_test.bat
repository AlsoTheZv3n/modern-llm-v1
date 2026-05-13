@echo off
setlocal enabledelayedexpansion

REM Set up MSVC + Windows SDK environment so nvcc can find cl.exe
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
    echo ERROR: vcvars64.bat not found at "%VCVARS%"
    exit /b 1
)
call "%VCVARS%" >nul

REM Resolve project root from this script's location (scripts\..)
set "ROOT=%~dp0.."
pushd "%ROOT%"

set "BUILD=llm\build"

if "%~1"=="clean" (
    echo Cleaning build directory...
    if exist "%BUILD%" rmdir /s /q "%BUILD%"
)

echo === Configure ===
cmake -S llm -B %BUILD% -G Ninja -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 goto :fail

echo.
echo === Build ===
cmake --build %BUILD%
if errorlevel 1 goto :fail

echo.
echo === Test ===
ctest --test-dir %BUILD% --output-on-failure
if errorlevel 1 goto :fail

popd
exit /b 0

:fail
popd
exit /b 1
