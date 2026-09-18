@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
cl /nologo /std:c++17 /O2 /EHsc /openmp /DNDEBUG /LD /arch:AVX2 "%~dp0native_mvt\native_mvt.cpp" /Fe:"%~dp0native_mvt\native_mvt.dll"
exit /b %errorlevel%
