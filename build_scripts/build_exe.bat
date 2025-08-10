@echo off
echo ==========================================
echo  InvisioVault Executable Builder
echo  Quick Build Script for Windows
echo ==========================================
echo.
echo This script uses the advanced Python builder.
echo For more options, use: python build_scripts\build_executable.py --help
echo.

echo Building InvisioVault executable...
python build_scripts\build_executable.py --clean

echo.
echo Build process completed!
echo.
echo Advanced options available:
echo   python build_scripts\build_executable.py --clean --debug
echo   python build_scripts\build_executable.py --clean --onedir
echo   python build_scripts\build_executable.py --clean --verbose
echo.
pause
