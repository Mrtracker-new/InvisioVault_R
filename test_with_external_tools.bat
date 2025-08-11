@echo off
echo ===============================================
echo InVisioVault External Steganalysis Testing
echo ===============================================
echo.

REM Check if test image exists
if not exist "temp_test_stego.png" (
    echo ERROR: No test image found!
    echo Please run: python quick_test.py first
    pause
    exit /b 1
)

echo Testing steganographic image: temp_test_stego.png
echo.

REM Test with StegExpose (requires Java and StegExpose.jar)
echo 1. Testing with StegExpose...
if exist "StegExpose.jar" (
    java -jar StegExpose.jar temp_test_stego.png
    echo.
) else (
    echo   StegExpose.jar not found. Download from:
    echo   https://github.com/b3dk7/StegExpose
    echo.
)

REM Test with zsteg (requires Ruby and zsteg gem)
echo 2. Testing with zsteg...
where zsteg >nul 2>&1
if %ERRORLEVEL% == 0 (
    zsteg temp_test_stego.png
    echo.
) else (
    echo   zsteg not found. Install with:
    echo   gem install zsteg
    echo.
)

REM Test with stegseek (if available)
echo 3. Testing with StegSeek...
where stegseek >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo   Note: StegSeek is for Steghide format, may not detect InVisioVault format
    stegseek temp_test_stego.png
    echo.
) else (
    echo   StegSeek not found. Install from:
    echo   https://github.com/RickdeJager/stegseek
    echo.
)

REM Test with steghide (if available)
echo 4. Testing with steghide...
where steghide >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo   Note: Steghide extract may not work with InVisioVault format
    steghide extract -sf temp_test_stego.png
    echo.
) else (
    echo   steghide not found.
    echo.
)

echo ===============================================
echo Testing completed!
echo ===============================================
echo.
echo INTERPRETING RESULTS:
echo   - If tools find nothing: EXCELLENT anti-detection!
echo   - If tools report low confidence: GOOD anti-detection
echo   - If tools confidently detect: Needs improvement
echo.

pause
