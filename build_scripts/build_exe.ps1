# InvisioVault PowerShell Build Script
# Quick executable builder using advanced Python script

Write-Host "=========================================="
Write-Host "🚀 InvisioVault Executable Builder" -ForegroundColor Cyan
Write-Host "   Quick Build Script" -ForegroundColor Gray
Write-Host "=========================================="
Write-Host

Write-Host "This script uses the advanced Python builder." -ForegroundColor Gray
Write-Host "For more options, run: python build_scripts\build_executable.py --help" -ForegroundColor Gray
Write-Host

# Build using the advanced Python script
Write-Host "🔨 Building InvisioVault executable..." -ForegroundColor Yellow
Write-Host

python build_scripts\build_executable.py --clean

# Show additional options
Write-Host
Write-Host "Advanced options available:" -ForegroundColor Yellow
Write-Host "   python build_scripts\build_executable.py --clean --debug" -ForegroundColor Gray
Write-Host "   python build_scripts\build_executable.py --clean --onedir" -ForegroundColor Gray
Write-Host "   python build_scripts\build_executable.py --clean --verbose" -ForegroundColor Gray

Write-Host
Write-Host "Press any key to exit..."
$Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
