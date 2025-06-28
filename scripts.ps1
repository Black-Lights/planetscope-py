# scripts.ps1 - Windows PowerShell development scripts for PlanetScope-py

param(
    [Parameter(Mandatory=$true)]
    [string]$Command
)

function Show-Help {
    Write-Host "PlanetScope-py Development Commands" -ForegroundColor Cyan
    Write-Host "===================================" -ForegroundColor Cyan
    Write-Host "install      Install package in development mode" -ForegroundColor Green
    Write-Host "install-dev  Install with development dependencies" -ForegroundColor Green
    Write-Host "test         Run all tests with coverage" -ForegroundColor Green
    Write-Host "test-quick   Run quick tests (unit tests only)" -ForegroundColor Green
    Write-Host "lint         Run code linting (flake8)" -ForegroundColor Green
    Write-Host "format       Format code with black" -ForegroundColor Green
    Write-Host "format-check Check code formatting without changing" -ForegroundColor Green
    Write-Host "clean        Clean build artifacts" -ForegroundColor Green
    Write-Host "build        Build package for distribution" -ForegroundColor Green
    Write-Host "env-info     Show environment information" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage: .\scripts.ps1 <command>" -ForegroundColor Yellow
    Write-Host "Example: .\scripts.ps1 test" -ForegroundColor Yellow
}

function Install-Package {
    Write-Host "Installing package in development mode..." -ForegroundColor Green
    pip install -e .
}

function Install-Dev {
    Write-Host "Installing with development dependencies..." -ForegroundColor Green
    pip install -e ".[dev]"
    pip install -e ".[docs]"
}

function Run-Tests {
    Write-Host "Running all tests with coverage..." -ForegroundColor Green
    pytest tests/ --cov=planetscope_py --cov-report=html --cov-report=term-missing -v
}

function Run-QuickTests {
    Write-Host "Running quick tests..." -ForegroundColor Green
    pytest tests/ -m "not slow" -x -v
}

function Run-Lint {
    Write-Host "Running code linting..." -ForegroundColor Green
    flake8 planetscope_py/ tests/ --max-line-length=88 --extend-ignore=E203,W503
}

function Format-Code {
    Write-Host "Formatting code with black..." -ForegroundColor Green
    black planetscope_py/ tests/
}

function Check-Format {
    Write-Host "Checking code formatting..." -ForegroundColor Green
    black --check planetscope_py/ tests/
}

function Clean-Build {
    Write-Host "Cleaning build artifacts..." -ForegroundColor Green
    if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
    if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
    if (Test-Path "*.egg-info") { Remove-Item -Recurse -Force "*.egg-info" }
    if (Test-Path "htmlcov") { Remove-Item -Recurse -Force "htmlcov" }
    if (Test-Path ".coverage") { Remove-Item -Force ".coverage" }
    if (Test-Path ".pytest_cache") { Remove-Item -Recurse -Force ".pytest_cache" }
    if (Test-Path "__pycache__") { Get-ChildItem -Recurse -Name "__pycache__" | Remove-Item -Recurse -Force }
    Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force
    Write-Host "Clean complete!" -ForegroundColor Green
}

function Build-Package {
    Write-Host "Building package..." -ForegroundColor Green
    Clean-Build
    python -m build
}

function Show-EnvInfo {
    Write-Host "Environment Information:" -ForegroundColor Cyan
    Write-Host "Python version:" -ForegroundColor Yellow
    python --version
    Write-Host "Pip version:" -ForegroundColor Yellow
    pip --version
    Write-Host "Current directory:" -ForegroundColor Yellow
    Get-Location
    Write-Host "Installed packages (planetscope related):" -ForegroundColor Yellow
    pip list | Select-String -Pattern "(planetscope|pytest|black|flake8)"
}

# Main command dispatcher
switch ($Command.ToLower()) {
    "help" { Show-Help }
    "install" { Install-Package }
    "install-dev" { Install-Dev }
    "test" { Run-Tests }
    "test-quick" { Run-QuickTests }
    "lint" { Run-Lint }
    "format" { Format-Code }
    "format-check" { Check-Format }
    "clean" { Clean-Build }
    "build" { Build-Package }
    "env-info" { Show-EnvInfo }
    default {
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Write-Host "Use '.\scripts.ps1 help' to see available commands" -ForegroundColor Yellow
    }
}