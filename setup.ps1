param(
    [switch]$RecreateVenv,
    [switch]$RunApp,
    [Nullable[int]]$CameraIndex = $null,
    [int]$FrameWidth = 640,
    [int]$FrameHeight = 480
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$RequirementsFile = Join-Path $ProjectRoot "requirements.txt"
$RequiredPythonMinor = "3.12"

function Get-PythonMinorVersion {
    param([string]$PythonExe)

    $versionValue = & $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to read Python version from: $PythonExe"
    }
    return $versionValue.Trim()
}

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' was not found. Install Python 3.12 and retry."
}

$launcherVersion = (& py -3.12 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
if ($launcherVersion -ne $RequiredPythonMinor) {
    throw "Python 3.12 is required. Install Python 3.12 and rerun setup.ps1"
}

$needsCreate = $RecreateVenv
if (-not (Test-Path $VenvPython)) {
    $needsCreate = $true
}

if (-not $needsCreate) {
    $existingMinor = Get-PythonMinorVersion -PythonExe $VenvPython
    if ($existingMinor -ne $RequiredPythonMinor) {
        Write-Host "[Setup] Existing .venv uses Python $existingMinor. Recreating with Python 3.12..."
        $needsCreate = $true
    }
}

if ($needsCreate) {
    if (Test-Path $VenvDir) {
        Remove-Item -Recurse -Force $VenvDir
    }

    Write-Host "[Setup] Creating .venv with Python 3.12..."
    & py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create virtual environment with Python 3.12"
    }
}

if (-not (Test-Path $RequirementsFile)) {
    throw "requirements.txt not found at: $RequirementsFile"
}

Write-Host "[Setup] Installing dependencies..."
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "pip upgrade failed"
}

& $VenvPython -m pip install -r $RequirementsFile
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed"
}

Write-Host "[Setup] Environment is ready."

if ($RunApp) {
    Write-Host "[Setup] Starting app..."
    $appArgs = @("body_framing_guidance/main.py", "--frame-width", "$FrameWidth", "--frame-height", "$FrameHeight")
    if ($null -ne $CameraIndex) {
        $appArgs += @("--camera-index", "$CameraIndex")
    }

    & $VenvPython @appArgs
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Next steps:"
Write-Host "1) Activate env: .venv\Scripts\activate"
Write-Host "2) Run app: python body_framing_guidance/main.py"
Write-Host ""
Write-Host "Or run directly from setup script:"
Write-Host "./setup.ps1 -RunApp"
