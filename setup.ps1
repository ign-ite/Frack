param(
    [switch]$RecreateVenv,
    [switch]$SetupOnly,
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
$PythonInstallerVersion = "3.12.10"
$PythonInstallerUrl = "https://www.python.org/ftp/python/$PythonInstallerVersion/python-$PythonInstallerVersion-amd64.exe"

function Get-PythonMinorVersion {
    param(
        [string]$PythonCommand,
        [switch]$UseLauncher
    )

    if ($UseLauncher) {
        $versionValue = & $PythonCommand -3.12 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    } else {
        $versionValue = & $PythonCommand -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    }

    if ($LASTEXITCODE -ne 0) {
        return $null
    }

    return $versionValue.Trim()
}

function Get-Python312Spec {
    $pyCommand = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $pyCommand) {
        $minorVersion = Get-PythonMinorVersion -PythonCommand $pyCommand.Source -UseLauncher
        if ($minorVersion -eq $RequiredPythonMinor) {
            return @{
                Mode = "launcher"
                Command = $pyCommand.Source
            }
        }
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $pythonCommand) {
        $minorVersion = Get-PythonMinorVersion -PythonCommand $pythonCommand.Source
        if ($minorVersion -eq $RequiredPythonMinor) {
            return @{
                Mode = "executable"
                Command = $pythonCommand.Source
            }
        }
    }

    $candidatePaths = @(
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:ProgramFiles\Python312\python.exe",
        "$env:ProgramFiles\Python\Python312\python.exe",
        "$env:ProgramFiles(x86)\Python312\python.exe"
    )

    foreach ($candidatePath in $candidatePaths) {
        if (-not (Test-Path $candidatePath)) {
            continue
        }

        $minorVersion = Get-PythonMinorVersion -PythonCommand $candidatePath
        if ($minorVersion -eq $RequiredPythonMinor) {
            return @{
                Mode = "executable"
                Command = $candidatePath
            }
        }
    }

    return $null
}

function Install-Python312 {
    $wingetCommand = Get-Command winget -ErrorAction SilentlyContinue
    if ($null -ne $wingetCommand) {
        Write-Host "[Setup] Installing Python 3.12 using winget..."
        & $wingetCommand.Source install --id Python.Python.3.12 --exact --accept-package-agreements --accept-source-agreements --silent --disable-interactivity
        if ($LASTEXITCODE -eq 0) {
            return
        }

        Write-Warning "[Setup] winget install failed (exit code: $LASTEXITCODE). Falling back to python.org installer."
    } else {
        Write-Host "[Setup] winget not found. Falling back to python.org installer..."
    }

    $installerPath = Join-Path $env:TEMP "python-$PythonInstallerVersion-amd64.exe"
    Write-Host "[Setup] Downloading Python $PythonInstallerVersion from python.org..."
    Invoke-WebRequest -Uri $PythonInstallerUrl -OutFile $installerPath

    Write-Host "[Setup] Running Python installer..."
    & $installerPath /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_test=0
    if ($LASTEXITCODE -ne 0) {
        throw "Python installer failed with exit code $LASTEXITCODE"
    }
}

function Invoke-RequiredPython {
    param(
        [hashtable]$PythonSpec,
        [string[]]$Arguments,
        [string]$FailureMessage
    )

    if ($PythonSpec.Mode -eq "launcher") {
        & $PythonSpec.Command -3.12 @Arguments
    } else {
        & $PythonSpec.Command @Arguments
    }

    if ($LASTEXITCODE -ne 0) {
        throw $FailureMessage
    }
}

$pythonSpec = Get-Python312Spec
if ($null -eq $pythonSpec) {
    Write-Host "[Setup] Python 3.12 not found. Installing now..."
    Install-Python312
    $pythonSpec = Get-Python312Spec
}

if ($null -eq $pythonSpec) {
    throw "Python 3.12 installation did not succeed. Please install Python 3.12 manually and rerun setup.ps1."
}

$needsCreate = $RecreateVenv
if (-not (Test-Path $VenvPython)) {
    $needsCreate = $true
}

if (-not $needsCreate) {
    $existingMinor = Get-PythonMinorVersion -PythonCommand $VenvPython
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
    Invoke-RequiredPython -PythonSpec $pythonSpec -Arguments @("-m", "venv", ".venv") -FailureMessage "Failed to create virtual environment with Python 3.12"
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

$shouldRunApp = $RunApp -or (-not $SetupOnly)
if ($shouldRunApp) {
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
Write-Host "By default setup.ps1 runs the app automatically."
Write-Host "Use -SetupOnly if you want to skip launching the app."
