param([string]$Python = 'python', [switch]$SkipTraining)
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
Push-Location $PSScriptRoot
try {
    & $Python 'src/download_data.py'; if ($LASTEXITCODE -ne 0) { throw 'Data download or verification failed' }
    if (-not $SkipTraining) { & $Python 'src/experiment.py'; if ($LASTEXITCODE -ne 0) { throw 'Experiment failed' } }
    & $Python 'src/visualize.py'; if ($LASTEXITCODE -ne 0) { throw 'Visualization failed' }
    & $Python 'src/build_dashboard.py'; if ($LASTEXITCODE -ne 0) { throw 'Dashboard failed' }
    & $Python -m unittest discover -s tests -v; if ($LASTEXITCODE -ne 0) { throw 'Verification failed' }
    Write-Output 'Experiment, figures, dashboard and verification complete.'
} finally { Pop-Location }
