<#
.SYNOPSIS
    Installs MeetScribe on this computer. No admin rights, no developer tools.

.DESCRIPTION
    Written for deploying to other people's machines: copy this repository (or just
    this script plus the `dist\` folder holding the .whl) to the target computer and
    run it once.

    Program and data are kept apart on purpose:

        %LOCALAPPDATA%\Programs\MeetScribe\runtime   the private Python environment
        %LOCALAPPDATA%\Programs\MeetScribe\bin       the `meetscribe` command
        %LOCALAPPDATA%\MeetScribe                    config, database, models, audio

    Deleting recordings can therefore never break the program, and reinstalling the
    program never touches the recordings.

    The script is idempotent: running it again upgrades in place and leaves data
    alone.

.PARAMETER Uninstall
    Remove the program. Recordings and configuration are kept unless -Purge is
    also given.

.PARAMETER Purge
    With -Uninstall, also delete %LOCALAPPDATA%\MeetScribe -- every recording,
    transcript and setting. Asks for confirmation first.

.PARAMETER NoSetup
    Skip the guided configuration and the model download. Install only.

.PARAMETER NoShortcut
    Do not create the Start Menu entry.

.PARAMETER Wheel
    Explicit path to a .whl to install, instead of searching `dist\`.

.PARAMETER Prefix
    Install the program somewhere other than %LOCALAPPDATA%\Programs\MeetScribe.
    Recordings still go to the app's own data directory.

.PARAMETER NoPath
    Do not touch the user's PATH. For machines where PATH is managed centrally,
    and for testing this script.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install.ps1 -Uninstall
#>

[CmdletBinding()]
param(
    [switch]$Uninstall,
    [switch]$Purge,
    [switch]$NoSetup,
    [switch]$NoShortcut,
    [string]$Wheel,
    [string]$Prefix,
    [switch]$NoPath
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$AppName     = 'MeetScribe'
if ($Prefix) { $ProgramRoot = $Prefix }
else         { $ProgramRoot = Join-Path $env:LOCALAPPDATA "Programs\$AppName" }
$RuntimeDir  = Join-Path $ProgramRoot 'runtime'
$BinDir      = Join-Path $ProgramRoot 'bin'
# Mirrors meetscribe.security.paths.data_dir(): the same override the app honours,
# so what this script reports -- and what -Purge would delete -- is the directory
# actually in use rather than the default one.
if ($env:MEETSCRIBE_DATA_DIR) { $DataRoot = $env:MEETSCRIBE_DATA_DIR }
else                          { $DataRoot = Join-Path $env:LOCALAPPDATA $AppName }
$ShimPath    = Join-Path $BinDir 'meetscribe.cmd'
$ExePath     = Join-Path $RuntimeDir 'Scripts\meetscribe.exe'
$RepoRoot    = Split-Path -Parent $MyInvocation.MyCommand.Path

# ----------------------------------------------------------------------- output

function Write-Step  ($m) { Write-Host ''; Write-Host "==> $m" -ForegroundColor Cyan }
function Write-Ok    ($m) { Write-Host "  ok  " -ForegroundColor Green -NoNewline; Write-Host $m }
function Write-Warn  ($m) { Write-Host " aviso" -ForegroundColor Yellow -NoNewline; Write-Host " $m" }
function Write-Fail  ($m) { Write-Host " error" -ForegroundColor Red -NoNewline; Write-Host " $m" }

function Invoke-Native {
    <#
        Run an external tool and return its exit code.

        Not just `& $exe $args`. `uv` and `pip` report progress on stderr, and
        Windows PowerShell turns a native command's stderr into ErrorRecords
        whenever stderr is redirected -- which then trips $ErrorActionPreference =
        'Stop' and aborts the install *on its own progress output*. It only happens
        when someone pipes the installer to a log file or runs it from CI, which is
        exactly how an unattended deployment is run, so it has to be handled here
        rather than left as a caveat.

        Exit code is the only success signal that is actually reliable.
    #>
    param(
        [Parameter(Mandatory)][string]$Exe,
        [Parameter(ValueFromRemainingArguments)][string[]]$Arguments
    )
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        # Straight to the host, not down the pipeline: the caller wants the exit
        # code, and the user wants to read the tool's own output (a doctor report,
        # a download progress bar) rather than have it captured into a variable.
        & $Exe @Arguments | Out-Host
        return $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
}

function Assert-Native ($code, $what) {
    if ($code -ne 0) {
        Write-Fail "$what falló (código $code)."
        exit 1
    }
}

# ------------------------------------------------------------------ prerequisites

function Get-Installer {
    <#
        Returns how to build the environment. `uv` is preferred because it can
        provision its own CPython, which removes "install Python first" from the
        instructions given to a non-technical user. An existing Python 3.12+ is the
        fallback. Nothing is downloaded and executed implicitly -- if neither is
        present the script says exactly what to run and stops.
    #>
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if ($uv) {
        Write-Ok "uv encontrado ($($uv.Source))"
        return @{ Kind = 'uv'; Exe = $uv.Source }
    }

    foreach ($candidate in @('python', 'python3', 'py')) {
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        if (-not $cmd) { continue }
        $raw = & $cmd.Source -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if (-not $raw) { continue }
        $parts = $raw.Trim().Split('.')
        if ([int]$parts[0] -eq 3 -and [int]$parts[1] -ge 12) {
            Write-Ok "Python $($raw.Trim()) encontrado ($($cmd.Source))"
            return @{ Kind = 'python'; Exe = $cmd.Source }
        }
    }

    Write-Fail 'No se encontró ni uv ni Python 3.12 o superior.'
    Write-Host ''
    Write-Host '  Instala uno de los dos y vuelve a ejecutar este script:'
    Write-Host ''
    Write-Host '    uv     (recomendado, trae su propio Python):' -ForegroundColor Gray
    Write-Host '      winget install --id astral-sh.uv -e'
    Write-Host ''
    Write-Host '    Python 3.12:' -ForegroundColor Gray
    Write-Host '      winget install --id Python.Python.3.12 -e'
    Write-Host ''
    exit 1
}

function Find-Package {
    <#
        A wheel is preferred: it is what you copy to a machine that has no
        repository on it. Falling back to the source tree keeps the script usable
        straight from a clone.
    #>
    if ($Wheel) {
        if (-not (Test-Path $Wheel)) { Write-Fail "No existe '$Wheel'."; exit 1 }
        return (Resolve-Path $Wheel).Path
    }
    $dist = Join-Path $RepoRoot 'dist'
    if (Test-Path $dist) {
        $found = Get-ChildItem -Path $dist -Filter 'meetscribe-*.whl' |
                 Sort-Object LastWriteTime -Descending |
                 Select-Object -First 1
        if ($found) {
            # A wheel older than the sources beside it silently deploys yesterday's
            # application, which is the kind of mistake that is only noticed when a
            # command the user was told about turns out not to exist.
            $src = Join-Path $RepoRoot 'src'
            if (Test-Path $src) {
                $newest = Get-ChildItem -Path $src -Recurse -File |
                          Sort-Object LastWriteTime -Descending |
                          Select-Object -First 1
                if ($newest -and $newest.LastWriteTime -gt $found.LastWriteTime) {
                    Write-Warn "El paquete $($found.Name) es más antiguo que el código en src\."
                    Write-Host  "        Reconstruyelo con 'uv build' o pasa -Wheel <ruta>." -ForegroundColor Gray
                    Write-Host  "        Instalando desde el código fuente en su lugar." -ForegroundColor Gray
                    return $RepoRoot
                }
            }
            return $found.FullName
        }
    }
    if (Test-Path (Join-Path $RepoRoot 'pyproject.toml')) {
        return $RepoRoot
    }
    Write-Fail 'No se encontró ni un .whl en dist\ ni un pyproject.toml junto a este script.'
    exit 1
}

# ---------------------------------------------------------------------- install

function New-Runtime ($installer) {
    <#
        Idempotent on purpose: re-running the installer is the upgrade path, and on
        an already-installed machine that has to succeed. Both `uv venv` and
        `python -m venv` refuse a directory that already holds an environment, so
        an existing healthy one is reused and only the package is reinstalled. A
        broken one is replaced rather than repaired -- rebuilding costs a minute and
        diagnosing someone else's half-deleted venv costs an afternoon.
    #>
    Write-Step 'Preparando el entorno de Python'
    New-Item -ItemType Directory -Force -Path $ProgramRoot, $BinDir | Out-Null

    $venvPython = Join-Path $RuntimeDir 'Scripts\python.exe'
    if (Test-Path $venvPython) {
        if ((Invoke-Native $venvPython -c 'pass') -eq 0) {
            Write-Ok "Reutilizando el entorno existente en $RuntimeDir"
            return
        }
        Write-Warn 'El entorno existente no funciona; se reconstruye.'
        Remove-Item -Recurse -Force $RuntimeDir
    } elseif (Test-Path $RuntimeDir) {
        # A directory with no interpreter in it: a partial or interrupted install.
        Remove-Item -Recurse -Force $RuntimeDir
    }

    if ($installer.Kind -eq 'uv') {
        Assert-Native (Invoke-Native $installer.Exe venv --python 3.12 $RuntimeDir) 'uv venv'
    } else {
        Assert-Native (Invoke-Native $installer.Exe -m venv $RuntimeDir) 'python -m venv'
    }
    Write-Ok "Entorno en $RuntimeDir"
}

function Install-Package ($installer, $package) {
    Write-Step 'Instalando MeetScribe y sus dependencias'
    Write-Host '  Puede tardar unos minutos la primera vez (descarga ~300 MB).'
    $venvPython = Join-Path $RuntimeDir 'Scripts\python.exe'

    # meetscribe itself is always reinstalled, its dependencies are not. Both
    # installers treat "same version already present" as nothing to do, so a
    # rebuilt 1.0.0 wheel would otherwise deploy nothing at all -- the upgrade
    # would appear to succeed and change nothing. Dependencies are left to the
    # resolver, which is what keeps a re-run fast.
    if ($installer.Kind -eq 'uv') {
        $code = Invoke-Native $installer.Exe pip install --python $venvPython `
                    --reinstall-package meetscribe $package
    } else {
        Invoke-Native $venvPython -m pip install --upgrade pip --quiet | Out-Null
        $code = Invoke-Native $venvPython -m pip install $package
        if ($code -eq 0) {
            $code = Invoke-Native $venvPython -m pip install --force-reinstall --no-deps $package
        }
    }
    Assert-Native $code 'la instalación de paquetes'

    if (-not (Test-Path $ExePath)) {
        Write-Fail "Se instaló, pero no apareció $ExePath."
        exit 1
    }
    $version = & $ExePath --version
    Write-Ok "$version instalado"
}

function New-Shim {
    # A one-line .cmd rather than a PATH entry pointing into the venv: the venv can
    # be rebuilt on upgrade without the user's PATH ever changing.
    Write-Step 'Creando el comando "meetscribe"'
    $lines = @(
        '@echo off',
        'rem MeetScribe launcher -- generated by install.ps1. Do not edit.',
        ('"' + $ExePath + '" %*')
    )
    Set-Content -Path $ShimPath -Value $lines -Encoding ASCII
    Write-Ok $ShimPath

    if ($NoPath) {
        Write-Warn "PATH sin tocar (-NoPath). Ejecuta MeetScribe con: $ShimPath"
        if ($env:Path -notlike "*$BinDir*") { $env:Path = "$env:Path;$BinDir" }
        return
    }

    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if ($null -eq $userPath) { $userPath = '' }
    $entries = $userPath.Split(';') | Where-Object { $_ -ne '' }
    if ($entries -notcontains $BinDir) {
        $updated = (@($entries) + $BinDir) -join ';'
        [Environment]::SetEnvironmentVariable('Path', $updated, 'User')
        Write-Ok "$BinDir añadido al PATH del usuario"
        Write-Warn 'Abre una terminal nueva para que "meetscribe" esté disponible.'
    } else {
        Write-Ok 'Ya estaba en el PATH'
    }
    # Make it work in *this* session too, so the steps below can use it.
    if ($env:Path -notlike "*$BinDir*") { $env:Path = "$env:Path;$BinDir" }
}

function New-Shortcut {
    if ($NoShortcut) { return }
    Write-Step 'Creando el acceso directo'
    $menu = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'
    $link = Join-Path $menu 'MeetScribe.lnk'
    try {
        $shell = New-Object -ComObject WScript.Shell
        $sc = $shell.CreateShortcut($link)
        $sc.TargetPath = "$env:SystemRoot\System32\cmd.exe"
        # /k keeps the window open after the session ends, so the printed summary
        # and the export command stay readable instead of vanishing.
        $sc.Arguments = "/k `"$ShimPath`" start"
        $sc.WorkingDirectory = $DataRoot
        $sc.IconLocation = "$env:SystemRoot\System32\SndVol.exe,0"
        $sc.Description = 'Transcribe y resume la reunión en curso'
        $sc.Save()
        Write-Ok $link
    } catch {
        Write-Warn "No se pudo crear el acceso directo: $($_.Exception.Message)"
    }
}

function Invoke-Setup {
    if ($NoSetup) {
        Write-Warn 'Configuración omitida (-NoSetup). Ejecuta "meetscribe setup" cuando quieras.'
        return
    }
    Write-Step 'Configurando'
    if ((Invoke-Native $ExePath setup --defaults --con-modelos) -ne 0) {
        Write-Warn 'La configuración terminó con avisos. Revisa "meetscribe doctor".'
    }
}

function Invoke-Doctor {
    Write-Step 'Comprobando el equipo'
    # --sin-audio here on purpose: opening the real devices during an unattended
    # install would fail on a machine with nothing playing, which is not a fault.
    # A non-zero exit is likewise informational: doctor reports, it does not gate.
    Invoke-Native $ExePath doctor --sin-audio | Out-Null  # code ignored on purpose
    Write-Host ''
    Write-Host '  Para probar los dispositivos de audio de verdad, con la reunión abierta:'
    Write-Host '    meetscribe doctor' -ForegroundColor Gray
}

# -------------------------------------------------------------------- uninstall

function Remove-Installation {
    Write-Step "Desinstalando $AppName"

    if (Test-Path $ProgramRoot) {
        Remove-Item -Recurse -Force $ProgramRoot
        Write-Ok "Programa eliminado de $ProgramRoot"
    } else {
        Write-Warn "No había nada instalado en $ProgramRoot"
    }

    # Only rewrite PATH if this installation is actually in it. Writing the same
    # value back would still normalise the user's PATH as a side effect, and
    # nothing about uninstalling justifies touching entries we did not add.
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if ($userPath -and $userPath.Split(';') -contains $BinDir) {
        $kept = $userPath.Split(';') | Where-Object { $_ -ne '' -and $_ -ne $BinDir }
        [Environment]::SetEnvironmentVariable('Path', ($kept -join ';'), 'User')
        Write-Ok 'PATH limpiado'
    } else {
        Write-Ok 'PATH sin cambios (no estaba)'
    }

    $link = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\MeetScribe.lnk'
    if (Test-Path $link) { Remove-Item -Force $link; Write-Ok 'Acceso directo eliminado' }

    if ($Purge) {
        if (-not (Test-Path $DataRoot)) { Write-Ok 'No había datos que borrar'; return }
        Write-Host ''
        Write-Warn "-Purge borrará TODO en ${DataRoot}: grabaciones, transcripciones y actas."
        $answer = Read-Host '  Escribe BORRAR para confirmar'
        if ($answer -ceq 'BORRAR') {
            Remove-Item -Recurse -Force $DataRoot
            Write-Ok 'Datos eliminados'
        } else {
            Write-Ok 'Datos conservados'
        }
    } else {
        Write-Ok "Datos conservados en $DataRoot (usa -Purge para borrarlos)"
    }
}

# ------------------------------------------------------------------------- main

Write-Host ''
Write-Host "  $AppName - instalador" -ForegroundColor White

if ($Uninstall) {
    Remove-Installation
    Write-Host ''
    Write-Host '  Desinstalado.' -ForegroundColor Green
    Write-Host ''
    exit 0
}

$installer = Get-Installer
$package   = Find-Package
Write-Ok "Paquete: $package"

New-Runtime  $installer
Install-Package $installer $package
New-Shim
New-Shortcut
Invoke-Setup
Invoke-Doctor

Write-Host ''
Write-Host '  Instalado.' -ForegroundColor Green
Write-Host ''
Write-Host '  Empezar una reunión :  meetscribe start'
Write-Host '  Ver lo grabado      :  meetscribe sessions'
Write-Host '  Ajustar preferencias:  meetscribe setup'
Write-Host '  Ayuda               :  meetscribe --help'
Write-Host ''
