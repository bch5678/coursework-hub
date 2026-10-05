<#
.SYNOPSIS
Unpack the BN5212 course archives into the layout the data pipeline scans.

.DESCRIPTION
The course ships MIMIC-CXR as one large zip whose entries carry a packaging
prefix, and MIMIC-IV as a zip holding ~75 GB of tables of which this project
reads only seven. This script extracts just what is needed, flattens the CXR
path, and verifies the result. It is idempotent: finished files are skipped, so
a re-run after an interruption resumes rather than restarts.

Deleting the redundant CXR archives is opt-in via -RemoveDuplicateArchives, and
only runs after the extraction has been verified.

.EXAMPLE
.\scripts\prepare_local_data.ps1
.\scripts\prepare_local_data.ps1 -RemoveDuplicateArchives
#>
param(
    [string]$DataRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) "data"),
    [string]$CxrArchive = "MIMIC-CXR\BN5212_MIMIC-CXR.zip",
    [string]$MimicArchive = "MIMIC-IV\MIMIC_IV.zip",
    [switch]$RemoveDuplicateArchives,
    [switch]$SkipMimicIv
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Write-Step($text) { Write-Host "`n=== $text ===" -ForegroundColor Cyan }
function Free-GB { [math]::Round((Get-PSDrive C).Free / 1GB, 1) }

$cxrZip = Join-Path $DataRoot $CxrArchive
$mimicZip = Join-Path $DataRoot $MimicArchive
$cxrRoot = Join-Path $DataRoot "mimic-cxr\2.1.0"
$mimicRoot = Join-Path $DataRoot "mimiciv"

if (-not (Test-Path $cxrZip)) { throw "CXR archive not found: $cxrZip" }

# ------------------------------------------------- repair a doubled CXR path
# An earlier extraction could leave files/mimic-cxr/<ver>/files/pXX/... instead
# of files/pXX/... . Flatten it with a same-volume rename rather than paying for
# a re-extraction of tens of gigabytes.
$nested = Get-ChildItem (Join-Path $cxrRoot "files") -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -notmatch '^p\d{2}$' } | Select-Object -First 1
if ($nested) {
    Write-Step "Repairing doubled CXR path"
    $inner = Get-ChildItem $nested.FullName -Recurse -Directory -Filter "files" -ErrorAction SilentlyContinue |
        Where-Object { (Get-ChildItem $_.FullName -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^p\d{2}$' }) } |
        Select-Object -First 1
    if ($inner) {
        $staging = Join-Path $cxrRoot "files_flat"
        Move-Item -LiteralPath $inner.FullName -Destination $staging
        Remove-Item -LiteralPath (Join-Path $cxrRoot "files") -Recurse -Force
        Move-Item -LiteralPath $staging -Destination (Join-Path $cxrRoot "files")
        Write-Host "flattened" -ForegroundColor Green
    }
}

# ---------------------------------------------------------------- CXR images
Write-Step "Extracting chest radiographs (DICOM only)"
$zip = [System.IO.Compression.ZipFile]::OpenRead($cxrZip)
$entries = $zip.Entries | Where-Object { $_.FullName -like "*.dcm" }
$expected = $entries.Count
Write-Host "archive holds $expected DICOM files"

$done = 0
foreach ($e in $entries) {
    # Keep the tail of the path from the LAST "files/" segment, which is the one
    # that starts the pXX/p<subject>/s<study> tree; earlier "files/" segments
    # belong to the packaging prefix.
    if ($e.FullName -match '.*/(files/p\d{2}/.+\.dcm)$') {
        $relative = $Matches[1]
    } elseif ($e.FullName -match '.*/(p\d{2}/p\d+/s\d+/[^/]+\.dcm)$') {
        $relative = "files/" + $Matches[1]
    } else {
        Write-Warning "unexpected entry, skipped: $($e.FullName)"
        continue
    }
    $out = Join-Path $cxrRoot $relative
    if ((Test-Path $out) -and (Get-Item $out).Length -eq $e.Length) { $done++; continue }
    $dir = Split-Path $out -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    [System.IO.Compression.ZipFileExtensions]::ExtractToFile($e, $out, $true)
    $done++
    if ($done % 500 -eq 0) { Write-Host ("  {0,5}/{1}   free {2} GB" -f $done, $expected, (Free-GB)) }
}
$zip.Dispose()

$actual = (Get-ChildItem $cxrRoot -Recurse -Filter *.dcm -File -ErrorAction SilentlyContinue).Count
Write-Host "on disk: $actual .dcm"
if ($actual -ne $expected) { throw "Expected $expected DICOM files, found $actual" }

$sample = Get-ChildItem $cxrRoot -Recurse -Filter *.dcm -File | Select-Object -First 1
$relative = $sample.FullName.Substring($cxrRoot.Length + 1)
if ($relative -notmatch '^files\\p\d{2}\\p\d+\\s\d+\\[^\\]+\.dcm$') {
    throw "Layout is not files/pXX/p<subject>/s<study>/<dicom>.dcm : $relative"
}
Write-Host "layout OK -> $relative" -ForegroundColor Green
Write-Host "image_root for the pipeline config: $cxrRoot"

# ------------------------------------------------- optional archive clean-up
# Runs before the MIMIC-IV extraction: those tables need ~45 GB, and freeing the
# redundant archives first keeps the disk from running close to full.
if ($RemoveDuplicateArchives) {
    Write-Step "Removing redundant CXR archives"
    # These repackage the same DICOM files without the pXX level, so the pipeline
    # cannot scan them directly. Safe only now that the extraction is verified.
    $dupes = @("dataset.zip", "data_subset1.zip", "data_subset2.zip",
               "data_subset3.zip", "data_subset4.zip")
    $freed = 0
    foreach ($d in $dupes) {
        $p = Join-Path $DataRoot "MIMIC-CXR\$d"
        if (Test-Path $p) {
            $freed += (Get-Item $p).Length
            Remove-Item -LiteralPath $p -Force
            Write-Host "  deleted $d"
        }
    }
    Write-Host ("  freed {0:N1} GB, free space now {1} GB" -f ($freed / 1GB), (Free-GB)) -ForegroundColor Green
}

# ------------------------------------------------------------ MIMIC-IV tables
if (-not $SkipMimicIv) {
    Write-Step "Extracting the MIMIC-IV tables this project reads"
    if (-not (Test-Path $mimicZip)) { throw "MIMIC-IV archive not found: $mimicZip" }
    # admissions/patients build the cohort; d_items/d_labitems resolve itemids;
    # icustays supports the ICU-level cohort option; chartevents/labevents/omr
    # feed the clinical branch. Everything else stays packed.
    $want = @(
        "hosp/admissions.csv", "hosp/patients.csv", "hosp/d_labitems.csv",
        "icu/d_items.csv", "icu/icustays.csv",
        "icu/chartevents.csv", "hosp/labevents.csv", "hosp/omr.csv"
    )
    $zip = [System.IO.Compression.ZipFile]::OpenRead($mimicZip)
    foreach ($w in $want) {
        $entry = $zip.Entries | Where-Object { $_.FullName -like "*$w" } | Select-Object -First 1
        if (-not $entry) { Write-Warning "not in archive: $w"; continue }
        $out = Join-Path $mimicRoot $w
        if ((Test-Path $out) -and (Get-Item $out).Length -eq $entry.Length) {
            Write-Host ("  skip {0,-24} already complete" -f $w); continue
        }
        $dir = Split-Path $out -Parent
        if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
        Write-Host ("  {0,-24} {1,6:N1} GB ..." -f $w, ($entry.Length / 1GB)) -NoNewline
        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $out, $true)
        if ((Get-Item $out).Length -ne $entry.Length) { throw "Size mismatch for $w" }
        Write-Host (" done, free {0} GB" -f (Free-GB))
    }
    $zip.Dispose()
    Write-Host "mimic_iv_root for the pipeline config: $mimicRoot"
}

Write-Step "Ready"
Write-Host ("free space: {0} GB" -f (Free-GB))
Write-Host "Next: run the pipeline with config/server.local.json in bn5212-data-pipeline."
