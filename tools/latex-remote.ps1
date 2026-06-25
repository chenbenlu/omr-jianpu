<#
.SYNOPSIS
    把 LaTeX 專案上傳到遠端 LaTeX 編譯 API 取回 PDF。
    設計成可被 VS Code LaTeX Workshop 當作「自訂 tool」呼叫（見同目錄 README.md）。

.DESCRIPTION
    流程：
      1. 把「主檔所在目錄」整包壓成 zip（排除 aux / 輸出 / 版控等雜訊，保留 figures/、.bib、.sty）。
      2. POST 到 <ApiUrl>/compile  (multipart: file=zip, engine, entry=主檔名)。
      3. HTTP 200 → 把回傳的 PDF 寫到 <OutDir>\<主檔名>.pdf，exit 0。
         HTTP 422 → 印出 latexmk log 尾段（純文字），exit 1（LaTeX Workshop 會標記失敗）。

    單檔專案也適用（zip 內只有一個 .tex）。遠端用 latexmk，會自動跑 bibtex/多趟編譯。

    注意：zip entry 一律用正斜線。.NET Framework 的 ZipFile.CreateFromDirectory 會寫成反斜線，
    Linux 端 Python zipfile 會把 "figures\x.png" 當成「含反斜線的單一檔名」而非子目錄，
    使伺服器找不到 figures/ 圖檔——故此處手動建立 entry 並強制正斜線。

.PARAMETER RootTex
    LaTeX 主檔完整路徑。LaTeX Workshop 對應的 placeholder 是 %DOC_EXT%。

.PARAMETER Engine
    auto | pdflatex | xelatex | lualatex（預設 auto）。
    auto＝讀主檔前 30 行的 magic comment「% !TEX program = ...」決定，找不到則 pdflatex。

.PARAMETER OutDir
    PDF 輸出目錄。LaTeX Workshop 對應 %OUTDIR%；留空或為 '.' 時改用主檔所在目錄。

.PARAMETER ApiUrl
    編譯服務根網址（預設 http://your-latex-compiler.example.com，需在內網或 VPN 上）。

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File latex-remote.ps1 `
        -RootTex "d:\DL_final_project\docs\proposal\proposal.tex"
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $RootTex,
    [string] $Engine = 'auto',
    [string] $OutDir = '',
    [string] $ApiUrl = 'http://your-latex-compiler.example.com'
)

$ErrorActionPreference = 'Stop'

# 讓中文訊息在 LaTeX Workshop 輸出頻道 / 終端機以 UTF-8 顯示（無 console 時忽略）
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
} catch { }

function Fail([string] $msg) {
    Write-Host "[latex-remote] $msg" -ForegroundColor Red
    exit 1
}

function Resolve-Engine([string] $texPath) {
    # 從主檔前 30 行的 magic comment 解析引擎；找不到則回 pdflatex。
    # 支援：% !TEX program = xelatex / % !TEX TS-program = lualatex（大小寫、空白寬鬆）
    try { $head = Get-Content -LiteralPath $texPath -TotalCount 30 -ErrorAction Stop } catch { return 'pdflatex' }
    foreach ($line in $head) {
        if ($line -match '(?i)!TEX\s+(?:TS-)?program\s*=\s*([A-Za-z]+)') {
            $p = $Matches[1].ToLower()
            if (@('pdflatex', 'xelatex', 'lualatex') -contains $p) { return $p }
        }
    }
    return 'pdflatex'
}

# --- 解析路徑 ------------------------------------------------------------
if (-not (Test-Path -LiteralPath $RootTex)) { Fail "找不到主檔：$RootTex" }
$root      = Get-Item -LiteralPath $RootTex
$rootDir   = $root.DirectoryName
$entryName = $root.Name                                              # 例：proposal.tex
$pdfName   = [IO.Path]::GetFileNameWithoutExtension($entryName) + '.pdf'

if ([string]::IsNullOrWhiteSpace($OutDir) -or $OutDir -eq '.') { $OutDir = $rootDir }
if (-not (Test-Path -LiteralPath $OutDir)) {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
}
$pdfOut = Join-Path $OutDir $pdfName

$validEngines = @('pdflatex', 'xelatex', 'lualatex')
if ($Engine -eq 'auto') { $Engine = Resolve-Engine $RootTex }
if ($validEngines -notcontains $Engine) { Fail "不支援的 engine：$Engine（可用 auto, $($validEngines -join ', ')）" }

Write-Host "[latex-remote] 主檔=$entryName  引擎=$Engine  API=$ApiUrl"

# --- 1) 打包專案目錄到暫存 zip（排除雜訊，保留子目錄結構，強制正斜線）-------
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$excludeFilePatterns = @('*.pdf','*.aux','*.bbl','*.blg','*.log','*.out','*.fls',
                         '*.fdb_latexmk','*.synctex.gz','*.toc','*.lof','*.lot',
                         '*.nav','*.snm','*.vrb')
$excludeDirNames     = @('.git','out','.vscode','_minted*')
$baseLen = $rootDir.TrimEnd('\','/').Length + 1

$files = Get-ChildItem -LiteralPath $rootDir -Recurse -File | Where-Object {
    $rel  = $_.FullName.Substring($baseLen)
    $segs = $rel -split '[\\/]'
    for ($i = 0; $i -lt $segs.Length - 1; $i++) {        # 父層目錄落在排除清單則跳過
        foreach ($d in $excludeDirNames) { if ($segs[$i] -like $d) { return $false } }
    }
    foreach ($p in $excludeFilePatterns) { if ($_.Name -like $p) { return $false } }
    return $true
}
if (-not $files) { Fail "主檔目錄沒有可上傳的檔案：$rootDir" }

$zip  = Join-Path $env:TEMP ("latexremote_" + [guid]::NewGuid().ToString('N') + '.zip')
$body = Join-Path $env:TEMP ("latexremote_resp_" + [guid]::NewGuid().ToString('N') + '.bin')

try {
    $archive = [IO.Compression.ZipFile]::Open($zip, [IO.Compression.ZipArchiveMode]::Create)
    try {
        foreach ($f in $files) {
            $entry = $f.FullName.Substring($baseLen) -replace '\\', '/'   # 強制正斜線
            [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive, $f.FullName, $entry) | Out-Null
        }
    }
    finally { $archive.Dispose() }

    # --- 2) 上傳編譯 -----------------------------------------------------
    $code = & curl.exe -sS -m 180 -X POST `
        -F "file=@$zip;type=application/zip" `
        -F "engine=$Engine" `
        -F "entry=$entryName" `
        -o "$body" -w "%{http_code}" `
        "$ApiUrl/compile"
    $curlExit = $LASTEXITCODE

    if ($curlExit -ne 0) {
        Fail "連線失敗（curl exit=$curlExit）：請確認已連 VPN/內網，且 $ApiUrl 可達（curl $ApiUrl/healthz）"
    }

    if ($code -eq '200') {
        Move-Item -LiteralPath $body -Destination $pdfOut -Force
        Write-Host "[latex-remote] 編譯成功 -> $pdfOut" -ForegroundColor Green
        exit 0
    }
    else {
        Write-Host "[latex-remote] 編譯失敗（HTTP $code）— latexmk log 尾段：" -ForegroundColor Red
        if (Test-Path -LiteralPath $body) { Get-Content -LiteralPath $body -Raw -Encoding UTF8 }
        exit 1
    }
}
finally {
    Remove-Item -LiteralPath $zip  -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $body -Force -ErrorAction SilentlyContinue
}
