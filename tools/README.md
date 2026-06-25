# tools/ — LaTeX 編譯輔助

本資料夾的 `latex-remote.ps1` 讓 VS Code **LaTeX Workshop** 把 `docs/proposal/` 的論文
上傳到遠端 **LaTeX 編譯 API** 編成 PDF，免在本機裝 TeX Live。

`.vscode/settings.json` 已預先設好兩個 recipe，按 VS Code 左側 **TeX → Build LaTeX project**
（或 `Ctrl+Alt+B`）即可：

| Recipe | 用途 | 需求 |
|--------|------|------|
| **遠端編譯 (pdflatex)** | 打包上傳編譯服務 `/compile`，取回 PDF | 連上內網或 VPN，並配置編譯服務網址 |
| **latexmk (本機備援)** | 純本機編譯 | 本機已裝 TeX Live（含 `latexmk`） |

- `autoBuild` 設為 **never**：只在你主動按 Build 時編譯，存檔不會自動觸發。
- 第一次按 Build 會用清單第一個 recipe（遠端）；之後記住你上次選的（`recipe.default: lastUsed`）。
  沒 VPN 的人改選「latexmk (本機備援)」一次即可，VS Code 會沿用。
- 切換 recipe：命令面板 → **LaTeX Workshop: Build with recipe**。

## 遠端編譯怎麼運作

`latex-remote.ps1`（被 LaTeX Workshop 當自訂 tool 呼叫）：

1. 把主檔所在目錄（`docs/proposal/`）整包壓成 zip，排除 `*.pdf/*.aux/*.log/...` 與 `.git`、`out`，
   保留 `figures/`、`reference.bib`、`neurips_2020.sty`。
2. `POST .../compile`（`file=zip`、`engine=pdflatex`、`entry=proposal.tex`）。
3. 200 → 寫出 `docs/proposal/proposal.pdf`；422 → 在 Build 輸出印出 latexmk log 尾段定位錯誤。

> zip entry 一律用正斜線：.NET Framework 的 `ZipFile.CreateFromDirectory` 會用反斜線，
> Linux 端解壓會把 `figures\x.png` 當成單一檔名而非子目錄，導致圖檔找不到——腳本已手動修正。

## 限制 / 注意

- **需內網/VPN**：服務可能僅對內網可見，請先確認編譯服務網址可達。可先測 `curl https://<compiler-host>/healthz`。
- **無 SyncTeX**：遠端不回傳 `.synctex.gz`，故 PDF↔原始碼的正/反向跳轉不可用（本機 recipe 才有）。
- **引擎**：論文用 `pdflatex`（neurips_2020 + `inputenc/fontenc`）。要 `xelatex/lualatex` 改 tool 的 `-Engine`。
- 服務端單次編譯逾時 120s；本腳本上傳 + 等待逾時 180s。
- 腳本須存成 **UTF-8 with BOM**，否則 Windows PowerShell 5.1 會以 Big5 解碼中文而 parse 失敗。

服務本身（部署、API、CI）請參考你組織內的部署說明。
