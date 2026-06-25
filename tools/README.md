# tools/ — LaTeX 編譯輔助

`latex-remote.py` 讓 VS Code **LaTeX Workshop** 把 `docs/proposal/` 的論文上傳到遠端
**LaTeX 編譯 API** 編成 PDF，免在本機裝 TeX Live。純 Python 標準函式庫，無第三方相依。

`.vscode/settings.json` 已設好兩個 recipe，按 VS Code 左側 **TeX → Build LaTeX project**
（或 `Ctrl+Alt+B`）即可：

| Recipe | 用途 | 需求 |
|--------|------|------|
| **遠端編譯 (pdflatex)** | 打包上傳編譯服務 `/compile`，取回 PDF | 連上內網或 VPN，並設好服務位址（見下） |
| **latexmk (本機備援)** | 純本機編譯（也才有 SyncTeX 跳轉） | 本機已裝 TeX Live（含 `latexmk`） |

- `autoBuild` 設為 **never**：只在你主動按 Build 時編譯，存檔不會自動觸發。
- 第一次按 Build 用清單第一個 recipe（遠端）；之後記住上次選的（`recipe.default: lastUsed`）。
  沒 VPN 的人改選「latexmk (本機備援)」一次即可，VS Code 會沿用。
- 切換 recipe：命令面板 → **LaTeX Workshop: Build with recipe**。

## 設定服務位址（一次）

服務位址**不寫進任何會 commit 的檔案**。擇一設定（腳本依此順序解析）：

1. 複製 `latex-remote.config.example.json` 為 **`latex-remote.config.json`**，把 `api_url`
   改成你的內網/VPN 編譯服務位址。此檔已列入 `.gitignore`，不會被提交。
2. 或設環境變數 `LATEX_REMOTE_API_URL`。

## 怎麼運作

`latex-remote.py`（被 LaTeX Workshop 當自訂 tool 呼叫，`command: python`）：

1. 把主檔所在目錄（`docs/proposal/`）整包壓成 zip，排除 `*.pdf/*.aux/*.log/...`、`.git`、`out`，
   保留 `figures/`、`reference.bib`、`neurips_2020.sty`（zip entry 以正斜線寫入）。
2. `POST <api_url>/compile`（`file=zip`、`engine`、`entry=proposal.tex`）。
3. 200 → 寫出 `docs/proposal/proposal.pdf`；422 → 在 Build 輸出印出 latexmk log 尾段以定位錯誤。

**引擎**：`--engine auto` 會讀主檔前 30 行的 `% !TEX program = xelatex`（或 `TS-program`）決定，
找不到則 pdflatex。本論文 recipe 直接固定 `--engine pdflatex`（neurips_2020 + `inputenc/fontenc`）。

## 限制 / 注意

- **需內網/VPN**：服務多半僅對內網可見；先確認服務位址可達。
- **無 SyncTeX**：遠端不回傳 `.synctex.gz`，PDF↔原始碼正/反向跳轉不可用（本機 recipe 才有）。
- **需要 `python` 在 PATH**：recipe 用 `command: python`。若你的 VS Code 環境沒有，改成可用的直譯器。
