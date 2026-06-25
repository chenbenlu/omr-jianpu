#!/usr/bin/env python3
"""把 LaTeX 專案上傳到遠端編譯 API（latex-compiler）取回 PDF。"""

from __future__ import annotations

import argparse
import fnmatch
import io
import json
import os
import re
import sys
import urllib.error
import urllib.request
import uuid
import zipfile

# 強制 UTF-8 輸出：LaTeX Workshop 以管線擷取 stdout 並用 UTF-8 解碼，
# 但 Windows 上 Python 預設用本地碼頁（cp950），不轉會在 Build 面板顯示亂碼。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001  pythonw 等無 stream 時略過
        pass

EXCLUDE_FILE_GLOBS = [
    "*.pdf",
    "*.aux",
    "*.bbl",
    "*.blg",
    "*.log",
    "*.out",
    "*.fls",
    "*.fdb_latexmk",
    "*.synctex.gz",
    "*.toc",
    "*.lof",
    "*.lot",
    "*.nav",
    "*.snm",
    "*.vrb",
]
EXCLUDE_DIR_NAMES = {".git", "out", ".vscode", "__pycache__"}
VALID_ENGINES = ("pdflatex", "xelatex", "lualatex")
TIMEOUT_SEC = 180


def log(msg: str) -> None:
    print(f"[latex-remote] {msg}", flush=True)


def die(msg: str):
    log(msg)
    sys.exit(1)


def resolve_api_url(cli_url, script_dir: str) -> str:
    """取得服務位址="""
    if cli_url:
        return cli_url.rstrip("/")
    env = os.environ.get("LATEX_REMOTE_API_URL")
    if env:
        return env.rstrip("/")
    cfg = os.path.join(script_dir, "latex-remote.config.json")
    if os.path.isfile(cfg):
        try:
            with open(cfg, "r", encoding="utf-8") as fh:
                url = (json.load(fh) or {}).get("api_url", "")
        except Exception as exc:  # noqa: BLE001
            die(f"讀取 latex-remote.config.json 失敗：{exc}")
        if url:
            return url.rstrip("/")
    die(
        "找不到編譯服務位址：請設定環境變數 LATEX_REMOTE_API_URL，或在 "
        "tools/latex-remote.config.json 填入 api_url"
        "（可複製 latex-remote.config.example.json 後修改）。"
    )


def detect_engine(root_tex: str) -> str:
    """讀主檔前 30 行的 magic comment 決定引擎；找不到回 pdflatex。

    支援 "% !TEX program = xelatex" / "% !TEX TS-program = lualatex"（大小寫寬鬆）。
    """
    pat = re.compile(r"!TEX\s+(?:TS-)?program\s*=\s*([A-Za-z]+)", re.IGNORECASE)
    try:
        with open(root_tex, "r", encoding="utf-8", errors="ignore") as fh:
            for _ in range(30):
                line = fh.readline()
                if not line:
                    break
                m = pat.search(line)
                if m and m.group(1).lower() in VALID_ENGINES:
                    return m.group(1).lower()
    except OSError:
        pass
    return "pdflatex"


def build_zip(root_dir: str) -> bytes:
    """把 root_dir 打包成 zip（排除雜訊，保留 figures/ 等子目錄；entry 用正斜線）。"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames[:] = [
                d
                for d in dirnames
                if d not in EXCLUDE_DIR_NAMES and not fnmatch.fnmatch(d, "_minted*")
            ]
            for name in filenames:
                if any(fnmatch.fnmatch(name, g) for g in EXCLUDE_FILE_GLOBS):
                    continue
                full = os.path.join(dirpath, name)
                arcname = os.path.relpath(full, root_dir).replace(os.sep, "/")
                zf.write(full, arcname)
    return buf.getvalue()


def build_multipart(fields: dict, zip_bytes: bytes) -> "tuple[bytes, str]":
    """手工組 multipart/form-data：文字欄位 + 一個 zip 檔案欄位（name=file）。"""
    boundary = uuid.uuid4().hex
    crlf = b"\r\n"
    out = bytearray()
    for key, val in fields.items():
        out += b"--" + boundary.encode() + crlf
        out += f'Content-Disposition: form-data; name="{key}"'.encode() + crlf + crlf
        out += str(val).encode("utf-8") + crlf
    out += b"--" + boundary.encode() + crlf
    out += b'Content-Disposition: form-data; name="file"; filename="project.zip"' + crlf
    out += b"Content-Type: application/zip" + crlf + crlf
    out += zip_bytes + crlf
    out += b"--" + boundary.encode() + b"--" + crlf
    return bytes(out), boundary


def main() -> int:
    ap = argparse.ArgumentParser(description="Remote LaTeX compile client")
    ap.add_argument(
        "--root", required=True, help="主檔完整路徑（LaTeX Workshop 的 %DOC_EXT%）"
    )
    ap.add_argument(
        "--out", default="", help="PDF 輸出目錄（%OUTDIR%）；空或 . 則用主檔目錄"
    )
    ap.add_argument(
        "--engine",
        default="auto",
        help="auto|pdflatex|xelatex|lualatex（預設 auto，依 magic comment）",
    )
    ap.add_argument("--api-url", default="", help="覆寫服務位址（一般不用，走設定檔）")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_tex = os.path.abspath(args.root)
    if not os.path.isfile(root_tex):
        die(f"找不到主檔：{root_tex}")
    root_dir = os.path.dirname(root_tex)
    entry = os.path.basename(root_tex)
    stem = os.path.splitext(entry)[0]

    out_dir = os.path.abspath(args.out) if args.out and args.out != "." else root_dir
    os.makedirs(out_dir, exist_ok=True)
    pdf_out = os.path.join(out_dir, stem + ".pdf")

    engine = args.engine.lower()
    if engine == "auto":
        engine = detect_engine(root_tex)
    if engine not in VALID_ENGINES:
        die(f"不支援的 engine：{engine}（可用 auto, {', '.join(VALID_ENGINES)}）")

    api_url = resolve_api_url(args.api_url or None, script_dir)
    log(f"主檔={entry}  引擎={engine}  目標=已設定的編譯服務")

    zip_bytes = build_zip(root_dir)
    body, boundary = build_multipart({"engine": engine, "entry": entry}, zip_bytes)
    req = urllib.request.Request(
        api_url + "/compile",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
            pdf = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        log(f"編譯失敗（HTTP {exc.code}）— latexmk log 尾段：")
        print(detail)
        return 1
    except urllib.error.URLError as exc:
        die(f"連線失敗：{exc.reason}（請確認已連內網/VPN，且設定檔中的服務位址可達）")

    with open(pdf_out, "wb") as fh:
        fh.write(pdf)
    log(f"編譯成功 -> {pdf_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
