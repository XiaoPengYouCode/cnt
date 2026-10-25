#!/usr/bin/env bash
# 离线优先：保留 data/raw/libime 原始词库，缺失时下载固定版本并校验。
# 只构建数据，不安装/注册/启动输入法。需要先有 data/lm.cntl。
set -euo pipefail
cd "$(dirname "$0")/.."
raw=data/raw/libime
archive=dict-20260703.tar.zst
checksum=c686cab6df8964c48d596f57d205bac31fc72870b06a83017e44503df8c09697
mkdir -p "$raw"
if [[ ! -f "$raw/dict_sc.txt" ]]; then
    curl -fL "https://download.fcitx-im.org/data/$archive" -o "$raw/$archive.part"
    printf '%s  %s\n' "$checksum" "$raw/$archive.part" | sha256sum -c -
    mv "$raw/$archive.part" "$raw/$archive"
    tar --zstd -xf "$raw/$archive" -C "$raw" dict_sc.txt dict_extb.txt
fi
if [[ ! -f data/lm.cntl ]]; then
    echo '缺少 data/lm.cntl，请先用 build-lm 生成语言模型。' >&2
    exit 1
fi
cargo build --release -p cnt-dict-tools
stage=$(mktemp -d data/.dict-build.XXXXXX)
trap 'rm -rf "$stage"' EXIT
target/release/cnt-dict-tools import-libime "$raw/dict_sc.txt" "$stage/wordlist.tsv" --lm data/lm.cntl
if [[ -f data/wordlist-domain.tsv ]]; then
    cat data/wordlist-domain.tsv >> "$stage/wordlist.tsv"
fi
target/release/cnt-dict-tools build "$stage/wordlist.tsv" "$stage/cnt.dict"
mv "$stage/wordlist.tsv" data/wordlist.tsv
mv "$stage/cnt.dict" data/cnt.dict
printf '词库已重建：data/wordlist.tsv → data/cnt.dict（未安装到系统）\n'
