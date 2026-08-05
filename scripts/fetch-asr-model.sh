#!/usr/bin/env bash
# 下载语音输入模型（Fun-ASR-Nano 的 CTC 导出，ONNX / int8，约 188 MB）
#
# 模型不入库、也不随程序分发（体积 + 许可），按 data/ 的规矩独立拉取：
#   data/asr/model.int8.onnx   声学模型（encoder + CTC 头）
#   data/asr/tokens.txt        token 表（base64 编码的字节级 BPE）
#
# 为什么是这个模型（2026-08 系统筛过一轮的结论）：
#
#   条件：本地 CPU RTF ≤ 0.3 / 常驻 ≤ 500 MB / **词表必须有中文标点与数字** /
#         中文质量尽量高 / ONNX 单次前向。
#
#   Fun-ASR-Nano（通义，2025-12，Apache-2.0，0.8B）在**工业**测试集上是开源第一：
#   平均 WER 16.72，优于 FireRedASR2(22.63) / Paraformer v2(23.49) /
#   GLM-ASR-Nano(26.13) / Whisper-large-v3(33.39)，逼近闭源 Seed-ASR(15.95)；
#   方言 28.18 对 FireRedASR2 的 52.82。本文件是它的 encoder+CTC 分支导出，
#   200M 量级、单次前向，词表带 `，。？！` 与 `0-9`。
#
#   落选的：FireRedASR2 全家词表无标点无数字（8667 token，实测确认）；
#   Qwen3-ASR-0.6B / GLM-ASR-Nano / funasr-nano 全量版都是自回归且 >800 MB。
#
# 用法：
#   bash scripts/fetch-asr-model.sh                      # → data/asr + data/punct
#   bash scripts/fetch-asr-model.sh <声学目录> <标点目录>
#
# 调试用（不是发布路径）：想换成 2024 一代的 SenseVoice-Small 做对照，
# 把 NAME 换成 sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09。

set -euo pipefail
cd "$(dirname "$0")/.."

DEST="${1:-data/asr}"
NAME="sherpa-onnx-sense-voice-funasr-nano-int8-2025-12-17"
SIZE="188 MB"
URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${NAME}.tar.bz2"

# 标点模型（CT-Transformer / FunASR ct-punc，int8 65 MB）。
# 声学模型的 CTC 头输出是光板汉字流（实测：两句拼接后中间没有任何标点），
# 而上屏文本必须有句读，所以标点是**独立的一段推理**（见 cnt-asr 的 Punctuator 端口）。
PUNCT_DEST="${2:-data/punct}"
PUNCT_NAME="sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8"
PUNCT_SIZE="65 MB"
PUNCT_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/${PUNCT_NAME}.tar.bz2"

_C_RST='\033[0m'; _C_BLD='\033[1m'; _C_DIM='\033[2m'; _C_CYN='\033[36m'; _C_GRN='\033[32m'
step() { printf "${_C_BLD}${_C_CYN}==>${_C_RST} ${_C_BLD}%s${_C_RST}\n" "$*"; }
info() { printf "  ${_C_DIM}%s${_C_RST}\n" "$*"; }
ok()   { printf "  ${_C_GRN}✔${_C_RST} %s\n" "$*"; }

TMP=$(mktemp -d)
trap 'rm -rf "${TMP}"' EXIT

# ---- 声学模型 ----
if [ -f "${DEST}/tokens.txt" ] && ls "${DEST}"/model*.onnx >/dev/null 2>&1; then
    ok "声学模型已存在：${DEST}（要换模型请先删掉这个目录）"
else
    step "下载声学模型（约 ${SIZE}，来源 sherpa-onnx）"
    info "${URL}"
    curl -L --progress-bar -o "${TMP}/model.tar.bz2" "${URL}"
    step "解压"
    tar -xjf "${TMP}/model.tar.bz2" -C "${TMP}"
    step "安装到 ${DEST}"
    mkdir -p "${DEST}"
    cp "${TMP}/${NAME}"/model*.onnx "${DEST}/"
    cp "${TMP}/${NAME}/tokens.txt" "${DEST}/"
    # 自带的测试音频（zh/en/ja/ko/yue）留着：离线验证不用自己录
    if [ -d "${TMP}/${NAME}/test_wavs" ]; then
        cp -r "${TMP}/${NAME}/test_wavs" "${DEST}/"
    fi
    ok "$(du -sh "${DEST}" | cut -f1) → ${DEST}"
fi

# ---- 标点模型 ----
if [ -f "${PUNCT_DEST}/model.int8.onnx" ] || [ -f "${PUNCT_DEST}/model.onnx" ]; then
    ok "标点模型已存在：${PUNCT_DEST}"
else
    step "下载标点模型（约 ${PUNCT_SIZE}）"
    info "${PUNCT_URL}"
    curl -L --progress-bar -o "${TMP}/punct.tar.bz2" "${PUNCT_URL}"
    tar -xjf "${TMP}/punct.tar.bz2" -C "${TMP}"
    mkdir -p "${PUNCT_DEST}"
    cp "${TMP}/${PUNCT_NAME}"/model*.onnx "${PUNCT_DEST}/"
    ok "$(du -sh "${PUNCT_DEST}" | cut -f1) → ${PUNCT_DEST}"
fi

echo
echo "  看模型契约（输入名 / dtype / metadata）："
echo "    cargo run --release -p cnt-asr-tools -- info ${DEST}"
echo
echo "  离线验证（不打开麦克风）："
echo "    cargo run --release -p cnt-asr-tools -- transcribe ${DEST} ${DEST}/test_wavs/zh.wav"
