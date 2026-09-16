#!/usr/bin/env bash
# cnt 输入法安装/升级脚本（拼音 + 语音，一条命令交付）
#
# 安装为「第三个输入法」，与英文输入和 Rime 互不冲突：
# - 主程序    → ~/.local/bin/cnt-daemon
# - 数据/模型 → ~/.local/share/cnt/{dict.cntd,lm.cntl,asr/,punct/}
# - 配置      → ~/.config/cnt/config.toml（缺 [voice] 时自动补上并开启语音）
# - 组件      → /usr/share/ibus/component/cnt.xml（唯一需要管理员密码的步骤）
#
# 升级时平滑替换：先切换到其他输入法 → 停掉旧版 → 放入新版 → 切回，
# 全程不重启输入法服务（重启瞬间系统可能恢复输入源失败，表现为输入法挂掉）。
#
# 参数：
#   -y, --yes      不询问，需要下载模型时直接下（约 324 MB）
#   --no-voice     只装拼音输入，不下模型、不开语音
#
# 卸载：
#   sudo rm /usr/share/ibus/component/cnt.xml
#   rm -f ~/.local/bin/cnt-daemon && rm -rf ~/.local/share/cnt && ibus restart

set -euo pipefail
cd "$(dirname "$0")/.."

BIN_DIR="${HOME}/.local/bin"
DATA_DIR="${HOME}/.local/share/cnt"
CONFIG_DIR="${XDG_CONFIG_HOME:-${HOME}/.config}/cnt"
CONFIG_FILE="${CONFIG_DIR}/config.toml"
COMPONENT_DIR="/usr/share/ibus/component"
EXEC="${BIN_DIR}/cnt-daemon"

ASSUME_YES=0
WITH_VOICE=1
for arg in "$@"; do
    case "${arg}" in
        -y|--yes) ASSUME_YES=1 ;;
        --no-voice) WITH_VOICE=0 ;;
        *) echo "未知参数：${arg}（可用：-y / --no-voice）" >&2; exit 1 ;;
    esac
done

# ── 输出样式（homebrew 风格：彩色 + emoji）────────────────────
_C_RST='\033[0m'; _C_BLD='\033[1m'; _C_DIM='\033[2m'
_C_CYN='\033[36m'; _C_GRN='\033[32m'; _C_YEL='\033[33m'; _C_RED='\033[31m'
step() { printf "${_C_BLD}${_C_CYN}==>${_C_RST} ${_C_BLD}%s${_C_RST}\n" "$*"; }
info() { printf "  ${_C_DIM}%s${_C_RST}\n" "$*"; }
ok()   { printf "  ${_C_GRN}✔${_C_RST} %s\n" "$*"; }
warn() { printf "  ${_C_YEL}⚠${_C_RST} %s\n" "$*"; }
err()  { printf "  ${_C_RED}✘${_C_RST} %s\n" "$*" >&2; }

# ── 1. 构建依赖 ───────────────────────────────────────────────
# 语音输入的麦克风采集走 ALSA（cpal → alsa-sys → libasound）。
# 系统默认只装运行时库，编译需要 dev 包里的头文件与 alsa.pc。
# PipeWire 自带 ALSA 兼容层，所以走 ALSA 在 PipeWire 上照样能录音。
step "检查构建依赖"
if pkg-config --exists alsa 2>/dev/null; then
    ok "alsa 开发库已就位"
else
    info "缺少 alsa 开发库，正在安装 libasound2-dev（需管理员密码）..."
    sudo apt-get install -y libasound2-dev
    ok "libasound2-dev 已安装"
fi

# ── 2. 编译 ──────────────────────────────────────────────────
step "编译程序（首次需要几分钟，请稍候）..."
# 发布构建：不带 --features 即全局关闭 fastrace（编译期零开销）。
# onnxruntime 静态链入二进制，安装时不需要额外拷动态库。
cargo build --release -p cnt-daemon

# 激活 cnt 输入法：首次启动偶尔会超时，自动重试最多 3 次。
activate_cnt() {
    for _ in 1 2 3; do
        if ibus engine cnt 2>/dev/null; then
            ok "cnt 输入法已激活"
            return 0
        fi
        info "正在启动 cnt 输入法，稍候自动重试..."
        sleep 3
    done
    warn "多次尝试仍未激活 cnt 输入法"
    return 1
}

# ── 3. 主程序 ─────────────────────────────────────────────────
step "安装主程序"
mkdir -p "${BIN_DIR}"
# 升级时平滑替换：先切走（若正在使用）→ 停旧版 → 删旧文件 → 放新版 → 切回。
CUR=$(ibus engine 2>/dev/null || true)
if [ "${CUR}" = "cnt" ]; then
    info "正在切换到英文输入法（几秒后自动切回）..."
    ibus engine xkb:us::eng || true
    sleep 1
fi
OLD=$(pgrep -f "${BIN_DIR}/cnt-daemon" | head -1 || true)
if [ -n "${OLD}" ]; then
    info "正在退出旧版本（输入习惯不会丢失）..."
    kill -INT "${OLD}"
    sleep 2
fi
rm -f "${BIN_DIR}/cnt-daemon"
cp target/release/cnt-daemon "${BIN_DIR}/cnt-daemon"
ok "cnt-daemon → ${BIN_DIR}"
activate_cnt || true

# ── 4. 词库与语言模型 ─────────────────────────────────────────
step "安装词库和语言模型数据"
mkdir -p "${DATA_DIR}"
if [ ! -f data/cnt.dict ] || [ ! -f data/lm.cntl ]; then
    err "缺少词库文件（data/cnt.dict / data/lm.cntl），请先按 README 生成。"
    exit 1
fi
cp data/cnt.dict "${DATA_DIR}/dict.cntd"
cp data/lm.cntl "${DATA_DIR}/lm.cntl"
ok "词库 + 语言模型已安装"

# ── 5. 语音模型（声学 + 标点）──────────────────────────────────
VOICE_READY=0
if [ "${WITH_VOICE}" = "1" ]; then
    step "语音输入模型"
    if [ ! -f data/asr/tokens.txt ] || ! ls data/punct/model*.onnx >/dev/null 2>&1; then
        DO_FETCH=1
        if [ "${ASSUME_YES}" != "1" ]; then
            printf "  需要下载语音模型（声学 188 MB + 标点 65 MB）。现在下载？[Y/n] "
            read -r reply || reply=""
            case "${reply}" in [nN]*) DO_FETCH=0 ;; esac
        fi
        if [ "${DO_FETCH}" = "1" ]; then
            bash scripts/fetch-asr-model.sh
        else
            info "跳过下载，语音输入将保持关闭（拼音输入不受影响）"
        fi
    fi

    if [ -f data/asr/tokens.txt ]; then
        mkdir -p "${DATA_DIR}/asr"
        cp data/asr/tokens.txt "${DATA_DIR}/asr/"
        cp data/asr/model*.onnx "${DATA_DIR}/asr/"
        [ -d data/asr/test_wavs ] && cp -r data/asr/test_wavs "${DATA_DIR}/asr/"
        ok "声学模型已安装（${DATA_DIR}/asr）"
        VOICE_READY=1
    fi
    # 标点模型缺了只是没有句读，语音仍可用，所以不阻断安装
    if ls data/punct/model*.onnx >/dev/null 2>&1; then
        mkdir -p "${DATA_DIR}/punct"
        cp data/punct/model*.onnx "${DATA_DIR}/punct/"
        ok "标点模型已安装（${DATA_DIR}/punct）"
    elif [ "${VOICE_READY}" = "1" ]; then
        warn "没有标点模型，语音上屏的文本将没有标点"
    fi
fi

# ── 6. 配置 ──────────────────────────────────────────────────
step "配置文件"
mkdir -p "${CONFIG_DIR}"
if [ ! -f "${CONFIG_FILE}" ]; then
    cat > "${CONFIG_FILE}" <<EOF
# cnt 输入法配置

# 每页候选词数（5~10；上限 10 = 数字选择键 1-9 加 0）
page_size = 8

[voice]
# 语音输入总开关
enabled = ${VOICE_READY}
# 按住说话（该键按住期间对应用无副作用；右 Ctrl 被换成 Copilot 键的键盘用 Alt_R）
ptt_key = "Alt_R"
# 切换式常开听写
toggle_key = "Control+Shift+space"
# 标点恢复（关掉后上屏文本没有句读）
punctuation = true
# 推理线程数
threads = 4
# 麦克风设备名子串，留空 = 系统默认
device = ""
EOF
    # TOML 的布尔要小写 true/false，上面写的是 1/0，这里修正
    sed -i "s/^enabled = 1$/enabled = true/; s/^enabled = 0$/enabled = false/" "${CONFIG_FILE}"
    ok "已生成 ${CONFIG_FILE}"
elif ! grep -q '^\[voice\]' "${CONFIG_FILE}"; then
    {
        echo ""
        echo "[voice]"
        echo "enabled = $([ "${VOICE_READY}" = "1" ] && echo true || echo false)"
        echo 'ptt_key = "Alt_R"'
        echo 'toggle_key = "Control+Shift+space"'
        echo "punctuation = true"
        echo "threads = 4"
    } >> "${CONFIG_FILE}"
    ok "已在 ${CONFIG_FILE} 追加 [voice] 段"
else
    # 已有 [voice] 段：不擅自改用户的配置，只提示
    if awk '/^\[voice\]/{f=1;next} /^\[/{f=0} f' "${CONFIG_FILE}" \
        | grep -qE '^\s*enabled\s*=\s*true'; then
        ok "配置已存在且语音已开启（${CONFIG_FILE}）"
    else
        warn "配置里语音未开启，如需开启请把 ${CONFIG_FILE} 的 [voice] 段改成 enabled = true"
    fi
fi

# ── 7. 注册 IBus 组件 ────────────────────────────────────────
step "注册输入法组件（如需修改，会要求管理员密码）"
XML=$(cat <<EOF
<?xml version="1.0" encoding="utf-8"?>
<!-- filename: cnt.xml -->
<component>
    <name>org.freedesktop.IBus.Cnt</name>
    <description>Cnt Pinyin Component</description>
    <exec>${EXEC}</exec>
    <version>0.1.0</version>
    <license>MIT</license>
    <author>cnt</author>
    <homepage>https://example.invalid/cnt</homepage>
    <engines>
        <engine>
            <name>cnt</name>
            <language>zh_CN</language>
            <license>MIT</license>
            <author>cnt</author>
            <icon></icon>
            <layout>us</layout>
            <longname>Cnt 拼音 (Rust)</longname>
            <description>一个简单的简体中文拼音输入法（Rust 实现）</description>
            <rank>50</rank>
        </engine>
    </engines>
</component>
EOF
)
if [ -f "${COMPONENT_DIR}/cnt.xml" ] \
    && cmp -s "${COMPONENT_DIR}/cnt.xml" <(printf '%s\n' "${XML}"); then
    ok "组件已是最新，无需修改"
else
    sudo install -m 644 /dev/stdin "${COMPONENT_DIR}/cnt.xml" <<< "${XML}"
fi

step "确认输入法可用"
# 升级场景已激活；首次安装需重启输入法服务让新组件生效。
if ibus engine 2>/dev/null | grep -q "cnt"; then
    ok "cnt 输入法已激活，无需重启输入法服务"
else
    info "首次安装：正在重启输入法服务（几秒后自动恢复）..."
    ibus restart
    sleep 2
    activate_cnt || true
fi

step "添加到系统输入法列表"
# GNOME 的输入法切换器（Super+Space）显示的是系统输入源列表，需显式加入。
if [ "${XDG_CURRENT_DESKTOP:-}" != "${XDG_CURRENT_DESKTOP#*GNOME}" ]; then
    SOURCES=$(gsettings get org.gnome.desktop.input-sources sources)
    if ! echo "$SOURCES" | grep -q "'cnt'"; then
        INNER="${SOURCES#\[}"; INNER="${INNER%\]}"
        if [ -n "$INNER" ]; then
            NEW="[${INNER}, ('ibus', 'cnt')]"
        else
            NEW="[('ibus', 'cnt')]"
        fi
        gsettings set org.gnome.desktop.input-sources sources "$NEW"
        ok "已添加到输入法列表"
    else
        ok "已在输入法列表中"
    fi
else
    info "非 GNOME 桌面：请到系统设置 → 输入法 手动添加 cnt"
fi

echo
printf "${_C_BLD}${_C_GRN}  🎉 安装完成${_C_RST}  切换输入法：Super+Space → Cnt\n"
if [ "${VOICE_READY}" = "1" ]; then
    PTT=$(awk -F'"' '/^ptt_key/{print $2}' "${CONFIG_FILE}" 2>/dev/null || true)
    echo "     语音：按住 ${PTT:-Alt_R} 说话，松手上屏；Ctrl+Shift+空格 常开；Esc 放弃"
fi
