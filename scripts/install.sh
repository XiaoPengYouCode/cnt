#!/usr/bin/env bash
# cnt 输入法安装/升级脚本
#
# 安装为「第三个输入法」，与英文输入和 Rime 互不冲突：
# - 主程序  → ~/.local/bin/cnt-daemon
# - 数据    → ~/.local/share/cnt/
# - 组件    → /usr/share/ibus/component/cnt.xml（唯一需要管理员密码的步骤）
#
# 升级时平滑替换：先切换到其他输入法 → 停掉旧版 → 放入新版 → 切回，
# 全程不重启输入法服务（重启瞬间系统可能恢复输入源失败，表现为输入法挂掉）。
#
# 卸载：
#   sudo rm /usr/share/ibus/component/cnt.xml && rm -rf ~/.local/bin/cnt-daemon ~/.local/share/cnt && ibus restart

set -euo pipefail
cd "$(dirname "$0")/.."

BIN_DIR="${HOME}/.local/bin"
DATA_DIR="${HOME}/.local/share/cnt"
COMPONENT_DIR="/usr/share/ibus/component"
EXEC="${BIN_DIR}/cnt-daemon"

# ── 输出样式（homebrew 风格：彩色 + emoji）────────────────────
_C_RST='\033[0m'; _C_BLD='\033[1m'; _C_DIM='\033[2m'
_C_CYN='\033[36m'; _C_GRN='\033[32m'; _C_YEL='\033[33m'; _C_RED='\033[31m'
step() { printf "${_C_BLD}${_C_CYN}==>${_C_RST} ${_C_BLD}%s${_C_RST}\n" "$*"; }
info() { printf "  ${_C_DIM}%s${_C_RST}\n" "$*"; }
ok()   { printf "  ${_C_GRN}✔${_C_RST} %s\n" "$*"; }
warn() { printf "  ${_C_YEL}⚠${_C_RST} %s\n" "$*"; }
err()  { printf "  ${_C_RED}✘${_C_RST} %s\n" "$*" >&2; }

step "编译程序（首次需要几分钟，请稍候）..."
# 发布构建：不带 --features 即全局关闭 fastrace（编译期零开销）。
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
activate_cnt || true

step "安装词库和语言模型数据"
mkdir -p "${DATA_DIR}"
if [ ! -f data/cnt.dict ] || [ ! -f data/lm.cntl ]; then
    err "缺少词库文件（data/cnt.dict / data/lm.cntl），请先按 README 生成。"
    exit 1
fi
cp data/cnt.dict "${DATA_DIR}/dict.cntd"
cp data/lm.cntl "${DATA_DIR}/lm.cntl"

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
printf "${_C_BLD}${_C_GRN}  🎉 Cnt 拼音输入法安装完成！${_C_RST}\n"
echo
echo "     按 Super+Space（或右上角面板）切换到 Cnt，即可开始输入。"
echo "     以后升级：再次运行 ./scripts/install.sh 即可。"
