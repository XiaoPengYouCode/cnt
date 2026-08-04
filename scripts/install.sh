#!/usr/bin/env bash
# cnt 输入法安装脚本
#
# 安装为「第三个输入法」，与英文输入（xkb 布局）和 Rime 互不冲突：
# - 二进制 → ~/.local/bin/cnt-daemon            （用户级）
# - 数据   → ~/.local/share/cnt/                （daemon 默认数据目录）
# - 配置   → ~/.config/cnt/config.toml          （可选，用户级）
# - 组件   → /usr/share/ibus/component/cnt.xml  （sudo，唯一特权操作）
#
# 关键机制：组件 XML 告诉 ibus「存在一个叫 cnt 的引擎，用这个命令启动它」。
# ibus 只在用户显式选中 cnt 时才启动 cnt-daemon，不会自启、不抢 Rime。
# 注：ibus 1.5.x 的用户目录组件扫描是 #if 0 注释掉的（见 ibusregistry.c），
#     因此 XML 必须装到系统目录。

set -euo pipefail
cd "$(dirname "$0")/.."

BIN_DIR="${HOME}/.local/bin"
DATA_DIR="${HOME}/.local/share/cnt"
COMPONENT_DIR="/usr/share/ibus/component"
EXEC="${BIN_DIR}/cnt-daemon"

echo "[1/5] 构建 release 版（发布构建：trace 全局关闭，span 编译期零开销）..."
# fastrace 的 enable 是编译期开关，且依赖 feature 在构建图内全局合并：
# 不带 flag 即全局关闭（发布构建，零开销）；诊断/bench 构建用
# `cargo build --release --features \"fastrace/enable\"` 全局开启。
cargo build --release -p cnt-daemon

# 平滑激活 cnt 引擎：ibus spawn 引擎进程存在握手竞态（首次 spawn 常超时），
# 失败则间隔重试，最多 3 次。
activate_cnt() {
    for _ in 1 2 3; do
        if ibus engine cnt 2>/dev/null; then
            echo "  已激活 cnt 引擎"
            return 0
        fi
        echo "  引擎激活失败，3s 后重试..."
        sleep 3
    done
    echo "  警告：多次尝试仍无法激活 cnt 引擎" >&2
    return 1
}

echo "[2/5] 安装二进制 -> ${BIN_DIR}"
mkdir -p "${BIN_DIR}"
# 升级路径：平滑替换，无 restart、无 kill。顺序为 切走→停→删→放→启：
# - 若 cnt 正被使用，先切走（销毁活动引擎），避免替换/退出阶段 gnome-shell
#   的 setEngine 撞上引擎消失（此前两次会话卡死即源于此）
# - SIGINT 让旧 daemon 走 ctrl_c 优雅退出（flush 用户库）；实测 ibus 引擎进程
#   不随引擎切换回收，必须先停，rm 才无占用
# - rm + cp 而非 mv：cargo 对 target/release 二进制建硬链接（与 deps/ 同 inode），
#   mv 覆盖已部署路径会报 "same file"；cp 产生独立副本无此问题
# - 切回 cnt（带重试；首次安装时组件未注册会失败，由 [5/6] 的 restart 兜底）
CUR=$(ibus engine 2>/dev/null || true)
if [ "${CUR}" = "cnt" ]; then
    echo "  当前引擎是 cnt，先切走（xkb）..."
    ibus engine xkb:us::eng || true
    sleep 1
fi
OLD=$(pgrep -f "${BIN_DIR}/cnt-daemon" | head -1 || true)
if [ -n "${OLD}" ]; then
    echo "  让旧 daemon (${OLD}) 优雅退出（SIGINT）..."
    kill -INT "${OLD}"
    sleep 2
fi
rm -f "${BIN_DIR}/cnt-daemon"
cp target/release/cnt-daemon "${BIN_DIR}/cnt-daemon"
activate_cnt || true

echo "[3/5] 安装数据 -> ${DATA_DIR}"
mkdir -p "${DATA_DIR}"
if [ ! -f data/cnt.dict ] || [ ! -f data/lm.cntl ]; then
    echo "错误：data/cnt.dict 或 data/lm.cntl 不存在，请先按 README 生成词库和语言模型。" >&2
    exit 1
fi
cp data/cnt.dict "${DATA_DIR}/dict.cntd"
cp data/lm.cntl "${DATA_DIR}/lm.cntl"

echo "[4/5] 写入组件 XML（内容已一致时自动跳过 sudo）..."
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
    echo "  ${COMPONENT_DIR}/cnt.xml 已存在且内容一致，跳过 sudo"
else
    sudo install -m 644 /dev/stdin "${COMPONENT_DIR}/cnt.xml" <<< "${XML}"
fi

echo "[5/6] 让组件生效..."
# 升级场景：组件 XML 已被 ibus 加载，[2/5] 已平滑激活新引擎，无需 restart。
# 首次安装（或激活失败）：restart 让 ibus 加载新组件 XML，再激活。
# （ibus restart 是“原子弹”：restart 瞬间 gnome-shell 会立即恢复上次输入源并
#   setEngine，若引擎尚未注册就绪则失败且不重试——输入法会话表现为“挂掉”。
#   因此升级一律避免 restart，只有首装才用它。）
if ibus engine 2>/dev/null | grep -q "cnt"; then
    echo "  cnt 引擎已激活，跳过 restart"
else
    echo "  首次安装/激活失败兜底：ibus restart"
    ibus restart
    sleep 2
    activate_cnt || true
fi

echo "[6/6] 把 cnt 加入桌面输入源..."
# 关键：ibus 引擎注册 ≠ 桌面切换器显示。GNOME 的 Super+Space 切换器显示的是
# org.gnome.desktop.input-sources sources，需要显式把 ('ibus', 'cnt') 加进去。
if [ "${XDG_CURRENT_DESKTOP:-}" != "${XDG_CURRENT_DESKTOP#*GNOME}" ]; then
    SOURCES=$(gsettings get org.gnome.desktop.input-sources sources)
    if ! echo "$SOURCES" | grep -q "'cnt'"; then
        # 去掉首尾 []，追加 ('ibus', 'cnt')
        INNER="${SOURCES#\[}"; INNER="${INNER%\]}"
        if [ -n "$INNER" ]; then
            NEW="[${INNER}, ('ibus', 'cnt')]"
        else
            NEW="[('ibus', 'cnt')]"
        fi
        gsettings set org.gnome.desktop.input-sources sources "$NEW"
        echo "已把 ('ibus', 'cnt') 加入 GNOME 输入源（切换器立即可见）"
    else
        echo "cnt 已在输入源列表中，跳过"
    fi
else
    echo "非 GNOME 会话：请手动在输入法设置中添加 cnt 引擎"
fi

echo
echo "完成！现在输入法切换器（Super+Space / 面板）里应有三个输入法："
echo "  1. 英文（xkb 布局）"
echo "  2. Rime"
echo "  3. Cnt 拼音 (Rust) ← 新安装"
echo
echo "手动卸载："
echo "  sudo rm /usr/share/ibus/component/cnt.xml && rm -rf ~/.local/bin/cnt-daemon ~/.local/share/cnt && ibus restart"
