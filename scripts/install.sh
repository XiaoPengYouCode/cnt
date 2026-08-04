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

echo "[1/5] 构建 release 版..."
cargo build --release

echo "[2/5] 安装二进制 -> ${BIN_DIR}"
mkdir -p "${BIN_DIR}"
cp target/release/cnt-daemon "${BIN_DIR}/cnt-daemon"

echo "[3/5] 安装数据 -> ${DATA_DIR}"
mkdir -p "${DATA_DIR}"
if [ ! -f data/cnt.dict ] || [ ! -f data/lm.cntl ]; then
    echo "错误：data/cnt.dict 或 data/lm.cntl 不存在，请先按 README 生成词库和语言模型。" >&2
    exit 1
fi
cp data/cnt.dict "${DATA_DIR}/dict.cntd"
cp data/lm.cntl "${DATA_DIR}/lm.cntl"

echo "[4/5] 写入组件 XML（需要 sudo）..."
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
sudo install -m 644 /dev/stdin "${COMPONENT_DIR}/cnt.xml" <<< "${XML}"

echo "[5/6] 重启 ibus 让组件生效（rime 会随会话自动重启，无冲突）..."
ibus restart

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
