#!/usr/bin/env bash
# 埋点规范自查（AGENTS.md「可观测性」一节的可执行版本）
#
# 规范是文档，检查得是代码 —— 否则下一个人（或下一个我）照样会犯同样的错。
# 这个脚本把三类硬约束变成可复现的检查：
#
#   1. 库 crate 不得用重的 `Span::enter_with_local_parent`（该用 `LocalSpan`）；
#      有正当理由的例外必须在该行上方 3 行内写 `trace-exception: 原因`。
#   2. 库 crate 不得 `set_reporter` / `fastrace::flush`（那是应用的职责）。
#   3. 应用 crate（有 main.rs 的）必须 `set_reporter` + `flush`。
#   4. 工作量指标不得挂在 `Event` 上（属性=事实，事件=期间发生的事）。
#
# 用法：bash scripts/check-tracing.sh    （有违规则退出码 1）

set -uo pipefail
cd "$(dirname "$0")/.."

_C_RST='\033[0m'; _C_BLD='\033[1m'; _C_GRN='\033[32m'; _C_RED='\033[31m'; _C_DIM='\033[2m'
fail=0
note() { printf "  ${_C_DIM}%s${_C_RST}\n" "$*"; }
bad()  { printf "  ${_C_RED}✘${_C_RST} %s\n" "$*"; fail=1; }
good() { printf "  ${_C_GRN}✔${_C_RST} %s\n" "$*"; }

printf "${_C_BLD}埋点规范自查${_C_RST}（规范见 AGENTS.md「可观测性」）\n\n"

# 工作量指标的常见名字：出现在 Event 里就是用错了容器
WORKLOAD_KEYS='count|frames|samples|nbest|expand|vocab|tokens|rtf|secs|chars'

for dir in crates/*/; do
    crate=$(basename "${dir}")
    src="${dir}src"
    [ -d "${src}" ] || continue

    if [ -f "${src}/main.rs" ]; then
        # ── 应用 crate ─────────────────────────────────────────
        miss=""
        grep -rq "set_reporter" "${src}" || miss="${miss} set_reporter"
        grep -rq "fastrace::flush" "${src}" || miss="${miss} flush"
        if [ -n "${miss}" ]; then
            bad "${crate}（应用）缺:${miss}"
            note "应用 crate 必须启动即装 reporter、退出/批次结束 flush，否则埋了也看不见"
        else
            good "${crate}（应用）reporter + flush 齐"
        fi
    else
        # ── 库 crate ───────────────────────────────────────────
        # 1. 重 Span 用法（允许带 trace-exception 注释的例外）
        hits=$(grep -rn "Span::enter_with_local_parent" "${src}" 2>/dev/null \
               | grep -v "LocalSpan::enter_with_local_parent" || true)
        if [ -n "${hits}" ]; then
            while IFS= read -r hit; do
                file="${hit%%:*}"; rest="${hit#*:}"; line="${rest%%:*}"
                # 例外声明写在该行上方 3 行内（理由通常要写两三行）
                from=$(( line > 3 ? line - 3 : 1 ))
                ctx=$(sed -n "${from},${line}p" "${file}")
                if echo "${ctx}" | grep -q "trace-exception:"; then
                    note "${crate}: ${file}:${line} 重 Span（已声明例外）"
                else
                    bad "${crate}: ${file}:${line} 用了重的 Span::enter_with_local_parent"
                    note "库代码该用 LocalSpan；确有理由请在上一行写 // trace-exception: 原因"
                fi
            done <<< "${hits}"
        fi
        # 2. 库里不该出现 reporter / flush
        if grep -rq "set_reporter\|fastrace::flush" "${src}" 2>/dev/null; then
            bad "${crate}（库）出现了 set_reporter/flush —— 那是应用 crate 的职责"
        fi
    fi

    # 3. 工作量指标不得塞进 Event（属性 vs 事件）
    evt=$(grep -rn -A 3 "Event::new" "${src}" 2>/dev/null \
          | grep -E "with_propert.*\"(${WORKLOAD_KEYS})\"" || true)
    if [ -n "${evt}" ]; then
        bad "${crate}: 工作量指标挂在 Event 上（应改为 span 属性 add_property）"
        echo "${evt}" | head -3 | while IFS= read -r l; do note "${l}"; done
    fi
done

echo
if [ "${fail}" = "0" ]; then
    printf "${_C_GRN}${_C_BLD}  埋点规范检查通过${_C_RST}\n"
else
    printf "${_C_RED}${_C_BLD}  存在违规，见上${_C_RST}\n"
fi
exit "${fail}"
