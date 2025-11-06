#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Notion Universal Importer — v3 (Max-Compat)
- 通用输入：JSON / 目录 / Markdown / HTML 片段
- 强制约束：rich_text ≤ 1900 字符；children ≤ 100；单次请求嵌套 ≤ 2
- 递归续传：对 >2 层嵌套，获取父块ID后继续 append
- 表格校验：simple table 至少1行，cells数=table_width
- 429重试：遵循 Retry-After，指数退避
- 日志：--verbose；预演：--dry-run
"""

import os, sys, json, time, argparse, re, pathlib, itertools
from typing import Any, Dict, List, Optional, Iterable, Tuple
import requests
from dataclasses import dataclass
from dotenv import load_dotenv

# ---------- 环境 ----------
load_dotenv()
NOTION_API = "https://api.notion.com/v1"
NOTION_VERSION = os.environ.get("NOTION_VERSION", "2025-09-03")  # 可覆盖，建议跟随官方最新
MAX_RICH_TEXT_CHARS = 1900           # 保守低于 2000 上限（Notion: text.content ≤ 2000）  # ref
MAX_CHILDREN_PER_REQUEST = 100       # 单次 append children 上限                      # ref
MAX_NESTING_PER_REQUEST = 2          # 单次请求嵌套层级上限                           # ref
DEFAULT_BATCH_SIZE = 90              # 留余量，避开100上限

# ---------- 基础工具 ----------
def eprint(*args):
    print(*args, file=sys.stderr)

def chunk(seq: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

def notion_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json"
    }

def http_request(method: str, url: str, headers: Dict[str, str], body: Optional[Dict]=None,
                 max_retries: int = 6, verbose: bool = False) -> Dict[str, Any]:
    """
    带429重试与 Retry-After 遵循；指数退避
    """
    payload = json.dumps(body) if body is not None else None
    backoff = 1.0
    for attempt in range(max_retries+1):
        if verbose:
            eprint(f"[HTTP] {method} {url}")
            if body is not None:
                eprint(f"[HTTP] body={json.dumps(body, ensure_ascii=False)[:5000]}...")
        r = requests.request(method, url, headers=headers, data=payload)
        if r.status_code == 429:
            # 遵循 Retry-After; 若无则指数退避
            ra = r.headers.get("Retry-After")
            wait = float(ra) if ra and ra.isdigit() else backoff
            eprint(f"[429] Rate limited. Wait {wait:.2f}s ...")
            time.sleep(wait)
            backoff = min(backoff*2, 30)
            continue
        if r.status_code >= 300:
            raise RuntimeError(f"{method} {url} failed: {r.status_code} {r.text}")
        return r.json() if r.text else {}
    raise RuntimeError(f"HTTP retries exhausted for {method} {url}")

# ---------- 富文本与块构造 ----------
def rt_text(content: str, *, bold=False, italic=False, strike=False, underline=False,
            code=False, color="default", href: Optional[str]=None) -> Dict[str, Any]:
    content = "" if content is None else str(content)
    return {
        "type": "text",
        "text": {"content": content, "link": {"url": href} if href else None},
        "annotations": {
            "bold": bool(bold),
            "italic": bool(italic),
            "strikethrough": bool(strike),
            "underline": bool(underline),
            "code": bool(code),
            "color": color
        }
    }

def split_to_rich_text_chunks(s: str, ann: Dict[str, Any]) -> List[Dict[str, Any]]:
    s = "" if s is None else str(s)
    if not s:
        return [rt_text("", **ann)]
    pieces = [s[i:i+MAX_RICH_TEXT_CHARS] for i in range(0, len(s), MAX_RICH_TEXT_CHARS)]
    return [rt_text(p, **ann) for p in pieces]

def parse_inline_markup(text: str) -> List[Dict[str, Any]]:
    """
    轻量 Markdown/HTML → rich_text[]
    支持 **bold** *italic* `code` ~~del~~ {red|文字} 以及 <b><i><span style="color:red">
    复杂HTML不解析，保留纯文本
    """
    if text is None:
        return [rt_text("")]
    s = str(text)

    # 粗暴去掉最简单HTML标签并转为标注
    html_patterns = [
        (r"<b>(.*?)</b>", {"bold": True}),
        (r"<strong>(.*?)</strong>", {"bold": True}),
        (r"<i>(.*?)</i>", {"italic": True}),
        (r"<em>(.*?)</em>", {"italic": True}),
        (r"<span\s+style=['\"]color\s*:\s*red['\"]>(.*?)</span>", {"color": "red"}),
    ]
    # 先转成占位形式，避免标记互相影响
    placeholder = []
    def keep(seg_ann):
        idx = len(placeholder)
        return f"§§SEG{idx}§§", placeholder.append(seg_ann)

    for pat, ann in html_patterns:
        while True:
            m = re.search(pat, s, flags=re.IGNORECASE|re.DOTALL)
            if not m: break
            seg = m.group(1)
            key, _ = keep({"text": seg, "ann": ann})
            s = s[:m.start()] + key + s[m.end():]

    # Markdown 风格：按嵌套优先级处理
    rules = [
        (r"\*\*(.+?)\*\*", {"bold": True}),
        (r"\*(.+?)\*", {"italic": True}),
        (r"~~(.+?)~~", {"strike": True}),
        (r"`(.+?)`", {"code": True}),
        (r"\{red\|(.+?)\}", {"color": "red"}),
    ]
    for pat, ann in rules:
        while True:
            m = re.search(pat, s, flags=re.DOTALL)
            if not m: break
            seg = m.group(1)
            key, _ = keep({"text": seg, "ann": ann})
            s = s[:m.start()] + key + s[m.end():]

    # 最后把字符串按占位切段
    parts: List[Dict[str, Any]] = []
    tokens = re.split(r"(§§SEG\d+§§)", s)
    for t in tokens:
        if not t:
            continue
        m = re.match(r"§§SEG(\d+)§§", t)
        if m:
            idx = int(m.group(1))
            seg = placeholder[idx]["text"]
            ann = placeholder[idx]["ann"]
            parts.extend(split_to_rich_text_chunks(seg, ann))
        else:
            parts.extend(split_to_rich_text_chunks(t, {}))
    return parts or [rt_text("")]

def paragraph_block(text: str) -> Dict[str, Any]:
    rich = parse_inline_markup(text)
    # 如果 rich_text 项超过100，拆成多段落（每100项一段）
    if len(rich) <= MAX_CHILDREN_PER_REQUEST:
        return {"type": "paragraph", "paragraph": {"rich_text": rich}}
    blocks = []
    for batch in chunk(rich, MAX_CHILDREN_PER_REQUEST):
        blocks.append({"type": "paragraph", "paragraph": {"rich_text": batch}})
    return {"type": "group", "children": blocks}  # 特殊标记，稍后展平

def heading_block(level: int, text: str) -> Dict[str, Any]:
    level = max(1, min(3, int(level or 2)))
    return {f"type": f"heading_{level}", f"heading_{level}": {"rich_text": parse_inline_markup(text), "color": "default", "is_toggleable": False}}

def bullet_block(text: str, children: Optional[List[Dict]]=None) -> Dict[str, Any]:
    blk = {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": parse_inline_markup(text)}}
    if children:
        blk["bulleted_list_item"]["children"] = children
    return blk

def code_block(text: str, lang: str = "plain text") -> Dict[str, Any]:
    # Notion 语言名需合法；未知则回退 plain text（脚本不强求完整枚举）
    return {"type": "code", "code": {"rich_text": parse_inline_markup(text), "language": (lang or "plain text").lower()}}

def equation_block(expr: str) -> Dict[str, Any]:
    # Notion 限制：equation.expression ≤ 1000 字符（脚本不切片，超限会报错，建议拆分）  # ref
    return {"type": "equation", "equation": {"expression": expr or ""}}

def table_from_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    输入：列表[dict] => simple table；自动生成 header；cells 与 table_width 匹配
    """
    if not rows:
        rows = [{}]
    keys = list({k for r in rows for k in (r.keys() if isinstance(r, dict) else [])})
    keys = keys or ["Col1"]
    width = len(keys)

    def to_cell(val: Any) -> List[Dict[str, Any]]:
        if isinstance(val, (str, int, float)):
            return parse_inline_markup(str(val))
        if isinstance(val, bool):
            return parse_inline_markup("✓" if val else "✗")
        if val is None:
            return parse_inline_markup("")
        return parse_inline_markup(json.dumps(val, ensure_ascii=False))

    table = {"type": "table", "table": {"table_width": width, "has_column_header": True, "has_row_header": False, "children": []}}
    # header row
    header_cells = [[rt] for rt in [rt_text(k) for k in keys]]
    table["table"]["children"].append({"type": "table_row", "table_row": {"cells": header_cells}})
    # data rows
    for r in rows:
        row_cells = []
        for k in keys:
            v = r.get(k, "")
            row_cells.append(to_cell(v))
        table["table"]["children"].append({"type": "table_row", "table_row": {"cells": row_cells}})
    return table

# ---------- 输入标准化 ----------
def load_any(path: str) -> Any:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    if p.suffix.lower() in {".md", ".markdown"}:
        return {"__markdown__": p.read_text(encoding="utf-8")}
    # 纯文本当作 markdown 处理
    return {"__markdown__": p.read_text(encoding="utf-8")}

def md_to_blocks(md: str) -> List[Dict[str, Any]]:
    """
    非完整Markdown，仅处理最常见结构；复杂Markdown建议预处理为JSON
    """
    lines = md.splitlines()
    blocks: List[Dict[str, Any]] = []
    buf: List[str] = []
    in_code = False
    code_lang = "plain text"
    for line in lines:
        fence = re.match(r"\s*```(\w+)?", line)
        if fence:
            if not in_code:
                in_code = True
                code_lang = fence.group(1) or "plain text"
                buf = []
            else:
                blocks.append(code_block("\n".join(buf), code_lang))
                in_code = False
                buf = []
            continue
        if in_code:
            buf.append(line)
            continue
        if line.startswith("### "):
            if buf:
                blocks.append(paragraph_block("\n".join(buf)))
                buf=[]
            blocks.append(heading_block(3, line[4:].strip()))
        elif line.startswith("## "):
            if buf:
                blocks.append(paragraph_block("\n".join(buf)))
                buf=[]
            blocks.append(heading_block(2, line[3:].strip()))
        elif line.startswith("# "):
            if buf:
                blocks.append(paragraph_block("\n".join(buf)))
                buf=[]
            blocks.append(heading_block(1, line[2:].strip()))
        elif re.match(r"^\s*-\s+", line):
            # 紧凑处理：连续的 - 列表合并为块序列
            txt = re.sub(r"^\s*-\s+", "", line).strip()
            blocks.append(bullet_block(txt))
        elif line.strip() == "":
            if buf:
                blocks.append(paragraph_block("\n".join(buf)))
                buf=[]
        else:
            buf.append(line)
    if buf:
        blocks.append(paragraph_block("\n".join(buf)))
    # 展平可能的 "group" 分段
    flat: List[Dict[str, Any]] = []
    for b in blocks:
        if b.get("type") == "group":
            flat.extend(b.get("children", []))
        else:
            flat.append(b)
    return flat

def normalize_payload(data: Any, fallback_title: str) -> Dict[str, Any]:
    """
    输出统一为：{"title": str, "children": [block,...]}
    """
    if isinstance(data, dict) and "__markdown__" in data:
        md = data["__markdown__"] or ""
        title = (md.splitlines()[0].lstrip("# ").strip()) or fallback_title
        return {"title": title, "children": md_to_blocks(md)}

    if isinstance(data, list):
        # 直接作为 blocks
        return {"title": fallback_title, "children": list(data)}

    if isinstance(data, dict):
        title = data.get("title") or data.get("name") or fallback_title
        for k in ("children", "blocks", "results"):
            if isinstance(data.get(k), list):
                return {"title": title, "children": list(data[k])}
        # 不规则对象 => 转成“键→值”的展开段落与表格尝试
        rows = []
        for key, val in data.items():
            if isinstance(val, (dict, list)):
                rows.append({str(key): json.dumps(val, ensure_ascii=False)})
            else:
                rows.append({str(key): val})
        return {"title": title, "children": [table_from_rows(rows)]}

    # 其他原子值
    return {"title": fallback_title, "children": [paragraph_block(str(data))]}

# ---------- 块清洗与嵌套拆解 ----------
def sanitize_block(b: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    宽容修复：确保所有块字段齐备；支持把 {type:"group",children:[...]} 展平
    返回 List 是为了处理 group → 多块展开
    """
    if not isinstance(b, dict):
        return [paragraph_block(json.dumps(b, ensure_ascii=False))]

    if b.get("type") == "group":
        out = []
        for c in b.get("children", []):
            out.extend(sanitize_block(c))
        return out

    t = b.get("type")
    if not t:
        return [paragraph_block(json.dumps(b, ensure_ascii=False))]

    # paragraph/bulleted_list_item/heading 支持
    if t == "paragraph":
        rich = b.get("paragraph", {}).get("rich_text")
        if not rich:
            # 若提供了纯 "text"
            txt = b.get("text") or ""
            return sanitize_block(paragraph_block(str(txt)))
        # 校正过长：由 parse_inline_markup 已切块，无需再处理
        return [b]

    if t.startswith("heading_"):
        return [b]

    if t in {"bulleted_list_item","numbered_list_item","quote","callout","to_do","toggle"}:
        # 递归清洗 children
        data = b.get(t, {})
        children = data.get("children")
        if isinstance(children, list):
            fixed = []
            for c in children:
                fixed.extend(sanitize_block(c))
            data["children"] = fixed
            b[t] = data
        return [b]

    if t in {"code","equation","divider","table_of_contents"}:
        return [b]

    if t == "table":
        data = b.get("table", {})
        width = int(data.get("table_width") or 1)
        children = data.get("children") or []
        if not children:
            # 空表：生成一行空 header
            header = {"type":"table_row","table_row":{"cells":[[rt_text("")] for _ in range(width)]}}
            data["children"] = [header]
        else:
            fixed_rows = []
            for r in children:
                if isinstance(r, dict) and r.get("type") == "table_row":
                    cells = r.get("table_row", {}).get("cells", [])
                    # 补齐/截断到 table_width
                    row_cells = []
                    for i in range(width):
                        cell = cells[i] if i < len(cells) else [rt_text("")]
                        # 规范化 cell 富文本
                        texts = []
                        for it in (cell or []):
                            if isinstance(it, dict) and it.get("type") == "text":
                                content = str(it.get("text", {}).get("content",""))
                                ann = it.get("annotations", {}) or {}
                                texts.extend(split_to_rich_text_chunks(content, ann))
                            elif isinstance(it, str):
                                texts.extend(parse_inline_markup(it))
                            else:
                                texts.extend(parse_inline_markup(json.dumps(it, ensure_ascii=False)))
                        row_cells.append(texts or [rt_text("")])
                    fixed_rows.append({"type":"table_row","table_row":{"cells": row_cells}})
            data["children"] = fixed_rows or [{"type":"table_row","table_row":{"cells":[[rt_text("")] for _ in range(width)]}}]
        b["table"] = data
        return [b]

    if t in {"image","file","pdf","video","embed","bookmark","link_preview"}:
        return [b]

    if t in {"child_page","child_database","column_list","column","template","synced_block"}:
        return [b]

    # 兜底
    return [paragraph_block(json.dumps(b, ensure_ascii=False))]

def sanitize_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for b in blocks:
        out.extend(sanitize_block(b))
    return out

# ---------- Notion API 封装 ----------
@dataclass
class NotionClient:
    token: str
    verbose: bool = False
    dry_run: bool = False

    def create_page(self, parent_page_id: str, title: str) -> str:
        """
        父页面模式：仅允许 title 属性
        """
        url = f"{NOTION_API}/pages"
        payload = {
            "parent": {"page_id": parent_page_id},
            "properties": {
                "title": {"title": [{"type":"text", "text": {"content": title}}]}
            }
        }
        if self.dry_run:
            eprint(f"[DRY-RUN] create_page title={title}")
            return "dry_page_id"
        j = http_request("POST", url, headers=notion_headers(self.token), body=payload, verbose=self.verbose)
        return j["id"]

    def append_children(self, parent_id: str, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        追加首层 children，返回 Notion 回包的 results（含新创建block的id）
        """
        if self.dry_run:
            eprint(f"[DRY-RUN] append {len(blocks)} blocks to {parent_id}")
            # 模拟返回
            return [{"id": f"dry_block_{i}", "type": b.get("type"), "has_children": bool(b.get(b.get("type"), {}).get("children"))} for i,b in enumerate(blocks)]
        url = f"{NOTION_API}/blocks/{parent_id}/children"
        body = {"children": blocks}
        j = http_request("PATCH", url, headers=notion_headers(self.token), body=body, verbose=self.verbose)
        return j.get("results", [])

# ---------- 上传（分层 ≤2，递归） ----------
def trim_children_for_request(block: Dict[str, Any], *, allow_depth: int = 2,
                              pending: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]], depth: int = 1) -> Dict[str, Any]:
    """
    若当前块含 children 且深度达到 allow_depth，则剥离 children 进入 pending；否则递归
    """
    t = block.get("type")
    if not t:
        return block
    data = block.get(t, {})
    ch = data.get("children")
    if isinstance(ch, list) and ch:
        if depth >= allow_depth:
            pending.append((block, ch))  # 记录：此“父块对象”与其 children（稍后拿真实ID再追加）
            data = dict(data)
            data.pop("children", None)
            block = {"type": t, t: data}
            return block
        else:
            # 递归处理下一层
            new_children = []
            for c in ch:
                new_children.append(trim_children_for_request(c, allow_depth=allow_depth, pending=pending, depth=depth+1))
            data = dict(data)
            data["children"] = new_children
            block = {"type": t, t: data}
            return block
    return block

def upload_blocks_recursive(nc: NotionClient, parent_id: str, blocks: List[Dict[str, Any]], batch_size: int = DEFAULT_BATCH_SIZE):
    """
    先把需要的 children 截断到 ≤2 层，首批 append；拿到每个父块的真实ID后，再对其 pending children 递归 append
    """
    # 1) 清洗 & 限制两层
    cleaned = sanitize_blocks(blocks)

    # 2) 分批提交
    for batch in chunk(cleaned, batch_size):
        pending: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
        request_blocks = [trim_children_for_request(b, allow_depth=MAX_NESTING_PER_REQUEST, pending=pending) for b in batch]

        # 若 request_blocks 某元素实为 "group" 已在 sanitize 时展平，不会出现

        # 3) 发送首批，拿到 results 顺序与 request_blocks 对齐
        results = []
        for sub in chunk(request_blocks, MAX_CHILDREN_PER_REQUEST):
            results.extend(nc.append_children(parent_id, sub))

        # 4) 将有 children 的父块映射到真实ID后，递归追加
        #    这里根据顺序映射：request_blocks[i] -> results[j]
        idx = 0
        id_map: Dict[int, str] = {}
        for i, rb in enumerate(request_blocks):
            # 每个 rb 对应 results[idx]
            if idx >= len(results): break
            id_map[i] = results[idx]["id"]
            idx += 1

        for (parent_block, children) in pending:
            # 找到 parent_block 在 request_blocks中的位置
            try:
                i = request_blocks.index(parent_block)
            except ValueError:
                continue
            pb_id = id_map.get(i)
            if not pb_id:
                continue
            upload_blocks_recursive(nc, pb_id, children, batch_size=batch_size)

# ---------- 主流程 ----------
def iter_files(path: str) -> Iterable[str]:
    p = pathlib.Path(path)
    if p.is_file():
        yield str(p)
    else:
        for root, _, files in os.walk(path):
            for fn in files:
                if fn.lower().endswith(".json") or fn.lower().endswith(".md"):
                    yield os.path.join(root, fn)

def main():
    ap = argparse.ArgumentParser(description="Notion Universal Importer — v3 (Max-Compat)")
    ap.add_argument("path", help="JSON/Markdown 文件或目录")
    ap.add_argument("--parent", dest="parent", default=os.environ.get("NOTION_PARENT_PAGE_ID"), help="父页面 Page ID（必填之一：parent/database）")
    ap.add_argument("--database", dest="database", default=None, help="数据库 Data Source ID（高级：此示例侧重父页面模式）")
    ap.add_argument("--token", dest="token", default=os.environ.get("NOTION_TOKEN"), help="Notion 集成令牌")
    ap.add_argument("--title", dest="title_override", default=None, help="覆盖标题")
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--from-md", dest="from_md", action="store_true", help="把输入当作 Markdown 解析")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true", help="仅打印将要发送的块，不真正请求")
    ap.add_argument("--verbose", dest="verbose", action="store_true", help="打印详细日志")
    args = ap.parse_args()

    if not args.token:
        eprint("错误：缺少 Notion token。--token 或设置 NOTION_TOKEN")
        sys.exit(1)
    if not (args.parent or args.database):
        eprint("错误：缺少父容器。请提供 --parent（父页面）或 --database（数据源ID）。")
        sys.exit(1)

    files = list(iter_files(args.path))
    if not files:
        eprint("未找到任何可处理的文件。")
        sys.exit(1)

    nc = NotionClient(token=args.token, verbose=args.verbose, dry_run=args.dry_run)

    ok, fail = 0, 0
    for fp in files:
        try:
            eprint(f"\n==> Importing: {fp}")
            raw = load_any(fp)
            fallback_title = pathlib.Path(fp).stem
            if args.from_md and isinstance(raw, str):
                raw = {"__markdown__": raw}
            normalized = normalize_payload(raw, args.title_override or fallback_title)
            title = args.title_override or normalized["title"] or fallback_title
            children = normalized.get("children", [])
            # 1) 先创建空页面（父页面模式）
            if args.parent:
                page_id = nc.create_page(args.parent, title)
                # 2) 递归上传 blocks
                upload_blocks_recursive(nc, page_id, children, batch_size=args.batch_size)
                eprint(f"✅ Done (page): {fp} → {page_id}")
            else:
                # 数据库模式（留作扩展；这里仅占位提示）
                eprint("⚠️ 当前脚本主要支持父页面模式。数据库模式请在 properties 映射后再启用。")
            ok += 1
        except Exception as e:
            eprint(f"❌ Failed: {fp} :: {e}")
            fail += 1
    eprint(f"\n完成：成功 {ok}，失败 {fail}")
    if fail:
        sys.exit(2)

if __name__ == "__main__":
    main()