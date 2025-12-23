import io
import re
from datetime import datetime
from typing import List, Dict, Optional, Any
from collections import defaultdict

import pdfplumber
import pandas as pd
import streamlit as st
import zipfile


# =========================================================
# SYSTEM TYPE (add more later)
# =========================================================
SYSTEM_TYPES = ["Cadre"]  # you can add more: ["Cadre", "SystemB", ...]
DEFAULT_BRAND_BY_SYSTEM = {"Cadre": "Cadre Wire Group"}


# =========================================================
# OUTPUT COLUMNS
# =========================================================
TARGET_COLUMNS = [
    "ReferralManager",
    "ReferralEmail",
    "Brand",
    "QuoteNumber",
    "QuoteDate",
    "Company",
    "FirstName",
    "LastName",
    "ContactEmail",
    "ContactPhone",
    "Address",
    "County",
    "City",
    "State",
    "ZipCode",
    "Country",
    "item_id",
    "item_desc",
    "UnitPrice",
    "TotalSales",
    "QuoteValidDate",
    "CustomerNumber",
    "manufacturer_Name",
    "PDF",
    "DemoQuote",
]


# =========================================================
# REGEX
# =========================================================
MONEY_RE = re.compile(r"^\$?[\d,]+\.\d{2,5}$")
QTY_RE = re.compile(r"^\d+(?:\.\d+)?$")
ITEM_ID_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-/_]+$")
ITEM_START_RE = re.compile(r"^(\d{1,4})\s+([A-Z0-9][A-Z0-9.\-/_]+)\b", re.ASCII)

# Stop only when the line clearly looks like totals header area (start of line), not anywhere in the line.
SUMMARY_STOP_LINE_RE = re.compile(
    r"^(Subtotal|Total\b|Grand\s+Total|Freight|Tax\b|Product\b)",
    re.IGNORECASE,
)


# =========================================================
# BASIC PDF TEXT HELPERS (header + tax)
# =========================================================
def extract_full_text(pdf_bytes: bytes) -> str:
    full_text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            full_text += txt + "\n"
    return full_text


def normalize_date_str(date_str: Optional[str]) -> Optional[str]:
    if not date_str:
        return None
    try:
        dt = datetime.strptime(date_str, "%m/%d/%Y")
        return dt.strftime("%m/%d/%Y")
    except Exception:
        return date_str


def extract_header_info(full_text: str) -> Dict[str, Optional[str]]:
    header: Dict[str, Optional[str]] = {}

    m_quote = re.search(r"Quote\s+(\d+)\s+Date\s+(\d{1,2}/\d{1,2}/\d{4})", full_text)
    if m_quote:
        header["QuoteNumber"] = m_quote.group(1)
        header["QuoteDate"] = m_quote.group(2)

    m_cust = re.search(r"Customer\s+(\d+)", full_text)
    if m_cust:
        header["CustomerNumber"] = m_cust.group(1)

    m_contact = re.search(r"Contact\s+([A-Za-z .'-]+)", full_text)
    if m_contact:
        name = m_contact.group(1).strip()
        parts = name.split()
        if len(parts) >= 2:
            header["FirstName"] = parts[0]
            header["LastName"] = " ".join(parts[1:])
        elif parts:
            header["FirstName"] = parts[0]

    m_sales = re.search(r"Salesperson\s+([A-Za-z .'-]+)", full_text)
    if m_sales:
        header["ReferralManager"] = m_sales.group(1).strip()

    if "Quoted For:" in full_text and "Quote Good Through" in full_text:
        try:
            start = full_text.index("Quoted For:")
            end = full_text.index("Quote Good Through")
            addr_block = full_text[start:end]
        except Exception:
            addr_block = ""

        m_company = re.search(r"Quoted For:\s*(.+?)\s+Ship To:", addr_block)
        if m_company:
            header["Company"] = m_company.group(1).strip()

        m_addr = re.search(r"(\d{3,6}\s+[A-Za-z0-9 .#-]+)", addr_block)
        if m_addr:
            header["Address"] = m_addr.group(1).strip()

        m_city = re.search(
            r"([A-Za-z .]+),\s*([A-Z]{2})\s+(\d{5})(?:-\d{4})?",
            addr_block,
        )
        if m_city:
            header["City"] = m_city.group(1).strip()
            header["State"] = m_city.group(2)
            header["ZipCode"] = m_city.group(3)

        if "United States of America" in addr_block:
            header["Country"] = "USA"

    m_valid = re.search(r"Quote Good Through\s+(\d{1,2}/\d{1,2}/\d{4})", full_text)
    if m_valid:
        header["QuoteValidDate"] = m_valid.group(1)

    return header


def extract_tax_amount(full_text: str) -> Optional[float]:
    # Try totals block first
    block = full_text
    if "Product" in full_text and "Total" in full_text:
        try:
            start = full_text.index("Product")
            end = full_text.index("Total", start)
            block = full_text[start:end]
        except Exception:
            block = full_text

    m = re.search(r"\bTax\s+([\d,]+\.\d{2})\b", block)
    if not m:
        m = re.search(r"\bTax\s+([\d,]+\.\d{2})\b", full_text)
        if not m:
            return None

    try:
        return float(m.group(1).replace(",", ""))
    except Exception:
        return None


def _safe_float(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    try:
        return float(str(s).replace("$", "").replace(",", ""))
    except Exception:
        return None


# =========================================================
# METHOD A: TABLE EXTRACTION (MOST RELIABLE WHEN IT WORKS)
# + IMPROVEMENT: also capture description from the NEXT ROW if the table row has no description
# =========================================================
def _clean_cell(x: Any) -> str:
    if x is None:
        return ""
    return str(x).replace("\n", " ").strip()


def _row_has_item_signature(cells: List[str]) -> bool:
    if len(cells) < 2:
        return False
    c0 = cells[0].strip()
    c1 = cells[1].strip()
    return c0.isdigit() and bool(ITEM_ID_RE.match(c1))


def _parse_item_from_table_row(cells: List[str]) -> Dict[str, str]:
    line_no = cells[0].strip()
    item_id = cells[1].strip()

    tokens: List[str] = []
    for c in cells[2:]:
        tokens.extend(c.split())

    qty = ""
    for t in tokens:
        if QTY_RE.match(t):
            qty = t
            break

    money = [t for t in tokens if MONEY_RE.match(t)]
    unit_price = money[-2] if len(money) >= 2 else (money[-1] if len(money) == 1 else "")
    total = money[-1] if len(money) >= 1 else ""

    desc_parts = []
    for c in cells[2:]:
        c_clean = c.strip()
        if not c_clean:
            continue
        toks = c_clean.split()
        non_num = [
            t
            for t in toks
            if not (
                QTY_RE.match(t)
                or MONEY_RE.match(t)
                or re.fullmatch(r"[A-Z/]+", t)
                or t.isdigit()
            )
        ]
        if non_num:
            desc_parts.append(c_clean)

    description = " ".join(desc_parts).strip()

    return {
        "line_no": line_no,
        "item_id": item_id,
        "qty": qty,
        "unit_price": unit_price,
        "total": total,
        "description": description,
    }


def _line_is_desc_continuation(line: str) -> bool:
    """
    True if the line looks like a description-only continuation line
    (i.e., not a new item line and not totals).
    """
    if not line:
        return False
    if ITEM_START_RE.match(line):
        return False
    if SUMMARY_STOP_LINE_RE.match(line):
        return False

    toks = line.split()
    # If it's mostly numeric/uom, it's not description
    wordy = any(
        not (QTY_RE.match(t) or MONEY_RE.match(t) or re.fullmatch(r"[A-Z/]+", t) or t.isdigit())
        for t in toks
    )
    return wordy


def extract_line_items_by_tables(pdf_bytes: bytes, debug: bool = False) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []

    table_settings_variants = [
        # Lattice-ish (lines)
        {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "edge_min_length": 20,
            "min_words_vertical": 1,
            "min_words_horizontal": 1,
            "intersection_tolerance": 3,
            "text_tolerance": 2,
        },
        # Stream-ish (no lines)
        {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "min_words_vertical": 1,
            "min_words_horizontal": 1,
            "text_tolerance": 2,
        },
    ]

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for pidx, page in enumerate(pdf.pages, start=1):
            page_items_found = False
            for settings in table_settings_variants:
                try:
                    tables = page.extract_tables(table_settings=settings) or []
                except Exception:
                    tables = []

                if debug and tables:
                    st.write(
                        f"DEBUG: page {pidx} tables found: {len(tables)} "
                        f"(settings={settings.get('vertical_strategy')}/{settings.get('horizontal_strategy')})"
                    )

                for table in tables:
                    if not table:
                        continue

                    # Convert to string cells
                    str_rows = []
                    for raw_row in table:
                        row_cells = [_clean_cell(c) for c in (raw_row or [])]
                        str_rows.append(row_cells)

                    # Parse rows, with description continuation support
                    r = 0
                    while r < len(str_rows):
                        cells = str_rows[r]
                        if cells and SUMMARY_STOP_LINE_RE.match(cells[0].strip()):
                            r += 1
                            continue

                        if _row_has_item_signature(cells):
                            item = _parse_item_from_table_row(cells)

                            # IMPROVEMENT: if description empty, take next row as description continuation (common in some PDFs)
                            if not item.get("description") and r + 1 < len(str_rows):
                                nxt_cells = str_rows[r + 1]
                                nxt_line = " ".join([c for c in nxt_cells if c]).strip()
                                if _line_is_desc_continuation(nxt_line):
                                    item["description"] = nxt_line.strip()
                                    r += 1  # consume next row

                            items.append(item)
                            page_items_found = True

                        r += 1

                if page_items_found:
                    break

    return items


# =========================================================
# METHOD B: WORD-COORDINATE ROW PARSER (FALLBACK)
# + IMPROVEMENT: allow 2 extra description lines AND allow "missing money" lines up to 4 lookahead
# =========================================================
def _group_words_into_rows(words, y_tol=2.5) -> List[List[dict]]:
    buckets = defaultdict(list)
    for w in words:
        key = round(w["top"] / y_tol) * y_tol
        buckets[key].append(w)

    rows = []
    for y in sorted(buckets.keys()):
        rows.append(sorted(buckets[y], key=lambda x: x["x0"]))
    return rows


def _row_to_text(row_words: List[dict]) -> str:
    return " ".join(w["text"] for w in row_words).strip()


def _parse_item_row_tokens(tokens: List[str]) -> Dict[str, Optional[str]]:
    line_no = tokens[0]
    item_id = tokens[1]
    after = tokens[2:]

    qty = None
    for t in after:
        if QTY_RE.match(t):
            qty = t
            break

    money_idxs = [i for i, t in enumerate(after) if MONEY_RE.match(t)]
    unit_price = None
    total = None
    if len(money_idxs) >= 2:
        total = after[money_idxs[-1]]
        unit_price = after[money_idxs[-2]]
    elif len(money_idxs) == 1:
        total = after[money_idxs[-1]]

    return {
        "line_no": line_no,
        "item_id": item_id,
        "qty": qty,
        "unit_price": unit_price,
        "total": total,
    }


def extract_line_items_by_words(pdf_bytes: bytes, debug: bool = False) -> List[Dict[str, str]]:
    visual_lines: List[str] = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for pidx, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(
                x_tolerance=2,
                y_tolerance=2,
                keep_blank_chars=False,
                use_text_flow=True,
            )
            if not words:
                continue

            rows = _group_words_into_rows(words, y_tol=2.5)
            for r in rows:
                txt = _row_to_text(r)
                if txt:
                    visual_lines.append(txt)

            if debug:
                st.write(f"DEBUG: page {pidx} word-rows: {len(rows)}")

    items: List[Dict[str, Optional[str]]] = []
    current: Optional[Dict[str, Optional[str]]] = None
    desc_parts: List[str] = []

    def flush():
        nonlocal current, desc_parts
        if current:
            current["description"] = " ".join([p for p in desc_parts if p]).strip() or ""
            items.append(current)
        current = None
        desc_parts = []

    i = 0
    while i < len(visual_lines):
        line = visual_lines[i].strip()

        if SUMMARY_STOP_LINE_RE.match(line):
            flush()
            break

        m = ITEM_START_RE.match(line)
        if m:
            flush()
            tokens = line.split()
            if len(tokens) >= 2:
                current = _parse_item_row_tokens(tokens)
                desc_parts = []

                # IMPROVEMENT: look further ahead for wrapped money lines (up to 4)
                need_unit = current.get("unit_price") is None
                need_total = current.get("total") is None
                if need_unit or need_total:
                    for j in (1, 2, 3, 4):
                        if i + j >= len(visual_lines):
                            break
                        nxt = visual_lines[i + j].strip()
                        if ITEM_START_RE.match(nxt) or SUMMARY_STOP_LINE_RE.match(nxt):
                            break
                        nxt_money = [t for t in nxt.split() if MONEY_RE.match(t)]
                        if nxt_money:
                            if need_total and current.get("total") is None:
                                current["total"] = nxt_money[-1]
                                need_total = False
                            if need_unit and len(nxt_money) >= 2 and current.get("unit_price") is None:
                                current["unit_price"] = nxt_money[-2]
                                need_unit = False

            i += 1
            continue

        if current:
            toks = line.split()
            wordy = any(
                not (
                    QTY_RE.match(t)
                    or MONEY_RE.match(t)
                    or re.fullmatch(r"[A-Z/]+", t)
                    or t.isdigit()
                )
                for t in toks
            )
            if wordy:
                # IMPROVEMENT: allow up to 3 description lines per item (some PDFs split desc)
                if len(desc_parts) < 3:
                    desc_parts.append(line)

        i += 1

    flush()

    cleaned: List[Dict[str, str]] = []
    for it in items:
        if not it.get("item_id"):
            continue
        # Require at least one money field to filter out false positives from headers/addresses
        if it.get("unit_price") or it.get("total"):
            cleaned.append(
                {
                    "line_no": str(it.get("line_no") or ""),
                    "item_id": str(it.get("item_id") or ""),
                    "qty": str(it.get("qty") or ""),
                    "unit_price": str(it.get("unit_price") or ""),
                    "total": str(it.get("total") or ""),
                    "description": str(it.get("description") or ""),
                }
            )
    return cleaned


# =========================================================
# HYBRID EXTRACTOR (TABLES FIRST, WORDS FALLBACK)
# =========================================================
def extract_line_items_hybrid(pdf_bytes: bytes, debug: bool = False) -> List[Dict[str, str]]:
    items = extract_line_items_by_tables(pdf_bytes, debug=debug)
    if not items:
        items = extract_line_items_by_words(pdf_bytes, debug=debug)

    # De-dup
    seen = set()
    out = []
    for it in items:
        k = (it.get("line_no", ""), it.get("item_id", ""), it.get("total", ""))
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


# =========================================================
# BUILD ROWS
# =========================================================
def build_rows_for_pdf(
    system_type: str,
    pdf_bytes: bytes,
    filename: str,
    debug: bool = False,
) -> List[Dict]:
    full_text = extract_full_text(pdf_bytes)
    header = extract_header_info(full_text)

    # HYBRID item extraction (tables first, then words)
    items = extract_line_items_hybrid(pdf_bytes, debug=debug)

    # TAX RULES:
    # - If tax == 0 or missing: DO NOT include
    # - If tax > 0: include, but item_desc must be blank
    tax_val = extract_tax_amount(full_text)
    if tax_val is not None and abs(tax_val) >= 0.005:
        tax_str = f"{tax_val:,.2f}"
        items.append(
            {
                "line_no": "TAX",
                "item_id": "Tax",
                "qty": "1",
                "unit_price": tax_str,
                "total": tax_str,
                "description": "",  # blank description
            }
        )

    quote_number_text = header.get("QuoteNumber")
    quote_date_text = normalize_date_str(header.get("QuoteDate"))
    quote_valid_text = normalize_date_str(header.get("QuoteValidDate"))

    brand = DEFAULT_BRAND_BY_SYSTEM.get(system_type, system_type)
    referral_manager = header.get("ReferralManager")

    rows: List[Dict] = []
    for it in items:
        unit_price_val = _safe_float(it.get("unit_price"))
        total_val = _safe_float(it.get("total"))

        rows.append(
            {
                "ReferralManager": referral_manager,
                "ReferralEmail": None,
                "Brand": brand,
                "QuoteNumber": quote_number_text,
                "QuoteDate": quote_date_text,
                "Company": header.get("Company"),
                "FirstName": header.get("FirstName"),
                "LastName": header.get("LastName"),
                "ContactEmail": None,
                "ContactPhone": None,
                "Address": header.get("Address"),
                "County": None,
                "City": header.get("City"),
                "State": header.get("State"),
                "ZipCode": header.get("ZipCode"),
                "Country": header.get("Country"),
                "item_id": it.get("item_id"),
                "item_desc": it.get("description"),
                "UnitPrice": unit_price_val,
                "TotalSales": total_val,
                "QuoteValidDate": quote_valid_text,
                "CustomerNumber": header.get("CustomerNumber"),
                "manufacturer_Name": None,
                "PDF": filename,
                "DemoQuote": None,
            }
        )

    return rows


# =========================================================
# STREAMLIT UI (CLEANER + ONE DOWNLOAD ZIP + CLEAR UPLOADS)
# - Only system type selection
# - Optional fields removed from UI
# - Uploads cleared after extraction
# - One download button to get CSV + PDFs in one ZIP
# =========================================================
st.set_page_config(page_title="Quote PDF Extractor", layout="wide")
st.title("Quote PDF Extractor")

st.markdown(
    """
1) Select **System type**  
2) Upload PDF(s)  
3) Click **Extract**  
4) Download **one ZIP** containing:
- extracted CSV
- uploaded PDFs
"""
)

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0
if "result_zip_bytes" not in st.session_state:
    st.session_state["result_zip_bytes"] = None
if "result_summary" not in st.session_state:
    st.session_state["result_summary"] = None
if "result_preview" not in st.session_state:
    st.session_state["result_preview"] = None

system_type = st.selectbox("System type", SYSTEM_TYPES, index=0)

debug_mode = st.checkbox(
    "Debug mode",
    value=False,
    help="Shows diagnostics about tables/words extraction for PDFs that miss line items.",
)

uploaded_files = st.file_uploader(
    "Upload PDF(s)",
    type=["pdf"],
    accept_multiple_files=True,
    key=f"pdf_uploader_{st.session_state['uploader_key']}",
)

extract_btn = st.button("Extract")

# Show results if available
if st.session_state["result_zip_bytes"] is not None:
    st.success(st.session_state["result_summary"] or "Done.")
    if st.session_state["result_preview"] is not None:
        st.subheader("Preview (first 50 rows)")
        st.dataframe(st.session_state["result_preview"], use_container_width=True)

    st.download_button(
        label="Download ZIP (CSV + PDFs)",
        data=st.session_state["result_zip_bytes"],
        file_name="extraction_output.zip",
        mime="application/zip",
    )

    if st.button("New extraction"):
        st.session_state["result_zip_bytes"] = None
        st.session_state["result_summary"] = None
        st.session_state["result_preview"] = None
        st.session_state["uploader_key"] += 1
        st.rerun()

if extract_btn:
    if not uploaded_files:
        st.error("Please upload at least one PDF.")
    else:
        file_data: List[Dict[str, bytes]] = []
        for f in uploaded_files:
            file_data.append({"name": f.name, "bytes": f.read()})

        all_rows: List[Dict] = []
        per_pdf_debug: List[Dict[str, Any]] = []

        progress = st.progress(0.0)
        status = st.empty()

        for idx, fd in enumerate(file_data, start=1):
            name = fd["name"]
            pdf_bytes = fd["bytes"]
            status.text(f"Processing {idx}/{len(file_data)}: {name}")

            try:
                rows = build_rows_for_pdf(
                    system_type=system_type,
                    pdf_bytes=pdf_bytes,
                    filename=name,
                    debug=debug_mode,
                )
                all_rows.extend(rows)

                if debug_mode:
                    per_pdf_debug.append(
                        {"PDF": name, "Rows": len(rows)}
                    )
            except Exception as e:
                st.warning(f"Error processing {name}: {e}")
                if debug_mode:
                    per_pdf_debug.append({"PDF": name, "Rows": 0, "Error": str(e)})

            progress.progress(idx / len(file_data))

        if not all_rows:
            st.error("No line items were found in the uploaded PDFs.")
        else:
            df = pd.DataFrame(all_rows, columns=TARGET_COLUMNS)

            # Build ONE ZIP: CSV + PDFs (+ optional debug CSV)
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(
                    "extracted/extracted_line_items.csv",
                    df.to_csv(index=False).encode("utf-8"),
                )

                for fd in file_data:
                    zf.writestr(f"pdfs/{fd['name']}", fd["bytes"])

                if debug_mode:
                    dbg_df = pd.DataFrame(per_pdf_debug)
                    zf.writestr(
                        "extracted/debug_counts.csv",
                        dbg_df.to_csv(index=False).encode("utf-8"),
                    )

            zip_buf.seek(0)

            st.session_state["result_zip_bytes"] = zip_buf.getvalue()
            st.session_state["result_summary"] = f"Parsed {len(file_data)} PDF(s) with {len(df)} total rows."
            st.session_state["result_preview"] = df.head(50)

            # Clear uploaded files
            st.session_state["uploader_key"] += 1
            st.rerun()
