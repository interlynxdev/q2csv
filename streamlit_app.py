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
# Money-like tokens sometimes include a unit suffix in these PDFs, e.g. "1,730.00000MFT" or "0.90000EAC".
MONEY_RE = re.compile(r"^\$?[\d,]+\.\d{2,5}[A-Za-z]{0,6}$")
# Quantities can be comma-formatted, e.g. "2,000"
QTY_RE = re.compile(r"^\d+(?:,\d{3})*(?:\.\d+)?$")
# Item IDs can include '#', e.g. "W.#2-TINNED/BARE/SOLID"
ITEM_ID_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-/_#]+$")
# Avoid \b (word boundary) because '.' and '#' are not word-chars; require whitespace after item token instead.
ITEM_START_RE = re.compile(r"^(\d{1,4})\s+([A-Z0-9][A-Z0-9.\-/_#]+)(?=\s)", re.ASCII)

# Short all-caps tokens are typically UOM codes (EAC/FT/MFT/EA/PK/etc). Longer all-caps words are often descriptions.
UOM_RE = re.compile(r"^[A-Z/]{1,4}$")

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


def extract_quoted_for_from_layout(pdf_bytes: bytes) -> Dict[str, Optional[str]]:
    """
    Robustly extract the *Quoted For* (left column) address using word coordinates.

    Why: In these Cadre PDFs, `page.extract_text()` often merges the left and right columns
    into the same lines (Quoted For + Ship To), which breaks regex-only parsing.
    """
    out: Dict[str, Optional[str]] = {
        "Company": None,
        "Address": None,
        "City": None,
        "State": None,
        "ZipCode": None,
        "Country": None,
    }

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if not pdf.pages:
                return out
            page = pdf.pages[0]
            words = page.extract_words(
                x_tolerance=2,
                y_tolerance=2,
                keep_blank_chars=False,
                use_text_flow=True,
            ) or []
            if not words:
                return out

            # Find the "Quoted" anchor to limit the vertical region we consider.
            anchor = next((w for w in words if w.get("text") == "Quoted"), None)
            if not anchor:
                return out

            y0 = anchor["top"] - 6
            y1 = anchor["top"] + 260  # enough to include country line, but not the entire page
            mid_x = page.width / 2

            left_words = [w for w in words if y0 <= w["top"] <= y1 and w["x0"] < mid_x]
            if not left_words:
                return out

            # Group words into visual lines by y buckets
            buckets: Dict[float, List[dict]] = defaultdict(list)
            for w in left_words:
                key = round(w["top"] / 3) * 3
                buckets[key].append(w)

            lines: List[str] = []
            for y in sorted(buckets.keys()):
                row = sorted(buckets[y], key=lambda ww: ww["x0"])
                line = " ".join(ww["text"] for ww in row).strip()
                if line:
                    lines.append(line)

            # Find start line with "Quoted For:" and stop before totals/header sections.
            start_idx = None
            for i, ln in enumerate(lines):
                if ln.lower().startswith("quoted for:"):
                    start_idx = i
                    break
            if start_idx is None:
                return out

            stop_re = re.compile(r"^(Quote\s+Good\s+Through|Ship\s+Via|Line\s+Item|Product\b|Tax\b|Total\b)", re.IGNORECASE)
            addr_lines: List[str] = []
            for ln in lines[start_idx:]:
                if stop_re.match(ln):
                    break
                addr_lines.append(ln)

            if not addr_lines:
                return out

            # addr_lines[0] is "Quoted For: <Company>"
            first = addr_lines[0]
            company = first.split(":", 1)[1].strip() if ":" in first else first.strip()
            if company:
                out["Company"] = company

            # Remaining lines should look like:c
            # street
            # optional suite
            # city, ST zip
            # optional country
            rest = [ln.strip() for ln in addr_lines[1:] if ln.strip()]
            city_re = re.compile(r"^([A-Za-z .]+),\s*([A-Z]{2})\s+(\d{5})(?:-\d{4})?$")

            city_idx = None
            for i, ln in enumerate(rest):
                if city_re.match(ln):
                    city_idx = i
                    break

            if city_idx is not None:
                m = city_re.match(rest[city_idx])
                if m:
                    out["City"] = m.group(1).strip()
                    out["State"] = m.group(2)
                    out["ZipCode"] = m.group(3)

                street_parts = rest[:city_idx]
                if street_parts:
                    out["Address"] = ", ".join(street_parts)

                if city_idx + 1 < len(rest):
                    # Whatever is on the next line is treated as country (if it doesn't look like a date/number)
                    c = rest[city_idx + 1].strip()
                    if c and not re.match(r"^\d", c):
                        out["Country"] = c
            else:
                # Fallback: if no city line, treat all remaining lines as Address
                if rest:
                    out["Address"] = ", ".join(rest)

    except Exception:
        return out

    return out


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

    # Extract "Quoted For" address using pattern matching (more robust for PDF column layouts)
    if "Quoted For:" in full_text:
        try:
            start = full_text.index("Quoted For:") + len("Quoted For:")
            # Find the end - either "Ship To:" or "Quote Good Through"
            end = len(full_text)
            if "Ship To:" in full_text:
                ship_to_pos = full_text.index("Ship To:")
                if ship_to_pos > start:
                    end = ship_to_pos
            if "Quote Good Through" in full_text:
                qgt_pos = full_text.index("Quote Good Through")
                if qgt_pos > start:
                    end = min(end, qgt_pos)
            quoted_for_block = full_text[start:end].strip()
        except Exception:
            quoted_for_block = ""

        if quoted_for_block:
            # Strategy: Use regex to find specific patterns rather than relying on line splits
            # This handles PDFs where columns are merged into one text blob

            # Company: first "word group" (text before first address-like pattern)
            # Address pattern: starts with digits (street number)
            m_company = re.match(r'^([A-Z][A-Za-z0-9 &.\'-]+?)(?=\s+\d{1,6}\s+[A-Z]|\s*$)', quoted_for_block)
            if m_company:
                header["Company"] = m_company.group(1).strip()

            # Street address: starts with number, capture until city/state/zip pattern
            # Pattern: "1234 Street Name" possibly with "Suite/Apt/Unit #"
            m_street = re.search(
                r'(\d{1,6}\s+[A-Za-z0-9 .#\'-]+?)(?:\s+((?:SUITE|STE|APT|UNIT|BLDG|FLOOR|FL|#)\s*[A-Za-z0-9\-]+))?'
                r'(?=\s+[A-Za-z .]+,\s*[A-Z]{2}\s+\d{5}|\s*$)',
                quoted_for_block,
                re.IGNORECASE
            )
            if m_street:
                addr = m_street.group(1).strip()
                if m_street.group(2):
                    addr += ", " + m_street.group(2).strip()
                header["Address"] = addr

            # City, State, Zip - find FIRST occurrence
            m_city = re.search(
                r'([A-Za-z .]+),\s*([A-Z]{2})\s+(\d{5})(?:-\d{4})?',
                quoted_for_block,
            )
            if m_city:
                header["City"] = m_city.group(1).strip()
                header["State"] = m_city.group(2)
                header["ZipCode"] = m_city.group(3)

            # Country: text after city/state/zip that looks like a country name
            if m_city:
                after_city = quoted_for_block[m_city.end():].strip()
                # Take first word group that could be a country (letters/spaces only, not starting with digit)
                m_country = re.match(r'^([A-Za-z ]+?)(?:\s+[A-Z]{2}\s+\d{5}|\s+\d|\s*$)', after_city)
                if m_country:
                    country = m_country.group(1).strip()
                    # Avoid capturing duplicate city names or other patterns
                    if country and len(country) > 2 and not re.match(r'^[A-Z]{2}$', country):
                        header["Country"] = country

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
        # Extract leading numeric portion from tokens like "1,730.00000MFT"
        m = re.search(r"[\d,]+(?:\.\d+)?", str(s))
        if not m:
            return None
        return float(m.group(0).replace("$", "").replace(",", ""))
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

    # Normalize money-like tokens by stripping any unit suffix.
    money = [re.sub(r"[A-Za-z]+$", "", t) for t in tokens if MONEY_RE.match(t)]
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
                or UOM_RE.match(t)
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
        not (QTY_RE.match(t) or MONEY_RE.match(t) or UOM_RE.match(t) or t.isdigit())
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
        total = re.sub(r"[A-Za-z]+$", "", after[money_idxs[-1]])
        unit_price = re.sub(r"[A-Za-z]+$", "", after[money_idxs[-2]])
    elif len(money_idxs) == 1:
        total = re.sub(r"[A-Za-z]+$", "", after[money_idxs[-1]])

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
                        nxt_money = [re.sub(r"[A-Za-z]+$", "", t) for t in nxt.split() if MONEY_RE.match(t)]
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
                    or UOM_RE.match(t)
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
    # Override Quoted For address using layout-based extraction (more reliable for 2-column PDFs)
    qf = extract_quoted_for_from_layout(pdf_bytes)
    for k, v in qf.items():
        if v:  # only overwrite when we successfully extracted something
            header[k] = v

    # HYBRID item extraction (tables first, then words)
    items = extract_line_items_hybrid(pdf_bytes, debug=debug)

    # NOTE: We intentionally do NOT append Tax as a line item.
    # Some downstream systems want only product line items; tax can be handled separately from totals.

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
