"""
=============================================================================
  VENDOR ASSESSMENT — DATA INGESTION SCRIPT
  v2 — corrected vendor relationships
=============================================================================

SOURCE FILES (all sit in the same VENDORANAYLSIS folder as this script):

  Performance_Tools.xlsx
      Sheet 1 : "PO_transactions"              <- PO history, lead times
      Sheet 2 : "Perfomance tools MASTER TABLE" <- Sage sales data (your SQL output)

  CPL35 Jan 02 2026 Price File.xlsx
      *** This is Performance Tool's OWN published price catalogue ***
      *** (Wilmar Corporation / Performance Tool brand) — NOT a separate vendor ***
      Sheet "Data"   : full item list with metadata + list prices
      Sheet "Sheet1" : tiered pricing columns CA35 / CA40

  251030 Lista precios METABO -JMC.xlsx
      Sheet "Price List" : Metabo price list (Germany, Caribbean distributor copy)

  price list -new (1).pdf
      Ronix price list (Iran, FOB pricing — needs freight factor adjustment)

OUTPUT  ->  VENDORANAYLSIS/processed/
    01_po_transactions.csv              Performance Tool PO history
    01b_po_lead_time_summary.csv        Avg/min/max lead time per item
    02_sage_master.csv                  Your internal sales data (from Sage)
    03_performance_tool_catalogue.csv   CPL35 Data sheet + Sheet1 tier prices MERGED
    04_metabo_prices.csv                Metabo price list
    05_ronix_prices.csv                 Ronix FOB prices + estimated landed cost
    06_combined_vendor_prices.csv       All 3 vendor price lists unified (for fuzzy match)

VENDOR SUMMARY:
    Performance Tool  ->  Wilmar Corporation, Kent WA, USA   (CURRENT vendor)
    Metabo            ->  Metabowerke GmbH, Nurtingen, Germany (NEW)
    Ronix             ->  Ronix Tools, Iran  — FOB pricing    (NEW)
    Workpro          ->  Workpro, China  — published price list  (NEW)

INSTALL ONCE:
    pip install pandas openpyxl pdfplumber
=============================================================================
"""

import warnings
import pandas as pd
import pdfplumber
from pathlib import Path

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATHS  — script must sit in VENDORANAYLSIS/ alongside all source files
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "processed"

# Your internal data (Sage export)
PT_INTERNAL_FILE = BASE_DIR / "Performance_Tools.xlsx"
PT_PO_SHEET      = "PO_transactions"
PT_MASTER_SHEET  = "Perfomance tools MASTER TABLE "   # with trailing space

# Workpro published price catalogue (China)
# NEW vendor — price list only, not yet a current supplier
PT_CATALOGUE_FILE  = BASE_DIR / "CPL35 Jan 02 2026 Price File.xlsx"
PT_CATALOGUE_SHEET = "Data"       # full item detail with metadata
PT_TIER_SHEET      = "Sheet1"     # CA35 / CA40 tiered pricing

# New vendors
METABO_FILE  = BASE_DIR / "251030 Lista precios METABO -JMC.xlsx"
METABO_SHEET = "Price List"

RONIX_PDF = BASE_DIR / "price list -new (1).pdf"

# Exchange rates for USD to JPY conversion
EXCHANGE_RATES_FILE = BASE_DIR / "Search BOJ Counter Rates.csv"

# ---------------------------------------------------------------------------
# Ronix is FOB (port of origin, Iran). Adjust this multiplier to reflect
# your actual freight + import duty rate for your territory.
# Example: 1.30 = you estimate 30% uplift to get to your landed cost.
# ---------------------------------------------------------------------------
RONIX_FREIGHT_FACTOR = 1.30


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Output folder: {OUTPUT_DIR}\n")


def save(df: pd.DataFrame, filename: str, label: str) -> pd.DataFrame:
    out = OUTPUT_DIR / filename
    df.to_csv(out, index=False)
    print(f"[OK] {label:<48}  {len(df):>6,} rows  ->  {filename}")
    return df


def check(path: Path) -> bool:
    if not path.exists():
        print(f"[!!] FILE NOT FOUND: {path.name}")
        return False
    return True


def clean_price_col(series: pd.Series) -> pd.Series:
    """Strip $, US$, commas, spaces then cast to float."""
    return pd.to_numeric(
        series.astype(str).str.replace(r"[US$,\s]", "", regex=True),
        errors="coerce"
    )


def load_exchange_rates():
    """Load USD to JMD exchange rates from BOJ (Bank of Jamaica) file. Returns the latest DMD & TT rate."""
    if not check(EXCHANGE_RATES_FILE):
        return None
    
    df = pd.read_csv(EXCHANGE_RATES_FILE)
    df["Date"] = pd.to_datetime(df["Date"], format="%d %b %Y")
    df = df.sort_values("Date")
    latest_rate = df.iloc[-1]["DMD & TT"]  # Latest rate
    print(f"[OK] Loaded exchange rates. Latest USD/JMD rate: {latest_rate}")
    return latest_rate


# ===========================================================================
# SECTION A — YOUR INTERNAL DATA  (Performance_Tools.xlsx)
# ===========================================================================

# ---------------------------------------------------------------------------
# A1.  PO TRANSACTIONS
#      Performance_Tools.xlsx  ->  sheet: PO_transactions
#
# Columns: ITEMNO | VENDORID | Vendor Item Number | PORHSEQ |
#          ORDDATE | RECPDATE | Days * Quantity Received | Lead time |
#          po_unit_cost | QTYORDERED | QTYRECEIVED
# ---------------------------------------------------------------------------
def load_po_transactions():
    if not check(PT_INTERNAL_FILE):
        return None

    try:
        df = pd.read_excel(
            PT_INTERNAL_FILE,
            sheet_name=PT_PO_SHEET,
            dtype={"ITEMNO": str, "VENDORID": str}
        )
    except ValueError as e:
        if "Worksheet named" in str(e):
            xls = pd.ExcelFile(PT_INTERNAL_FILE)
            available = [s.strip() for s in xls.sheet_names]
            print(f"[SKIP] PO transactions sheet '{PT_PO_SHEET}' not found in {PT_INTERNAL_FILE}")
            print(f"[INFO] Available sheets: {available}")
            alt = next((s for s in available if "po" in s.lower() or "transaction" in s.lower()), None)
            if alt:
                print(f"[INFO] Found alternate sheet '{alt}', trying that instead.")
                df = pd.read_excel(
                    PT_INTERNAL_FILE,
                    sheet_name=alt,
                    dtype={"ITEMNO": str, "VENDORID": str}
                )
            else:
                return None
        else:
            raise

    df.columns = df.columns.str.strip()

    df.rename(columns={
        "Vendor Item Number":       "vendor_item_number",
        "Days * Quantity Received": "days_x_qty_received",
        "Days x Quantity Received": "days_x_qty_received",
        "Lead time":                "lead_time_days",
        "QTYORDERED":               "qty_ordered",
        "QTYRECEIVED":              "qty_received",
    }, inplace=True)

    # Handle truncated column name  QTYRE...
    for col in list(df.columns):
        if col.upper().startswith("QTYRE") and col != "qty_received":
            df.rename(columns={col: "qty_received"}, inplace=True)
            break

    df["ITEMNO"] = df["ITEMNO"].astype(str).str.strip()

    # --- Clean vendor_item_number: flag status values as empty ----------------
    # Vendor item numbers that are status flags (NYA, No Longer Available,
    # Obsolete, Discontinued, N/A etc.) are set to empty string so they are
    # not used in fuzzy matching. Valid item numbers are kept as-is.
    if "vendor_item_number" in df.columns:
        import re as _re
        _VIN_STATUS = _re.compile(
            r"^\s*(n/?a|no longer available|not available|nya|obsolete|"
            r"discontinued|discont|unavailable|none|null|nan|"
            r"replaced|delisted|phase.?out|tbd|pending|unknown)\s*$",
            _re.IGNORECASE,
        )
        _VIN_SUFFIX = _re.compile(
            r"\s*[-–]\s*(discontinued|discont|n/?a|no longer|not available|"
            r"nya|obsolete|replaced|phase.?out|unavail).*$",
            _re.IGNORECASE,
        )
        def _clean_vin(raw):
            s = str(raw).strip()
            if not s or s.lower() in ("nan", "none", ""):
                return ""
            if _VIN_STATUS.match(s):
                return ""          # entire value is a status flag — discard
            s = _VIN_SUFFIX.sub("", s).strip()   # strip trailing status text
            return s

        original = df["vendor_item_number"].copy()
        df["vendor_item_number"] = df["vendor_item_number"].apply(_clean_vin)
        flagged = (original.astype(str).str.strip() != "") & (df["vendor_item_number"] == "")
        kept    = (df["vendor_item_number"] != "").sum()
        print(f"[INFO] vendor_item_number: {kept} valid, {flagged.sum()} status flags cleared (NYA/Obsolete/etc.)")

    for dc in ["ORDDATE", "RECPDATE"]:
        if dc in df.columns:
            df[dc] = pd.to_datetime(df[dc], errors="coerce")

    for nc in ["lead_time_days", "po_unit_cost", "qty_ordered",
               "qty_received", "days_x_qty_received"]:
        if nc in df.columns:
            df[nc] = pd.to_numeric(df[nc], errors="coerce")

    save(df, "01_po_transactions.csv", "PO transactions (Performance Tool)")

    # Per-item lead time summary — used in convenience score
    if "lead_time_days" in df.columns:
        summary = (
            df.groupby("ITEMNO")["lead_time_days"]
            .agg(
                avg_lead_time_days="mean",
                min_lead_time_days="min",
                max_lead_time_days="max",
                po_order_count="count",
            )
            .reset_index()
            .round(1)
        )
        save(summary, "01b_po_lead_time_summary.csv",
             "PO lead time summary per item")

    return df


# ---------------------------------------------------------------------------
# A2.  SAGE MASTER TABLE
#      Performance_Tools.xlsx  ->  sheet: Perfomance tools MASTER TABLE
#
# Columns: ITEMNO | GL Account | Sub Category Name | item_description |
#          CATEGORY | vendor_name | performance_tool_price |
#          sales_frequency_YYYY | total_volume_YYYY | total_revenue_YYYY |
#          avg_price_YYYY  (2020 - 2026) | lifetime_* totals
# ---------------------------------------------------------------------------
def load_sage_master(po_df=None):
    if not check(PT_INTERNAL_FILE):
        return None

    df = pd.read_excel(
        PT_INTERNAL_FILE,
        sheet_name=PT_MASTER_SHEET,
        dtype={"ITEMNO": str}
    )
    df.columns = df.columns.str.strip()
    df["ITEMNO"] = df["ITEMNO"].astype(str).str.strip()

    numeric_keywords = [
        "revenue", "price", "volume", "frequency",
        "performance_tool_price", "lifetime"
    ]
    for col in df.columns:
        if any(kw in col.lower() for kw in numeric_keywords):
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r"[$,]", "", regex=True),
                errors="coerce"
            ).fillna(0)

    # --- Map vendor_item_number from PO transactions onto Sage master --------
    # This gives the fuzzy-match pass 2 what it needs without loading PO sheet
    # again at match time.  We take the most common (mode) vendor item number
    # per ITEMNO in case there are multiple POs with different values.
    if po_df is not None and "vendor_item_number" in po_df.columns:
        vin_map = (
            po_df[po_df["vendor_item_number"].str.strip() != ""]
            .groupby("ITEMNO")["vendor_item_number"]
            .agg(lambda x: x.mode().iloc[0] if len(x.mode()) else "")
            .reset_index()
            .rename(columns={"vendor_item_number": "vendor_item_number"})
        )
        df = df.merge(vin_map, on="ITEMNO", how="left")
        df["vendor_item_number"] = df["vendor_item_number"].fillna("")
        mapped_count = (df["vendor_item_number"] != "").sum()
        print(f"[INFO] vendor_item_number mapped onto Sage master: {mapped_count} of {len(df)} items")

    return save(df, "02_sage_master.csv", "Sage master table (internal sales data)")


# ===========================================================================
# SECTION B — CPL PRICE CATALOGUE
#             CPL35 Jan 02 2026 Price File.xlsx
#             Workpro published price list.
# ===========================================================================

# ---------------------------------------------------------------------------
# B1.  CPL35 DATA SHEET  — full item catalogue with metadata
#
# Columns: Item | Description | Current price | CPL35 (1.1.26) | Change % |
#          Standard Pack | Inner Qty | Master Qty | UPC | Class | Class Desc |
#          Brand | Brand Desc | Subclass | Subclass Desc | Group | New
# ---------------------------------------------------------------------------
def load_pt_catalogue():
    if not check(PT_CATALOGUE_FILE):
        return None

    # --- Data sheet ---
    df_data = pd.read_excel(
        PT_CATALOGUE_FILE,
        sheet_name=PT_CATALOGUE_SHEET,
        dtype={"Item": str}
    )
    df_data.columns = df_data.columns.str.strip()

    df_data.rename(columns={
        "Item":            "item_number",
        "Description":     "item_description",
        "Current price":   "pt_current_price",
        "CPL35 Current":   "pt_current_price",
        "CPL35 (1.1.26)":  "pt_cpl35_price",
        "CPL35 1.1.26":    "pt_cpl35_price",
        "CPL35 1.1.26 ":   "pt_cpl35_price",
        "Change %":        "pt_price_change_pct",
        "Change":          "pt_price_change_pct",
        "Standard Pack":   "standard_pack",
        "Inner Qty":       "inner_qty",
        "Master Qty":      "master_qty",
        "UPC":             "upc",
        "Class":           "class_code",
        "Class Desc":      "class_desc",
        "Brand":           "brand_code",
        "Brand Desc":      "brand_desc",
        "Subclass":        "subclass_code",
        "Subclass Desc":   "subclass_desc",
        "Group":           "group_name",
        "New":             "is_new_item",
    }, inplace=True)

    df_data["item_number"] = df_data["item_number"].astype(str).str.strip()
    df_data.dropna(subset=["item_number"], inplace=True)

    for col in ["pt_current_price", "pt_cpl35_price", "pt_price_change_pct"]:
        if col in df_data.columns:
            df_data[col] = clean_price_col(df_data[col])

    # --- Sheet1: tier pricing  (Item | CA35 | CA40) ---
    df_tier = pd.read_excel(
        PT_CATALOGUE_FILE,
        sheet_name=PT_TIER_SHEET,
        dtype={"Item": str}
    )
    df_tier.columns = df_tier.columns.str.strip()

    df_tier.rename(columns={
        "Item":  "item_number",
        "CA35":  "pt_price_ca35",
        "CA40":  "pt_price_ca40",
    }, inplace=True)
    df_tier["item_number"] = df_tier["item_number"].astype(str).str.strip()

    for col in ["pt_price_ca35", "pt_price_ca40"]:
        if col in df_tier.columns:
            df_tier[col] = clean_price_col(df_tier[col])

    # Merge data sheet + tier pricing on item_number
    df = df_data.merge(df_tier, on="item_number", how="left")

    # Tag clearly so there is no confusion downstream
    df["vendor"]       = "Workpro"
    df["vendor_brand"] = "Workpro"
    df["country"]      = "China"
    df["price_type"]   = "landed"
    df["source_file"]  = "CPL35 Jan 02 2026 Price File.xlsx"

    return save(df, "03_CPL_catalogue.csv",
                "Workpro catalogue (CPL35 file + Sheet1 tiers)")


# ===========================================================================
# SECTION C — NEW VENDORS
# ===========================================================================

# ---------------------------------------------------------------------------
# C1.  METABO PRICE LIST
#      251030 Lista precios METABO -JMC.xlsx  ->  sheet: Price List
#      Metabowerke GmbH, Nurtingen, Germany
#      "JMC" = Caribbean/Latin America distributor copy
#
# Columns: Code | Description | Group | Price 2024/2025 US$ | Valid till | Notes
# ---------------------------------------------------------------------------
def load_metabo():
    if not check(METABO_FILE):
        return None

    df = pd.read_excel(
        METABO_FILE,
        sheet_name=METABO_SHEET,
        dtype={"Code": str},
        header=0
    )
    df.columns = df.columns.str.strip()
    df.dropna(how="all", inplace=True)

    df.rename(columns={
        "Code":                 "item_number",
        "Description":          "item_description",
        "Group":                "group_name",
        "Price 2024/2025 U$S":  "vendor_price_usd",
        "Valid till":           "price_valid_till",
        "Notes":                "notes",
    }, inplace=True)

    df["item_number"] = df["item_number"].astype(str).str.strip()

    if "vendor_price_usd" in df.columns:
        df["vendor_price_usd"] = clean_price_col(df["vendor_price_usd"])

    if "price_valid_till" in df.columns:
        df["price_valid_till"] = pd.to_datetime(
            df["price_valid_till"], errors="coerce"
        )
        df["price_expired"] = df["price_valid_till"] < pd.Timestamp.today()

    # Flag items noted as phasing out or needing availability confirmation
    if "notes" in df.columns:
        df["phase_out_flag"] = df["notes"].astype(str).str.contains(
            r"phase out|discontinu|confirmar|disponibil",
            case=False, na=False
        )

    df["vendor"]       = "Metabo"
    df["vendor_brand"] = "Metabowerke GmbH"
    df["country"]      = "Germany"
    df["price_type"]   = "landed"
    df["source_file"]  = "251030 Lista precios METABO -JMC.xlsx"

    return save(df, "04_metabo_prices.csv", "Metabo price list")


# ---------------------------------------------------------------------------
# C2.  RONIX PRICE LIST  (PDF)
#      price list -new (1).pdf
#      Ronix Tools, Iran
#      Price is FOB — NOT landed. Use RONIX_FREIGHT_FACTOR to estimate
#      your actual landed cost for fair comparison with other vendors.
#
# Columns: ItemCode | Product Name | Picture | Description |
#          Quantity | Quantity Per Carton | CBM Per Carton |
#          Total Cartons | Total CBM | FOB Unit Price (USD) | Total Amount
# ---------------------------------------------------------------------------
def load_ronix():
    if not check(RONIX_PDF):
        return None

    all_rows = []
    header   = None

    with pdfplumber.open(RONIX_PDF) as pdf:
        for page in pdf.pages:
            for tbl in (page.extract_tables() or []):
                for row in tbl:
                    if not any(c for c in row if c and str(c).strip()):
                        continue
                    if header is None:
                        header = [
                            str(c).strip().replace("\n", " ") if c else f"col_{i}"
                            for i, c in enumerate(row)
                        ]
                    else:
                        all_rows.append(row)

    if not all_rows or header is None:
        print("[!!] Ronix PDF: could not extract tables.")
        print("     Try: open in Excel -> Save As -> .xlsx, place in same folder,")
        print("     rename to match RONIX_PDF path, then re-run.")
        return None

    df = pd.DataFrame(all_rows, columns=header)

    # Drop columns that aren't useful for pricing analysis
    drop_kw = ["picture", "total amount", "total carton", "total cbm"]
    df.drop(
        columns=[c for c in df.columns if any(k in c.lower() for k in drop_kw)],
        errors="ignore", inplace=True
    )

    # Normalise column names (PDF text can vary slightly page to page)
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if any(x in cl for x in ["itemcode", "item code"]):
            col_map[col] = "item_number"
        elif "product name" in cl:
            col_map[col] = "item_description"
        elif "description" in cl and "product" not in cl:
            col_map[col] = "product_description"
        elif "qty per carton" in cl or "quantity per carton" in cl:
            col_map[col] = "qty_per_carton"
        elif "cbm per" in cl:
            col_map[col] = "cbm_per_carton"
        elif "fob unit" in cl or ("unit price" in cl):
            col_map[col] = "fob_unit_price_usd"
        elif "quantity" in cl and "per" not in cl and "total" not in cl:
            col_map[col] = "order_quantity"
    df.rename(columns=col_map, inplace=True)

    # Keep only rows with a valid numeric item code
    if "item_number" in df.columns:
        df["item_number"] = df["item_number"].astype(str).str.strip()
        df = df[df["item_number"].str.match(r"^\d{3,}", na=False)].copy()

    if "fob_unit_price_usd" in df.columns:
        df["fob_unit_price_usd"] = clean_price_col(df["fob_unit_price_usd"])

        # Estimated landed cost = FOB x freight factor
        # Update RONIX_FREIGHT_FACTOR at the top of this file for your market
        df["estimated_landed_price_usd"] = (
            df["fob_unit_price_usd"] * RONIX_FREIGHT_FACTOR
        ).round(4)
        df["freight_factor_applied"] = RONIX_FREIGHT_FACTOR
        df["NOTE_ronix_pricing"] = (
            f"FOB Iran. Landed estimate uses x{RONIX_FREIGHT_FACTOR} multiplier. "
            "Update RONIX_FREIGHT_FACTOR for your actual freight + duty rate."
        )

    df["vendor"]       = "Ronix"
    df["vendor_brand"] = "Ronix Tools"
    df["country"]      = "China"
    df["price_type"]   = "fob"
    df["source_file"]  = "price list -new (1).pdf"

    return save(df, "05_ronix_prices.csv", "Ronix price list (PDF extracted)")


# ===========================================================================
# SECTION D — COMBINED VENDOR PRICE FILE
# Stacks Performance Tool, Metabo and Ronix into one file.
# This is what the fuzzy matching script will join against your Sage master.
# ===========================================================================
def build_combined(pt_df, metabo_df, ronix_df, usd_jpy_rate=None):

    frames = []

    if pt_df is not None:
        slim = pt_df[["item_number", "item_description",
                       "vendor", "vendor_brand", "country", "price_type"]].copy()
        # Use the new CPL35 price as the primary cost; fall back to current price
        if "pt_cpl35_price" in pt_df.columns:
            slim["vendor_price_usd"] = pt_df["pt_cpl35_price"].values
        elif "pt_current_price" in pt_df.columns:
            slim["vendor_price_usd"] = pt_df["pt_current_price"].values
        # Keep tier prices for reference
        for tier in ["pt_price_ca35", "pt_price_ca40"]:
            if tier in pt_df.columns:
                slim[tier] = pt_df[tier].values
        frames.append(slim)

    if metabo_df is not None:
        slim = metabo_df[["item_number", "item_description", "vendor_price_usd",
                           "vendor", "vendor_brand", "country", "price_type"]].copy()
        if "phase_out_flag" in metabo_df.columns:
            slim["phase_out_flag"] = metabo_df["phase_out_flag"].values
        frames.append(slim)

    if ronix_df is not None:
        # Use estimated landed price (not raw FOB) for fair comparison
        price_col = (
            "estimated_landed_price_usd"
            if "estimated_landed_price_usd" in ronix_df.columns
            else "fob_unit_price_usd"
        )
        slim = ronix_df[["item_number", "item_description",
                          price_col, "vendor", "vendor_brand",
                          "country", "price_type"]].copy()
        slim.rename(columns={price_col: "vendor_price_usd"}, inplace=True)
        # Preserve raw FOB as reference column
        if "fob_unit_price_usd" in ronix_df.columns and price_col != "fob_unit_price_usd":
            slim["ronix_fob_price_usd"] = ronix_df["fob_unit_price_usd"].values
        frames.append(slim)

    if not frames:
        print("[!!] No vendor data loaded — combined file skipped.")
        return None

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["vendor_price_usd"] = pd.to_numeric(
        combined["vendor_price_usd"], errors="coerce"
    )
    combined.dropna(subset=["item_number", "item_description"], inplace=True)
    combined = combined[
        combined["item_description"].astype(str).str.strip() != ""
    ].copy()

    # Add JMD prices using BOJ exchange rates
    if usd_jpy_rate is not None:
        combined["vendor_price_jmd"] = combined["vendor_price_usd"] * usd_jpy_rate

    return save(combined, "06_combined_vendor_prices.csv",
                "Combined vendor prices (all vendors unified)")



# ===========================================================================
# SECTION E — PT vs VENDOR MATCH  (Sage ITEMNO → each vendor separately)
#
# For each Sage item, each vendor is searched in strict priority order:
#   Step 1 — Sage ITEMNO       vs vendor item_number       (keep >= 80%)
#   Step 2 — Sage vendor_item_number vs vendor item_number  (keep >= 80%)
#   Step 3 — Sage description  vs vendor description        (keep >= 55%)
#
# Steps run sequentially per Sage item per vendor — if Step 1 finds a
# match the item is confirmed and Steps 2 & 3 are skipped for that vendor.
# Each vendor is searched independently so there is no cross-contamination.
#
# Output: 07_matched_vendor_prices.csv
#   One row per (Sage item × vendor) confirmed match.
#   Columns include: sage_ITEMNO, sage_description, sage_pt_cost,
#                    vendor, item_number, item_description, vendor_price_usd,
#                    match_score, match_step
# ===========================================================================
PT_ITEMNO_THRESHOLD = 80   # item-number match floor (Steps 1 & 2)
PT_DESC_THRESHOLD   = 55   # description match floor (Step 3 fallback)


def build_fuzzy_matched(sage_df, po_df,
                        workpro_df=None, metabo_df=None, ronix_df=None):
    """
    Match each Sage item against each vendor separately.
    Step 1: Sage ITEMNO vs vendor item_number.
    Step 2: Sage vendor_item_number vs vendor item_number.
    Step 3: Sage description vs vendor description (fallback only).
    """
    if sage_df is None:
        print("[!!] PT-vs-vendor match skipped — no Sage master.")
        return None

    try:
        from rapidfuzz import process as rf_process, fuzz as rf_fuzz
    except ImportError:
        print("[!!] rapidfuzz not installed — run: pip install rapidfuzz")
        return None

    import numpy as np

    # ── Resolve vendor frames and their price columns ────────────────────────
    vendor_frames = {}
    if workpro_df is not None:
        pcol = ("pt_cpl35_price"   if "pt_cpl35_price"   in workpro_df.columns else
                "pt_current_price" if "pt_current_price" in workpro_df.columns else None)
        if pcol:
            vendor_frames["Workpro"] = (workpro_df.copy().reset_index(drop=True), pcol)

    if metabo_df is not None and "vendor_price_usd" in metabo_df.columns:
        vendor_frames["Metabo"] = (metabo_df.copy().reset_index(drop=True), "vendor_price_usd")

    if ronix_df is not None:
        rcol = ("estimated_landed_price_usd"
                if "estimated_landed_price_usd" in ronix_df.columns else
                "fob_unit_price_usd"
                if "fob_unit_price_usd"          in ronix_df.columns else None)
        if rcol:
            vendor_frames["Ronix"] = (ronix_df.copy().reset_index(drop=True), rcol)

    if not vendor_frames:
        print("[!!] PT-vs-vendor match skipped — no vendor dataframes.")
        return None

    # ── Sage side ────────────────────────────────────────────────────────────
    sage = sage_df.copy().reset_index(drop=True)
    sage["_ino"]  = sage["ITEMNO"].astype(str).str.strip().str.lower()
    sage["_desc"] = sage["item_description"].fillna("").astype(str).str.strip().str.lower()

    # vendor_item_number from master (baked in by load_sage_master) + PO fallback
    sage["_vin"] = sage.get("vendor_item_number", pd.Series("", index=sage.index))
    sage["_vin"] = sage["_vin"].fillna("").astype(str).str.strip().str.lower()
    sage["_vin"] = sage["_vin"].replace({"nan": "", "none": ""})
    if po_df is not None and "vendor_item_number" in po_df.columns:
        po_map = (
            po_df.dropna(subset=["ITEMNO", "vendor_item_number"])
            .assign(
                _k=lambda d: d["ITEMNO"].astype(str).str.strip().str.lower(),
                _v=lambda d: d["vendor_item_number"].astype(str).str.strip().str.lower(),
            )
            .groupby("_k")["_v"]
            .agg(lambda x: x.mode().iloc[0] if len(x.mode()) else "")
        )
        missing = sage["_vin"] == ""
        sage.loc[missing, "_vin"] = sage.loc[missing, "_ino"].map(po_map).fillna("")

    all_matched = []

    # ── Match against each vendor independently ──────────────────────────────
    for vendor_name, (vdf, price_col) in vendor_frames.items():
        print(f"[..] PT vs {vendor_name}: {len(vdf):,} vendor items …")

        # Vendor index
        vdf["_vino"]  = vdf["item_number"].astype(str).str.strip().str.lower()
        vdf["_vdesc"] = vdf["item_description"].astype(str).str.strip().str.lower()
        vdf["_vprice"]= pd.to_numeric(vdf[price_col], errors="coerce")

        vendor_inos  = vdf["_vino"].tolist()
        vendor_descs = vdf["_vdesc"].tolist()

        # ── Step 1: Sage ITEMNO vs vendor item_number ────────────────────────
        sage_inos = sage["_ino"].tolist()
        mat1 = rf_process.cdist(
            sage_inos, vendor_inos,
            scorer=rf_fuzz.token_sort_ratio,
            score_cutoff=0, workers=-1,
        )   # shape: (n_sage, n_vendor)

        # ── Step 2: Sage vendor_item_number vs vendor item_number ────────────
        vin_queries = sage["_vin"].tolist()
        active_vin  = [(i, q) for i, q in enumerate(vin_queries) if q]
        mat2 = None
        if active_vin:
            mat2 = rf_process.cdist(
                [q for _, q in active_vin], vendor_inos,
                scorer=rf_fuzz.token_sort_ratio,
                score_cutoff=0, workers=-1,
            )

        # ── Step 3: Sage description vs vendor description ───────────────────
        sage_descs    = sage["_desc"].tolist()
        active_descs  = [(i, d) for i, d in enumerate(sage_descs) if d]
        mat3 = None
        if active_descs:
            mat3 = rf_process.cdist(
                [d for _, d in active_descs], vendor_descs,
                scorer=rf_fuzz.token_set_ratio,
                score_cutoff=0, workers=-1,
            )

        # ── Build step-2 and step-3 score lookup (sage_idx -> best_vendor_idx, score) ──
        step2_best = {}   # sage_idx -> (vendor_idx, score)
        if mat2 is not None:
            for pi, (si, _) in enumerate(active_vin):
                best_col = int(np.argmax(mat2[pi]))
                sc       = float(mat2[pi][best_col])
                step2_best[si] = (best_col, sc)

        step3_best = {}   # sage_idx -> (vendor_idx, score)
        if mat3 is not None:
            for pi, (si, _) in enumerate(active_descs):
                best_col = int(np.argmax(mat3[pi]))
                sc       = float(mat3[pi][best_col])
                step3_best[si] = (best_col, sc)

        # ── For each Sage item: step 1 → step 2 → step 3 ────────────────────
        vendor_matched = 0
        for si in range(len(sage)):
            sr = sage.iloc[si]

            # Step 1 — Sage ITEMNO
            best1_col = int(np.argmax(mat1[si]))
            sc1       = float(mat1[si][best1_col])
            if sc1 >= PT_ITEMNO_THRESHOLD:
                vi = best1_col
                step = "step1_itemno"
                sc   = sc1
            else:
                # Step 2 — Sage vendor_item_number
                vi2, sc2 = step2_best.get(si, (-1, 0.0))
                if sc2 >= PT_ITEMNO_THRESHOLD and vi2 >= 0:
                    vi   = vi2
                    step = "step2_vendor_itemno"
                    sc   = sc2
                else:
                    # Step 3 — description fallback
                    vi3, sc3 = step3_best.get(si, (-1, 0.0))
                    if sc3 >= PT_DESC_THRESHOLD and vi3 >= 0:
                        vi   = vi3
                        step = "step3_description"
                        sc   = sc3
                    else:
                        continue   # no match for this Sage item / vendor pair

            vrow = vdf.iloc[vi]
            vp   = float(vrow["_vprice"]) if pd.notna(vrow["_vprice"]) else None
            all_matched.append({
                "sage_ITEMNO":      sr["ITEMNO"],
                "sage_description": sr.get("item_description", ""),
                "sage_pt_cost":     sr.get("performance_tool_price", None),
                "vendor":           vendor_name,
                "item_number":      vrow["item_number"],
                "item_description": vrow["item_description"],
                "vendor_price_usd": vp,
                "match_score":      round(sc, 1),
                "match_step":       step,
            })
            vendor_matched += 1

        print(f"[INFO]   {vendor_name}: {vendor_matched} Sage items matched")

    if not all_matched:
        print("[!!] No PT-vs-vendor matches found.")
        return None

    df_matched = pd.DataFrame(all_matched).reset_index(drop=True)
    save(df_matched, "07_matched_vendor_prices.csv",
         "PT-vs-vendor matches (step1=itemno, step2=vendor_ino, step3=desc)")

    # Unmatched — Sage items with no match in ANY vendor
    matched_inos = set(df_matched["sage_ITEMNO"].astype(str).str.strip())
    unmatched = sage[~sage["ITEMNO"].astype(str).str.strip().isin(matched_inos)]
    if len(unmatched):
        save(
            unmatched[["ITEMNO", "item_description"]].rename(
                columns={"ITEMNO": "sage_ITEMNO", "item_description": "sage_description"}
            ),
            "07b_unmatched_sage_items.csv",
            "Sage items with no vendor match across all vendors"
        )

    total   = len(sage)
    matched = len(matched_inos)
    print(f"[INFO] Total: {matched}/{total} Sage items matched in at least one vendor "
          f"({100*matched/total:.1f}%)")
    print(f"[INFO] Total rows: {len(df_matched)} "
          f"(step1={sum(1 for r in all_matched if r['match_step']=='step1_itemno')}, "
          f"step2={sum(1 for r in all_matched if r['match_step']=='step2_vendor_itemno')}, "
          f"step3={sum(1 for r in all_matched if r['match_step']=='step3_description')})")

    return df_matched



# ===========================================================================
# SECTION F — VENDOR vs VENDOR MATCH  (Workpro / Metabo / Ronix pairwise)
#
# For each vendor pair, items are matched in two steps then validated:
#   Step 1 — item_number vs item_number  (keep >= 80%)
#   Step 2 — description vs description  (keep >= 55%, only for items that
#             did NOT get a Step 1 match)
#   Step 3 — LOW-SCORE item_number matches (60–79%) are re-checked against
#             description: if description >= 50% the match is kept and
#             tagged "itemno_desc_confirmed"; otherwise discarded
#
# Output: 08_cross_vendor_matches.csv
# ===========================================================================
CV_ITEMNO_HIGH    = 80   # item-number floor — confirmed without extra check
CV_ITEMNO_LOW     = 60   # item-number "maybe" range — needs description check
CV_DESC_THRESHOLD = 55   # description-only match floor (Step 2)
CV_DESC_CONFIRM   = 50   # description score needed to promote low itemno match


def _strip_status_cv(raw: str) -> str:
    """Strip status suffixes from vendor item_number."""
    import re as _re
    _STATUS = _re.compile(
        r"\s*[-–—]\s*(discontinued|discont|disc|n/?a|no longer|not available|"
        r"nya|obsolete|delisted|replaced|phase.?out|unavail)[^$]*$",
        _re.IGNORECASE,
    )
    _NA_ONLY = _re.compile(
        r"^\s*(n/?a|no longer available|not available|nya|obsolete|discontinued|"
        r"unavailable|none|null|nan)\s*$",
        _re.IGNORECASE,
    )
    s = str(raw).strip()
    if _NA_ONLY.match(s):
        return ""
    s = _STATUS.sub("", s).strip()
    import re
    m = re.match(r"([A-Za-z0-9][A-Za-z0-9\-\._]*)", s)
    return m.group(1).lower() if m else ""


def build_cross_vendor_matches(workpro_df, metabo_df, ronix_df):
    """
    Match vendors against each other pairwise.
    Step 1: item_number → item_number (>=80% keep, 60-79% check desc).
    Step 2: description → description for items with no item_number match.
    Step 3: 60-79% item_number matches boosted by description confirmation.
    """
    try:
        from rapidfuzz import process as rf_process, fuzz as rf_fuzz
    except ImportError:
        print("[!!] rapidfuzz not installed — cross-vendor match skipped.")
        return None

    import numpy as np

    def _prep(df, vendor_name, price_col):
        if df is None or df.empty:
            return None
        out = df[["item_number", "item_description"]].copy().reset_index(drop=True)
        out["vendor"]   = vendor_name
        out["price"]    = pd.to_numeric(df[price_col], errors="coerce")
        out["_ino"]     = out["item_number"].astype(str).str.strip().str.lower()
        out["_ino_cl"]  = out["_ino"].apply(_strip_status_cv)
        out["_desc"]    = out["item_description"].astype(str).str.strip().str.lower()
        return out

    # Resolve vendor frames
    frames = {}
    if workpro_df is not None:
        pcol = ("pt_cpl35_price"   if "pt_cpl35_price"   in workpro_df.columns else
                "pt_current_price" if "pt_current_price" in workpro_df.columns else None)
        if pcol:
            frames["Workpro"] = _prep(workpro_df, "Workpro", pcol)

    if metabo_df is not None and "vendor_price_usd" in metabo_df.columns:
        frames["Metabo"] = _prep(metabo_df, "Metabo", "vendor_price_usd")

    if ronix_df is not None:
        rcol = ("estimated_landed_price_usd"
                if "estimated_landed_price_usd" in ronix_df.columns else
                "fob_unit_price_usd"
                if "fob_unit_price_usd"          in ronix_df.columns else None)
        if rcol:
            frames["Ronix"] = _prep(ronix_df, "Ronix", rcol)

    if len(frames) < 2:
        print("[!!] Need at least 2 vendors for cross-vendor matching.")
        return None

    vendor_names = list(frames.keys())
    pairs = [(vendor_names[i], vendor_names[j])
             for i in range(len(vendor_names))
             for j in range(i + 1, len(vendor_names))]

    all_matches = []
    seen_pairs  = set()

    def _emit(row_a, row_b, va, vb, score, method):
        key = (va, row_a["_ino"], vb, row_b["_ino"])
        rev = (vb, row_b["_ino"], va, row_a["_ino"])
        if key in seen_pairs or rev in seen_pairs:
            return
        seen_pairs.add(key)
        pa = row_a["price"]
        pb = row_b["price"]
        all_matches.append({
            "vendor_a":         va,
            "item_number_a":    row_a["item_number"],
            "description_a":    row_a["item_description"],
            "price_a_usd":      round(float(pa), 2) if pd.notna(pa) else None,
            "vendor_b":         vb,
            "item_number_b":    row_b["item_number"],
            "description_b":    row_b["item_description"],
            "price_b_usd":      round(float(pb), 2) if pd.notna(pb) else None,
            "item_match_score": round(score, 1),
            "match_method":     method,
        })

    for va, vb in pairs:
        fa = frames[va]
        fb = frames[vb]
        print(f"[..] Cross-matching {va} ({len(fa):,}) vs {vb} ({len(fb):,}) …")

        inos_a  = fa["_ino"].tolist()
        inos_b  = fb["_ino"].tolist()
        descs_a = fa["_desc"].tolist()
        descs_b = fb["_desc"].tolist()

        # ── Step 1: item_number A vs item_number B ───────────────────────────
        mat_ino = rf_process.cdist(
            inos_a, inos_b,
            scorer=rf_fuzz.token_sort_ratio,
            score_cutoff=0, workers=-1,
        )

        # ── Step 2 / 3: description A vs description B ───────────────────────
        mat_desc = rf_process.cdist(
            descs_a, descs_b,
            scorer=rf_fuzz.token_set_ratio,
            score_cutoff=0, workers=-1,
        )

        step1_confirmed = set()   # row indices in fa that got a Step 1 match
        low_score_rows  = []      # (ia, ib, sc_ino) for 60-79% item_number hits

        for ia in range(len(fa)):
            best_ib  = int(np.argmax(mat_ino[ia]))
            sc_ino   = float(mat_ino[ia][best_ib])

            if sc_ino >= CV_ITEMNO_HIGH:
                # High-confidence item_number match — keep directly
                _emit(fa.iloc[ia], fb.iloc[best_ib], va, vb, sc_ino, "itemno")
                step1_confirmed.add(ia)

            elif sc_ino >= CV_ITEMNO_LOW:
                # Low-confidence item_number match — queue for Step 3 desc check
                low_score_rows.append((ia, best_ib, sc_ino))

        # ── Step 2: description match for unmatched rows ─────────────────────
        for ia in range(len(fa)):
            if ia in step1_confirmed:
                continue   # already matched by item_number
            best_ib  = int(np.argmax(mat_desc[ia]))
            sc_desc  = float(mat_desc[ia][best_ib])
            if sc_desc >= CV_DESC_THRESHOLD:
                _emit(fa.iloc[ia], fb.iloc[best_ib], va, vb, sc_desc, "description")

        # ── Step 3: re-check low item_number scores with description ─────────
        for ia, ib, sc_ino in low_score_rows:
            sc_desc = float(mat_desc[ia][ib])   # desc score for the SAME pair
            if sc_desc >= CV_DESC_CONFIRM:
                # Both item_number and description agree — promote
                _emit(fa.iloc[ia], fb.iloc[ib], va, vb,
                      (sc_ino + sc_desc) / 2, "itemno_desc_confirmed")

        n_pair = sum(1 for m in all_matches
                     if m["vendor_a"] in (va, vb) and m["vendor_b"] in (va, vb))
        print(f"[INFO]   {va} <-> {vb}: {n_pair} matched pairs")

    if not all_matches:
        print("[!!] No cross-vendor matches found.")
        return None

    df_cross = pd.DataFrame(all_matches).reset_index(drop=True)
    save(df_cross, "08_cross_vendor_matches.csv",
         "Cross-vendor matches (Workpro / Metabo / Ronix pairwise)")
    print(f"[INFO] Total cross-vendor pairs: {len(df_cross)}")
    return df_cross



# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("\n" + "=" * 65)
    print("  VENDOR ASSESSMENT — DATA INGESTION  v2")
    print("=" * 65)

    print("""
VENDOR MAP:
  Performance Tool  (CURRENT)  Wilmar Corp, Kent WA, USA
    Internal data  <- Performance_Tools.xlsx  (both sheets)
  Workpro  (NEW)               Workpro, China
    Price list     <- CPL35 Jan 02 2026 Price File.xlsx  (Workpro catalogue)
  Metabo  (NEW)               Metabowerke GmbH, Germany
    Price list     <- 251030 Lista precios METABO -JMC.xlsx
  Ronix  (NEW)                Ronix Tools, Iran  [FOB pricing]
    Price list     <- price list -new (1).pdf
""")

    ensure_output_dir()

    print("--- Loading exchange rates ---")
    usd_jpy_rate = load_exchange_rates()

    print("--- A: Internal data (Performance_Tools.xlsx) ---")
    po_df     = load_po_transactions()
    master_df = load_sage_master(po_df=po_df)

    print("\n--- B: CPL catalogue (CPL35 Jan 02 2026 Price File.xlsx) ---")
    pt_df     = load_pt_catalogue()

    print("\n--- C1: Metabo prices ---")
    metabo_df = load_metabo()

    print("\n--- C2: Ronix prices (PDF) ---")
    ronix_df  = load_ronix()

    print("\n--- D: Building combined vendor price file ---")
    combined_df = build_combined(pt_df, metabo_df, ronix_df, usd_jpy_rate)

    print("\n--- E: PT vs Vendor match (Sage -> each vendor separately) ---")
    print("    Step 1: Sage ITEMNO          -> vendor item_number  (>= 80%)")
    print("    Step 2: Sage vendor_item_no  -> vendor item_number  (>= 80%)")
    print("    Step 3: Sage description     -> vendor description   (>= 55%, fallback)")
    _matched_df = build_fuzzy_matched(
        master_df, po_df,
        workpro_df=pt_df,
        metabo_df=metabo_df,
        ronix_df=ronix_df,
    )

    print("\n--- F: Vendor vs Vendor match (Workpro / Metabo / Ronix pairwise) ---")
    print("    Step 1: item_number vs item_number  (>= 80% keep, 60-79% check desc)")
    print("    Step 2: description vs description   (>= 55%, for unmatched items)")
    print("    Step 3: low item_number (60-79%) + description >= 50% = confirmed")
    _cross_df = build_cross_vendor_matches(pt_df, metabo_df, ronix_df)

    print("\n" + "=" * 65)
    print("  INGESTION COMPLETE")
    print("=" * 65)
    print(f"\nAll CSVs written to:  {OUTPUT_DIR.resolve()}\n")

    print("Files:")
    for f in sorted(OUTPUT_DIR.glob("*.csv")):
        kb = f.stat().st_size / 1024
        print(f"  {f.name:<55} {kb:>7.1f} KB")

    print("\nNext step -> streamlit run vendor_app.py\n")


if __name__ == "__main__":
    main()