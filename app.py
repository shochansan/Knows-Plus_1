import io
import re
import json
import math
import zipfile
import tempfile
import datetime as dt
import unicodedata
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import streamlit as st

try:
    from numbers_parser import Document
    HAS_NUMBERS = True
except Exception:
    HAS_NUMBERS = False

# =========================================================
# Unified Config
# =========================================================
APP_TITLE = "GPS統合分析アプリ"
APP_VERSION = "2026-04-11_unified_tabs_v2_points_labels"
POSITION_ORDER = ["GK", "CB", "SB", "MF", "SH", "FW"]
STREAMLIT_FILE_TYPES = ["csv", "xlsx", "xls", "numbers"]
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".numbers"}
ZERO_WIDTH = ["\u200b", "\u200c", "\u200d", "\ufeff"]

RATIO_BOOKLET_METRICS = [
    "Distance_vsGame_pct",
    "SI_D_vsGame_pct",
    "HI_D_vsGame_pct",
    "Sprint_vsGame_pct",
    "SPD_MX_vsGame_pct",
    "High Agility_vsGame_pct",
    "dis(m)/min_vsGame_pct",
    "SI D/min_vsGame_pct",
    "HI D/min_vsGame_pct",
    "High Agility/min_vsGame_pct",
    "Accel >2m/s/s(n)_vsGame_pct",
    "Decel>2m/s/s(n)_vsGame_pct",
]

WEEK_PATTERNS = {
    "月〜土": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
    "火〜日": ["Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
}

WEEKLY_METRICS = [
    "Distance_vsGame_pct",
    "HI_D_vsGame_pct",
    "SI_D_vsGame_pct",
    "Accel >2m/s/s(n)_vsGame_pct",
    "Decel>2m/s/s(n)_vsGame_pct",
    "HighAgility_vsGame_pct",
    "SPD_MX_vsGame_pct",
    "dis(m)/min_vsGame_pct_All",
    "dis(m)/min_vsGame_pct_MX",
    "High Agility/min(n/min)_vsGame_pct_All",
    "High Agility/min(n/min)_vsGame_pct_MX",
]

ACWR_METRICS = [
    ("Distance", "Distance"),
    ("HI_D", "HI_D"),
    ("Accel", "Accel"),
    ("Decel", "Decel"),
    ("High_Agility", "High Agility"),
]

ACUTE_DAYS = 7
CHRONIC_DAYS = 28
ACUTE_MIN_DAYS = 3
CHRONIC_MIN_DAYS = 14
DANGER_LOW = 0.8
DANGER_HIGH = 1.3
LAMBDA_OPTIONS = {
    "弱め (λ=0.1)": 0.1,
    "標準 (λ=0.2)": 0.2,
    "強め (λ=0.3)": 0.3,
}

POINT_ALPHA = 0.25
LABEL_ALPHA = 0.55
POINT_SIZE = 22
LABEL_FONTSIZE = 7
MEDIAN_FONTSIZE = 8
JITTER_WIDTH = 0.16
SHOW_VALUE_IN_LABEL = False
WEEKLY_POINT_ALPHA = 0.28
WEEKLY_LABEL_ALPHA = 0.18
WEEKLY_POINT_SIZE = 18
WEEKLY_LABEL_FONTSIZE = 7

WEEKLY_METRIC_COLORS = {
    "Distance_vsGame_pct": "#66bb6a",
    "HI_D_vsGame_pct": "#9ccc65",
    "SI_D_vsGame_pct": "#4dd0e1",
    "Accel >2m/s/s(n)_vsGame_pct": "#ffffff",
    "Decel>2m/s/s(n)_vsGame_pct": "#ffffff",
    "HighAgility_vsGame_pct": "#ffee58",
    "SPD_MX_vsGame_pct": "#ffffff",
    "dis(m)/min_vsGame_pct_All": "#ef5350",
    "dis(m)/min_vsGame_pct_MX": "#f8bbd0",
    "High Agility/min(n/min)_vsGame_pct_All": "#ef5350",
    "High Agility/min(n/min)_vsGame_pct_MX": "#f8bbd0",
}


# =========================================================
# Matplotlib / text utils
# =========================================================
def setup_matplotlib_japanese_font():
    import matplotlib
    from matplotlib import font_manager

    preferred = [
        "IPAexGothic", "IPAGothic",
        "Noto Sans CJK JP", "Noto Sans JP",
        "Hiragino Sans", "Hiragino Kaku Gothic ProN",
        "Yu Gothic", "YuGothic",
        "Meiryo", "MS Gothic",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42


def nfkc(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x)
    for zw in ZERO_WIDTH:
        s = s.replace(zw, "")
    s = s.replace("\u00A0", " ").replace("\u3000", " ")
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\t\r\n]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def norm_colname(x) -> str:
    s = nfkc(x)
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("％", "%").replace("－", "-").replace("ー", "-")
    s = s.replace("/", "").replace("\\", "")
    s = s.replace("_", "").replace("-", "")
    s = s.replace(">", "")
    s = re.sub(r"\s+", "", s)
    return s.lower()


def metric_key(x) -> str:
    s = nfkc(x).lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def normalize_person_key(x) -> str:
    return nfkc(x).replace(" ", "").replace("\u3000", "")


def normalize_position(x) -> str:
    s = nfkc(x).upper()
    mapping = {
        "GOALKEEPER": "GK",
        "CENTER BACK": "CB",
        "SIDE BACK": "SB",
        "MIDFIELDER": "MF",
        "SIDE HALF": "SH",
        "FORWARD": "FW",
        "DF": "CB",
        "GK": "GK",
        "CB": "CB",
        "SB": "SB",
        "MF": "MF",
        "SH": "SH",
        "FW": "FW",
    }
    if s in mapping:
        return mapping[s]
    return s if s else "Unknown"


def sort_positions_fixed(values: List[str]) -> List[str]:
    vals = []
    for v in values:
        p = normalize_position(v)
        if p not in vals:
            vals.append(p)
    base = [p for p in POSITION_ORDER if p in vals]
    rest = sorted([p for p in vals if p not in POSITION_ORDER and p != "Unknown"])
    if "Unknown" in vals:
        return base + rest + ["Unknown"]
    return base + rest


def normalize_session_name(x) -> str:
    return nfkc(x)


def normalize_session_compact(x) -> str:
    s = nfkc(x).lower()
    s = re.sub(r"[\s_/\\()\[\]{}\-]+", "", s)
    return s


def is_all_session(x) -> bool:
    s = normalize_session_compact(x)
    return s.startswith("all") and "roundall" not in s


def is_round_all_session(x) -> bool:
    s = normalize_session_compact(x)
    return "roundall" in s


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._ -]+", "_", str(name)).replace(" ", "_")


def nice_ylim(values, base_min=0.0, default_max=100.0):
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if len(arr) == 0:
        return base_min, default_max
    vmax = float(arr.max())
    upper = max(default_max, math.ceil(vmax / 10.0) * 10.0)
    return base_min, upper


def nice_sym_ylim(values, default_abs=100.0):
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if len(arr) == 0:
        return -default_abs, default_abs
    m = max(default_abs, math.ceil(float(np.abs(arr).max()) / 10.0) * 10.0)
    return -m, m


def make_columns_unique(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.loc[:, ~pd.Index(df.columns).duplicated(keep="first")].copy()


# =========================================================
# File readers
# =========================================================
def validate_uploaded_file(uploaded_file, label: str):
    if uploaded_file is None:
        return
    ext = Path(uploaded_file.name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"{label} は未対応形式です: {uploaded_file.name}")


def read_csv_any(raw: bytes) -> pd.DataFrame:
    last_error = None
    for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis", "latin1"]:
        try:
            return make_columns_unique(pd.read_csv(io.BytesIO(raw), encoding=enc))
        except Exception as e:
            last_error = e
    raise last_error


def detect_header_row(raw_df: pd.DataFrame, required_tokens: List[str], scan_rows: int = 80) -> Optional[int]:
    tokens = [norm_colname(t) for t in required_tokens]
    best_row, best_hit = None, -1
    n = min(len(raw_df), scan_rows)
    for i in range(n):
        row_texts = [norm_colname(v) for v in raw_df.iloc[i].tolist() if nfkc(v) != ""]
        hit = sum(1 for t in tokens if t in row_texts)
        if hit > best_hit:
            best_hit = hit
            best_row = i
    if best_row is not None and best_hit >= max(2, int(len(tokens) * 0.3)):
        return best_row
    return None


def read_excel_any(raw: bytes, required_tokens: Optional[List[str]] = None) -> pd.DataFrame:
    try:
        return make_columns_unique(pd.read_excel(io.BytesIO(raw)))
    except Exception:
        raw_df = pd.read_excel(io.BytesIO(raw), header=None)
        if required_tokens:
            h = detect_header_row(raw_df, required_tokens)
        else:
            h = 0
        header = [nfkc(v) for v in raw_df.iloc[h].tolist()]
        data = raw_df.iloc[h + 1:].copy()
        data.columns = header
        return make_columns_unique(data.reset_index(drop=True))


def _clean_numbers_cell(x):
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    return str(x)


def _df_from_numbers_rows(rows: List[List[object]], required_tokens: Optional[List[str]] = None) -> pd.DataFrame:
    cleaned = [[_clean_numbers_cell(c) for c in r] for r in rows]
    cleaned = [r for r in cleaned if any(nfkc(c) != "" for c in r)]
    if len(cleaned) < 2:
        return pd.DataFrame()
    if required_tokens:
        raw = pd.DataFrame(cleaned)
        h = detect_header_row(raw, required_tokens)
        if h is None:
            h = 0
    else:
        h = 0
    header = [nfkc(c) if nfkc(c) != "" else f"col_{i}" for i, c in enumerate(cleaned[h])]
    data = cleaned[h + 1:]
    if not data:
        return pd.DataFrame(columns=header)
    max_len = len(header)
    padded = [(list(r) + [None] * max(0, max_len - len(r)))[:max_len] for r in data]
    df = pd.DataFrame(padded, columns=header)
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return make_columns_unique(df)


def read_numbers_any(raw: bytes, required_tokens: Optional[List[str]] = None) -> pd.DataFrame:
    if not HAS_NUMBERS:
        raise ValueError("numbers_parser がインストールされていないため .numbers を読み込めません。")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".numbers") as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        doc = Document(tmp_path)
        candidates = []
        for sh in doc.sheets:
            for tbl in sh.tables:
                rows = list(tbl.rows(values_only=True))
                if not rows:
                    continue
                df = _df_from_numbers_rows(rows, required_tokens=required_tokens)
                if df.empty:
                    continue
                score = 0
                cols = {norm_colname(c) for c in df.columns}
                if required_tokens:
                    score += sum(1 for t in required_tokens if norm_colname(t) in cols)
                score += len(df.columns) * 0.01
                candidates.append((score, df))
        if not candidates:
            raise ValueError("Numbersファイルから表を取得できませんでした。")
        candidates.sort(key=lambda x: x[0], reverse=True)
        return make_columns_unique(candidates[0][1])
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def read_any_file(uploaded_file, required_tokens: Optional[List[str]] = None) -> pd.DataFrame:
    validate_uploaded_file(uploaded_file, uploaded_file.name)
    raw = uploaded_file.getvalue()
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return read_csv_any(raw)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return read_excel_any(raw, required_tokens=required_tokens)
    if name.endswith(".numbers"):
        return read_numbers_any(raw, required_tokens=required_tokens)
    raise ValueError("未対応形式です。")


# =========================================================
# Column helpers
# =========================================================
def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cmap = {norm_colname(c): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        nc = norm_colname(cand)
        if nc in cmap:
            return cmap[nc]
    return None


def require_cols(df: pd.DataFrame, needed: Dict[str, List[str]], label: str) -> Dict[str, str]:
    out = {}
    missing = []
    for logical, cands in needed.items():
        col = find_col(df, cands)
        if col is None:
            missing.append((logical, cands))
        else:
            out[logical] = col
    if missing:
        detail = "\n".join([f"- {logical}: {cands}" for logical, cands in missing])
        raise ValueError(f"{label} に必要列が見つかりません。\n{detail}")
    return out


def standardize_match_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        c = nfkc(col).lower()
        if c in ["name", "player", "player name", "選手名", "氏名", "名前"]:
            rename[col] = "Name"
        elif c in ["position", "pos", "ポジション"]:
            rename[col] = "Position"
        elif c in ["session", "メニュー", "セッション"]:
            rename[col] = "Session"
        elif c in ["duration_tf", "duration", "時間", "duration tf"]:
            rename[col] = "Duration_TF"
        elif c in ["distance", "td", "total distance", "距離"]:
            rename[col] = "Distance"
        elif c in ["si_d", "si d", "sid"]:
            rename[col] = "SI_D"
        elif c in ["hi_d", "hi d", "hid"]:
            rename[col] = "HI_D"
        elif c in ["sprint", "sprint(n)", "sprint count"]:
            rename[col] = "Sprint"
        elif c in ["accel_z2", "accel z2", "accel_z2_x"]:
            rename[col] = "Accel_Z2"
        elif c in ["accel_z3", "accel z3", "accel_z3_x"]:
            rename[col] = "Accel_Z3"
        elif c in ["decel_z2", "decel z2", "decel_z2_x"]:
            rename[col] = "Decel_Z2"
        elif c in ["decel_z3", "decel z3", "decel_z3_x"]:
            rename[col] = "Decel_Z3"
        elif c in ["spd mx", "spd_mx", "spd_max", "max speed", "最高速度"]:
            rename[col] = "SPD MX"
    return make_columns_unique(df.rename(columns=rename))


def to_num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def parse_duration_to_min(x) -> float:
    if x is None:
        return np.nan
    try:
        if pd.isna(x):
            return np.nan
    except Exception:
        pass
    if isinstance(x, (int, float, np.integer, np.floating)):
        v = float(x)
        if v < 0:
            return np.nan
        if v < 1.5:
            return v * 24 * 60
        if v >= 300:
            return v / 60.0
        return v
    s = nfkc(x)
    if s == "":
        return np.nan
    if re.fullmatch(r"\d+:\d{1,2}:\d{1,2}", s):
        h, m, sec = map(int, s.split(":"))
        return h * 60 + m + sec / 60.0
    if re.fullmatch(r"\d+:\d{1,2}", s):
        m, sec = map(int, s.split(":"))
        return m + sec / 60.0
    s2 = s.replace(",", "")
    try:
        return parse_duration_to_min(float(s2))
    except Exception:
        return np.nan


# =========================================================
# Shared calculations for GPS ratio app
# =========================================================
def game_required_tokens() -> List[str]:
    return [
        "Name", "名前",
        "Distance", "Distance_90",
        "SI_D", "SI D", "SID", "SI_D_90",
        "HI_D", "HI D", "HID", "HI_D_90",
        "SPD_Max", "SPD MX", "SPD_MX", "Max Speed",
        "Accel", "Accel_Total", "Accel_Total_90",
        "Decel", "Decel_Total", "Decel_Total_90",
        "High Agility", "HighAgility", "High Agility_90", "HighAgility_90",
        "Distance_per_min", "Distance per min", "dis(m)/min",
        "SI D/min", "HI D/min",
        "HighAgility_per_min", "High Agility_per_min", "High Agility per min", "High Agility/min(n/min)",
    ]


def build_session_metrics(train_df: pd.DataFrame) -> pd.DataFrame:
    need = {
        "Name": ["Name", "名前"],
        "Session": ["Session", "セッション"],
        "Duration_TF": ["Duration_TF", "Duration", "時間", "Duration (TF)"],
        "Distance": ["Distance", "距離"],
        "SI_D": ["SI_D", "SI D", "SID"],
        "HI_D": ["HI_D", "HI D", "HID"],
        "Sprint": ["Sprint", "スプリント"],
        "Accel_Z2": ["Accel_Z2", "Accel Z2", "Accel_Z2_x"],
        "Accel_Z3": ["Accel_Z3", "Accel Z3", "Accel_Z3_x"],
        "Decel_Z2": ["Decel_Z2", "Decel Z2", "Decel_Z2_x"],
        "Decel_Z3": ["Decel_Z3", "Decel Z3", "Decel_Z3_x"],
        "SPD_MX": ["SPD MX", "SPD_MX", "SPD_Max", "Max Speed", "最高速度"],
    }
    m = require_cols(train_df, need, "練習データ")
    pos_col = find_col(train_df, ["Position", "Pos", "ポジション", "position"])

    df = train_df.copy()
    df[m["Name"]] = df[m["Name"]].astype(str)
    df[m["Session"]] = df[m["Session"]].map(normalize_session_name)
    if pos_col is None:
        df["Position__tmp"] = "Unknown"
        pos_col = "Position__tmp"
    df[pos_col] = df[pos_col].map(normalize_position)

    for k in ["Distance", "SI_D", "HI_D", "Sprint", "Accel_Z2", "Accel_Z3", "Decel_Z2", "Decel_Z3", "SPD_MX"]:
        df[m[k]] = to_num(df[m[k]])
    df["Duration_min"] = df[m["Duration_TF"]].map(parse_duration_to_min)
    df["High Agility"] = df[m["Accel_Z3"]].fillna(0) + df[m["Decel_Z3"]].fillna(0)
    df["Accel >2m/s/s(n)"] = df[m["Accel_Z2"]].fillna(0) + df[m["Accel_Z3"]].fillna(0)
    df["Decel>2m/s/s(n)"] = df[m["Decel_Z2"]].fillna(0) + df[m["Decel_Z3"]].fillna(0)

    group_cols = [m["Name"], pos_col, m["Session"]]
    agg = df.groupby(group_cols, dropna=False).agg({
        "Duration_min": "sum",
        m["Distance"]: "sum",
        m["SI_D"]: "sum",
        m["HI_D"]: "sum",
        m["Sprint"]: "sum",
        m["SPD_MX"]: "max",
        "High Agility": "sum",
        "Accel >2m/s/s(n)": "sum",
        "Decel>2m/s/s(n)": "sum",
    }).reset_index()

    agg = agg.rename(columns={
        m["Name"]: "Name",
        pos_col: "Position",
        m["Session"]: "Session",
        m["Distance"]: "Distance",
        m["SI_D"]: "SI_D",
        m["HI_D"]: "HI_D",
        m["Sprint"]: "Sprint",
        m["SPD_MX"]: "SPD_MX",
    })

    dur = agg["Duration_min"].replace(0, np.nan)
    agg["dis(m)/min"] = agg["Distance"] / dur
    agg["SI D/min"] = agg["SI_D"] / dur
    agg["HI D/min"] = agg["HI_D"] / dur
    agg["High Agility/min"] = agg["High Agility"] / dur
    return agg


def detect_game_metric_cols(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    metric_colmap = {metric_key(c): c for c in df.columns}
    candidate_map = {
        "Name": ["Name", "名前"],
        "Distance": ["Distance", "Distance_90", "Distance90"],
        "SI_D": ["SI_D", "SI D", "SID", "SI_D_90"],
        "HI_D": ["HI_D", "HI D", "HID", "HI_D_90"],
        "Sprint": ["Sprint", "Sprint_90"],
        "SPD_MX": ["SPD MX", "SPD_MX", "SPD_Max", "Max Speed"],
        "High Agility": ["High Agility", "HighAgility", "High Agility_90", "HighAgility_90"],
        "Accel >2m/s/s(n)": ["Accel", "Accel_Total", "Accel_Total_90", "Accel >2m/s/s(n)"],
        "Decel>2m/s/s(n)": ["Decel", "Decel_Total", "Decel_Total_90", "Decel>2m/s/s(n)"],
        "dis(m)/min": ["Distance_per_min", "Distance per min", "dis(m)/min"],
        "SI D/min": ["SI D/min", "SI_D_per_min", "SI_D/min"],
        "HI D/min": ["HI D/min", "HI_D_per_min", "HI_D/min"],
        "High Agility/min": ["High Agility/min(n/min)", "HighAgility_per_min", "High Agility per min", "High Agility/min"],
    }
    out = {}
    for logical, aliases in candidate_map.items():
        found = None
        for alias in aliases:
            mk = metric_key(alias)
            for key, col in metric_colmap.items():
                if key == mk or mk in key or key in mk:
                    found = col
                    break
            if found:
                break
        out[logical] = found
    return out


def build_ratio_df(train_df: pd.DataFrame, game_df: pd.DataFrame) -> pd.DataFrame:
    session_df = build_session_metrics(train_df)
    gm = detect_game_metric_cols(game_df)
    if gm["Name"] is None:
        raise ValueError("試合平均の Name 列が見つかりません。")

    g = game_df.copy()
    g[gm["Name"]] = g[gm["Name"]].astype(str)
    g["Name_key"] = g[gm["Name"]].map(normalize_person_key)
    session_df["Name_key"] = session_df["Name"].map(normalize_person_key)

    keep_cols = ["Name_key"]
    rename_cols = {}
    for logical, col in gm.items():
        if logical == "Name" or col is None:
            continue
        keep_cols.append(col)
        rename_cols[col] = f"{logical}__game"

    game_use = g[keep_cols].copy().rename(columns=rename_cols)
    game_use = game_use.groupby("Name_key", as_index=False).first()
    ratio = session_df.merge(game_use, on="Name_key", how="left")

    metrics = [
        "Distance", "SI_D", "HI_D", "Sprint", "SPD_MX",
        "High Agility", "dis(m)/min", "SI D/min", "HI D/min", "High Agility/min",
        "Accel >2m/s/s(n)", "Decel>2m/s/s(n)",
    ]
    for m in metrics:
        base = f"{m}__game"
        out = f"{m}_vsGame_pct"
        if base in ratio.columns:
            ratio[out] = np.where(to_num(ratio[base]) > 0, to_num(ratio[m]) / to_num(ratio[base]) * 100, np.nan)
        else:
            ratio[out] = np.nan
    return ratio


def ratio_stats_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    s = to_num(df[metric]).dropna()
    if len(s) == 0:
        return pd.DataFrame({"項目": ["N"], "値": [0]})
    q1 = s.quantile(0.25)
    med = s.quantile(0.5)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    low = max(float(s.min()), float(q1 - 1.5 * iqr))
    high = min(float(s.max()), float(q3 + 1.5 * iqr))
    return pd.DataFrame({
        "項目": ["N", "Mean", "SD", "Min", "Whisker Low", "Q1", "Median", "Q3", "Whisker High", "Max"],
        "値": [
            len(s), s.mean(), s.std(ddof=1) if len(s) >= 2 else 0.0,
            s.min(), low, q1, med, q3, high, s.max()
        ]
    })


def plot_ratio_boxplot(df: pd.DataFrame, metric: str, session_order: List[str], title: str):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    data = []
    labels = []
    per_session_rows = []

    work = df.copy()
    if "Name" not in work.columns:
        work["Name"] = ""

    for sess in session_order:
        sub = work.loc[work["Session"] == sess].copy()
        vals = to_num(sub[metric])
        arr = vals.dropna().values
        if len(arr) == 0:
            continue
        data.append(arr)
        labels.append(sess)
        valid = sub.loc[vals.notna(), ["Name", metric]].copy()
        per_session_rows.append(valid)

    if not data:
        ax.text(0.5, 0.5, "データがありません", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return fig

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showfliers=False,
        widths=0.55,
        boxprops=dict(linewidth=2.2, edgecolor="black"),
        whiskerprops=dict(linewidth=2.0, color="black"),
        capprops=dict(linewidth=2.0, color="black"),
        medianprops=dict(linewidth=2.6, color="black"),
    )
    for b in bp["boxes"]:
        b.set_facecolor("white")
        b.set_alpha(1.0)

    rng = np.random.default_rng(0)
    for i, valid in enumerate(per_session_rows, start=1):
        yvals = pd.to_numeric(valid[metric], errors="coerce")
        names = valid["Name"].fillna("").astype(str).tolist()
        for yv, nm in zip(yvals, names):
            if pd.isna(yv):
                continue
            xj = i + rng.uniform(-JITTER_WIDTH, JITTER_WIDTH)
            ax.scatter([xj], [yv], alpha=POINT_ALPHA, s=POINT_SIZE)
            label_text = nm
            if SHOW_VALUE_IN_LABEL:
                label_text = f"{nm} ({yv:.0f})"
            ax.text(
                xj + 0.02, yv, label_text,
                fontsize=LABEL_FONTSIZE,
                alpha=LABEL_ALPHA,
                va="center"
            )

        med = pd.to_numeric(valid[metric], errors="coerce").dropna().median()
        if pd.notna(med):
            ax.text(
                i, med, f"{med:.1f}",
                fontsize=MEDIAN_FONTSIZE,
                ha="center", va="bottom"
            )

    ymin, ymax = nice_ylim(np.concatenate(data) if data else [])
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.tick_params(axis="x", labelsize=10, pad=6)
    ax.set_ylabel("% of Game Avg")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.93, bottom=0.18)
    return fig


def build_ratio_pdf_booklet(ratio_df: pd.DataFrame, metrics: List[str], positions: List[str]) -> bytes:
    buf = io.BytesIO()
    sessions = list(dict.fromkeys(ratio_df["Session"].dropna().astype(str).tolist()))
    with PdfPages(buf) as pdf:
        # cover
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        ax.text(0.5, 0.7, "GPS Ratio Booklet", ha="center", va="center", fontsize=24)
        ax.text(0.5, 0.58, APP_VERSION, ha="center", va="center", fontsize=12)
        ax.text(0.5, 0.48, "All + Position pages", ha="center", va="center", fontsize=14)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for metric in metrics:
            pdf.savefig(plot_ratio_boxplot(ratio_df, metric, sessions, f"{metric} | All"), bbox_inches="tight")
            plt.close()
            for pos in positions:
                sub = ratio_df[ratio_df["Position"] == pos].copy()
                fig = plot_ratio_boxplot(sub, metric, sessions, f"{metric} | {pos}")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# =========================================================
# ACWR helpers
# =========================================================
def parse_date_from_filename(filename: str) -> Optional[dt.date]:
    base = Path(filename).stem
    m = re.search(r"(20\d{2})[-_\.](\d{1,2})[-_\.](\d{1,2})", base)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return dt.date(y, mo, d)
        except ValueError:
            return None
    m = re.search(r"(20\d{2})(\d{2})(\d{2})", base)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return dt.date(y, mo, d)
        except ValueError:
            return None
    return None


def extract_daily_metrics_for_acwr(df_raw: pd.DataFrame, use_date: dt.date) -> pd.DataFrame:
    need = {
        "player": ["Name", "Player", "Athlete", "選手", "氏名", "名前"],
        "position": ["Position", "Pos", "ポジション"],
        "session": ["Session", "SESSION", "Drill", "Set", "メニュー", "セッション"],
        "distance": ["Distance", "距離"],
        "hi_d": ["HI_D", "HI D", "HID"],
        "accel_z2": ["Accel_Z2_x", "Accel_Z2", "Accel Z2"],
        "accel_z3": ["Accel_Z3_x", "Accel_Z3", "Accel Z3"],
        "decel_z2": ["Decel_Z2_x", "Decel_Z2", "Decel Z2"],
        "decel_z3": ["Decel_Z3_x", "Decel_Z3", "Decel Z3"],
    }
    m = require_cols(df_raw, need, "ACWR日次データ")
    df = df_raw.copy()
    df["session_norm"] = df[m["session"]].map(normalize_session_compact)
    df = df[df["session_norm"].str.contains("roundall", na=False)].copy()
    if len(df) == 0:
        raise ValueError("ROUND ALL 行が見つかりません。")
    df["player"] = df[m["player"]].map(nfkc)
    df["position"] = df[m["position"]].map(normalize_position)
    for key in ["distance", "hi_d", "accel_z2", "accel_z3", "decel_z2", "decel_z3"]:
        df[key] = to_num(df[m[key]])
    out = pd.DataFrame({
        "date": pd.to_datetime(use_date),
        "player": df["player"],
        "position": df["position"],
        "Distance": df["distance"],
        "HI_D": df["hi_d"],
        "Accel": df["accel_z2"].fillna(0) + df["accel_z3"].fillna(0),
        "Decel": df["decel_z2"].fillna(0) + df["decel_z3"].fillna(0),
        "High_Agility": df["accel_z3"].fillna(0) + df["decel_z3"].fillna(0),
    })
    out = out.groupby(["date", "player", "position"], as_index=False).last()
    return out


def build_player_timeseries(df_daily: pd.DataFrame, metric_key_name: str) -> Tuple[pd.DataFrame, pd.Series]:
    dfx = df_daily.copy()
    dfx["date"] = pd.to_datetime(dfx["date"]).dt.normalize()
    first_dates = dfx.groupby("player")["date"].min()
    last_pos = dfx.sort_values(["date", "player"]).groupby("player")["position"].last()
    wide = dfx.pivot_table(index="date", columns="player", values=metric_key_name, aggfunc="last")
    wide = wide.sort_index()
    full_dates = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
    wide = wide.reindex(full_dates)
    for player in wide.columns:
        fd = first_dates.get(player)
        if pd.isna(fd):
            continue
        mask_after = wide.index >= pd.to_datetime(fd)
        wide.loc[mask_after, player] = wide.loc[mask_after, player].fillna(0)
    return wide, last_pos


def compute_acwr(wide: pd.DataFrame) -> pd.DataFrame:
    acute = wide.rolling(ACUTE_DAYS, min_periods=ACUTE_MIN_DAYS).mean()
    chronic = wide.rolling(CHRONIC_DAYS, min_periods=CHRONIC_MIN_DAYS).mean()
    acwr = acute / chronic.replace(0, np.nan)
    acwr = acwr.mask(chronic <= 0)
    return acwr


def plot_acwr_scatter(acwr_today: pd.Series, positions: pd.Series, title: str, actual_values: Optional[pd.Series] = None):
    df = pd.DataFrame({
        "player": acwr_today.index,
        "acwr": acwr_today.values,
        "position": positions.reindex(acwr_today.index).fillna("Unknown").values,
    })
    if actual_values is not None:
        df["actual"] = df["player"].map(actual_values.to_dict())
    else:
        df["actual"] = np.nan
    d = df.dropna(subset=["acwr"]).copy()
    d["position"] = d["position"].map(normalize_position)
    d["pos_order"] = d["position"].apply(lambda x: POSITION_ORDER.index(x) if x in POSITION_ORDER else 999)
    d = d.sort_values(["pos_order", "position", "player"]).reset_index(drop=True)
    d["y"] = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(10, max(5, len(d) * 0.35)))
    ax.scatter(d["acwr"], d["y"])
    for _, row in d.iterrows():
        label = row["player"]
        if pd.notna(row["actual"]):
            try:
                label += f" ({int(round(float(row['actual'])) )})"
            except Exception:
                pass
        ax.text(float(row["acwr"]) + 0.02, float(row["y"]), label, va="center", fontsize=9)
    ax.axvline(DANGER_LOW, linestyle="--", alpha=0.6)
    ax.axvline(DANGER_HIGH, linestyle="--", alpha=0.6)
    xmin = min(0.0, float(d["acwr"].min()) - 0.2) if len(d) else 0.0
    xmax = max(2.0, float(d["acwr"].max()) + 0.2) if len(d) else 2.0
    ax.set_xlim(xmin, xmax)
    ax.set_yticks([])
    ax.set_xlabel("ACWR")
    ax.set_title(title)
    fig.tight_layout()
    return fig, d


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def build_acwr_bundle(df_daily: pd.DataFrame, display_date: dt.date) -> Tuple[bytes, bytes]:
    idx = pd.to_datetime(display_date).normalize()
    pdf_buf = io.BytesIO()
    zip_buf = io.BytesIO()
    zf = zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED)
    with PdfPages(pdf_buf) as pdf:
        for metric_key_name, metric_label in ACWR_METRICS:
            wide, latest_pos = build_player_timeseries(df_daily, metric_key_name)
            acwr = compute_acwr(wide)
            if acwr.empty or idx not in acwr.index:
                continue
            players_today = (
                df_daily.loc[pd.to_datetime(df_daily["date"]).dt.normalize() == idx, "player"]
                .dropna().astype(str).unique().tolist()
            )
            acwr_today = acwr.loc[idx].reindex(players_today)
            pos_today = latest_pos.reindex(players_today).fillna("Unknown")
            actual = (
                df_daily.loc[pd.to_datetime(df_daily["date"]).dt.normalize() == idx, ["player", metric_key_name]]
                .groupby("player")[metric_key_name].last()
                .reindex(players_today)
            )
            fig, df_status = plot_acwr_scatter(acwr_today, pos_today, f"{metric_label} / {display_date.isoformat()}", actual)
            pdf.savefig(fig, bbox_inches="tight")
            zf.writestr(f"ACWR_{safe_filename(metric_label)}_{display_date.isoformat()}.png", fig_to_png_bytes(fig))
            csv_df = df_status[["player", "position", "acwr"]].copy()
            csv_df[f"{metric_label}_actual"] = csv_df["player"].map(actual.to_dict())
            zf.writestr(f"ACWR_{safe_filename(metric_label)}_{display_date.isoformat()}.csv", csv_df.to_csv(index=False, encoding="utf-8-sig"))
            plt.close(fig)
    zf.close()
    pdf_buf.seek(0)
    zip_buf.seek(0)
    return pdf_buf.getvalue(), zip_buf.getvalue()


# =========================================================
# Match average builder helpers
# =========================================================
def is_full_session_match(value) -> bool:
    s = nfkc(value).lower()
    return bool(re.search(r"\b(full|full time|フルタイム)\b", s))


def validate_match_required_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    required = [
        "Name", "Position", "Session", "Duration_TF",
        "Distance", "SI_D", "HI_D", "Sprint",
        "Accel_Z2", "Accel_Z3", "Decel_Z2", "Decel_Z3", "SPD MX",
    ]
    miss = [c for c in required if c not in df.columns]
    return len(miss) == 0, miss


def build_one_match_record(df_raw: pd.DataFrame, match_date: dt.date) -> pd.DataFrame:
    df = standardize_match_columns(df_raw.copy())
    ok, missing = validate_match_required_columns(df)
    if not ok:
        raise ValueError("必須列不足: " + ", ".join(missing))
    df["Session_norm"] = df["Session"].map(nfkc)
    df = df[df["Session_norm"].map(is_full_session_match)].copy()
    if len(df) == 0:
        raise ValueError("full / full time / フルタイム に該当する Session が見つかりません。")
    for c in ["Distance", "SI_D", "HI_D", "Sprint", "Accel_Z2", "Accel_Z3", "Decel_Z2", "Decel_Z3", "SPD MX"]:
        df[c] = to_num(df[c])
    df["Duration_min"] = df["Duration_TF"].map(parse_duration_to_min)
    df["Position"] = df["Position"].map(normalize_position)
    out = df.groupby(["Name", "Position"], as_index=False).agg({
        "Duration_min": "sum",
        "Distance": "sum",
        "SI_D": "sum",
        "HI_D": "sum",
        "Sprint": "sum",
        "Accel_Z2": "sum",
        "Accel_Z3": "sum",
        "Decel_Z2": "sum",
        "Decel_Z3": "sum",
        "SPD MX": "max",
    })
    out["High Agility"] = out["Accel_Z3"] + out["Decel_Z3"]
    out["Accel >2m/s/s(n)"] = out["Accel_Z2"] + out["Accel_Z3"]
    out["Decel>2m/s/s(n)"] = out["Decel_Z2"] + out["Decel_Z3"]
    out["dis(m)/min"] = out["Distance"] / out["Duration_min"].replace(0, np.nan)
    out["High Agility/min(n/min)"] = out["High Agility"] / out["Duration_min"].replace(0, np.nan)
    out["date"] = pd.to_datetime(match_date)
    out["name_key"] = out["Name"].map(normalize_person_key)
    return out


def weighted_average_records(records: pd.DataFrame, target_date: dt.date, lambda_value: float = 0.2) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()
    df = records.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    target_ts = pd.to_datetime(target_date).normalize()
    df = df[df["date"] <= target_ts].copy()
    if len(df) == 0:
        return pd.DataFrame()
    df["days_ago"] = (target_ts - df["date"]).dt.days
    df["weight"] = np.exp(-lambda_value * df["days_ago"])
    metric_cols = [
        "Duration_min", "Distance", "SI_D", "HI_D", "Sprint", "SPD MX",
        "High Agility", "Accel >2m/s/s(n)", "Decel>2m/s/s(n)",
        "dis(m)/min", "High Agility/min(n/min)",
    ]
    rows = []
    for nk, g in df.groupby("name_key"):
        row = {
            "Name": g["Name"].iloc[-1],
            "Position": g["Position"].iloc[-1],
            "matches_used": len(g),
        }
        w = g["weight"].values
        for c in metric_cols:
            vals = pd.to_numeric(g[c], errors="coerce").values
            mask = np.isfinite(vals) & np.isfinite(w)
            row[c] = np.average(vals[mask], weights=w[mask]) if mask.any() else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out = out.sort_values(["Position", "Name"]).reset_index(drop=True)
    return out


def build_change_comparison(df_avg: pd.DataFrame, session_df: pd.DataFrame, session1: str, session2: str, metric: str) -> pd.DataFrame:
    s1 = session_df[session_df["Session"] == session1][["Name", "Position", metric]].rename(columns={metric: f"{metric}_1"})
    s2 = session_df[session_df["Session"] == session2][["Name", "Position", metric]].rename(columns={metric: f"{metric}_2"})
    merged = s1.merge(s2, on=["Name", "Position"], how="outer")
    merged["Change_pct"] = np.where(
        pd.to_numeric(merged[f"{metric}_1"], errors="coerce") != 0,
        (pd.to_numeric(merged[f"{metric}_2"], errors="coerce") - pd.to_numeric(merged[f"{metric}_1"], errors="coerce"))
        / pd.to_numeric(merged[f"{metric}_1"], errors="coerce") * 100,
        np.nan,
    )
    return merged


def plot_change_bar(plot_df: pd.DataFrame, metric: str, session1: str, session2: str):
    fig, ax = plt.subplots(figsize=(max(8, len(plot_df) * 0.7), 5))
    ax.bar(plot_df["Name"], plot_df["Change_pct"])
    ax.axhline(0, linewidth=1)
    ymin, ymax = nice_sym_ylim(plot_df["Change_pct"])
    ax.set_ylim(ymin, ymax)
    ax.set_title(f"{metric} 変動率（{session1} → {session2}）")
    ax.set_ylabel("変動率 (%)")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    return fig


# =========================================================
# Weekly helpers
# =========================================================
def detect_weekly_required_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cmap = {norm_colname(c): c for c in df.columns}
    def pick(cands):
        for c in cands:
            nc = norm_colname(c)
            for k, v in cmap.items():
                if k == nc or nc in k or k in nc:
                    return v
        return None
    return {
        "player": pick(["Name", "Player", "選手", "氏名", "名前"]),
        "position": pick(["Position", "Pos", "ポジション"]),
        "session": pick(["Session", "メニュー", "セッション"]),
    }


def detect_weekly_metric_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cmap = {metric_key(c): c for c in df.columns}
    aliases = {
        "Distance_vsGame_pct": ["Distance_vsGame_pct"],
        "HI_D_vsGame_pct": ["HI_D_vsGame_pct"],
        "SI_D_vsGame_pct": ["SI_D_vsGame_pct"],
        "Accel >2m/s/s(n)_vsGame_pct": ["Accel >2m/s/s(n)_vsGame_pct"],
        "Decel>2m/s/s(n)_vsGame_pct": ["Decel>2m/s/s(n)_vsGame_pct"],
        "HighAgility_vsGame_pct": ["HighAgility_vsGame_pct", "High Agility_vsGame_pct"],
        "SPD_MX_vsGame_pct": ["SPD_MX_vsGame_pct"],
        "dis(m)/min_vsGame_pct": ["dis(m)/min_vsGame_pct", "Distance/min_vsGame_pct"],
        "High Agility/min(n/min)_vsGame_pct": ["High Agility/min(n/min)_vsGame_pct", "HighAgility/min(n/min)_vsGame_pct"],
    }
    out = {}
    for logical, cands in aliases.items():
        found = None
        for cand in cands:
            mk = metric_key(cand)
            for k, v in cmap.items():
                if k == mk or mk in k or k in mk:
                    found = v
                    break
            if found:
                break
        out[logical] = found
    return out


def max_value_and_session(df: pd.DataFrame, value_col: str, raw_session_col: str) -> Tuple[float, str]:
    tmp = df[[value_col, raw_session_col]].copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=[value_col])
    if len(tmp) == 0:
        return np.nan, ""
    idx = tmp[value_col].idxmax()
    return float(tmp.loc[idx, value_col]), str(tmp.loc[idx, raw_session_col])


def summarize_weekly_one_day(df_raw: pd.DataFrame, weekday: str) -> pd.DataFrame:
    required = detect_weekly_required_columns(df_raw)
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"{weekday}: 必須列不足 {missing}")
    metric_cols = detect_weekly_metric_columns(df_raw)
    df = df_raw.copy()
    df["player_name"] = df[required["player"]].map(normalize_person_key)
    df["position"] = df[required["position"]].map(normalize_position)
    df["session_norm"] = df[required["session"]].map(normalize_session_compact)
    df["session_raw"] = df[required["session"]].astype(str)
    for logical, real_col in metric_cols.items():
        df[logical] = to_num(df[real_col]) if real_col is not None else np.nan
    rows = []
    for player_name, g in df.groupby("player_name", dropna=False):
        if player_name.strip() == "":
            continue
        pos = g["position"].dropna().iloc[0] if g["position"].notna().any() else "Unknown"
        row = {"weekday": weekday, "player_name": player_name, "position": pos}
        all_mask = g["session_norm"].map(is_all_session)
        base_map = {
            "Distance_vsGame_pct": "Distance_vsGame_pct",
            "HI_D_vsGame_pct": "HI_D_vsGame_pct",
            "SI_D_vsGame_pct": "SI_D_vsGame_pct",
            "Accel >2m/s/s(n)_vsGame_pct": "Accel >2m/s/s(n)_vsGame_pct",
            "Decel>2m/s/s(n)_vsGame_pct": "Decel>2m/s/s(n)_vsGame_pct",
            "HighAgility_vsGame_pct": "HighAgility_vsGame_pct",
            "SPD_MX_vsGame_pct": "SPD_MX_vsGame_pct",
        }
        for out_col, src_col in base_map.items():
            row[out_col] = g.loc[all_mask, src_col].dropna().iloc[0] if g.loc[all_mask, src_col].dropna().any() else np.nan
        row["dis(m)/min_vsGame_pct_All"] = g.loc[all_mask, "dis(m)/min_vsGame_pct"].dropna().iloc[0] if g.loc[all_mask, "dis(m)/min_vsGame_pct"].dropna().any() else np.nan
        row["High Agility/min(n/min)_vsGame_pct_All"] = g.loc[all_mask, "High Agility/min(n/min)_vsGame_pct"].dropna().iloc[0] if g.loc[all_mask, "High Agility/min(n/min)_vsGame_pct"].dropna().any() else np.nan
        mx1, s1 = max_value_and_session(g.loc[~all_mask], "dis(m)/min_vsGame_pct", "session_raw")
        mx2, s2 = max_value_and_session(g.loc[~all_mask], "High Agility/min(n/min)_vsGame_pct", "session_raw")
        row["dis(m)/min_vsGame_pct_MX"] = mx1
        row["dis(m)/min_vsGame_pct_MX_session"] = s1
        row["High Agility/min(n/min)_vsGame_pct_MX"] = mx2
        row["High Agility/min(n/min)_vsGame_pct_MX_session"] = s2
        rows.append(row)
    return pd.DataFrame(rows)


def build_weekly_summary(uploaded_files: Dict[str, object]) -> pd.DataFrame:
    daily = []
    for weekday, uf in uploaded_files.items():
        if uf is None:
            continue
        df = read_any_file(uf)
        daily.append(summarize_weekly_one_day(df, weekday))
    if not daily:
        return pd.DataFrame()
    full = pd.concat(daily, ignore_index=True)
    return full


def plot_weekly_metric_page(df: pd.DataFrame, metric: str, weekdays: List[str], title: str):
    fig = plt.figure(figsize=(11.69, 8.27))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.6, 1.4])
    ax = fig.add_subplot(gs[0, 0])
    ax_tbl = fig.add_subplot(gs[1, 0])
    ax_tbl.axis("off")

    data, labels = [], []
    stat_rows = []
    per_day_names = []
    work = df.copy()
    if "player_name" not in work.columns:
        if "Name" in work.columns:
            work["player_name"] = work["Name"].fillna("").astype(str)
        else:
            work["player_name"] = ""

    for wd in weekdays:
        sub = work.loc[work["weekday"] == wd].copy()
        vals = to_num(sub[metric])
        arr = vals.dropna()
        if len(arr):
            data.append(arr.values)
            labels.append(wd)
        valid = sub.loc[vals.notna(), ["player_name", metric]].copy()
        per_day_names.append(valid)
        stat_rows.append([
            wd,
            int(arr.count()),
            round(float(arr.mean()), 1) if len(arr) else np.nan,
            round(float(arr.median()), 1) if len(arr) else np.nan,
            round(float(arr.min()), 1) if len(arr) else np.nan,
            round(float(arr.max()), 1) if len(arr) else np.nan,
        ])

    if data:
        try:
            box = ax.boxplot(
                data,
                tick_labels=labels,
                patch_artist=True,
                widths=0.55,
                showfliers=False,
                medianprops=dict(color="black", linewidth=1.8),
                whiskerprops=dict(color="black", linewidth=1.2),
                capprops=dict(color="black", linewidth=1.2),
                boxprops=dict(color="black", linewidth=1.2),
            )
        except TypeError:
            box = ax.boxplot(
                data,
                labels=labels,
                patch_artist=True,
                widths=0.55,
                showfliers=False,
                medianprops=dict(color="black", linewidth=1.8),
                whiskerprops=dict(color="black", linewidth=1.2),
                capprops=dict(color="black", linewidth=1.2),
                boxprops=dict(color="black", linewidth=1.2),
            )
        facecolor = WEEKLY_METRIC_COLORS.get(metric, "#90caf9")
        for patch in box["boxes"]:
            patch.set_facecolor(facecolor)
            patch.set_edgecolor("black")
            patch.set_alpha(0.95)

        rng = np.random.default_rng(42)
        for i, valid in enumerate(per_day_names, start=1):
            if len(valid) == 0:
                continue
            y = pd.to_numeric(valid[metric], errors="coerce").values
            y = y[~np.isnan(y)]
            if len(y) == 0:
                continue
            x = rng.normal(loc=i, scale=0.04, size=len(y))
            ax.scatter(x, y, s=WEEKLY_POINT_SIZE, alpha=WEEKLY_POINT_ALPHA)
            valid_names = valid.loc[pd.to_numeric(valid[metric], errors="coerce").notna(), "player_name"].tolist()
            for xx, yy, name in zip(x, y, valid_names):
                ax.text(
                    xx + 0.01, yy, str(name),
                    fontsize=WEEKLY_LABEL_FONTSIZE,
                    alpha=WEEKLY_LABEL_ALPHA,
                    va="center"
                )

        for i, arr in enumerate(data, start=1):
            arr = pd.Series(arr).dropna()
            if len(arr) == 0:
                continue
            med = float(np.median(arr))
            ax.text(i, med, f"{med:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ymin, ymax = nice_ylim(pd.concat([pd.Series(x) for x in data]))
        ax.set_ylim(ymin, ymax)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
    else:
        ax.text(0.5, 0.5, "データがありません", ha="center", va="center", transform=ax.transAxes)

    ax.set_title(title)
    ax.set_xlabel("曜日")
    ax.set_ylabel("vsGame_pct")
    table = ax_tbl.table(
        cellText=stat_rows,
        colLabels=["Weekday", "N", "Mean", "Median", "Min", "Max"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)
    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.98])
    return fig


def build_weekly_pdf(full_df: pd.DataFrame, weekdays: List[str], positions: List[str]) -> bytes:
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        ax.text(0.5, 0.68, "Weekly Booklet", ha="center", va="center", fontsize=24)
        ax.text(0.5, 0.55, APP_VERSION, ha="center", va="center", fontsize=12)
        ax.text(0.5, 0.45, f"Weekdays: {' / '.join(weekdays)}", ha="center", va="center", fontsize=12)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        for metric in WEEKLY_METRICS:
            fig = plot_weekly_metric_page(full_df, metric, weekdays, f"{metric} | All")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            for pos in positions:
                sub = full_df[full_df["position"] == pos].copy()
                fig = plot_weekly_metric_page(sub, metric, weekdays, f"{metric} | {pos}")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# =========================================================
# Session state init
# =========================================================
def init_state():
    defaults = {
        "acwr_daily": pd.DataFrame(),
        "match_records": pd.DataFrame(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# =========================================================
# UI sections
# =========================================================
def render_ratio_tab():
    st.subheader("1. 練習データ vs 試合平均（GPS比較）")
    col1, col2 = st.columns(2)
    with col1:
        train_upload = st.file_uploader("練習データ", type=STREAMLIT_FILE_TYPES, key="ratio_train")
    with col2:
        game_upload = st.file_uploader("試合平均", type=STREAMLIT_FILE_TYPES, key="ratio_game")

    if train_upload is None or game_upload is None:
        st.info("練習データと試合平均をアップロードすると、選手×セッションごとの試合比率を算出できます。")
        return

    try:
        train_df = read_any_file(train_upload)
        game_df = read_any_file(game_upload, required_tokens=game_required_tokens())
        ratio_df = build_ratio_df(train_df, game_df)
    except Exception as e:
        st.error(str(e))
        return

    st.success(f"集計完了: {len(ratio_df)} 行")
    st.dataframe(ratio_df.head(50), use_container_width=True)

    sessions = list(dict.fromkeys(ratio_df["Session"].dropna().astype(str).tolist()))
    positions = ["All"] + sort_positions_fixed(ratio_df["Position"].dropna().astype(str).tolist())
    metrics = [m for m in RATIO_BOOKLET_METRICS if m in ratio_df.columns]

    c1, c2, c3 = st.columns(3)
    with c1:
        metric = st.selectbox("表示指標", metrics, key="ratio_metric")
    with c2:
        position = st.selectbox("Position", positions, key="ratio_position")
    with c3:
        make_pdf = st.checkbox("冊子PDFを作る", value=False, key="ratio_make_pdf")

    plot_df = ratio_df.copy() if position == "All" else ratio_df[ratio_df["Position"] == position].copy()
    fig = plot_ratio_boxplot(plot_df, metric, sessions, f"{metric} | {position}")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.dataframe(ratio_stats_table(plot_df, metric), use_container_width=True)

    csv_bytes = ratio_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button("比率データCSVをダウンロード", data=csv_bytes, file_name="ratio_output.csv", mime="text/csv")

    if make_pdf:
        pdf_bytes = build_ratio_pdf_booklet(ratio_df, metrics, [p for p in positions if p != "All"])
        st.download_button("冊子PDFをダウンロード", data=pdf_bytes, file_name="ratio_booklet.pdf", mime="application/pdf")


def render_acwr_tab():
    st.subheader("2. ACWR")
    st.caption("日別ファイルを追加し、表示日を選んで ACWR を確認します。ROUND ALL 行を利用します。")

    with st.expander("日別ファイル登録", expanded=True):
        uploaded_files = st.file_uploader(
            "複数ファイルアップロード", type=STREAMLIT_FILE_TYPES, accept_multiple_files=True, key="acwr_uploads"
        )
        selected_date = st.date_input("ファイル名から日付が取れない場合に使う日付", value=dt.date.today(), key="acwr_fallback_date")
        if st.button("登録", key="acwr_register"):
            added, errors = [], []
            records = []
            for uf in uploaded_files or []:
                try:
                    inferred = parse_date_from_filename(uf.name)
                    use_date = inferred or selected_date
                    df = read_any_file(uf)
                    rec = extract_daily_metrics_for_acwr(df, use_date)
                    records.append(rec)
                    added.append((uf.name, use_date.isoformat(), "推定" if inferred else "選択日", rec["player"].nunique()))
                except Exception as e:
                    errors.append((uf.name, str(e)))
            if records:
                new_df = pd.concat(records, ignore_index=True)
                base = st.session_state.acwr_daily.copy()
                if not base.empty:
                    combo = pd.concat([base, new_df], ignore_index=True)
                    combo["date"] = pd.to_datetime(combo["date"]).dt.normalize()
                    combo = combo.sort_values(["date", "player"]).drop_duplicates(subset=["date", "player"], keep="last")
                    st.session_state.acwr_daily = combo.reset_index(drop=True)
                else:
                    st.session_state.acwr_daily = new_df.reset_index(drop=True)
            if added:
                st.success("登録成功")
                st.dataframe(pd.DataFrame(added, columns=["file", "date", "date_source", "players"]), use_container_width=True)
            if errors:
                st.error("一部エラーがありました")
                st.dataframe(pd.DataFrame(errors, columns=["file", "error"]), use_container_width=True)
        if st.button("ACWRデータを全削除", key="acwr_reset"):
            st.session_state.acwr_daily = pd.DataFrame()
            st.rerun()

    df_daily = st.session_state.acwr_daily.copy()
    if df_daily.empty:
        st.info("まだデータがありません。")
        return

    df_daily["date"] = pd.to_datetime(df_daily["date"]).dt.normalize()
    latest_date = df_daily["date"].max().date()
    display_date = st.date_input("表示日", value=latest_date, key="acwr_display_date")
    metrics = [x[0] for x in ACWR_METRICS]
    labels = {k: v for k, v in ACWR_METRICS}
    metric_key_name = st.selectbox("指標", metrics, format_func=lambda x: labels[x], key="acwr_metric")

    players_today = df_daily.loc[df_daily["date"].dt.date == display_date, "player"].dropna().astype(str).unique().tolist()
    if not players_today:
        st.warning("この日の選手データがありません。")
        return

    wide, latest_pos = build_player_timeseries(df_daily, metric_key_name)
    acwr = compute_acwr(wide)
    idx = pd.to_datetime(display_date).normalize()
    if idx not in acwr.index:
        st.warning("履歴不足のため ACWR を計算できません。")
        return

    acwr_today = acwr.loc[idx].reindex(players_today)
    pos_today = latest_pos.reindex(players_today).fillna("Unknown")
    actual = df_daily.loc[df_daily["date"] == idx, ["player", metric_key_name]].groupby("player")[metric_key_name].last().reindex(players_today)
    fig, status_df = plot_acwr_scatter(acwr_today, pos_today, f"{labels[metric_key_name]} / {display_date.isoformat()}", actual)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.dataframe(status_df[["player", "position", "acwr", "actual"]], use_container_width=True)

    pdf_bytes, zip_bytes = build_acwr_bundle(df_daily, display_date)
    st.download_button("ACWR PDFをダウンロード", data=pdf_bytes, file_name=f"acwr_{display_date.isoformat()}.pdf", mime="application/pdf")
    st.download_button("ACWR ZIPをダウンロード", data=zip_bytes, file_name=f"acwr_{display_date.isoformat()}.zip", mime="application/zip")


def render_matchavg_tab():
    st.subheader("3. 試合平均作成アプリ")
    st.caption("full / full time / フルタイム の行から試合データを蓄積し、指数減衰で加重平均を作成します。")
    col1, col2 = st.columns([1, 1])
    with col1:
        match_date = st.date_input("試合日", value=dt.date.today(), key="matchavg_date")
    with col2:
        uploaded_file = st.file_uploader("試合ファイル", type=STREAMLIT_FILE_TYPES, key="matchavg_upload")

    if uploaded_file is not None and st.button("試合データを登録", key="matchavg_register"):
        try:
            df = read_any_file(uploaded_file)
            rec = build_one_match_record(df, match_date)
            base = st.session_state.match_records.copy()
            combined = pd.concat([base, rec], ignore_index=True) if not base.empty else rec
            combined["date"] = pd.to_datetime(combined["date"]).dt.normalize()
            st.session_state.match_records = combined.sort_values(["date", "Name"]).reset_index(drop=True)
            st.success(f"登録しました: {uploaded_file.name} / {match_date}")
        except Exception as e:
            st.error(str(e))

    if st.button("試合履歴を全削除", key="matchavg_reset"):
        st.session_state.match_records = pd.DataFrame()
        st.rerun()

    records = st.session_state.match_records.copy()
    if records.empty:
        st.info("まだ試合履歴がありません。")
        return

    with st.expander("現在の保存データ", expanded=False):
        st.dataframe(records.head(100), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        lambda_label = st.selectbox("指数減衰 λ", list(LAMBDA_OPTIONS.keys()), index=1, key="matchavg_lambda")
    with col2:
        target_date = st.date_input("平均を作る基準日", value=records["date"].max().date(), key="matchavg_target")

    avg_df = weighted_average_records(records, target_date, LAMBDA_OPTIONS[lambda_label])
    st.write(f"出力人数: {len(avg_df)}")
    st.dataframe(avg_df, use_container_width=True)
    st.download_button("加重平均CSVをダウンロード", data=avg_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"), file_name="weighted_match_average.csv", mime="text/csv")

    st.divider()
    st.markdown("#### セッション変動率")
    current_upload = st.file_uploader("比較用ファイル（1試合/1練習のセッション別）", type=STREAMLIT_FILE_TYPES, key="matchavg_compare_upload")
    if current_upload is None:
        return
    try:
        current_df = read_any_file(current_upload)
        session_df = build_session_metrics(current_df)
    except Exception as e:
        st.error(str(e))
        return
    sessions = list(dict.fromkeys(session_df["Session"].dropna().astype(str).tolist()))
    if len(sessions) < 2:
        st.warning("比較できる Session が2つ以上必要です。")
        return
    c1, c2, c3 = st.columns(3)
    with c1:
        session1 = st.selectbox("比較元", sessions, key="matchavg_s1")
    with c2:
        default_idx = 1 if len(sessions) > 1 else 0
        session2 = st.selectbox("比較先", sessions, index=default_idx, key="matchavg_s2")
    with c3:
        metric = st.selectbox("比較指標", ["Distance", "SI_D", "HI_D", "Sprint", "SPD_MX", "dis(m)/min", "High Agility", "High Agility/min"], key="matchavg_metric")
    comp_df = build_change_comparison(avg_df, session_df, session1, session2, metric)
    st.dataframe(comp_df, use_container_width=True)
    fig = plot_change_bar(comp_df.dropna(subset=["Change_pct"]), metric, session1, session2)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_weekly_tab():
    st.subheader("4. Weekly")
    week_pattern_name = st.radio("週構成", list(WEEK_PATTERNS.keys()), index=0, horizontal=True, key="weekly_pattern")
    weekdays = WEEK_PATTERNS[week_pattern_name]
    st.caption("曜日ごとの集計済みファイルを読み込み、週全体の冊子を作ります。")
    uploaded_files = {}
    cols = st.columns(len(weekdays))
    for col, wd in zip(cols, weekdays):
        with col:
            uploaded_files[wd] = st.file_uploader(wd, type=STREAMLIT_FILE_TYPES, key=f"weekly_{wd}")
    if not st.button("集計して表示", key="weekly_run"):
        return
    try:
        full_df = build_weekly_summary(uploaded_files)
    except Exception as e:
        st.error(str(e))
        return
    if full_df.empty:
        st.warning("有効なデータがありません。")
        return

    st.success(f"集計完了: {len(full_df)} 行")
    st.dataframe(full_df, use_container_width=True)
    positions = ["All"] + sort_positions_fixed(full_df["position"].dropna().astype(str).tolist())
    c1, c2 = st.columns(2)
    with c1:
        metric = st.selectbox("指標", WEEKLY_METRICS, key="weekly_metric")
    with c2:
        position = st.selectbox("Position", positions, key="weekly_pos")
    plot_df = full_df.copy() if position == "All" else full_df[full_df["position"] == position].copy()
    fig = plot_weekly_metric_page(plot_df, metric, weekdays, f"{metric} | {position}")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    pdf_bytes = build_weekly_pdf(full_df, weekdays, [p for p in positions if p != "All"])
    st.download_button("Weekly PDFをダウンロード", data=pdf_bytes, file_name="weekly_booklet.pdf", mime="application/pdf")
    st.download_button("Weekly CSVをダウンロード", data=full_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"), file_name="weekly_output.csv", mime="text/csv")


# =========================================================
# Main UI
# =========================================================
def main():
    setup_matplotlib_japanese_font()
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_state()

    st.title(APP_TITLE)
    st.caption(f"Version: {APP_VERSION}")
    st.info(
        "4つのアプリを1つに統合しています。"
        "タブ切替で『GPS比較』『ACWR』『試合平均』『Weekly』を使い分けできます。"
    )

    tabs = st.tabs([
        "GPS比較",
        "ACWR",
        "試合平均",
        "Weekly",
    ])
    with tabs[0]:
        render_ratio_tab()
    with tabs[1]:
        render_acwr_tab()
    with tabs[2]:
        render_matchavg_tab()
    with tabs[3]:
        render_weekly_tab()


if __name__ == "__main__":
    main()
