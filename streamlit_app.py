import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from xgboost import XGBRegressor
from streamlit_ketcher import st_ketcher
from molfeat.calc import FPCalculator

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold

def get_murcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    else:
        return None
    

def scale_ic50(ic50):
    if ic50 > 100:
        return 100
    else:
        return ic50
    
def class_ic50(ic50):
    if ic50 < 10:
        return 1
    else:
        return 0
    
def hamming_distance(fp1, fp2):
    return np.sum(fp1 != fp2)

def draw_molecule(smiles, size=(300, 300)):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.Draw.MolToImage(mol, size=size)

def canonize_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

def _render_lines_table(rows_df, metal_color, global_min_ic50=None):
    """Render cell line rows as an HTML table inside a compound card."""
    _MONO = "DM Mono, monospace"
    _SYNE = "Syne, sans-serif"

    header = (
        "<table style='width:100%;border-collapse:collapse;margin-top:6px;'>"
        "<thead><tr>"
        "<th style='font-family:\"DM Mono\",monospace;font-size:0.6rem;letter-spacing:0.08em;"
        "text-transform:uppercase;color:#4a5568;text-align:left;padding:4px 8px 4px 0;border-bottom:1px solid rgba(255,255,255,0.06);'>Cell line</th>"
        "<th style='font-family:\"DM Mono\",monospace;font-size:0.6rem;letter-spacing:0.08em;"
        "text-transform:uppercase;color:#4a5568;text-align:left;padding:4px 8px 4px 0;border-bottom:1px solid rgba(255,255,255,0.06);'>Time</th>"
        "<th style='font-family:\"DM Mono\",monospace;font-size:0.6rem;letter-spacing:0.08em;"
        "text-transform:uppercase;color:#4a5568;text-align:left;padding:4px 8px 4px 0;border-bottom:1px solid rgba(255,255,255,0.06);'>IC&#8325;&#8320;, μM</th>"
        "<th style='font-family:\"DM Mono\",monospace;font-size:0.6rem;letter-spacing:0.08em;"
        "text-transform:uppercase;color:#4a5568;text-align:left;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.06);'>Cisplatin</th>"
        "<th style='font-family:\"DM Mono\",monospace;font-size:0.6rem;letter-spacing:0.08em;"
        "text-transform:uppercase;color:#4a5568;text-align:left;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.06);'>DOI</th>"
        "<th style='font-family:\"DM Mono\",monospace;font-size:0.6rem;letter-spacing:0.08em;"
        "text-transform:uppercase;color:#4a5568;text-align:left;padding:4px 0 4px 8px;border-bottom:1px solid rgba(255,255,255,0.06);'>Year</th>"
        "</tr></thead><tbody>"
    )

    rows_html = ""
    min_ic50 = global_min_ic50 if global_min_ic50 is not None else rows_df['IC50_Dark_value'].min()
    for _, row in rows_df.iterrows():
        ic50_val = row['IC50_Dark(M*10^-6)']
        ic50_num = row['IC50_Dark_value']
        time_val = row['Time(h)']
        cell_val = row['Cell_line']
        doi_val = row['DOI']
        year_val = row['Year']
        cis_val = row['IC50_Cisplatin(M*10^-6)']

        try:
            cis_str = f"{float(cis_val):.2f}" if not pd.isna(cis_val) else "—"
            cis_color = "#fbbf24" if not pd.isna(cis_val) else "#4a5568"
        except (ValueError, TypeError):
            cis_str = str(cis_val)
            cis_color = "#4a5568"

        doi_short = doi_val[:30] + ("…" if len(str(doi_val)) > 30 else "")
        is_best = ic50_num == min_ic50
        ic50_color = "#3de8a0" if is_best else "#e8ecf4"
        dot = "<span style='display:inline-block;width:5px;height:5px;border-radius:50%;background:#3de8a0;margin-right:5px;vertical-align:middle;'></span>" if is_best else ""

        rows_html += (
            f"<tr>"
            f"<td style='font-family:\"DM Mono\",monospace;font-size:0.75rem;color:#8892a4;"
            f"padding:5px 8px 5px 0;border-bottom:1px solid rgba(255,255,255,0.04);'>"
            f"{dot}{cell_val}"
            + (f"<span style='font-family:DM Mono,monospace;font-size:0.6rem;color:#4a5568;margin-left:6px;'>" + _CELL_LINE_ORGAN[cell_val] + "</span>" if cell_val in _CELL_LINE_ORGAN else "")
            + "</td>"
            f"<td style='font-family:\"DM Mono\",monospace;font-size:0.75rem;color:#8892a4;"
            f"padding:5px 8px 5px 0;border-bottom:1px solid rgba(255,255,255,0.04);'>{int(time_val) if not pd.isna(time_val) else '—'}h</td>"
            f"<td style='font-family:\"Syne\",sans-serif;font-size:0.9rem;font-weight:700;"
            f"color:{ic50_color};padding:5px 8px 5px 0;border-bottom:1px solid rgba(255,255,255,0.04);'>{format_ic50(ic50_val)}</td>"
            f"<td style='font-family:\"DM Mono\",monospace;font-size:0.75rem;color:{cis_color};"
            f"padding:5px 8px 5px 0;border-bottom:1px solid rgba(255,255,255,0.04);'>{cis_str}</td>"
            f"<td style='padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.04);'>"
            f"<a href='https://doi.org/{doi_val}' target='_blank' "
            f"style='font-family:\"DM Mono\",monospace;font-size:0.68rem;color:#5b8fff;"
            f"text-decoration:none;'>{doi_short}</a></td>"
            f"<td style='font-family:\"DM Mono\",monospace;font-size:0.75rem;color:#4a5568;"
            f"padding:5px 0 5px 8px;border-bottom:1px solid rgba(255,255,255,0.04);'>{int(year_val)}</td>"
            f"</tr>"
        )

    return header + rows_html + "</tbody></table>"


@st.cache_data
def run_ligand_search(query_smi, scaffold_mode, metal, line, time_val, tanimoto_threshold=0.7):
    """Cached ligand search — only reruns when inputs change."""
    smiles_inputs = [canonize_smiles(s) for s in query_smi.split(".") if s]

    if scaffold_mode == "Full molecule match":
        result = df[df["SMILES_Ligands"].apply(
            lambda x: all(s in x.split(".") for s in smiles_inputs)
        )].sort_values(by="SMILES_Ligands")
    elif scaffold_mode == "Scaffold-based search":
        scaffolds = [get_murcko_scaffold(s) if get_murcko_scaffold(s) != "" else s for s in smiles_inputs]
        result = df[df["Scaffold"].apply(
            lambda x: all(s in x.split(".") for s in scaffolds)
        )].sort_values(by="SMILES_Ligands")
    elif scaffold_mode == "Similarity search":
        from rdkit.DataStructs import TanimotoSimilarity
        from rdkit.Chem import RDKFingerprint as _RDKFingerprint2
        query_mol = Chem.MolFromSmiles(query_smi)
        if query_mol is None:
            return df.iloc[0:0]
        query_fp = _RDKFingerprint2(query_mol)
        def _max_tanimoto(smiles_ligands):
            fps = _ligand_fps.get(smiles_ligands)
            if not fps:
                return 0.0
            return max(TanimotoSimilarity(query_fp, fp) for fp in fps)
        similarities = df["SMILES_Ligands"].apply(_max_tanimoto)
        result = df[similarities >= tanimoto_threshold].copy()
        result["_similarity"] = similarities[result.index]
        result = result.sort_values("_similarity", ascending=False).drop(columns=["_similarity"])
    else:
        def _has_sub(smiles_ligands):
            qmols = [Chem.MolFromSmiles(s) for s in smiles_inputs]
            qmols = [m for m in qmols if m is not None]
            parts = [Chem.MolFromSmiles(s) for s in smiles_ligands.split(".")]
            parts = [p for p in parts if p is not None]
            for qmol in qmols:
                if not any(p.HasSubstructMatch(qmol) for p in parts):
                    return False
            return True
        result = df[df["SMILES_Ligands"].apply(_has_sub)].sort_values(by="SMILES_Ligands")

    if metal != "All metals":
        result = result[result["Metal"] == metal]
    if line != "All cell lines":
        result = result[result["Cell_line"] == line]
    if time_val != "All time ranges":
        result = result[result["Time(h)"] == float(time_val)]
    return result


def render_metal_bars(counts, total, metal_clr):
    bars_html = '<div style="display:flex;flex-direction:column;gap:6px;margin-bottom:20px;">'
    for metal, count in counts.items():
        clr = metal_clr.get(str(metal), "#8892a4")
        pct = count / total * 100
        bars_html += (
            f'<div style="display:flex;align-items:center;gap:10px;">'            f'<span style="font-family:DM Mono,monospace;font-size:12px;font-weight:500;'            f'color:{clr};width:24px;text-align:right;flex-shrink:0;">{metal}</span>'            f'<div style="flex:1;height:4px;background:rgba(255,255,255,0.07);border-radius:2px;">'            f'<div style="width:{pct:.1f}%;height:100%;background:{clr};border-radius:2px;"></div></div>'            f'<span style="font-family:DM Mono,monospace;font-size:11px;color:#4a5568;width:46px;text-align:right;flex-shrink:0;">{count:,}</span>'            f'<span style="font-family:DM Mono,monospace;font-size:11px;color:#4a5568;width:36px;text-align:right;flex-shrink:0;">{pct:.1f}%</span>'            f'</div>'
        )
    bars_html += '</div>'
    st.markdown(bars_html, unsafe_allow_html=True)

def format_ic50(val, show_unit=False):
    """Format IC50 value: if < 0.01 μM, show in nM."""
    try:
        f = float(val)
        if f < 0.01:
            return f"{f * 1000:.2g} nM"
        return f"{val} μM" if show_unit else str(val)
    except (ValueError, TypeError):
        return str(val)

def show_search_results(search_df, ascending=True, sort_by_year=False, page_key="page_num"):

    # ── Summary bar ──────────────────────────────────────────────────────────
    num_complexes = search_df.drop_duplicates(subset=['SMILES_Ligands', 'Metal']).shape[0]
    num_ic50 = search_df.drop_duplicates(subset=['SMILES_Ligands', 'Counterion', 'IC50_Dark(M*10^-6)', 'Cell_line', 'Time(h)', 'DOI', 'Metal']).shape[0]
    num_sources = search_df['DOI'].nunique()

    # Compute pagination info for summary bar
    _PAGE_SIZE = 20
    _total_pages = max(1, (num_complexes + _PAGE_SIZE - 1) // _PAGE_SIZE)
    _cur_page = st.session_state.get(page_key, 0)
    _showing_from = _cur_page * _PAGE_SIZE + 1
    _showing_to = min((_cur_page + 1) * _PAGE_SIZE, num_complexes)
    _paging_html = (
        f"<div style='margin-left:auto;text-align:right;'>"
        f"<div style='font-family:DM Mono,monospace;font-size:0.65rem;letter-spacing:0.1em;"
        f"text-transform:uppercase;color:#4a5568;margin-bottom:3px;'>Showing</div>"
        f"<div style='font-family:DM Mono,monospace;font-size:0.85rem;font-weight:500;color:#8892a4;'>"
        f"{_showing_from}–{_showing_to} of {num_complexes}</div>"
        f"</div>"
    ) if num_complexes > _PAGE_SIZE else ""

    col_stats, col_csv = st.columns([5, 1])
    with col_stats:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:24px;padding:16px 20px;
                    background:#0f1420;border:1px solid rgba(255,255,255,0.07);
                    border-radius:12px;margin-bottom:20px;">
            <div>
                <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.1em;
                            text-transform:uppercase;color:#4a5568;margin-bottom:3px;">Complexes</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;
                            color:#e8ecf4;">{num_complexes}</div>
            </div>
            <div style="width:1px;height:36px;background:rgba(255,255,255,0.07);"></div>
            <div>
                <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.1em;
                            text-transform:uppercase;color:#4a5568;margin-bottom:3px;">IC&#8325;&#8320; values</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;
                            color:#3de8a0;">{num_ic50}</div>
            </div>
            <div style="width:1px;height:36px;background:rgba(255,255,255,0.07);"></div>
            <div>
                <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.1em;
                            text-transform:uppercase;color:#4a5568;margin-bottom:3px;">Sources</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;
                            color:#e8ecf4;">{num_sources}</div>
            </div>
            {_paging_html}
        </div>
        """, unsafe_allow_html=True)
    with col_csv:
        csv = search_df.to_csv(index=False).encode('utf-8')
        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
        st.download_button(label="Download CSV", data=csv, file_name="ic50.csv", mime="text/csv", use_container_width=True)

    # ── Metal color map ───────────────────────────────────────────────────────
    _METAL_CLR = {"Ru": "#3de8a0", "Ir": "#5b8fff", "Os": "#ff7c5b", "Rh": "#c084fc", "Re": "#fbbf24"}
    _MONO = "DM Mono, monospace"
    _SYNE = "Syne, sans-serif"

    # ── Group by compound (SMILES_Ligands + Metal) ────────────────────────────
    # Sort within each group by IC50 ascending so top-3 are the most active
    search_df = search_df.copy()
    search_df['IC50_Dark_value'] = pd.to_numeric(search_df['IC50_Dark_value'], errors='coerce')
    search_df.sort_values('IC50_Dark_value', ascending=ascending, inplace=True)

    grouped = search_df.groupby(['SMILES_Ligands', 'Metal'], sort=False)

    # Sort groups by min IC50 explicitly
    # Sort groups by min IC50 or year, then paginate
    if sort_by_year:
        group_order = search_df.groupby(['SMILES_Ligands', 'Metal'])['Year'].max()
        all_keys = group_order.sort_values(ascending=ascending).index.tolist()
    else:
        group_min = search_df.groupby(['SMILES_Ligands', 'Metal'])['IC50_Dark_value'].min()
        all_keys = group_min.sort_values(ascending=ascending).index.tolist()

    PAGE_SIZE = 20
    total_pages = max(1, (len(all_keys) + PAGE_SIZE - 1) // PAGE_SIZE)
    if page_key not in st.session_state:
        st.session_state[page_key] = 0
    if st.session_state[page_key] >= total_pages:
        st.session_state[page_key] = 0
    page_num = st.session_state[page_key]
    group_keys = all_keys[page_num * PAGE_SIZE : (page_num + 1) * PAGE_SIZE]


    _ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}

    for (smi, metal) in group_keys:
        group = grouped.get_group((smi, metal))
        metal_color = _METAL_CLR.get(str(metal), "#8892a4")

        min_ic50 = group['IC50_Dark_value'].min()
        n_lines = len(group)

        # Abbreviation: take first non-null value in the group
        abbr_series = group['Abbreviation_in_the_article'].dropna()
        abbr_val = str(abbr_series.iloc[0]) if len(abbr_series) > 0 else ""

        # Oxidation state: int -> Roman numeral
        ox_series = group['Oxidation_state'].dropna()
        ox_str = ""
        if len(ox_series) > 0:
            try:
                ox_roman = _ROMAN.get(int(ox_series.iloc[0]), str(int(ox_series.iloc[0])))
                ox_str = f"({ox_roman})"
            except (ValueError, TypeError):
                ox_str = ""

        col_img, col_data = st.columns([1, 4])

        with col_img:
            col_img.image(draw_molecule(smi), use_container_width=True)

        with col_data:
            # Header: metal(ox) badge + abbreviation + min IC50 + line count
            abbr_html = (
                f"<span style='font-family:{_MONO};font-size:0.6rem;color:#4a5568;margin-right:2px;'>Name from article:</span>"
                f"<span style='font-family:{_SYNE};font-size:1rem;font-weight:700;color:#e8ecf4;'>{abbr_val}</span>"
                if abbr_val else ""
            )
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:4px;flex-wrap:wrap;'>"
                f"<span style='font-family:{_MONO};font-size:0.75rem;font-weight:600;"
                f"padding:3px 8px;border-radius:4px;background:{metal_color}18;color:{metal_color};'>{metal}{ox_str}</span>"
                f"<span style='font-family:{_SYNE};font-size:1.2rem;font-weight:800;color:#e8ecf4;'>{format_ic50(min_ic50, show_unit=True)}</span>"
                f"<span style='font-family:{_MONO};font-size:0.65rem;color:#4a5568;'>min IC&#8325;&#8320; · {n_lines} cell line{'s' if n_lines > 1 else ''}</span>"
                f"</div>"
                + (f"<div style='margin-bottom:3px;'>{abbr_html}</div>" if abbr_val else "")
                + f"<div style='font-family:{_MONO};font-size:0.65rem;color:#4a5568;word-break:break-all;margin-bottom:4px;'>{smi}</div>",
                unsafe_allow_html=True
            )

            # Top-3 rows always visible
            group_min_ic50 = group['IC50_Dark_value'].min()
            top3 = group.head(3)
            st.markdown(_render_lines_table(top3, metal_color, global_min_ic50=group_min_ic50), unsafe_allow_html=True)

            # Remaining rows under expander
            rest = group.iloc[3:]
            if len(rest) > 0:
                with st.expander(f"Show {len(rest)} more"):
                    st.markdown(_render_lines_table(rest, metal_color, global_min_ic50=group_min_ic50), unsafe_allow_html=True)

        st.markdown("<div style='height:1px;background:rgba(255,255,255,0.06);margin:8px 0;'></div>", unsafe_allow_html=True)

    # ── Pagination ────────────────────────────────────────────────────────────
    if total_pages > 1:
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        col_prev, col_info, col_next = st.columns([1, 2, 1])
        if col_prev.button("← Previous", key=f"{page_key}_prev", disabled=(page_num == 0), use_container_width=True):
            st.session_state[page_key] -= 1
            st.rerun()
        col_info.markdown(
            f'<div style="text-align:center;font-family:DM Mono,monospace;font-size:0.75rem;color:#4a5568;padding-top:8px;">'
            f'Page {page_num + 1} of {total_pages}</div>',
            unsafe_allow_html=True
        )
        if col_next.button("Next →", key=f"{page_key}_next", disabled=(page_num >= total_pages - 1), use_container_width=True):
            st.session_state[page_key] += 1
            st.rerun()

calc = FPCalculator("ecfp")

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

st.set_page_config(
    page_title='MetalCytoToxDB',
    page_icon='🧪',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    /* ── Base ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0a0d14 !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    [data-testid="stMain"] {
        background-color: #0a0d14 !important;
    }

    /* ── Hide Streamlit branding ── */
    #MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }
    [data-testid="stDecoration"] { display: none !important; }
    [data-testid="stSidebarCollapseButton"] { display: none !important; }
    [data-testid="stCacheSpinner"], .stCacheSpinner { display: none !important; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #0f1420 !important;
        border-right: 1px solid rgba(255,255,255,0.07) !important;
    }

    [data-testid="stSidebarContent"] {
        padding-top: 24px;
    }

    /* ── Top header area ── */
    [data-testid="stHeader"] { display: none !important; }
    [data-testid="stMainBlockContainer"] { padding-top: 0.5rem !important; }
    [data-testid="stMain"] > div { padding-top: 0 !important; }
    .main .block-container { padding-top: 0.5rem !important; }

    /* ── Typography ── */
    h1 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 800 !important;
        font-size: 2rem !important;
        color: #e8ecf4 !important;
        letter-spacing: -0.02em !important;
        line-height: 1.1 !important;
    }

    h2 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
        color: #e8ecf4 !important;
        letter-spacing: -0.01em !important;
    }

    h3 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        color: #c8d0e0 !important;
    }

    p, li, [data-testid="stMarkdownContainer"] p {
        font-family: 'DM Sans', sans-serif !important;
        color: #8892a4 !important;
        font-size: 0.92rem !important;
        line-height: 1.6 !important;
    }

    /* ── Tabs ── */
    [data-testid="stTabs"] [role="tablist"] {
        background: #0f1420 !important;
        border-bottom: 1px solid rgba(255,255,255,0.07) !important;
        padding: 0 8px !important;
        gap: 4px !important;
    }

    [data-testid="stTabs"] [role="tab"] {
        font-family: 'DM Mono', monospace !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        color: #4a5568 !important;
        padding: 12px 16px !important;
        border-radius: 0 !important;
        border-bottom: 2px solid transparent !important;
        transition: all 0.15s !important;
        background: transparent !important;
    }

    [data-testid="stTabs"] [role="tab"]:hover {
        color: #8892a4 !important;
        background: transparent !important;
    }

    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: #3de8a0 !important;
        border-bottom-color: #3de8a0 !important;
        background: transparent !important;
        font-weight: 500 !important;
    }

    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    [data-testid="stTabPanel"] {
        padding-top: 24px !important;
    }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: #0f1420 !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 12px !important;
        padding: 20px 24px !important;
        border-top: 2px solid #3de8a0 !important;
    }

    [data-testid="stMetricLabel"] {
        font-family: 'DM Mono', monospace !important;
        font-size: 0.7rem !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: #4a5568 !important;
    }

    [data-testid="stMetricValue"] {
        font-family: 'Syne', sans-serif !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: #e8ecf4 !important;
    }

    /* ── Buttons ── */
    [data-testid="stButton"] button {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        background: transparent !important;
        color: #8892a4 !important;
        padding: 8px 18px !important;
        transition: all 0.15s !important;
    }

    [data-testid="stButton"] button:hover {
        background: rgba(255,255,255,0.05) !important;
        color: #e8ecf4 !important;
        border-color: rgba(255,255,255,0.25) !important;
    }

    /* Primary button (Download) */
    [data-testid="stDownloadButton"] button {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        border-radius: 8px !important;
        background: #3de8a0 !important;
        color: #0a0d14 !important;
        border: none !important;
        padding: 8px 18px !important;
        transition: all 0.15s !important;
    }

    [data-testid="stDownloadButton"] button:hover {
        background: #5cf5b5 !important;
    }

    /* ── Inputs ── */
    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] > div > div {
        font-family: 'DM Mono', monospace !important;
        font-size: 0.82rem !important;
        background: #141926 !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 8px !important;
        color: #e8ecf4 !important;
        transition: border-color 0.15s !important;
    }

    [data-testid="stTextInput"] input:focus {
        border-color: #3de8a0 !important;
        box-shadow: 0 0 0 2px rgba(61,232,160,0.12) !important;
    }

    [data-testid="stTextInput"] label,
    [data-testid="stSelectbox"] label {
        font-family: 'DM Mono', monospace !important;
        font-size: 0.7rem !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: #4a5568 !important;
    }

    /* ── Selectbox ── */
    [data-testid="stSelectbox"] [data-baseweb="select"] {
        background: #141926 !important;
        border-radius: 8px !important;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background: #0f1420 !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 10px !important;
    }

    [data-testid="stExpander"] summary {
        font-family: 'DM Mono', monospace !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.06em !important;
        color: #8892a4 !important;
        padding: 12px 16px !important;
    }

    [data-testid="stExpander"] summary:hover {
        color: #e8ecf4 !important;
    }

    /* ── Error / Success / Info ── */
    [data-testid="stAlert"] {
        border-radius: 10px !important;
        border: 1px solid !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.85rem !important;
    }

    /* ── Plotly charts — transparent background ── */
    .js-plotly-plot .plotly,
    .js-plotly-plot .plotly .svg-container {
        background: transparent !important;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }

    /* ── Caption under images ── */
    [data-testid="stImage"] figcaption {
        font-family: 'DM Mono', monospace !important;
        font-size: 0.68rem !important;
        color: #4a5568 !important;
        text-align: center !important;
    }

    /* ── Columns gap ── */
    [data-testid="stHorizontalBlock"] {
        gap: 16px !important;
    }

    /* ── Code blocks (SMILES display) ── */
    code {
        font-family: 'DM Mono', monospace !important;
        font-size: 0.82rem !important;
        background: #141926 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 6px !important;
        padding: 2px 8px !important;
        color: #3de8a0 !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 2px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

    </style>
    """, unsafe_allow_html=True)

inject_css()
# ──────────────────────────────────────────────────────────────────────────────

df = pd.read_csv('MetalCytoToxDB.csv')
authors = pd.read_csv('Authors.csv')
df['IC50_Dark_value'] = df['IC50_Dark_value'].apply(scale_ic50)
df['IC50_class'] = df['IC50_Dark_value'].apply(class_ic50)

cells = df['Cell_line'].value_counts().reset_index().loc[:19]
years = df.drop_duplicates(subset=['DOI'])['Year'].value_counts().reset_index()
times = df['Time(h)'].value_counts().reset_index().loc[:5]
ic50_class = df['IC50_class'].value_counts().reset_index()
ic50_class['IC50_class'].replace({0: '≥10 μM', 1: '<10μM'}, inplace=True)
line_list = df['Cell_line'].value_counts().nlargest(50).index.tolist()
time_list = [int(t) for t in df['Time(h)'].value_counts().nlargest(6).index.tolist()]
metal_list = df['Metal'].value_counts().index.tolist()

# ── Load precomputed per-ligand RDKit fingerprints ───────────────────────
import pickle as _pickle
from rdkit.Chem import RDKFingerprint as _RDKFingerprint
try:
    with open('ligand_fps.pkl', 'rb') as _f:
        _ligand_fps = _pickle.load(_f)
except FileNotFoundError:
    # Fallback: compute on the fly if pkl not found
    def _precompute_fps(df):
        fps = {}
        for smi in df['SMILES_Ligands'].unique():
            lig_fps = []
            for part in smi.split('.'):
                mol = Chem.MolFromSmiles(part)
                if mol is not None:
                    lig_fps.append(_RDKFingerprint(mol))
            if lig_fps:
                fps[smi] = lig_fps
        return fps
    _ligand_fps = _precompute_fps(df)

_CELL_LINE_ORGAN = {
    '143B': 'Bone',
    '16HBE': 'Lung',
    '1BR.3.G': 'Fibroblast/Connective',
    '22Rv1': 'Prostate',
    '4T1': 'Breast',
    '518A2': 'Skin',
    '5637': 'Bladder',
    '8505C': 'Thyroid',
    '95D': 'Lung',
    'A172': 'Brain/CNS',
    'A2058': 'Skin',
    'A253': 'Salivary gland',
    'A2780': 'Ovary',
    'A2780-CP70': 'Ovary',
    'A2780/ADR': 'Ovary',
    'A2780TR': 'Ovary',
    'A2780cisR': 'Ovary',
    'A2780sens': 'Ovary',
    'A375': 'Skin',
    'A375P': 'Skin',
    'A427': 'Lung',
    'A431': 'Skin',
    'A498': 'Kidney',
    'A549': 'Lung',
    'A549-TAX': 'Lung',
    'A549R MCTSs': 'Lung',
    'A549cisR': 'Lung',
    'ACHN': 'Kidney',
    'ADHF': 'Fibroblast/Connective',
    'AGS': 'Stomach',
    'ARPE-19': 'Eye',
    'AsPC-1': 'Pancreas',
    'B16': 'Skin',
    'B16-F10': 'Skin',
    'BALB/3T3': 'Fibroblast/Connective',
    'BALB/3T3 clone A31': 'Fibroblast/Connective',
    'BE': 'Lung',
    'BEAS-2B': 'Lung',
    'BEL-7402': 'Liver',
    'BEL-7402cisR': 'Liver',
    'BEL-7404': 'Liver',
    'BEL-7404/CP20': 'Liver',
    'BEL-7404R': 'Liver',
    'BEL-7704': 'Liver',
    'BGC823': 'Stomach',
    'BGM': 'Kidney',
    'BHK': 'Kidney',
    'BHK-21': 'Kidney',
    'BJ': 'Fibroblast/Connective',
    'BT-474': 'Breast',
    'BT549': 'Breast',
    'BeWo': 'Lung',
    'BxPC-3': 'Pancreas',
    'C6': 'Brain/CNS',
    'CA-M75': 'Colon',
    'CAKI-1': 'Kidney',
    'CAL-27': 'Oral/Head&Neck',
    'CAPAN-1': 'Pancreas',
    'CCD-1029Sk': 'Skin',
    'CCD-1059Sk': 'Skin',
    'CCD-18Co': 'Colon',
    'CCD-19Co': 'Colon',
    'CCD-841': 'Colon',
    'CCD-841-CON': 'Colon',
    'CCL228': 'Colon',
    'CCRF-CEM': 'Blood',
    'CEM': 'Blood',
    'CFPAC': 'Pancreas',
    'CFPAC-1': 'Pancreas',
    'CH-1': 'Ovary',
    'CH1/PA-1': 'Ovary',
    'CHL-1': 'Skin',
    'CHO': 'Ovary',
    'CHO p-40': 'Ovary',
    'CI80-13S': 'Lung',
    'CNE': 'Nasopharynx',
    'CNE-1': 'Nasopharynx',
    'CNE-2': 'Nasopharynx',
    'CNE-2Z': 'Nasopharynx',
    'COLO-201': 'Colon',
    'COLO-205': 'Colon',
    'COLO-320': 'Colon',
    'COLO-829': 'Colon',
    'COV362': 'Ovary',
    'CRL7687': 'Ovary',
    'CT-26': 'Colon',
    'CT-26 LUC': 'Colon',
    'CaSki': 'Cervix',
    'Caco-2': 'Colon',
    'Capan-2': 'Pancreas',
    'DAN-G': 'Pancreas',
    'DLD-1': 'Colon',
    'DU-145': 'Prostate',
    'DU-145cisR': 'Prostate',
    'EA.hy926': 'Endothelium/Vascular',
    'EAC': 'Breast',
    'EC-1': 'Esophagus',
    'EC109': 'Esophagus',
    'EC9706': 'Esophagus',
    'ECRF24': 'Endothelium/Vascular',
    'ES-2': 'Ovary',
    'EVSA-T': 'Breast',
    'Ect1/E6E7': 'Cervix',
    'FaDu': 'Oral/Head&Neck',
    'Fem-X': 'Skin',
    'G361': 'Skin',
    'GES-1': 'Stomach',
    'GM07492A': 'Fibroblast/Connective',
    'H23': 'Lung',
    'H358': 'Lung',
    'H460/MX20': 'Lung',
    'H4IIE': 'Liver',
    'H9c2': 'Heart',
    'HBL-100': 'Breast',
    'HCC1806': 'Breast',
    'HCC1937': 'Breast',
    'HCC38': 'Breast',
    'HCC70': 'Breast',
    'HCC827': 'Lung',
    'HCEC': 'Eye',
    'HCT-116': 'Colon',
    'HCT-116 p53+/+': 'Colon',
    'HCT-116 p53-/-': 'Colon',
    'HCT-116(p53/ko)': 'Colon',
    'HCT-116/Dox': 'Colon',
    'HCT-116CD44+': 'Colon',
    'HCT-116N': 'Colon',
    'HCT-116O': 'Colon',
    'HCT-116Ox': 'Colon',
    'HCT-15': 'Colon',
    'HCT-8': 'Colon',
    'HDFa': 'Fibroblast/Connective',
    'HEC-1-A': 'Endometrium/Uterus',
    'HEK293': 'Kidney',
    'HEK293T': 'Kidney',
    'HEp-2': 'Larynx/Head&Neck',
    'HFF': 'Fibroblast/Connective',
    'HFF-1': 'Fibroblast/Connective',
    'HFL-1': 'Lung',
    'HGC-27': 'Stomach',
    'HIO80': 'Ovary',
    'HK-2': 'Kidney',
    'HL-60/DR': 'Blood',
    'HL-7702': 'Liver',
    'HL60': 'Blood',
    'HL60/adr': 'Blood',
    'HL60/vinc': 'Blood',
    'HLF': 'Liver',
    'HMLER': 'Breast',
    'HOF': 'Ovary',
    'HOS': 'Bone',
    'HPAF-II': 'Pancreas',
    'HPL1D': 'Lung',
    'HS683': 'Brain/CNS',
    'HSAEC': 'Lung',
    'HSC-3': 'Oral/Head&Neck',
    'HT-1080': 'Fibroblast/Connective',
    'HT-29': 'Colon',
    'HUVEC': 'Endothelium/Vascular',
    'HaCaT': 'Skin',
    'HdFa': 'Fibroblast/Connective',
    'HeLa': 'Cervix',
    'HeLa S3': 'Cervix',
    'HeLa229': 'Cervix',
    'Hep3B': 'Liver',
    'HepG2': 'Liver',
    'Hs294T': 'Skin',
    'Hs578T': 'Breast',
    'Hs68': 'Fibroblast/Connective',
    'Hs683': 'Brain/CNS',
    'HuH-7': 'Liver',
    'HuT78': 'Blood',
    'ID8': 'Ovary',
    'IGROV-1': 'Ovary',
    'IMR-32': 'Brain/CNS',
    'IMR-90': 'Lung',
    'Ishikawa': 'Endometrium/Uterus',
    'Jurkat': 'Blood',
    'K-562': 'Blood',
    'KATO III': 'Stomach',
    'KB': 'Oral/Head&Neck',
    'KB-1019N': 'Oral/Head&Neck',
    'KB-3-1': 'Oral/Head&Neck',
    'KB-C2': 'Oral/Head&Neck',
    'KB-CV60': 'Oral/Head&Neck',
    'KB-V1': 'Oral/Head&Neck',
    'KB-V1/Vbl': 'Oral/Head&Neck',
    'KB/ATO': 'Oral/Head&Neck',
    'KBC-1': 'Oral/Head&Neck',
    'KBCP20': 'Oral/Head&Neck',
    'KCP4': 'Ovary',
    'KG-1': 'Blood',
    'KG-1a': 'Blood',
    'KMST-6': 'Fibroblast/Connective',
    'KYSE-140': 'Esophagus',
    'L-929': 'Fibroblast/Connective',
    'L1210': 'Blood',
    'L1210/0': 'Blood',
    'L1210/2': 'Blood',
    'L132': 'Lung',
    'LA795': 'Lung',
    'LCLC': 'Lung',
    'LCLC-103H': 'Lung',
    'LLC': 'Lung',
    'LLC1': 'Lung',
    'LN-18': 'Brain/CNS',
    'LN229': 'Brain/CNS',
    'LNCaP': 'Prostate',
    'LNCaP-Clone-FGC': 'Prostate',
    'LNZ308': 'Brain/CNS',
    'LO2': 'Liver',
    'LS 174T': 'Colon',
    'LS-174': 'Colon',
    'LoVo': 'Colon',
    'LoVo/Dox': 'Colon',
    'M-14': 'Skin',
    'M19': 'Skin',
    'M19-MEL': 'Skin',
    'MAgEC 10.5': 'Stomach',
    'MB49': 'Bladder',
    'MCF-10A': 'Breast',
    'MCF-12A': 'Breast',
    'MCF-7': 'Breast',
    'MCF-7/DOX': 'Breast',
    'MCF-7/Topo': 'Breast',
    'MCF-7CR': 'Breast',
    'MDA-MB-134-VI': 'Breast',
    'MDA-MB-216': 'Breast',
    'MDA-MB-231': 'Breast',
    'MDA-MB-231/Adr': 'Breast',
    'MDA-MB-361': 'Breast',
    'MDA-MB-435': 'Breast',
    'MDA-MB-435S': 'Breast',
    'MDA-MB-436': 'Breast',
    'MDA-MB-453': 'Breast',
    'MDA-MB-468': 'Breast',
    'MDBK': 'Kidney',
    'MDCK': 'Kidney',
    'ME-180': 'Cervix',
    'MES-SA': 'Endometrium/Uterus',
    'MES-SA/Dx5': 'Endometrium/Uterus',
    'MG-63': 'Bone',
    'MGC-803': 'Stomach',
    'MIA PaCa-2': 'Pancreas',
    'MIA PaCa-2cisR': 'Pancreas',
    'ML2': 'Blood',
    'MLuMEC': 'Breast',
    'MM418c5': 'Skin',
    'MM96L': 'Skin',
    'MO59J': 'Brain/CNS',
    'MOLM-13': 'Blood',
    'MOLT-4': 'Blood',
    'MOR': 'Blood',
    'MOR/CPR': 'Blood',
    'MRC-5': 'Lung',
    'MRC5pd30': 'Lung',
    'MS1': 'Endothelium/Vascular',
    'Molt4/C8': 'Blood',
    'NALM6': 'Blood',
    'NB4': 'Blood',
    'NCI-H1299': 'Lung',
    'NCI-H1975': 'Lung',
    'NCI-H226': 'Lung',
    'NCI-H292': 'Lung',
    'NCI-H295R': 'Adrenal/Neuroendocrine',
    'NCI-H460': 'Lung',
    'NCI-N87': 'Stomach',
    'NCM460': 'Colon',
    'NFF': 'Fibroblast/Connective',
    'NHDF': 'Fibroblast/Connective',
    'NIH-3T3': 'Fibroblast/Connective',
    'NP69': 'Nasopharynx',
    'Neuro-2a': 'Brain/CNS',
    'OE19': 'Esophagus',
    'OE21': 'Esophagus',
    'OE33': 'Esophagus',
    'OSE': 'Ovary',
    'OVCAR-3': 'Ovary',
    'OVCAR-4': 'Ovary',
    'OVCAR-5': 'Ovary',
    'OVCAR-8': 'Ovary',
    'PANC-1': 'Pancreas',
    'PANC-10-05': 'Pancreas',
    'PBMC': 'Blood',
    'PBMCs': 'Blood',
    'PC-12': 'Adrenal/Neuroendocrine',
    'PC-3': 'Prostate',
    'PNT1A': 'Prostate',
    'PNT2': 'Prostate',
    'PSN1': 'Pancreas',
    'PT45': 'Pancreas',
    'RAW 264.7': 'Blood/Immune',
    'RD': 'Muscle',
    'RDM4': 'Blood',
    'REH': 'Blood',
    'RKO': 'Colon',
    'RPE-1': 'Eye',
    'RPMI-8226': 'Blood',
    'RT-112': 'Bladder',
    'RT112': 'Bladder',
    'RT112 cP': 'Bladder',
    'S180': 'Connective tissue',
    'SCC-25': 'Oral/Head&Neck',
    'SCC-4': 'Oral/Head&Neck',
    'SCC-9': 'Oral/Head&Neck',
    'SF-268': 'Brain/CNS',
    'SGC-7901': 'Stomach',
    'SGC-7901/DDP': 'Stomach',
    'SH-SY5Y': 'Brain/CNS',
    'SISO': 'Cervix',
    'SK-BR-3': 'Breast',
    'SK-Hep1': 'Liver',
    'SK-MEL-147': 'Skin',
    'SK-MEL-28': 'Skin',
    'SK-MEL-5': 'Skin',
    'SK-Mel-103': 'Skin',
    'SK-OV-3': 'Ovary',
    'SK-OV-3cisR': 'Ovary',
    'SKMel2': 'Skin',
    'SKOV/Pt': 'Ovary',
    'SMMC-7721': 'Liver',
    'SNU-1': 'Stomach',
    'SUNE1': 'Nasopharynx',
    'SW480': 'Colon',
    'SW480 ': 'Colon',
    'SW620': 'Colon',
    'SW620/AD300': 'Colon',
    'SW620C': 'Colon',
    'SW620D': 'Colon',
    'SW620E': 'Colon',
    'SW620M': 'Colon',
    'SW620Mito': 'Colon',
    'SW620V': 'Colon',
    'SW626': 'Ovary',
    'Saos-2': 'Bone',
    'SiHa': 'Cervix',
    'SiHa ': 'Cervix',
    'Sk-mel': 'Skin',
    'SkMel-29': 'Skin',
    'T24': 'Bladder',
    'T47D': 'Breast',
    'T98G': 'Brain/CNS',
    'THP-1': 'Blood',
    'TK-10': 'Kidney',
    'TS/A': 'Breast',
    'Toledo': 'Blood',
    'Toledo-cis': 'Blood',
    'U-87 MG': 'Brain/CNS',
    'U251': 'Brain/CNS',
    'U266': 'Blood',
    'U266B1': 'Blood',
    'U2OS': 'Bone',
    'U2OS/Pt': 'Bone',
    'U373': 'Brain/CNS',
    'U937': 'Blood',
    'UACC-62': 'Skin',
    'UWB1.289': 'Ovary',
    'UWB1.289+BRCA1': 'Ovary',
    'V79': 'Fibroblast/Connective',
    'WI-38': 'Lung',
    'WM115': 'Skin',
    'WRL68': 'Liver',
    'WiDr': 'Colon',
    'cmt167': 'Lung',
}

_TISSUE_LIST = ['Adrenal/Neuroendocrine', 'Bladder', 'Blood', 'Blood/Immune', 'Bone',
                'Brain/CNS', 'Breast', 'Cervix', 'Colon', 'Connective tissue',
                'Endometrium/Uterus', 'Endothelium/Vascular', 'Esophagus', 'Eye',
                'Fibroblast/Connective', 'Heart', 'Kidney', 'Larynx/Head&Neck', 'Liver',
                'Lung', 'Muscle', 'Nasopharynx', 'Oral/Head&Neck', 'Ovary', 'Pancreas',
                'Prostate', 'Salivary gland', 'Skin', 'Stomach', 'Thyroid']

# Normal cell lines (manually annotated)
_NORMAL_LINES = {'HEK293', 'MRC-5', 'LO2', 'BEAS-2B', 'MCF-10A', 'NIH-3T3', 'HEK293T',
                 'RPE-1', 'L-929', 'HL-7702', 'HLF', 'HaCaT', 'HUVEC', '16HBE', 'Vero',
                 'NFF', 'ARPE-19', 'EA.hy926', 'HK-2', 'PBMC', 'CHO', 'V79', 'WI-38',
                 'IMR-90', 'GES-1', 'NCM460', 'NP69', 'PBMCs', 'BHK-21', 'CCD-19Lu',
                 'HELF', 'W138', 'HMrSv5', 'PBMC-PHA', 'PBMC+PHA', 'FLS', 'DPSC', 'BMSC', 'C8-D1A'}

# Light toxicity subset
light_df = df[df['IC50_Light(M*10^-6)'].notna()].copy()
light_df['IC50_Light_value'] = pd.to_numeric(light_df['IC50_Light(M*10^-6)'], errors='coerce')
light_df['PI'] = pd.to_numeric(light_df['IC50_Dark_value'], errors='coerce') / light_df['IC50_Light_value']

n_entries = df.drop_duplicates(subset=['SMILES_Ligands', 'Counterion', 'IC50_Dark(M*10^-6)', 'Cell_line', 'Time(h)', 'DOI', 'Metal']).shape[0]
n_smiles = df.drop_duplicates(['SMILES_Ligands', 'Metal']).shape[0]
n_sources = df['DOI'].nunique()
n_cell = df['Cell_line'].nunique()
metal_counts = df.drop_duplicates(subset=['SMILES_Ligands', 'Counterion', 'IC50_Dark(M*10^-6)', 'Cell_line', 'Time(h)', 'DOI', 'Metal']).groupby('Metal').size().sort_values(ascending=False)
metal_total = metal_counts.sum()



# ── Sidebar navigation ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 8px;">
        <div style="font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.2em;
                    color:#3de8a0;text-transform:uppercase;margin-bottom:6px;">database</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:800;
                    color:#e8ecf4;line-height:1.1;">
            Metal<span style="color:#3de8a0;">Cyto</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _nav_options = [
        "🔍  Search complexes",
        "📚  Literature",
        "☀️  Phototoxicity",
        "⚖️  Selectivity Index",
        "📊  Statistics",
    ]
    _default_idx = 0
    if "_nav" in st.session_state and st.session_state["_nav"] in _nav_options:
        _default_idx = _nav_options.index(st.session_state["_nav"])
        del st.session_state["_nav"]

    page = st.radio(
        label="Navigation",
        options=_nav_options,
        index=_default_idx,
        label_visibility="hidden",
    )

    st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.07);margin:20px 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#4a5568;line-height:1.6;">
        <div style="color:#8892a4;margin-bottom:6px;">Cite this work:</div>
        <div style="margin-bottom:8px;color:#4a5568;">Krasnov et al. <em>Machine Learning for Anticancer Activity Prediction of Transition Metal Complexes</em></div>
        <a href="https://doi.org/10.26434/chemrxiv-2025-1nqvm-v2" target="_blank"
           style="color:#5b8fff;text-decoration:none;word-break:break-all;">
            ChemRxiv 2025 ↗
        </a>
    </div>
    """, unsafe_allow_html=True)

# ── CSS: style sidebar radio as nav items ────────────────────────────────────
st.markdown("""
<style>
/* ── Sidebar radio → nav items ── */
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: column !important;
    gap: 2px !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 400 !important;
    color: #8892a4 !important;
    padding: 9px 14px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: background 0.15s, color 0.15s !important;
    width: 100% !important;
    display: block !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(255,255,255,0.05) !important;
    color: #e8ecf4 !important;
}
/* Active state via :has */
[data-testid="stSidebar"] [data-testid="stRadio"] div[data-baseweb="radio"]:has(input:checked) label {
    background: rgba(61,232,160,0.1) !important;
    color: #3de8a0 !important;
    font-weight: 500 !important;
}
/* Hide radio dots completely */
[data-testid="stSidebar"] [data-testid="stRadio"] [data-baseweb="radio"] > div:first-child,
[data-testid="stSidebar"] [data-testid="stRadio"] [data-baseweb="radio"] svg {
    display: none !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [data-baseweb="radio"] {
    padding-left: 0 !important;
    gap: 0 !important;
}

/* ── Search & predict button ── */
[data-testid="stButton"] button[kind="primary"],
[data-testid="stButton"]:has(button:contains("Search")) button,
[data-testid="stButton"]:has(button:contains("Predict")) button {
    background: #3de8a0 !important;
    color: #0a0d14 !important;
    border: none !important;
    font-weight: 600 !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    background: #5cf5b5 !important;
}

/* ── Search input placeholder ── */
input::placeholder {
    color: #8892a4 !important;
    opacity: 1 !important;
}

/* ── Nothing found message ── */
.stMarkdown p:has-text("Nothing found") {
    color: #4a5568 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
}

/* ── Section dividers ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.07) !important;
    margin: 24px 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Pages ─────────────────────────────────────────────────────────────────────

@st.dialog("Draw a structure", width="large")
def draw_structure_dialog():
    drawn = st_ketcher(height=480, key="ketcher_home")
    if drawn:
        st.session_state["_home_smi_val"] = drawn
        st.markdown('<div style="font-family:DM Mono,monospace;font-size:0.62rem;letter-spacing:0.08em;text-transform:uppercase;color:#6c757d;margin-top:4px;margin-bottom:2px;">Your SMILES — click Apply then close</div>', unsafe_allow_html=True)
        st.code(drawn, language=None)

if page == "🔍  Search complexes":

    _MONO = "DM Mono, monospace"
    _SYNE = "Syne, sans-serif"

    # ── Hero ─────────────────────────────────────────────────────────────────
    col_hero, col_zenodo = st.columns([5, 1])
    with col_hero:
        st.markdown(f"""
        <div style="margin-bottom:24px;">
            <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.15em;
                        text-transform:uppercase;color:#3de8a0;margin-bottom:8px;">
                ML-assisted cytotoxicity database
            </div>
            <div style="font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;
                        color:#e8ecf4;line-height:1.1;margin-bottom:12px;">
                Metal<span style="color:#3de8a0;">Cyto</span>ToxDB
            </div>
            <div style="font-family:'DM Sans',sans-serif;font-size:0.9rem;color:#8892a4;
                        max-width:560px;line-height:1.7;">
                Explore IC&#8325;&#8320; cytotoxicity data for transition metal complexes.
                Search by cell line, ligand structure, or literature source.
            </div>
            <div style="display:flex;gap:32px;margin-top:20px;flex-wrap:wrap;">
                <div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.1em;
                                text-transform:uppercase;color:#4a5568;margin-bottom:3px;">IC&#8325;&#8320; values</div>
                    <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:#3de8a0;">{n_entries:,}</div>
                </div>
                <div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.1em;
                                text-transform:uppercase;color:#4a5568;margin-bottom:3px;">Complexes</div>
                    <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:#e8ecf4;">{n_smiles:,}</div>
                </div>
                <div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.1em;
                                text-transform:uppercase;color:#4a5568;margin-bottom:3px;">Cell lines</div>
                    <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:#e8ecf4;">{n_cell:,}</div>
                </div>
                <div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.1em;
                                text-transform:uppercase;color:#4a5568;margin-bottom:3px;">Sources</div>
                    <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:#e8ecf4;">{n_sources:,}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_zenodo:
        st.markdown("""
        <div style="display:flex;justify-content:flex-end;padding-top:8px;">
            <a href="https://doi.org/10.5281/zenodo.15853577" target="_blank"
               style="font-family:'DM Mono',monospace;font-size:0.7rem;font-weight:500;
                      letter-spacing:0.06em;color:#0a0d14;background:#3de8a0;
                      text-decoration:none;padding:8px 14px;border-radius:8px;
                      white-space:nowrap;">
                Zenodo ↗
            </a>
        </div>
        """, unsafe_allow_html=True)

    # ── Metal distribution bars ──────────────────────────────────────────────
    _METAL_CLR = {"Ru": "#3de8a0", "Ir": "#5b8fff", "Os": "#ff7c5b", "Rh": "#c084fc", "Re": "#fbbf24"}
    bars_html = '<div style="display:flex;flex-direction:column;gap:6px;margin-bottom:20px;">'
    for metal, count in metal_counts.items():
        clr = _METAL_CLR.get(str(metal), "#8892a4")
        pct = count / metal_total * 100
        bars_html += (
            f'<div style="display:flex;align-items:center;gap:10px;">'
            f'<span style="font-family:DM Mono,monospace;font-size:12px;font-weight:500;'
            f'color:{clr};width:24px;text-align:right;flex-shrink:0;">{metal}</span>'
            f'<div style="flex:1;height:4px;background:rgba(255,255,255,0.07);border-radius:2px;">'
            f'<div style="width:{pct:.1f}%;height:100%;background:{clr};border-radius:2px;"></div></div>'
            f'<span style="font-family:DM Mono,monospace;font-size:11px;color:#4a5568;width:46px;text-align:right;flex-shrink:0;">{count:,}</span>'
            f'<span style="font-family:DM Mono,monospace;font-size:11px;color:#4a5568;width:36px;text-align:right;flex-shrink:0;">{pct:.1f}%</span>'
            f'</div>'
        )
    bars_html += '</div>'
    st.markdown(bars_html, unsafe_allow_html=True)

    # ── Search bar ───────────────────────────────────────────────────────────
    # ── Examples (rendered BEFORE text_input so clicks set value first) ────────
    _EXAMPLES = [
        ("p-cymene · Cl⁻", "Cc1ccc(C(C)C)cc1.[Cl-]"),
        ("ppy", "[c-]1ccccc1-c1ccccn1"),
        ("bpy", "c1ccc(-c2ccccn2)nc1"),
        ("PPh3", "c1ccc(P(c2ccccc2)c2ccccc2)cc1"),
        ("phen · phen", "c1cnc2c(c1)ccc1cccnc12.c1cnc2c(c1)ccc1cccnc12"),
        ("Cp", "c1cc[cH-]c1"),
        ("dfppy", "Fc1c[c-]c(-c2ccccn2)c(F)c1"),
    ]

    # Set widget value from example/clear clicks before widget renders
    for i, (_, smi) in enumerate(_EXAMPLES):
        if st.session_state.get(f"ex_{i}"):
            st.session_state["home_smiles_input"] = smi
    if st.session_state.get("home_clear"):
        st.session_state["home_smiles_input"] = ""

    col_search, col_clear, col_draw, col_search_btn = st.columns([5, 0.4, 1, 1])
    with col_search:
        home_smiles_input = st.text_input(
            label="",
            placeholder='SMILES for ligand — use "." to combine multiple ligands',
            label_visibility="collapsed",
            key="home_smiles_input",
        )
    with col_clear:
        st.button("✕", use_container_width=True, key="home_clear")
    with col_draw:
        if st.button("✏ Draw", use_container_width=True):
            draw_structure_dialog()
    with col_search_btn:
        st.button("🔍 Search", use_container_width=True, type="primary")

    # ── Structure preview ─────────────────────────────────────────────────────
    preview_smi = home_smiles_input.strip()
    if preview_smi and Chem.MolFromSmiles(preview_smi):
        prev_col_img, prev_col_smi = st.columns([1, 6])
        with prev_col_img:
            st.image(draw_molecule(preview_smi, size=(120, 120)), width=120)
        with prev_col_smi:
            st.markdown(
                f'<div style="font-family:DM Mono,monospace;font-size:0.62rem;color:#4a5568;padding-top:4px;margin-bottom:2px;">Drawn structure</div>'
                f'<div style="font-family:DM Mono,monospace;font-size:0.72rem;color:#3de8a0;word-break:break-all;">{preview_smi}</div>',
                unsafe_allow_html=True
            )

    st.markdown('<span style="font-family:DM Mono,monospace;font-size:0.65rem;color:#4a5568;">Try search with popular ligands:</span>', unsafe_allow_html=True)
    ex_cols = st.columns(len(_EXAMPLES))
    for i, (label, _) in enumerate(_EXAMPLES):
        with ex_cols[i]:
            st.button(label, key=f"ex_{i}", use_container_width=True)

    # ── Search regime ─────────────────────────────────────────────────────────
    home_scaffold = st.radio(
        "", ["Full molecule match", "Scaffold-based search", "Substructure search", "Similarity search"],
        index=0, horizontal=True, key="home_scaffold", label_visibility="collapsed"
    )

    # ── Tanimoto threshold (only for similarity search) ──────────────────────
    if home_scaffold == "Similarity search":
        home_tanimoto = st.slider(
            "Tanimoto similarity threshold", min_value=0.3, max_value=1.0,
            value=0.7, step=0.05, key="home_tanimoto",
            format="%.2f"
        )
    else:
        home_tanimoto = 0.7

    # ── Search filters ───────────────────────────────────────────────────────
    col1f, col2f, col3f, col4f, col5f, col6f, col7f = st.columns(7)
    home_metal    = col1f.selectbox("Metal",          ["All metals"] + metal_list, index=0, key="home_metal")
    home_tissue   = col2f.selectbox("Tissue",         ["All tissues"] + _TISSUE_LIST, index=0, key="home_tissue")
    _home_line_list = sorted([l for l, o in _CELL_LINE_ORGAN.items() if o == home_tissue]) if home_tissue != "All tissues" else line_list
    home_line     = col3f.selectbox("Cell line",      ["All cell lines"] + _home_line_list, index=0, key="home_line")
    home_time     = col4f.selectbox("Exposure time",  ["All time ranges"] + time_list, index=0, key="home_time")
    home_ic50_min = col5f.number_input("IC₅₀ min, μM", min_value=0.0, step=1.0, value=None, placeholder="e.g. 0", format="%.4g", key="home_ic50_min")
    home_ic50_max = col6f.number_input("IC₅₀ max, μM", min_value=0.0, step=1.0, value=None, placeholder="e.g. 10", format="%.4g", key="home_ic50_max")
    home_sorting  = col7f.selectbox("Sorting",        ["Most cytotoxic above", "Least cytotoxic above", "Newest first", "Oldest first"], index=0, key="home_sorting")

    # ── Run search ────────────────────────────────────────────────────────────
    sort_by_year = home_sorting in ("Newest first", "Oldest first")
    ascending = (home_sorting == "Most cytotoxic above") if not sort_by_year else (home_sorting == "Oldest first")
    query_smi = home_smiles_input.strip()
    has_smiles = bool(query_smi)

    if has_smiles:
        mol = Chem.MolFromSmiles(query_smi)
        if mol is None:
            st.error("Invalid SMILES — please check your input.")
        else:
            _metal_symbols = {'Ru', 'Ir', 'Os', 'Rh', 'Re'}
            _atoms_in_query = {atom.GetSymbol() for atom in mol.GetAtoms()}
            if _atoms_in_query & _metal_symbols:
                _found = ', '.join(_atoms_in_query & _metal_symbols)
                st.warning(f"⚠️ Your SMILES contains a metal atom ({_found}). Please enter ligand SMILES only — use the Metal filter above to filter by metal.")
            _cache_key = (query_smi, home_scaffold, home_metal, home_line, home_time)
            _is_cached = st.session_state.get("_last_search_key") == _cache_key
            _spinner_msg = {
                "Full molecule match": "Searching...",
                "Scaffold-based search": "Computing scaffolds...",
                "Substructure search": "Running substructure search...",
            }.get(home_scaffold, "Searching...")
            if _is_cached:
                search_df = run_ligand_search(
                    query_smi, home_scaffold, home_metal, home_line, home_time, home_tanimoto
                )
            else:
                with st.spinner(_spinner_msg):
                    search_df = run_ligand_search(
                        query_smi, home_scaffold, home_metal, home_line, home_time
                    )
                st.session_state["_last_search_key"] = _cache_key
            if home_tissue != "All tissues":
                tissue_lines = {l for l, o in _CELL_LINE_ORGAN.items() if o == home_tissue}
                search_df = search_df[search_df["Cell_line"].isin(tissue_lines)]
            if home_ic50_min is not None:
                search_df = search_df[search_df["IC50_Dark_value"] >= home_ic50_min]
            if home_ic50_max is not None:
                search_df = search_df[search_df["IC50_Dark_value"] <= home_ic50_max]
            if search_df.shape[0] == 0:
                st.markdown("Nothing found")
            else:
                show_search_results(search_df, ascending=ascending, sort_by_year=sort_by_year)
    else:
        search_df = df.copy()
        if home_metal != "All metals":
            search_df = search_df[search_df["Metal"] == home_metal]
        if home_tissue != "All tissues":
            tissue_lines = {l for l, o in _CELL_LINE_ORGAN.items() if o == home_tissue}
            search_df = search_df[search_df["Cell_line"].isin(tissue_lines)]
        if home_line != "All cell lines":
            search_df = search_df[search_df["Cell_line"] == home_line]
        if home_time != "All time ranges":
            search_df = search_df[search_df["Time(h)"] == float(home_time)]
        if home_ic50_min is not None:
            search_df = search_df[search_df["IC50_Dark_value"] >= home_ic50_min]
        if home_ic50_max is not None:
            search_df = search_df[search_df["IC50_Dark_value"] <= home_ic50_max]
        if sort_by_year:
            search_df = search_df.sort_values("Year", ascending=ascending)
        show_search_results(search_df, ascending=ascending, sort_by_year=sort_by_year)



elif page == "📚  Literature":

    st.markdown("""
    <div style="margin-bottom:24px;">
        <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;
                    color:#e8ecf4;margin-bottom:8px;">Literature</div>
        <div style="font-family:'DM Sans',sans-serif;font-size:0.88rem;color:#8892a4;line-height:1.6;">
            Look up all IC&#8325;&#8320; data from a specific article or author.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1select, col2select = st.columns([1, 1])
    doi = col1select.text_input("DOI", key="DOI")
    author = col2select.text_input("Author surname", key="Author", placeholder="Keppler")

    if doi:
        doi = doi.replace('https://doi.org/', '').replace('http://doi.org/', '')
        doi = doi.replace('http://dx.doi.org/', '').replace('https://dx.doi.org/', '')
        doi = doi.replace('https://www.doi.org/', '').replace('https://www.dx.doi.org/', '')
        doi = doi.replace('doi.org/', '').lower()
        search_df = df[df['DOI'].apply(lambda x: x.lower() == doi)]
        if search_df.shape[0] == 0:
            st.markdown('Nothing found')
        else:
            show_search_results(search_df)
    if author:
        author = author.lower()
        author_df = authors[authors['Authors'].apply(lambda x: author in x.lower())]
        search_df = df[df['DOI'].isin(author_df['DOI'])]
        if search_df.shape[0] == 0:
            st.markdown('Nothing found')
        else:
            show_search_results(search_df)

elif page == "☀️  Phototoxicity":

    _MONO = "DM Mono, monospace"
    _SYNE = "Syne, sans-serif"
    _METAL_CLR = {"Ru": "#3de8a0", "Ir": "#5b8fff", "Os": "#ff7c5b", "Rh": "#c084fc", "Re": "#fbbf24"}
    _ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}

    st.markdown("""
    <div style="margin-bottom:24px;">
        <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;
                    color:#e8ecf4;margin-bottom:8px;">Phototoxicity</div>
        <div style="font-family:'DM Sans',sans-serif;font-size:0.88rem;color:#8892a4;line-height:1.6;">
            IC&#8325;&#8320; data measured under light irradiation. PI = IC&#8325;&#8320;(dark) / IC&#8325;&#8320;(light).
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metal distribution bars ──────────────────────────────────────────────
    pt_metal_counts = light_df.drop_duplicates(subset=['SMILES_Ligands', 'Counterion', 'IC50_Light(M*10^-6)', 'Cell_line', 'Time(h)', 'DOI', 'Metal']).groupby('Metal').size().sort_values(ascending=False)
    pt_metal_total = pt_metal_counts.sum()
    render_metal_bars(pt_metal_counts, pt_metal_total, _METAL_CLR)

    # ── Search bar ───────────────────────────────────────────────────────────
    if "_pt_smi_val" not in st.session_state:
        st.session_state["_pt_smi_val"] = ""

    col_search, col_clear, col_draw, col_search_btn = st.columns([5, 0.4, 1, 1])
    with col_search:
        pt_smiles_input = st.text_input(
            label="",
            placeholder='SMILES for ligand — use "." to combine multiple ligands',
            label_visibility="collapsed",
            key="pt_smiles_input",
            value=st.session_state["_pt_smi_val"],
        )
        st.session_state["_pt_smi_val"] = pt_smiles_input
    with col_clear:
        if st.button("✕", key="pt_clear", use_container_width=True):
            st.session_state["_pt_smi_val"] = ""
            st.rerun()
    with col_draw:
        if st.button("✏ Draw", key="pt_draw", use_container_width=True):
            draw_structure_dialog()
    with col_search_btn:
        st.button("🔍 Search", key="pt_search_btn", use_container_width=True, type="primary")

    pt_scaffold = st.radio(
        "", ["Full molecule match", "Scaffold-based search", "Substructure search", "Similarity search"],
        index=0, horizontal=True, key="pt_scaffold", label_visibility="collapsed"
    )

    # ── Tanimoto threshold for similarity search ─────────────────────────────
    if pt_scaffold == "Similarity search":
        pt_tanimoto = st.slider("Tanimoto similarity threshold", min_value=0.3, max_value=1.0,
                                value=0.7, step=0.05, key="pt_tanimoto", format="%.2f")
    else:
        pt_tanimoto = 0.7

    # ── Filters ──────────────────────────────────────────────────────────────
    col1f, col2f, col3f, col4f, col5f = st.columns(5)
    pt_metal   = col1f.selectbox("Metal",        ["All metals"] + metal_list, index=0, key="pt_metal")
    pt_tissue  = col2f.selectbox("Tissue",       ["All tissues"] + _TISSUE_LIST, index=0, key="pt_tissue")
    _pt_line_list = sorted([l for l, o in _CELL_LINE_ORGAN.items() if o == pt_tissue]) if pt_tissue != "All tissues" else line_list
    pt_line    = col3f.selectbox("Cell line",    ["All cell lines"] + _pt_line_list, index=0, key="pt_line")
    pt_pi_min  = col4f.selectbox("Min PI",       ["Any", "≥2", "≥5", "≥10", "≥50"], index=0, key="pt_pi_min")
    pt_sorting = col5f.selectbox("Sorting",      ["Highest PI first", "Lowest IC₅₀(light) first"], index=0, key="pt_sorting")
    pt_time = "All time ranges"

    # ── Filter light_df ───────────────────────────────────────────────────────
    pt_query = st.session_state.get("_pt_smi_val", "").strip()
    fdf = light_df.copy()
    if pt_query:
        mol = Chem.MolFromSmiles(pt_query)
        if mol is None:
            st.error("Invalid SMILES — please check your input.")
            fdf = fdf.iloc[0:0]
        else:
            pt_smiles_inputs = [canonize_smiles(s) for s in pt_query.split(".") if s]
            if pt_scaffold == "Full molecule match":
                fdf = fdf[fdf["SMILES_Ligands"].apply(
                    lambda x: all(s in x.split(".") for s in pt_smiles_inputs)
                )]
            elif pt_scaffold == "Scaffold-based search":
                pt_smiles_inputs = [get_murcko_scaffold(s) if get_murcko_scaffold(s) != "" else s for s in pt_smiles_inputs]
                fdf = fdf[fdf["Scaffold"].apply(
                    lambda x: all(s in x.split(".") for s in pt_smiles_inputs)
                )]
            elif pt_scaffold == "Similarity search":
                from rdkit.DataStructs import TanimotoSimilarity
                from rdkit.Chem import RDKFingerprint as _RDKFP
                pt_query_mol = Chem.MolFromSmiles(pt_query)
                pt_query_fp = _RDKFP(pt_query_mol)
                def _pt_max_tanimoto(smiles_ligands):
                    fps = _ligand_fps.get(smiles_ligands)
                    if not fps:
                        return 0.0
                    return max(TanimotoSimilarity(pt_query_fp, fp) for fp in fps)
                pt_sims = fdf["SMILES_Ligands"].apply(_pt_max_tanimoto)
                fdf = fdf[pt_sims >= pt_tanimoto]
            else:
                pt_query_mols = [Chem.MolFromSmiles(s) for s in pt_smiles_inputs]
                pt_query_mols = [m for m in pt_query_mols if m is not None]
                fdf = fdf[fdf["SMILES_Ligands"].apply(
                    lambda x: has_all_substructures(x, pt_query_mols)
                )]
    if pt_metal != "All metals":
        fdf = fdf[fdf["Metal"] == pt_metal]
    if pt_tissue != "All tissues":
        pt_tissue_lines = {l for l, o in _CELL_LINE_ORGAN.items() if o == pt_tissue}
        fdf = fdf[fdf["Cell_line"].isin(pt_tissue_lines)]
    if pt_line != "All cell lines":
        fdf = fdf[fdf["Cell_line"] == pt_line]
    if pt_time != "All time ranges":
        fdf = fdf[fdf["Time(h)"] == float(pt_time)]
    if pt_pi_min != "Any":
        pi_threshold = float(pt_pi_min.replace("≥", ""))
        fdf = fdf[fdf["PI"] >= pi_threshold]

    # ── Summary ───────────────────────────────────────────────────────────────
    n_pt_cpx = fdf.drop_duplicates(subset=["SMILES_Ligands", "Metal"]).shape[0]
    n_pt_ic50 = len(fdf)
    n_pt_sources = fdf['DOI'].nunique()
    col_stats, col_csv = st.columns([5, 1])
    with col_stats:
        _pt_cur_page = st.session_state.get("pt_page", 0)
        _pt_total = n_pt_cpx
        _pt_from = _pt_cur_page * 20 + 1
        _pt_to = min((_pt_cur_page + 1) * 20, _pt_total)
        _pt_showing = (
            f"<div style='margin-left:auto;text-align:right;'>"
            f"<div style='font-family:DM Mono,monospace;font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;color:#4a5568;margin-bottom:3px;'>Showing</div>"
            f"<div style='font-family:DM Mono,monospace;font-size:0.85rem;font-weight:500;color:#8892a4;'>{_pt_from}–{_pt_to} of {_pt_total}</div>"
            f"</div>"
        ) if _pt_total > 20 else ""
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:24px;padding:16px 20px;
                    background:#0f1420;border:1px solid rgba(255,255,255,0.07);
                    border-radius:12px;margin-bottom:20px;">
            <div>
                <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.1em;
                            text-transform:uppercase;color:#4a5568;margin-bottom:3px;">Complexes</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;
                            color:#e8ecf4;">{n_pt_cpx}</div>
            </div>
            <div style="width:1px;height:36px;background:rgba(255,255,255,0.07);"></div>
            <div>
                <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.1em;
                            text-transform:uppercase;color:#4a5568;margin-bottom:3px;">IC&#8325;&#8320; values</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;
                            color:#3de8a0;">{n_pt_ic50}</div>
            </div>
            <div style="width:1px;height:36px;background:rgba(255,255,255,0.07);"></div>
            <div>
                <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.1em;
                            text-transform:uppercase;color:#4a5568;margin-bottom:3px;">Sources</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;
                            color:#e8ecf4;">{n_pt_sources}</div>
            </div>
            {_pt_showing}
        </div>
        """, unsafe_allow_html=True)
    with col_csv:
        csv_pt = fdf.to_csv(index=False).encode('utf-8')
        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
        st.download_button("Download CSV", data=csv_pt, file_name="phototoxicity.csv", mime="text/csv", use_container_width=True)

    # ── Group and render ──────────────────────────────────────────────────────
    fdf['IC50_Light_value'] = pd.to_numeric(fdf['IC50_Light_value'], errors='coerce')
    fdf['PI'] = pd.to_numeric(fdf['PI'], errors='coerce')

    if pt_sorting == "Highest PI first":
        group_order = fdf.groupby(["SMILES_Ligands", "Metal"])["PI"].max().sort_values(ascending=False)
    else:
        group_order = fdf.groupby(["SMILES_Ligands", "Metal"])["IC50_Light_value"].min().sort_values(ascending=True)

    group_keys = group_order.index.tolist()[:50]
    grouped = fdf.groupby(["SMILES_Ligands", "Metal"], sort=False)

    for (smi, metal) in group_keys:
        if (smi, metal) not in grouped.groups:
            continue
        group = grouped.get_group((smi, metal))
        metal_color = _METAL_CLR.get(str(metal), "#8892a4")

        min_light = group['IC50_Light_value'].min()
        max_pi = group['PI'].max()
        n_lines = len(group)

        abbr_series = group['Abbreviation_in_the_article'].dropna()
        abbr_val = str(abbr_series.iloc[0]) if len(abbr_series) > 0 else ""
        ox_series = group['Oxidation_state'].dropna()
        ox_str = ""
        if len(ox_series) > 0:
            try:
                ox_str = f"({_ROMAN.get(int(ox_series.iloc[0]), str(int(ox_series.iloc[0])))})"
            except (ValueError, TypeError):
                ox_str = ""

        pi_color = "#3de8a0" if (not pd.isna(max_pi) and max_pi >= 10) else "#fbbf24" if (not pd.isna(max_pi) and max_pi >= 2) else "#e8ecf4"
        pi_str = f"{max_pi:.1f}" if not pd.isna(max_pi) else "—"
        abbr_html = (f"<span style='font-family:{_MONO};font-size:0.6rem;color:#4a5568;margin-right:2px;'>Name from article:</span>"
                      f"<span style='font-family:{_SYNE};font-size:1rem;font-weight:700;color:#e8ecf4;'>{abbr_val}</span>") if abbr_val else ""

        col_img, col_data = st.columns([1, 4])
        with col_img:
            st.image(draw_molecule(smi), use_container_width=True)
        with col_data:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:4px;flex-wrap:wrap;'>"
                f"<span style='font-family:{_MONO};font-size:0.75rem;font-weight:600;"
                f"padding:3px 8px;border-radius:4px;background:{metal_color}18;color:{metal_color};'>{metal}{ox_str}</span>"
                f"<span style='font-family:{_SYNE};font-size:1.2rem;font-weight:800;color:#e8ecf4;'>{min_light:.2f} μM</span>"
                f"<span style='font-family:{_MONO};font-size:0.65rem;color:#4a5568;'>IC&#8325;&#8320;(light) min</span>"
                f"<span style='font-family:{_MONO};font-size:0.85rem;font-weight:600;color:{pi_color};'>PI {pi_str}</span>"
                f"<span style='font-family:{_MONO};font-size:0.65rem;color:#4a5568;'>· {n_lines} cell line{'s' if n_lines > 1 else ''}</span>"
                f"</div>"
                + (f"<div style='margin-bottom:3px;'>{abbr_html}</div>" if abbr_val else "")
                + f"<div style='font-family:{_MONO};font-size:0.65rem;color:#4a5568;word-break:break-all;margin-bottom:4px;'>{smi}</div>",
                unsafe_allow_html=True
            )

            # Table header
            table_html = (
                "<table style='width:100%;border-collapse:collapse;margin-top:6px;'>"
                "<thead><tr>"
                "<th style='font-family:DM Mono,monospace;font-size:0.6rem;letter-spacing:0.08em;text-transform:uppercase;color:#4a5568;text-align:left;padding:4px 8px 4px 0;border-bottom:1px solid rgba(255,255,255,0.06);'>Cell line</th>"
                "<th style='font-family:DM Mono,monospace;font-size:0.6rem;letter-spacing:0.08em;text-transform:uppercase;color:#4a5568;text-align:left;padding:4px 8px 4px 0;border-bottom:1px solid rgba(255,255,255,0.06);'>Time</th>"
                "<th style='font-family:DM Mono,monospace;font-size:0.6rem;letter-spacing:0.08em;text-transform:uppercase;color:#4a5568;text-align:left;padding:4px 8px 4px 0;border-bottom:1px solid rgba(255,255,255,0.06);'>IC&#8325;&#8320;(dark)</th>"
                "<th style='font-family:DM Mono,monospace;font-size:0.6rem;letter-spacing:0.08em;text-transform:uppercase;color:#4a5568;text-align:left;padding:4px 8px 4px 0;border-bottom:1px solid rgba(255,255,255,0.06);'>IC&#8325;&#8320;(light)</th>"
                "<th style='font-family:DM Mono,monospace;font-size:0.6rem;letter-spacing:0.08em;text-transform:uppercase;color:#4a5568;text-align:left;padding:4px 8px 4px 0;border-bottom:1px solid rgba(255,255,255,0.06);'>PI</th>"
                "<th style='font-family:DM Mono,monospace;font-size:0.6rem;letter-spacing:0.08em;text-transform:uppercase;color:#4a5568;text-align:left;padding:4px 8px 4px 0;border-bottom:1px solid rgba(255,255,255,0.06);'>λ, nm</th>"
                "<th style='font-family:DM Mono,monospace;font-size:0.6rem;letter-spacing:0.08em;text-transform:uppercase;color:#4a5568;text-align:left;padding:4px 8px 4px 0;border-bottom:1px solid rgba(255,255,255,0.06);'>Time, min</th>"
                "<th style='font-family:DM Mono,monospace;font-size:0.6rem;letter-spacing:0.08em;text-transform:uppercase;color:#4a5568;text-align:left;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.06);'>DOI</th>"
                "<th style='font-family:DM Mono,monospace;font-size:0.6rem;letter-spacing:0.08em;text-transform:uppercase;color:#4a5568;text-align:left;padding:4px 0 4px 8px;border-bottom:1px solid rgba(255,255,255,0.06);'>Year</th>"
                "</tr></thead><tbody>"
            )

            group = group.sort_values('PI', ascending=False)
            top3 = group.head(3)
            rest = group.iloc[3:]
            min_light_val = group['IC50_Light_value'].min()

            def render_pt_rows(rows):
                html = ""
                for _, row in rows.iterrows():
                    dark_val = row['IC50_Dark(M*10^-6)']
                    light_val = row['IC50_Light(M*10^-6)']
                    light_num = row['IC50_Light_value']
                    pi_val = row['PI']
                    wl = row['Excitation_Wavelength(nm)']
                    irr = row['Irradiation_Time(minutes)']
                    doi_v = row['DOI']
                    doi_short = str(doi_v)[:28] + ("…" if len(str(doi_v)) > 28 else "")
                    is_best = light_num == min_light_val
                    dot = "<span style='display:inline-block;width:5px;height:5px;border-radius:50%;background:#3de8a0;margin-right:5px;vertical-align:middle;'></span>" if is_best else ""
                    light_color = "#3de8a0" if is_best else "#e8ecf4"
                    pi_str_row = f"{pi_val:.1f}" if not pd.isna(pi_val) else "—"
                    pi_c = "#3de8a0" if (not pd.isna(pi_val) and pi_val >= 10) else "#fbbf24" if (not pd.isna(pi_val) and pi_val >= 2) else "#8892a4"
                    wl_str = str(wl) if not pd.isna(wl) else "—"
                    irr_str = f"{int(irr)}" if not pd.isna(irr) else "—"
                    year_v = row['Year']
                    td = "border-bottom:1px solid rgba(255,255,255,0.04);"
                    html += (
                        f"<tr>"
                        "<td style='font-family:DM Mono,monospace;font-size:0.75rem;color:#8892a4;padding:5px 8px 5px 0;" + td + "'>"
                        + f"{dot}{row['Cell_line']}"
                        + (f"<span style='font-family:DM Mono,monospace;font-size:0.6rem;color:#4a5568;margin-left:6px;'>{_CELL_LINE_ORGAN[row['Cell_line']]}</span>" if row['Cell_line'] in _CELL_LINE_ORGAN else "")
                        + "</td>"
                        f"<td style='font-family:DM Mono,monospace;font-size:0.75rem;color:#8892a4;padding:5px 8px 5px 0;{td}'>{int(row['Time(h)'])}h</td>"
                        f"<td style='font-family:DM Mono,monospace;font-size:0.75rem;color:#8892a4;padding:5px 8px 5px 0;{td}'>{dark_val}</td>"
                        f"<td style='font-family:Syne,sans-serif;font-size:0.9rem;font-weight:700;color:{light_color};padding:5px 8px 5px 0;{td}'>{light_val}</td>"
                        f"<td style='font-family:DM Mono,monospace;font-size:0.8rem;font-weight:600;color:{pi_c};padding:5px 8px 5px 0;{td}'>{pi_str_row}</td>"
                        f"<td style='font-family:DM Mono,monospace;font-size:0.75rem;color:#8892a4;padding:5px 8px 5px 0;{td}'>{wl_str}</td>"
                        f"<td style='font-family:DM Mono,monospace;font-size:0.75rem;color:#8892a4;padding:5px 8px 5px 0;{td}'>{irr_str}</td>"
                        f"<td style='padding:5px 0;{td}'><a href='https://doi.org/{doi_v}' target='_blank' style='font-family:DM Mono,monospace;font-size:0.68rem;color:#5b8fff;text-decoration:none;'>{doi_short}</a></td>"
                        f"<td style='font-family:DM Mono,monospace;font-size:0.75rem;color:#4a5568;padding:5px 0 5px 8px;{td}'>{int(year_v)}</td>"
                        f"</tr>"
                    )
                return html

            st.markdown(table_html + render_pt_rows(top3) + "</tbody></table>", unsafe_allow_html=True)
            if len(rest) > 0:
                with st.expander(f"Show {len(rest)} more"):
                    st.markdown("<table style='width:100%;border-collapse:collapse;'><tbody>" + render_pt_rows(rest) + "</tbody></table>", unsafe_allow_html=True)

        st.markdown("<div style='height:1px;background:rgba(255,255,255,0.06);margin:8px 0;'></div>", unsafe_allow_html=True)

elif page == "⚖️  Selectivity Index":

    _MONO = "DM Mono, monospace"
    _SYNE = "Syne, sans-serif"
    _METAL_CLR = {"Ru": "#3de8a0", "Ir": "#5b8fff", "Os": "#ff7c5b", "Rh": "#c084fc", "Re": "#fbbf24"}
    _ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}

    st.markdown("""
    <div style="margin-bottom:24px;">
        <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;
                    color:#e8ecf4;margin-bottom:8px;">Selectivity Index</div>
        <div style="font-family:'DM Sans',sans-serif;font-size:0.88rem;color:#8892a4;line-height:1.6;">
            SI = IC&#8325;&#8320;(normal cell line) / IC&#8325;&#8320;(cancer cell line).
            Higher SI means more selective toward cancer cells.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metal distribution bars ──────────────────────────────────────────────
    _si_metal_counts = df[df['Cell_line'].isin(_NORMAL_LINES)].drop_duplicates(
        subset=['SMILES_Ligands', 'Metal']
    ).groupby('Metal').size().sort_values(ascending=False)
    _si_metal_total = _si_metal_counts.sum()
    render_metal_bars(_si_metal_counts, _si_metal_total, _METAL_CLR)

    # ── Search bar ───────────────────────────────────────────────────────────
    if "_si_smi_val" not in st.session_state:
        st.session_state["_si_smi_val"] = ""

    if st.session_state.get("si_clear"):
        st.session_state["si_smiles_input"] = ""

    col_search, col_clear, col_draw, col_search_btn = st.columns([5, 0.4, 1, 1])
    with col_search:
        si_smiles_input = st.text_input(
            label="",
            placeholder='SMILES for ligand — use "." to combine multiple ligands',
            label_visibility="collapsed",
            key="si_smiles_input",
        )
    with col_clear:
        st.button("✕", use_container_width=True, key="si_clear")
    with col_draw:
        if st.button("✏ Draw", key="si_draw", use_container_width=True):
            draw_structure_dialog()
    with col_search_btn:
        st.button("🔍 Search", key="si_search_btn", use_container_width=True, type="primary")

    si_scaffold = st.radio(
        "", ["Full molecule match", "Scaffold-based search", "Substructure search", "Similarity search"],
        index=0, horizontal=True, key="si_scaffold", label_visibility="collapsed"
    )

    # ── Tanimoto threshold for similarity search ─────────────────────────────
    if si_scaffold == "Similarity search":
        si_tanimoto = st.slider("Tanimoto similarity threshold", min_value=0.3, max_value=1.0,
                                value=0.7, step=0.05, key="si_tanimoto", format="%.2f")
    else:
        si_tanimoto = 0.7

    # ── Filters ──────────────────────────────────────────────────────────────
    normal_line_list = sorted(_NORMAL_LINES)
    cancer_line_list = [l for l in df['Cell_line'].value_counts().index.tolist() if l not in _NORMAL_LINES]

    col1f, col2f, col3f, col4f, col5f = st.columns(5)
    si_metal   = col1f.selectbox("Metal",              ["All metals"] + metal_list, index=0, key="si_metal")
    si_normal  = col2f.selectbox("Normal cell line",   ["All normal lines"] + normal_line_list, index=0, key="si_normal")
    si_cancer  = col3f.selectbox("Cancer cell line",   ["All cancer lines"] + cancer_line_list, index=0, key="si_cancer")
    si_si_min  = col4f.selectbox("Min SI",             ["Any", "≥2", "≥5", "≥10", "≥50"], index=0, key="si_si_min")
    si_sorting = col5f.selectbox("Sorting",            ["Highest SI first", "Lowest SI first", "Newest first", "Oldest first"], index=0, key="si_sorting")

    # ── Build SI dataframe ────────────────────────────────────────────────────
    normal_df = df[df['Cell_line'].isin(_NORMAL_LINES)][['SMILES_Ligands', 'Metal', 'Cell_line', 'Time(h)', 'IC50_Dark_value', 'IC50_Dark(M*10^-6)', 'DOI', 'Year', 'Abbreviation_in_the_article', 'Oxidation_state']].copy()
    cancer_df = df[~df['Cell_line'].isin(_NORMAL_LINES)][['SMILES_Ligands', 'Metal', 'Cell_line', 'Time(h)', 'IC50_Dark_value', 'IC50_Dark(M*10^-6)', 'DOI', 'Year']].copy()

    normal_df.columns = ['SMILES_Ligands', 'Metal', 'Normal_line', 'Time_n', 'IC50_normal_val', 'IC50_normal', 'DOI', 'Year', 'Abbreviation_in_the_article', 'Oxidation_state']
    cancer_df.columns = ['SMILES_Ligands', 'Metal', 'Cancer_line', 'Time_c', 'IC50_cancer_val', 'IC50_cancer', 'DOI_cancer', 'Year_cancer']

    si_df = normal_df.merge(cancer_df, on=['SMILES_Ligands', 'Metal'])
    si_df['SI'] = pd.to_numeric(si_df['IC50_normal_val'], errors='coerce') / pd.to_numeric(si_df['IC50_cancer_val'], errors='coerce')
    si_df = si_df[si_df['SI'].notna() & (si_df['SI'] > 0)]

    # Apply ligand search filter
    si_query = si_smiles_input.strip()
    if si_query:
        mol_si = Chem.MolFromSmiles(si_query)
        if mol_si is None:
            st.error("Invalid SMILES — please check your input.")
            si_query = ""
        else:
            si_smiles_inputs = [canonize_smiles(s) for s in si_query.split(".") if s]
            if si_scaffold == "Full molecule match":
                valid_smi = set(df[df["SMILES_Ligands"].apply(
                    lambda x: all(s in x.split(".") for s in si_smiles_inputs)
                )]["SMILES_Ligands"])
            elif si_scaffold == "Scaffold-based search":
                si_scaffolds = [get_murcko_scaffold(s) if get_murcko_scaffold(s) != "" else s for s in si_smiles_inputs]
                valid_smi = set(df[df["Scaffold"].apply(
                    lambda x: all(s in x.split(".") for s in si_scaffolds)
                )]["SMILES_Ligands"])
            elif si_scaffold == "Similarity search":
                from rdkit.DataStructs import TanimotoSimilarity
                from rdkit.Chem import RDKFingerprint as _RDKFP2
                si_query_mol = Chem.MolFromSmiles(si_query)
                si_query_fp = _RDKFP2(si_query_mol)
                def _si_max_tanimoto(smiles_ligands):
                    fps = _ligand_fps.get(smiles_ligands)
                    if not fps:
                        return 0.0
                    return max(TanimotoSimilarity(si_query_fp, fp) for fp in fps)
                si_sims = df["SMILES_Ligands"].apply(_si_max_tanimoto)
                valid_smi = set(df[si_sims >= si_tanimoto]["SMILES_Ligands"])
            else:
                def _si_has_sub(smiles_ligands):
                    qmols = [Chem.MolFromSmiles(s) for s in si_smiles_inputs]
                    qmols = [m for m in qmols if m is not None]
                    parts = [Chem.MolFromSmiles(s) for s in smiles_ligands.split(".")]
                    parts = [p for p in parts if p is not None]
                    return all(any(p.HasSubstructMatch(q) for p in parts) for q in qmols)
                valid_smi = set(df[df["SMILES_Ligands"].apply(_si_has_sub)]["SMILES_Ligands"])
            si_df = si_df[si_df["SMILES_Ligands"].isin(valid_smi)]

    # Apply filters
    if si_metal != "All metals":
        si_df = si_df[si_df['Metal'] == si_metal]
    if si_normal != "All normal lines":
        si_df = si_df[si_df['Normal_line'] == si_normal]
    if si_cancer != "All cancer lines":
        si_df = si_df[si_df['Cancer_line'] == si_cancer]
    if si_si_min != "Any":
        si_threshold = float(si_si_min.replace("≥", ""))
        si_df = si_df[si_df['SI'] >= si_threshold]

    si_df = si_df.sort_values('SI', ascending=(si_sorting == "Lowest SI first"))

    # ── Summary ───────────────────────────────────────────────────────────────
    n_si_cpx = si_df['SMILES_Ligands'].nunique()
    n_si_pairs = len(si_df)
    col_si_stats, col_si_csv = st.columns([5, 1])
    with col_si_stats:
        _si_cur_page = st.session_state.get("si_page", 0)
        _si_total = n_si_cpx
        _si_from = _si_cur_page * 20 + 1
        _si_to = min((_si_cur_page + 1) * 20, _si_total)
        _si_showing = (
            f"<div style='margin-left:auto;text-align:right;'>"
            f"<div style='font-family:DM Mono,monospace;font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;color:#4a5568;margin-bottom:3px;'>Showing</div>"
            f"<div style='font-family:DM Mono,monospace;font-size:0.85rem;font-weight:500;color:#8892a4;'>{_si_from}–{_si_to} of {_si_total}</div>"
            f"</div>"
        ) if _si_total > 20 else ""
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:24px;padding:16px 20px;
                    background:#0f1420;border:1px solid rgba(255,255,255,0.07);
                    border-radius:12px;margin-bottom:20px;">
            <div>
                <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.1em;
                            text-transform:uppercase;color:#4a5568;margin-bottom:3px;">Complexes</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;
                            color:#e8ecf4;">{n_si_cpx}</div>
            </div>
            <div style="width:1px;height:36px;background:rgba(255,255,255,0.07);"></div>
            <div>
                <div style="font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.1em;
                            text-transform:uppercase;color:#4a5568;margin-bottom:3px;">SI pairs</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;
                            color:#3de8a0;">{n_si_pairs}</div>
            </div>
            {_si_showing}
        </div>
        """, unsafe_allow_html=True)
    with col_si_csv:
        csv_si = si_df.drop(columns=['IC50_normal_val', 'IC50_cancer_val'], errors='ignore').to_csv(index=False).encode('utf-8')
        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
        st.download_button("Download CSV", data=csv_si, file_name="selectivity_index.csv", mime="text/csv", use_container_width=True)

    # ── Render cards grouped by compound ─────────────────────────────────────
    PAGE_SIZE_SI = 20
    si_sort_by_year = si_sorting in ("Newest first", "Oldest first")
    si_ascending = si_sorting == "Oldest first"
    if si_sort_by_year:
        all_keys_si = si_df.groupby(['SMILES_Ligands', 'Metal'])['Year'].max().sort_values(
            ascending=si_ascending
        ).index.tolist()
    else:
        all_keys_si = si_df.groupby(['SMILES_Ligands', 'Metal'])['SI'].max().sort_values(
            ascending=(si_sorting == "Lowest SI first")
        ).index.tolist()
    total_pages_si = max(1, (len(all_keys_si) + PAGE_SIZE_SI - 1) // PAGE_SIZE_SI)
    if 'si_page' not in st.session_state:
        st.session_state['si_page'] = 0
    if st.session_state['si_page'] >= total_pages_si:
        st.session_state['si_page'] = 0
    page_si = st.session_state['si_page']
    keys_page_si = all_keys_si[page_si * PAGE_SIZE_SI : (page_si + 1) * PAGE_SIZE_SI]

    grouped_si = si_df.groupby(['SMILES_Ligands', 'Metal'], sort=False)

    for (smi, metal) in keys_page_si:
        if (smi, metal) not in grouped_si.groups:
            continue
        group = grouped_si.get_group((smi, metal))
        metal_color = _METAL_CLR.get(str(metal), "#8892a4")
        max_si = group['SI'].max()
        n_pairs = len(group)

        abbr_series = group['Abbreviation_in_the_article'].dropna()
        abbr_val = str(abbr_series.iloc[0]) if len(abbr_series) > 0 else ""
        ox_series = group['Oxidation_state'].dropna()
        ox_str = ""
        if len(ox_series) > 0:
            try:
                ox_str = f"({_ROMAN.get(int(ox_series.iloc[0]), str(int(ox_series.iloc[0])))})"
            except (ValueError, TypeError):
                ox_str = ""

        si_color = "#3de8a0" if max_si >= 10 else "#fbbf24" if max_si >= 2 else "#e8ecf4"
        abbr_html = (
            f"<span style='font-family:{_MONO};font-size:0.6rem;color:#4a5568;margin-right:2px;'>Name from article:</span>"
            f"<span style='font-family:{_SYNE};font-size:1rem;font-weight:700;color:#e8ecf4;'>{abbr_val}</span>"
            if abbr_val else ""
        )

        col_img, col_data = st.columns([1, 4])
        with col_img:
            st.image(draw_molecule(smi), use_container_width=True)
        with col_data:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:4px;flex-wrap:wrap;'>"
                f"<span style='font-family:{_MONO};font-size:0.75rem;font-weight:600;"
                f"padding:3px 8px;border-radius:4px;background:{metal_color}18;color:{metal_color};'>{metal}{ox_str}</span>"
                f"<span style='font-family:{_MONO};font-size:0.85rem;font-weight:600;color:{si_color};'>SI {max_si:.1f}</span>"
                f"<span style='font-family:{_MONO};font-size:0.65rem;color:#4a5568;'>max · {n_pairs} pair{'s' if n_pairs > 1 else ''}</span>"
                f"</div>"
                + (f"<div style='margin-bottom:3px;'>{abbr_html}</div>" if abbr_val else "")
                + f"<div style='font-family:{_MONO};font-size:0.65rem;color:#4a5568;word-break:break-all;margin-bottom:4px;'>{smi}</div>",
                unsafe_allow_html=True
            )

            # Table
            top3 = group.head(3)
            rest = group.iloc[3:]
            table_html = (
                "<table style='width:100%;border-collapse:collapse;margin-top:6px;'>"
                "<thead><tr>"
                "<th style='font-family:DM Mono,monospace;font-size:0.6rem;letter-spacing:0.08em;text-transform:uppercase;color:#4a5568;text-align:left;padding:4px 8px 4px 0;border-bottom:1px solid rgba(255,255,255,0.06);'>Normal line</th>"
                "<th style='font-family:DM Mono,monospace;font-size:0.6rem;letter-spacing:0.08em;text-transform:uppercase;color:#4a5568;text-align:left;padding:4px 8px 4px 0;border-bottom:1px solid rgba(255,255,255,0.06);'>IC&#8325;&#8320;(normal)</th>"
                "<th style='font-family:DM Mono,monospace;font-size:0.6rem;letter-spacing:0.08em;text-transform:uppercase;color:#4a5568;text-align:left;padding:4px 8px 4px 0;border-bottom:1px solid rgba(255,255,255,0.06);'>Cancer line</th>"
                "<th style='font-family:DM Mono,monospace;font-size:0.6rem;letter-spacing:0.08em;text-transform:uppercase;color:#4a5568;text-align:left;padding:4px 8px 4px 0;border-bottom:1px solid rgba(255,255,255,0.06);'>IC&#8325;&#8320;(cancer)</th>"
                "<th style='font-family:DM Mono,monospace;font-size:0.6rem;letter-spacing:0.08em;text-transform:uppercase;color:#4a5568;text-align:left;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.06);'>SI</th>"
                "<th style='font-family:DM Mono,monospace;font-size:0.6rem;letter-spacing:0.08em;text-transform:uppercase;color:#4a5568;text-align:left;padding:4px 0 4px 8px;border-bottom:1px solid rgba(255,255,255,0.06);'>DOI</th>"
                "<th style='font-family:DM Mono,monospace;font-size:0.6rem;letter-spacing:0.08em;text-transform:uppercase;color:#4a5568;text-align:left;padding:4px 0 4px 8px;border-bottom:1px solid rgba(255,255,255,0.06);'>Year</th>"
                "</tr></thead><tbody>"
            )

            def render_si_rows(rows):
                html = ""
                for _, row in rows.iterrows():
                    si_val = row['SI']
                    si_c = "#3de8a0" if si_val >= 10 else "#fbbf24" if si_val >= 2 else "#8892a4"
                    td = "border-bottom:1px solid rgba(255,255,255,0.04);"
                    doi_short = str(row['DOI'])[:28] + ("…" if len(str(row['DOI'])) > 28 else "")
                    html += (
                        f"<tr>"
                        f"<td style='font-family:DM Mono,monospace;font-size:0.75rem;color:#8892a4;padding:5px 8px 5px 0;{td}'>"
                        + row['Normal_line']
                        + (f"<span style='font-family:DM Mono,monospace;font-size:0.6rem;color:#4a5568;margin-left:6px;'>{_CELL_LINE_ORGAN[row['Normal_line']]}</span>" if row['Normal_line'] in _CELL_LINE_ORGAN else "")
                        + "</td>"
                        f"<td style='font-family:DM Mono,monospace;font-size:0.75rem;color:#8892a4;padding:5px 8px 5px 0;{td}'>{row['IC50_normal']}</td>"
                        f"<td style='font-family:DM Mono,monospace;font-size:0.75rem;color:#8892a4;padding:5px 8px 5px 0;{td}'>"
                        + row['Cancer_line']
                        + (f"<span style='font-family:DM Mono,monospace;font-size:0.6rem;color:#4a5568;margin-left:6px;'>{_CELL_LINE_ORGAN[row['Cancer_line']]}</span>" if row['Cancer_line'] in _CELL_LINE_ORGAN else "")
                        + "</td>"
                        f"<td style='font-family:DM Mono,monospace;font-size:0.75rem;color:#8892a4;padding:5px 8px 5px 0;{td}'>{row['IC50_cancer']}</td>"
                        f"<td style='font-family:DM Mono,monospace;font-size:0.8rem;font-weight:600;color:{si_c};padding:5px 8px 5px 0;{td}'>{si_val:.1f}</td>"
                        f"<td style='padding:5px 0 5px 8px;{td}'><a href='https://doi.org/{row["DOI"]}' target='_blank' style='font-family:DM Mono,monospace;font-size:0.68rem;color:#5b8fff;text-decoration:none;'>{doi_short}</a></td>"
                        f"<td style='font-family:DM Mono,monospace;font-size:0.75rem;color:#4a5568;padding:5px 0 5px 8px;{td}'>{int(row['Year']) if pd.notna(row['Year']) else '—'}</td>"
                        f"</tr>"
                    )
                return html

            st.markdown(table_html + render_si_rows(top3) + "</tbody></table>", unsafe_allow_html=True)
            if len(rest) > 0:
                with st.expander(f"Show {len(rest)} more"):
                    st.markdown("<table style='width:100%;border-collapse:collapse;'><tbody>" + render_si_rows(rest) + "</tbody></table>", unsafe_allow_html=True)

        st.markdown("<div style='height:1px;background:rgba(255,255,255,0.06);margin:8px 0;'></div>", unsafe_allow_html=True)

    # ── Pagination ────────────────────────────────────────────────────────────
    if total_pages_si > 1:
        col_prev, col_info, col_next = st.columns([1, 2, 1])
        if col_prev.button("← Previous", key="si_prev", disabled=(page_si == 0), use_container_width=True):
            st.session_state['si_page'] -= 1
            st.rerun()
        col_info.markdown(
            f'<div style="text-align:center;font-family:DM Mono,monospace;font-size:0.75rem;color:#4a5568;padding-top:8px;">'
            f'Page {page_si + 1} of {total_pages_si}</div>',
            unsafe_allow_html=True
        )
        if col_next.button("Next →", key="si_next", disabled=(page_si >= total_pages_si - 1), use_container_width=True):
            st.session_state['si_page'] += 1
            st.rerun()

elif page == "📊  Statistics":

    st.markdown("""
    <div style="margin-bottom:24px;">
        <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;
                    color:#e8ecf4;margin-bottom:8px;">Database Statistics</div>
        <div style="font-family:'DM Sans',sans-serif;font-size:0.88rem;color:#8892a4;line-height:1.6;">
            Overview of IC&#8325;&#8320; distributions, cell line coverage, and publication trends
            across Ru, Ir, Os, Rh, and Re complexes.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── shared Plotly dark theme ──────────────────────────────────────────────
    _LAYOUT = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f1420",
        font=dict(family="DM Sans, sans-serif", color="#8892a4", size=12),
        title_font=dict(family="Syne, sans-serif", size=14, color="#e8ecf4"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.07)", tickfont=dict(size=11)),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.07)", tickfont=dict(size=11)),
        margin=dict(l=40, r=20, t=48, b=40),
    )
    _METAL_COLORS = {"Ru": "#3de8a0", "Ir": "#5b8fff", "Os": "#ff7c5b", "Rh": "#c084fc", "Re": "#fbbf24"}
    # ─────────────────────────────────────────────────────────────────────────

    col1fig, col2fig = st.columns([1, 1])

    fig_ic50 = px.histogram(df, x='IC50_Dark_value', nbins=32, title='Distribution of the IC₅₀ values')
    fig_ic50.update_traces(marker_color="#3de8a0", marker_opacity=0.75)
    fig_ic50.update_layout(**_LAYOUT, yaxis_title='Number of entries', xaxis_title='IC₅₀, μM')
    col1fig.plotly_chart(fig_ic50, use_container_width=True)

    fig_year = px.bar(years, x='Year', y='count', text='count', title="Distribution of the years of the source articles")
    fig_year.update_traces(marker_color="#5b8fff", marker_opacity=0.8, textfont_size=10)
    fig_year.update_layout(**_LAYOUT, yaxis_title='Number of articles', xaxis_title='Publication year', xaxis_tickangle=45)
    fig_year.update_xaxes(tickvals=[i for i in range(df['Year'].min(), df['Year'].max()+1)])
    col2fig.plotly_chart(fig_year, use_container_width=True)

    fig_cell = px.bar(cells, x='Cell_line', y='count', text='count', color='Cell_line', title="Number of entries for 20 most popular cell lines")
    fig_cell.update_layout(**_LAYOUT, yaxis_title='Number of entries', xaxis_title='Cell line', showlegend=False)
    fig_cell.update_traces(textfont_size=10)
    st.plotly_chart(fig_cell, use_container_width=True)

    fig_time = px.bar(times, x='Time(h)', y='count', text='count', title="Distribution of complexes exposure time")
    fig_time.update_traces(marker_color="#ff7c5b", marker_opacity=0.8, textfont_size=10)
    fig_time.update_layout(**_LAYOUT, yaxis_title='Number of entries', xaxis_title='Exposure time (h)')
    fig_time.update_xaxes(tickvals=[24, 48, 72, 96, 120, 144])
    col1fig.plotly_chart(fig_time, use_container_width=True)

    fig_class = px.bar(ic50_class, x='IC50_class', y='count', text='count',
                       title="Distribution of IC₅₀ values between two classes (toxic: <10 μM and non-toxic: ≥10 μM)",
                       color='IC50_class',
                       color_discrete_map={'<10μM': '#3de8a0', '≥10 μM': '#5b8fff'})
    fig_class.update_layout(**_LAYOUT, yaxis_title='Number of entries', xaxis_title='IC₅₀ class', showlegend=False)
    fig_class.update_traces(textfont_size=11)
    col2fig.plotly_chart(fig_class, use_container_width=True)

    # ── PI distribution ───────────────────────────────────────────────────────
    pi_data = light_df['PI'].dropna()
    pi_data = pi_data[pi_data < 500]
    fig_pi = px.histogram(pi_data, nbins=40, title="Distribution of Phototherapeutic Index (PI)")
    fig_pi.update_traces(marker_color="#fbbf24", marker_opacity=0.8)
    fig_pi.update_layout(**_LAYOUT, yaxis_title='Number of entries', xaxis_title='PI = IC₅₀(dark) / IC₅₀(light)')
    col1fig.plotly_chart(fig_pi, use_container_width=True)

    # ── SI distribution ───────────────────────────────────────────────────────
    _si_all = df[df['Cell_line'].isin(_NORMAL_LINES)].merge(
        df[~df['Cell_line'].isin(_NORMAL_LINES)][['SMILES_Ligands', 'Metal', 'IC50_Dark_value']],
        on=['SMILES_Ligands', 'Metal']
    )
    _si_all['SI'] = pd.to_numeric(_si_all['IC50_Dark_value_x'], errors='coerce') / pd.to_numeric(_si_all['IC50_Dark_value_y'], errors='coerce')
    si_data = _si_all['SI'].dropna()
    si_data = si_data[(si_data > 0) & (si_data < 200)]
    fig_si = px.histogram(si_data, nbins=40, title="Distribution of Selectivity Index (SI)")
    fig_si.update_traces(marker_color="#c084fc", marker_opacity=0.8)
    fig_si.update_layout(**_LAYOUT, yaxis_title='Number of pairs', xaxis_title='SI = IC₅₀(normal) / IC₅₀(cancer)')
    col2fig.plotly_chart(fig_si, use_container_width=True)


