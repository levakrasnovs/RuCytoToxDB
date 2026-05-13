# OpenBioMetalDB

**ML-assisted cytotoxicity database for transition metal complexes**

[![JMedChem](https://img.shields.io/badge/J.%20Med.%20Chem.-2026-blue)](https://doi.org/10.1021/acs.jmedchem.5c02755)
[![Streamlit](https://img.shields.io/badge/Live-biometaldb.streamlit.app-3de8a0)](https://biometaldb.streamlit.app/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-OpenBioMetalDB-0077B5)](https://www.linkedin.com/company/openbiometaldb)

OpenBioMetalDB is an open-access database of IC₅₀ cytotoxicity values for transition metal complexes, curated from peer-reviewed literature. The database covers Ru, Ir, Os, Rh, Re, Au and other metals, and is designed to support medicinal chemistry research, structure–activity relationship (SAR) analysis, and machine learning model development.

---

## Contents

| Metric | Value |
|---|---|
| IC₅₀ values | 35000+ |
| Unique complexes | 9000+ |
| Cell lines | 900+ |
| Literature sources | 2400+ |
| Metals | Ru, Ir, Os, Rh, Re, Au |

---

## Features

- **Structural search** — full molecule match, scaffold-based, substructure, and similarity search (Tanimoto/RDKit fingerprints) across ligands
- **Phototoxicity** — dedicated page for photocytotoxicity data with Phototherapeutic Index (PI = IC₅₀(dark) / IC₅₀(light))
- **Selectivity Index** — SI = IC₅₀(normal cell line) / IC₅₀(cancer cell line), with 39 manually annotated normal cell lines and ~15k pairs
- **Tissue filter** — 30 tissues annotated for 364 cell lines, with cascading cell line filter
- **Filters** — metal, tissue, cell line, exposure time, IC₅₀ range, sorting by IC₅₀ or publication year
- **Literature search** — look up all data from a specific DOI or author
- **Statistics** — IC₅₀ distribution, metals, top cell lines, PI and SI distributions
- **CSV export** — download any filtered result set

---

## Data

The main dataset is stored in `MetalCytoToxDB.csv`. Key columns:

| Column | Description |
|---|---|
| `SMILES_Ligands` | Ligand SMILES (`.`-separated for multiple ligands) |
| `Metal` | Metal symbol (Ru, Ir, Os, Rh, Re, Au) |
| `Oxidation_state` | Metal oxidation state |
| `IC50_Dark(M*10^-6)` | IC₅₀ in μM |
| `IC50_Light(M*10^-6)` | IC₅₀ under light irradiation (μM), if available |
| `Cell_line` | Cell line name |
| `Time(h)` | Exposure time (hours) |
| `IC50_Cisplatin(M*10^-6)` | Cisplatin IC₅₀ in the same assay, if reported |
| `DOI` | Source article DOI |
| `Year` | Publication year |
| `Abbreviation_in_the_article` | Compound name/label as used in the source article |

---


## Citation

If you use OpenBioMetalDB in your research, please cite:

> Krasnov L., Malikov D., Kiseleva M. et al. *Machine Learning Approach to Anticancer Activity Prediction of Transition-Metal Complexes Based on a Large-Scale Experimental Database.* J. Med. Chem. 2026. https://doi.org/10.1021/acs.jmedchem.5c02755

---

## Changelog

**13 May 2026**
- 8442 new IC₅₀ values from 1998–2026 literature for Au(I) and Au(III) complexes, covering 2,220 unique complexes

**24 Apr 2026**
- 666 new IC₅₀ values from recent 2025–2026 literature for Ir(III) complexes

**13 Apr 2026**
- Published in Journal of Medicinal Chemistry

---

## License

Data is freely available for academic and commercial use. Please cite the paper if you use the database.
