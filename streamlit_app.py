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

def draw_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.Draw.MolToImage(mol)

def canonize_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

def check_ligands(mol1, mol2, mol3):
    allowed_atoms = ["C", "O", "N", "H", "Cl", "F", "S", "P"]
    def contains_only_allowed_atoms(mol):
        return any(atom.GetSymbol() not in allowed_atoms for atom in mol.GetAtoms())

    canonize_l1 = Chem.MolToSmiles(mol1)
    canonize_l2 = Chem.MolToSmiles(mol2)
    canonize_l3 = Chem.MolToSmiles(mol3)

    if (len(mol1.GetAtoms()) < 6) | (len(mol2.GetAtoms()) < 6) | (len(mol3.GetAtoms()) < 6):
        st.error("Only ligands with more than 5 atoms are available for input.")
        return False
    elif (contains_only_allowed_atoms(mol1) | contains_only_allowed_atoms(mol2) | contains_only_allowed_atoms(mol3)):
        st.error("The model can predict molecules containing atoms: C, O, N, Cl, F, S, P.")
        return False
    elif ('[c-]' not in canonize_l1) | ('[c-]' not in canonize_l2):
        st.error("The complex should contain TWO cyclometalated ligands, i.e. TWO ligands with deprotonated carbon as L1 and L2.")
        return False
    elif canonize_l1 == canonize_l2 == canonize_l3:
        st.error("The complex should contain TWO cyclometalated ligands, i.e. TWO ligands with deprotonated carbon as L1 and L2. Your query contains deprotonated carbon in the L3 section. Please correct it.")
        return False
    else:
        return True

calc = FPCalculator("ecfp")

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

st.set_page_config(page_title='RuCytoToxDB', layout="wide")

df = pd.read_csv('RuCytoToxDB.csv')
df['IC50_Dark_value'] = df['IC50_Dark_value'].apply(scale_ic50)
df['IC50_class'] = df['IC50_Dark_value'].apply(class_ic50)

cells = df['Cell_line'].value_counts().reset_index().loc[:19]
years = df.drop_duplicates(subset=['DOI'])['Year'].value_counts().reset_index()
times = df['Time(h)'].value_counts().reset_index().loc[:5]
ic50_class = df['IC50_class'].value_counts().reset_index()
ic50_class['IC50_class'].replace({0: '≥10 μM', 1: '<10μM'}, inplace=True)
line_list = df['Cell_line'].value_counts().nlargest(50).index.tolist()
time_list = df['Time(h)'].value_counts().nlargest(5).index.tolist()

n_entries = df.shape[0]
n_smiles = df.drop_duplicates(['SMILES_Ligands', 'Counterion']).shape[0]
n_sources = df['DOI'].nunique()
n_cell = df['Cell_line'].nunique()

col1intro, col2intro, col3intro = st.columns([1, 1, 2])
col1intro.markdown(f"""
# RuCytoToxDB App v1.0

The ”RuCytoToxDB App” is an ML-based service integrated with the experimental database to explore literature cytotoxicity data and predict cytotoxicity (IC50) of ruthenium complexes requiring only molecular formula of the ligands as a feature.

Download RuCytoToxDB: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15853577.svg)](https://doi.org/10.5281/zenodo.15853577)
                   
""")

col2intro.markdown(f"""# Overall stats: 
* **{n_entries}** number of entries
* **{n_smiles}** unique ruthenium complexes
* **{n_sources}** literature sources
* **{n_cell}** cell lines""")

col3intro.image('TOC.png')

st.markdown("""### There are currently two operation modes:
* exploration of the database (**“Explore statistics”** window)
* prediction of **IC₅₀** (**“search and predict”** window)""")

tabs = st.tabs(["Explore statistics", "Search and Predict", "Advanced search", "Search by fixed ligand subset"])

with tabs[0]:

    col1fig, col2fig = st.columns([1, 1])
    fig_ic50 = px.histogram(df, x='IC50_Dark_value', nbins=32, title='Distribution of the IC₅₀ values')
    fig_ic50.update_layout(yaxis_title='Number of entries')
    fig_ic50.update_layout(xaxis_title='IC₅₀,μM')
    col1fig.plotly_chart(fig_ic50)

    fig_year = px.bar(years, x='Year', y='count', text='count', title="Distribution of the years of the source articles")
    fig_year.update_layout(yaxis_title='Number of articles')
    fig_year.update_layout(xaxis_title='Publication year')
    fig_year.update_layout(xaxis_tickangle=45)
    fig_year.update_xaxes(
        tickvals=[i for i in range(df['Year'].min(),df['Year'].max()+1)])
    col2fig.plotly_chart(fig_year, use_container_width=True)

    fig_cell = px.bar(cells, x='Cell_line', y='count', text='count', color='Cell_line', title="Number of entries for 20 most popular cell lines")
    fig_cell.update_layout(yaxis_title='Number of entries')
    fig_cell.update_layout(xaxis_title='Cell line')
    fig_cell.update_layout(showlegend=False)
    st.plotly_chart(fig_cell, use_container_width=True)

    fig_time = px.bar(times, x='Time(h)', y='count', text='count', title="Distribution of complexes exposure time")
    fig_time.update_layout(yaxis_title='Number of entries')
    fig_time.update_layout(xaxis_title='Exposure time (h)')
    fig_time.update_xaxes(
            tickvals=[24, 48, 72, 96, 120, 144])
    col1fig.plotly_chart(fig_time, use_container_width=True)

    fig_class = px.bar(ic50_class, x='IC50_class', y='count', text='count', title="Distribution of IC50 values between two classes (toxic: <10μM and non-toxic: ≥10μM)")
    fig_class.update_layout(yaxis_title='Number of entries')
    fig_class.update_layout(xaxis_title='IC₅₀,μM')
    col2fig.plotly_chart(fig_class, use_container_width=True)


with tabs[1]:

    st.markdown("""Please enter SMILES of the ligands (or draw the structural formula in the corresponding window) and press “**Search in the database and predict properties**” button to perform the prediction. If the complex exists in the database, experimental data will be displayed. If the complex does not exist in the database, the predicted **IC₅₀** will appear.""")
    st.markdown("""### To get SMILES of your ligand, draw custom molecule and click **"Apply"** button or copy SMILES from popular ligands:""")
    exp = st.expander("Popular ligands")
    exp1col, exp2col, exp3col = exp.columns(3)
    with exp:
        exp1col.markdown('### p-cymene')
        exp1col.image(draw_molecule('Cc1ccc(C(C)C)cc1'), caption='Cc1ccc(C(C)C)cc1')
        exp2col.markdown('### bpy')
        exp2col.image(draw_molecule('c1ccc(-c2ccccn2)nc1'), caption='c1ccc(-c2ccccn2)nc1')
        exp3col.markdown('### phen')
        exp3col.image(draw_molecule('c1cnc2c(c1)ccc1cccnc12'), caption='c1cnc2c(c1)ccc1cccnc12')
        exp2col.markdown('### bphen')
        exp2col.image(draw_molecule('c1ccc(-c2ccnc3c2ccc2c(-c4ccccc4)ccnc23)cc1'), caption='c1ccc(-c2ccnc3c2ccc2c(-c4ccccc4)ccnc23)cc1')
        exp3col.markdown('### PPh3')
        exp3col.image(draw_molecule('P(c1ccccc1)(c1ccccc1)c1ccccc1'), caption='P(c1ccccc1)(c1ccccc1)c1ccccc1')
        exp1col.markdown('### [Cl-]')
        exp1col.image(draw_molecule('[Cl-]'), caption='[Cl-]')
        exp2col.markdown('### PTA')
        exp2col.image(draw_molecule('C1N2CN3CN1CP(C2)C3'), caption='C1N2CN3CN1CP(C2)C3')
        exp3col.markdown('### dppb')
        exp3col.image(draw_molecule('c1ccc(P(CCCCP(c2ccccc2)c2ccccc2)c2ccccc2)cc1'), caption='c1ccc(P(CCCCP(c2ccccc2)c2ccccc2)c2ccccc2)cc1')

    smile_code = st_ketcher(height=400, key="ketcher_1")
    st.markdown(f"""### Your SMILES:""")
    st.markdown(f"``{smile_code}``")

    st.markdown(f"""### 1) Select number of ligands:""")

    num_ligands = st.selectbox(
        "Select number of ligands in your complex:",
        options=[2, 3, 4, 5, 6],
        index=1
    )

    st.markdown("""### 2) Paste this SMILES into the corresponding boxes below:""")

    if num_ligands:
        smiles_inputs = []
        for num in range(int(num_ligands)):
            smiles = st.text_input(f"SMILES L{num+1}", key=f"search_{num}")
            smiles_inputs.append(smiles)

        smiles_complex = '.'.join(smiles_inputs)

        if st.button("Search in the database and predict IC50"):
            mol = Chem.MolFromSmiles(smiles_complex)
            if (mol is not None):
                canonize_smiles = Chem.MolToSmiles(mol)
                st.image(draw_molecule(canonize_smiles), caption=canonize_smiles)
                search_df = df[(df['SMILES_Ligands'] == canonize_smiles)]
                if search_df.shape[0] == 0:
                    st.markdown('Nothing found')
        #                     L1_res_ecfp = calc(mol1)
        #                     L2_res_ecfp = calc(mol2)
        #                     L3_res_ecfp = calc(mol3)
        #                     L_res = L1_res_ecfp + L2_res_ecfp + L3_res_ecfp
        #                     L_res = L_res.reshape(1, -1)
        #                     pred_lum = str(int(round(model_lum.predict(L_res)[0], 0)))
        #                     pred_plqy = round(model_plqy.predict(L_res)[0]*100, 1)
        #                     str_plqy = str(pred_plqy)
        #                     predcol1, predcol2 = st.columns(2)
        #                     predcol1.markdown(f'## Predicted luminescence wavelength:')
        #                     predcol2.markdown(f'## Predicted PLQY:')
        #                     predcol1.markdown(f'### {pred_lum} nm in dichloromethane')
        #                     predcol2.markdown(f'### {str_plqy}% in dichloromethane')
        #                     if pred_plqy <= 10:
        #                         predcol2.image('low_qy.png', width=200)
        #                         predcol2.markdown(f'### Low PLQY (0-10%)')
        #                     elif 50 >= pred_plqy > 10:
        #                         predcol2.image('moderate_qy.png', width=200)
        #                         predcol2.markdown(f'### Moderate PLQY (10-50%)')
        #                     else:
        #                         predcol2.image('high_qy.png', width=200)
        #                         predcol2.markdown(f'### High PLQY (50-100%)')
        #                     df['res_dist'] = df['L1_ecfp'].apply(lambda ecfp1: hamming_distance(L1_res_ecfp, ecfp1)) + df['L1_ecfp'].apply(lambda ecfp2: hamming_distance(L2_res_ecfp, ecfp2)) + df['L3_ecfp'].apply(lambda ecfp3: hamming_distance(L3_res_ecfp, ecfp3))
        #                     search_df = df[df['res_dist'] == df['res_dist'].min()]

        #                     st.markdown(f'### Below are shown the most similar complexes found in the IrLumDB:')
        #                     col1search, col2search, col3search, col4search, col5search, col6search, col7search, col8search = st.columns([1, 1, 1, 1, 1, 2, 2, 2])
        #                     col1search.markdown(f'**λlum,nm**')
        #                     col2search.markdown(f'**PLQY**')
        #                     col3search.markdown(f'**Solvent**')
        #                     col4search.markdown(f'**Abbreviation**')
        #                     col5search.markdown(f'**Source**')
        #                     col6search.markdown(f'**L1**')
        #                     col7search.markdown(f'**L2**')
        #                     col8search.markdown(f'**L3**')
        #                     for lam, qy, solvent, doi, abbr, L1_df, L2_df, L3_df in zip(search_df['Max_wavelength(nm)'], search_df['PLQY'], search_df['Solvent'], search_df['DOI'], search_df['Abbreviation_in_the_article'], search_df['L1'], search_df['L2'], search_df['L3']):
        #                         col1result, col2result, col3result, col4result, col5result, col6result, col7result, col8result = st.columns([1, 1, 1, 1, 1, 2, 2, 2])
        #                         col1result.markdown(f'**{lam} nm**')
        #                         col2result.markdown(f'**{qy}**')
        #                         col3result.markdown(f'**{solvent}**')
        #                         col4result.markdown(f'**{abbr}**')
        #                         col5result.markdown(f'**https://doi.org/{doi}**')
        #                         col6result.image(draw_molecule(L1_df), caption=L1_df)
        #                         col7result.image(draw_molecule(L2_df), caption=L2_df)
        #                         col8result.image(draw_molecule(L3_df), caption=L3_df)
                else:
                    st.markdown(f'### Found this complex in RuCytoToxDB:')
                    col1search, col2search, col3search, col4search, col5search = st.columns([1, 1, 1, 3, 4])
                    col1search.markdown(f'**IC₅₀,μM**')
                    col2search.markdown(f'**Сell line**')
                    col3search.markdown(f'**Time(h)**')
                    col4search.markdown(f'**Abbreviation in the source:**')
                    col5search.markdown(f'**Source**')

                    for ic50, cell_line, time, doi, abbr in zip(search_df['IC50_Dark(M*10^-6)'], search_df['Cell_line'], search_df['Time(h)'], search_df['DOI'], search_df['Abbreviation_in_the_article']):
                        col1result, col2result, col3result, col4result, col5result = st.columns([1, 1, 1, 3, 4])
                        col1result.markdown(f'**{ic50}**')
                        col2result.markdown(f'**{cell_line}**')
                        col3result.markdown(f'**{time}**')
                        col4result.markdown(f'**{abbr}**')
                        col5result.markdown(f'**https://doi.org/{doi}**')
                    st.dataframe(search_df)
        #         else:
        #             st.error("Incorrect SMILES entered")
        #     else:
        #         st.error("Please enter all three ligands")

with tabs[2]:
    col1select, col2select, col3select = st.columns([1, 1, 1])
    selected_line = col1select.selectbox(label='Choose line', options=line_list, index=None, placeholder='A549')
    selected_time = col2select.selectbox(label='Choose exposure time (h)', options=['All time ranges'] + time_list, index=0)
    select_sorting = col3select.selectbox(label='Choose the sorting type', options=['Most cytooxic above', 'Least cytooxic above'], index=0)

    if selected_line:
        if selected_time == 'All time ranges':
            search_df = df[(df['Cell_line'] == selected_line)]
        else:
            search_df = df[(df['Cell_line'] == selected_line) & (df['Time(h)'] == selected_time)]
        if select_sorting == 'Least cytooxic above':
            search_df.sort_values(by='IC50_Dark_value', ascending=False, inplace=True)
        else:
            search_df.sort_values(by='IC50_Dark_value', ascending=True, inplace=True)
        num_compexes = search_df.drop_duplicates(subset=['SMILES_Ligands', 'Counterion']).shape[0]
        st.markdown(f'# Found {num_compexes} complexes')
        col1search, col2search, col3search, col4search, col5search, col6search, col7search = st.columns([1, 1, 1, 1, 1, 1, 1])
        col1search.markdown(f'**Ligands of Ru complexes**')
        col2search.markdown(f'**IC₅₀,μM**')
        col3search.markdown(f'**Сell line**')
        col4search.markdown(f'**Time(h)**')
        col5search.markdown(f'**Abbreviation in the source:**')
        col6search.markdown(f'**Source**')
        col7search.markdown(f'**Cisplatin**')

        for smi, ic50, cell_line, time, doi, abbr, cis in zip(search_df['SMILES_Ligands'], search_df['IC50_Dark(M*10^-6)'], search_df['Cell_line'], search_df['Time(h)'], search_df['DOI'], search_df['Abbreviation_in_the_article'], search_df['IC50_Cisplatin(M*10^-6)']):
            col1result, col2result, col3result, col4result, col5result, col6result, col7result = st.columns([1, 1, 1, 1, 1, 1, 1])
            col1result.image(draw_molecule(smi), caption=smi, use_container_width=True)
            col2result.markdown(f'**{ic50}**')
            col3result.markdown(f'**{cell_line}**')
            col4result.markdown(f'**{time}**')
            col5result.markdown(f'**{abbr}**')
            col6result.markdown(f'**https://doi.org/{doi}**')
            if not pd.isna(cis):
                col7result.markdown(f'**{cis}**')
            else:
                col7result.markdown(f'**No data**')

with tabs[3]:
    st.markdown("""### To get SMILES of your ligand, draw custom molecule and click **"Apply"** button or copy SMILES from popular ligands:""")
    exp = st.expander("Popular ligands")
    exp1col, exp2col, exp3col = exp.columns(3)
    with exp:
        exp1col.markdown('### p-cymene')
        exp1col.image(draw_molecule('Cc1ccc(C(C)C)cc1'), caption='Cc1ccc(C(C)C)cc1')
        exp2col.markdown('### bpy')
        exp2col.image(draw_molecule('c1ccc(-c2ccccn2)nc1'), caption='c1ccc(-c2ccccn2)nc1')
        exp3col.markdown('### phen')
        exp3col.image(draw_molecule('c1cnc2c(c1)ccc1cccnc12'), caption='c1cnc2c(c1)ccc1cccnc12')
        exp2col.markdown('### bphen')
        exp2col.image(draw_molecule('c1ccc(-c2ccnc3c2ccc2c(-c4ccccc4)ccnc23)cc1'), caption='c1ccc(-c2ccnc3c2ccc2c(-c4ccccc4)ccnc23)cc1')
        exp3col.markdown('### PPh3')
        exp3col.image(draw_molecule('P(c1ccccc1)(c1ccccc1)c1ccccc1'), caption='P(c1ccccc1)(c1ccccc1)c1ccccc1')
        exp1col.markdown('### [Cl-]')
        exp1col.image(draw_molecule('[Cl-]'), caption='[Cl-]')
        exp2col.markdown('### PTA')
        exp2col.image(draw_molecule('C1N2CN3CN1CP(C2)C3'), caption='C1N2CN3CN1CP(C2)C3')
        exp3col.markdown('### dppb')
        exp3col.image(draw_molecule('c1ccc(P(CCCCP(c2ccccc2)c2ccccc2)c2ccccc2)cc1'), caption='c1ccc(P(CCCCP(c2ccccc2)c2ccccc2)c2ccccc2)cc1')

    smile_code = st_ketcher(height=400, key="ketcher_2")
    st.markdown(f"""### Your SMILES:""")
    st.markdown(f"``{smile_code}``")

    st.markdown(f"""### 1) Select number of ligands:""")

    num_ligands = st.selectbox(
        "Select number of ligands in your substructure search query:",
        options=[1, 2, 3],
        index=0
    )

    st.markdown("""### 2) Paste this SMILES into the corresponding boxes below:""")

    if num_ligands:
        smiles_inputs = []
        for num in range(int(num_ligands)):
            smiles = st.text_input(f"SMILES L{num+1}", key=f"substructure_{num}")
            smiles_inputs.append(smiles)

        smiles_complex = '.'.join(smiles_inputs)

        col1select, col2select, col3select, col4select = st.columns([1, 1, 1, 1])
        selected_line = col1select.selectbox(label='Choose line', options=['All cell lines'] + line_list, index=0, placeholder='A549', key="substructure_line")
        selected_time = col2select.selectbox(label='Choose exposure time (h)', options=['All time ranges'] + time_list, index=0, key="substructure_time")
        selected_sorting = col3select.selectbox(label='Choose the sorting type', options=['Most cytooxic above', 'Least cytooxic above'], index=0, key="substructure_sort")
        selected_scaffold = col4select.selectbox(label='Choose the search regime', options=['Full molecule match', 'Scaffold-based search'], index=0, key="substructure_scaffold")

        if st.button("Search in the database"):
            mol = Chem.MolFromSmiles(smiles_complex)
            if (mol is not None):
                smiles_inputs = [canonize_smiles(smi) for smi in smiles_inputs]
                if selected_scaffold == 'Full molecule match': 
                    search_df = df[(df['SMILES_Ligands'].apply(lambda x: all([smi in x.split('.') for smi in smiles_inputs])))].sort_values(by='SMILES_Ligands')
                else:
                    smiles_inputs = [get_murcko_scaffold(smi) if get_murcko_scaffold(smi) != '' else smi for smi in smiles_inputs]
                    st.markdown("""### Your scaffold:""")
                    st.image(draw_molecule('.'.join(smiles_inputs)), caption='.'.join(smiles_inputs))
                    search_df = df[(df['Scaffold'].apply(lambda x: all([smi in x.split('.') for smi in smiles_inputs])))].sort_values(by='SMILES_Ligands')
                if selected_line != 'All cell lines':
                    search_df = search_df[(search_df['Cell_line'] == selected_line)]
                if selected_time != 'All time ranges':
                    search_df = search_df[(search_df['Time(h)'] == selected_time)]
                if selected_sorting == 'Least cytooxic above':
                    search_df.sort_values(by='IC50_Dark_value', ascending=False, inplace=True)
                else:
                    search_df.sort_values(by='IC50_Dark_value', ascending=True, inplace=True)

                if search_df.shape[0] == 0:
                    st.markdown('Nothing found')
                else:
                    csv = search_df.to_csv(index=False).encode('utf-8')

                    st.download_button(
                        label="📥 Download this data in CSV",
                        data=csv,
                        file_name="ic50.csv",
                        mime="text/csv"
                    )

                    num_compexes = search_df.drop_duplicates(subset=['SMILES_Ligands', 'Counterion']).shape[0]
                    num_ic50 = search_df.shape[0]
                    st.markdown(f'# Found {num_compexes} complexes and {num_ic50} IC₅₀ values')
                    col1search, col2search, col3search, col4search, col5search, col6search, col7search = st.columns([1, 1, 1, 1, 1, 1, 1])
                    col1search.markdown(f'**Ligands of Ru complexes**')
                    col2search.markdown(f'**IC₅₀,μM**')
                    col3search.markdown(f'**Сell line**')
                    col4search.markdown(f'**Time(h)**')
                    col5search.markdown(f'**Abbreviation in the source:**')
                    col6search.markdown(f'**Source**')
                    col7search.markdown(f'**Cisplatin**')

                    for smi, ic50, cell_line, time, doi, abbr, cis in zip(search_df['SMILES_Ligands'], search_df['IC50_Dark(M*10^-6)'], search_df['Cell_line'], search_df['Time(h)'], search_df['DOI'], search_df['Abbreviation_in_the_article'], search_df['IC50_Cisplatin(M*10^-6)']):
                        col1result, col2result, col3result, col4result, col5result, col6result, col7result = st.columns([1, 1, 1, 1, 1, 1, 1])
                        col1result.image(draw_molecule(smi), caption=smi, use_container_width=True)
                        col2result.markdown(f'**{ic50}**')
                        col3result.markdown(f'**{cell_line}**')
                        col4result.markdown(f'**{time}**')
                        col5result.markdown(f'**{abbr}**')
                        col6result.markdown(f'**https://doi.org/{doi}**')
                        if not pd.isna(cis):
                            col7result.markdown(f'**{cis}**')
                        else:
                            col7result.markdown(f'**No data**')