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
    
def show_search_results(search_df):

    csv = search_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download this data in CSV",
        data=csv,
        file_name="ic50.csv",
        mime="text/csv"
    )

    num_compexes = search_df.drop_duplicates(subset=['SMILES_Ligands', 'Metal']).shape[0]
    num_ic50 = search_df.drop_duplicates(subset=['SMILES_Ligands', 'Counterion', 'IC50_Dark(M*10^-6)', 'Cell_line', 'Time(h)', 'DOI', 'Metal']).shape[0]
    num_sources = search_df['DOI'].nunique()
    st.markdown(f'# Found {num_compexes} complexes and {num_ic50} IC₅₀ values from {num_sources} sources')
    search_df = search_df[:100]
    col1search, col2search, col3search, col4search, col5search, col6search, col7search, col8search = st.columns([1, 1, 1, 1, 1, 1, 1, 1])
    col1search.markdown(f'**Ligands of complexes**')
    col2search.markdown(f'**Metal**')
    col3search.markdown(f'**IC₅₀,μM**')
    col4search.markdown(f'**Сell line**')
    col5search.markdown(f'**Time(h)**')
    col6search.markdown(f'**Abbreviation in the source:**')
    col7search.markdown(f'**Source**')
    col8search.markdown(f'**Cisplatin**')

    for smi, metal, ic50, cell_line, time, doi, abbr, cis in zip(search_df['SMILES_Ligands'], search_df['Metal'], search_df['IC50_Dark(M*10^-6)'], search_df['Cell_line'], search_df['Time(h)'], search_df['DOI'], search_df['Abbreviation_in_the_article'], search_df['IC50_Cisplatin(M*10^-6)']):
        col1result, col2result, col3result, col4result, col5result, col6result, col7result, col8result = st.columns([1, 1, 1, 1, 1, 1, 1, 1])
        col1result.image(draw_molecule(smi), caption=smi, use_container_width=True)
        col2result.markdown(f'**{metal}**')
        col3result.markdown(f'**{ic50}**')
        col4result.markdown(f'**{cell_line}**')
        col5result.markdown(f'**{time}**')
        col6result.markdown(f'**{abbr}**')
        col7result.markdown(f'**https://doi.org/{doi}**')
        if not pd.isna(cis):
            col8result.markdown(f'**{cis}**')
        else:
            col8result.markdown(f'**No data**')


calc = FPCalculator("ecfp")

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

st.set_page_config(page_title='MetalCytoToxDB', layout="wide")

df = pd.read_csv('MetalCytoToxDB.csv')
authors = pd.read_excel('Authors.xlsx')
df['IC50_Dark_value'] = df['IC50_Dark_value'].apply(scale_ic50)
df['IC50_class'] = df['IC50_Dark_value'].apply(class_ic50)

cells = df['Cell_line'].value_counts().reset_index().loc[:19]
years = df.drop_duplicates(subset=['DOI'])['Year'].value_counts().reset_index()
times = df['Time(h)'].value_counts().reset_index().loc[:5]
ic50_class = df['IC50_class'].value_counts().reset_index()
ic50_class['IC50_class'].replace({0: '≥10 μM', 1: '<10μM'}, inplace=True)
line_list = df['Cell_line'].value_counts().nlargest(50).index.tolist()
time_list = df['Time(h)'].value_counts().nlargest(6).index.tolist()
metal_list = df['Metal'].value_counts().index.tolist()

n_entries = df.drop_duplicates(subset=['SMILES_Ligands', 'Counterion', 'IC50_Dark(M*10^-6)', 'Cell_line', 'Time(h)', 'DOI', 'Metal']).shape[0]
n_smiles = df.drop_duplicates(['SMILES_Ligands', 'Metal']).shape[0]
n_sources = df['DOI'].nunique()
n_cell = df['Cell_line'].nunique()

col1intro, col2intro, col3intro = st.columns([1, 1, 2])
col1intro.markdown(f"""
# MetalCytoToxDB
The ”MetalCytoToxDB App” is an ML-based service integrated with the experimental database to explore literature cytotoxicity data (IC₅₀) of metal complexes.

Download MetalCytoToxDB: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15853577.svg)](https://doi.org/10.5281/zenodo.15853577)
                   
""")

col2intro.markdown(f"""# Overall stats: 
* **{n_entries}** IC₅₀ values
* **{n_smiles}** unique metal complexes (Ru, Ir, Os, Rh, Re)
* **{n_sources}** literature sources
* **{n_cell}** cell lines""")

col3intro.image('TOC.png')

tabs = st.tabs(["Explore statistics", "Search and Predict", "Search by cell line", "Search by fixed ligand subset", "Search by DOI and Authors"])

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
                search_df = df[(df['SMILES_Ligands'] == canonize_smiles)]
                if search_df.shape[0] == 0:
                    st.markdown('Nothing found')

                else:
                    st.markdown(f'### Found this complex in RuCytoToxDB:')
                    show_search_results(search_df)

with tabs[2]:
    col1select, col2select, col3select, col4select = st.columns([1, 1, 1, 1])
    selected_line = col1select.selectbox(label='Choose line', options=line_list, index=None, placeholder='A549')
    selected_metal = col2select.selectbox(label='Choose Metal', options=['All metals'] + metal_list, index=0)
    selected_time = col3select.selectbox(label='Choose exposure time (h)', options=['All time ranges'] + time_list, index=0)
    select_sorting = col4select.selectbox(label='Choose the sorting type', options=['Most cytooxic above', 'Least cytooxic above'], index=0)

    if selected_metal == 'All metals':
        search_df = df
    else:
        search_df = df[(df['Metal'] == selected_metal)]
    if selected_line:
        if selected_time == 'All time ranges':
            search_df = search_df[(search_df['Cell_line'] == selected_line)]
        else:
            search_df = search_df[(search_df['Cell_line'] == selected_line) & (search_df['Time(h)'] == selected_time)]
        if select_sorting == 'Least cytooxic above':
            search_df.sort_values(by='IC50_Dark_value', ascending=False, inplace=True)
        else:
            search_df.sort_values(by='IC50_Dark_value', ascending=True, inplace=True)
        show_search_results(search_df)

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
        selected_sorting = col3select.selectbox(label='Choose the sorting type', options=['Most cytotoxic above', 'Least cytotoxic above'], index=0, key="substructure_sort")
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
                    show_search_results(search_df)

with tabs[4]:
    col1select, col2select = st.columns([1, 1])
    doi = col1select.text_input(f"DOI", key=f"DOI")
    author = col2select.text_input(f"Author surname", key=f"Author", placeholder='Keppler')
    if st.button("Search in the database", key=f"DOI_button"):
        if doi:
            doi = doi.replace('https://doi.org/', '')
            doi = doi.replace('http://doi.org/', '')
            doi = doi.replace('http://dx.doi.org/', '')
            doi = doi.replace('https://dx.doi.org/', '')
            doi = doi.replace('https://www.doi.org/', '')
            doi = doi.replace('https://www.dx.doi.org/', '')
            doi = doi.replace('doi.org/', '')
            doi = doi.lower()
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