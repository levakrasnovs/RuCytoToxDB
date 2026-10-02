"""Pre-render all MetalLipoDB complexes with metal2d into lipo_structures.pkl ({smiles_complex: PNG bytes}).

Rerun after every update of MetalLipoDB.csv:  python prerender_lipo.py
"""
import io, pickle, pandas as pd
from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D
import metal2d
RDLogger.DisableLog('rdApp.*')
smiles = pd.read_csv('MetalLipoDB.csv')['smiles_complex'].drop_duplicates().tolist()
out, failed = {}, []
for k, s in enumerate(smiles, 1):
    try:
        m = Chem.MolFromSmiles(s, sanitize=False); m.UpdatePropertyCache(strict=False)
        p = metal2d.prepare_for_drawing(metal2d.depict(m))
        d = rdMolDraw2D.MolDraw2DCairo(300, 300); metal2d.style_options(d.drawOptions()); d.drawOptions().padding = 0.05
        d.DrawMolecule(rdMolDraw2D.PrepareMolForDrawing(p, kekulize=False, wedgeBonds=False)); d.FinishDrawing()
        im = Image.open(io.BytesIO(d.GetDrawingText())).convert('P', palette=Image.ADAPTIVE, colors=32)
        buf = io.BytesIO(); im.save(buf, 'PNG', optimize=True); out[s] = buf.getvalue()
    except Exception as e:
        failed.append(s)
    if k % 500 == 0: print(k, len(smiles), flush=True)
pickle.dump(out, open('lipo_structures.pkl', 'wb'))
print('rendered', len(out), 'failed', len(failed), 'MB', round(sum(map(len, out.values())) / 1e6, 1))
