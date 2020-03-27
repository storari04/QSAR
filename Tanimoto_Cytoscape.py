from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

mols = [m for m in Chem.SDMolSupplier("./hoge.sdf")]

#Tanimotosimilarity 0.7 mapping
def Tanimotosimilarity():
    out = open("hogehoge.txt","w")
    fps = [AllChem.GetMorganFingerprint(m,2) for m in mols]

    simmat = []
    for i in range(len(fps)):
        tsims = DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])
        simmat.append(tsims)
        if tsims >= 0.7:
            out.write("%s\tsim\t%s\n"%(mols[i].GetProp("ID"),mols[j].GetProp("ID")))
    out.close()

#Removing Core
def removing_core():
    core = Chem.MolFromSmiles('c1ccccc1')
    for m in mols:
        r = Chem.ReplaceCore(m, core)
        #print(Chem.MolToSmiles(r))

    return r

#Generation of 3D molecules
def gen_3Dmolecules(mols):
    f = open( name , "w")
    f.write("molid\tconfId\tmin_energy\n")
    mols = [Chem.AddHs(m) for m in mols]
    out = Chem.SDWriter("genconf.sdf")
    for molid, mol in enumerate(mols):
        cids = AllChem.EmbedMultipleConfs(mol, numConfs = 10, clearConfs = True, pruneRmsThresh = 0.5)
        energy = []
        for cid in cids:
            mmff = AllChem.MMFFGetMoleculeForceField(m_h, prop, confId=cid)
            mmff.Minimize()
            energy.append(mmff.CalcEnergy())
            f.write("%s\t%s\t%s\n"%(molid, cid, energy))
        min_cid = energy.index(min(e_list))
        #print min(e_list)
        mol.SetProp("molid","%s"%molid)
        mol.SetProp("Min_energy","%s" % min(e_list))
        mol.SetProp("confId","%s"%min_cid)
        out.write(mol, confId = min_cid)
    out.close()
    f.close()
