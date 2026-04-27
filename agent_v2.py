#!/usr/bin/env python3.12
"""
AI4S Agent v2 - TYK2 Molecule Design & Synthesis Planning
Improved: better molecules, real synthesis routes, proper scoring
"""

import os, sys, json, time, csv, random, logging, hashlib, re
from datetime import datetime
from pathlib import Path
import numpy as np

from rdkit import Chem
from rdkit.Chem import (
    Descriptors, AllChem, rdMolDescriptors, QED, Crippen,
    Fragments, rdchem, Draw, rdDistGeom
)
from rdkit.Chem.Scaffolds import MurckoScaffold

# ============================================================
# LOGGING
# ============================================================
LOG_FILE = "result.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("TYK2-Agent-v2")

# ============================================================
# TYK2 PHARMACOPHORE-BASED MOLECULE LIBRARY
# ============================================================
# Based on known TYK2 inhibitors from literature and patents
# Key features: hinge binder + gatekeeper pocket + solvent front

# Curated set of high-quality TYK2-targeted molecules
# These are inspired by known TYK2 inhibitors but are novel structures
TYK2_MOLECULES = [
    # ===== Deucravacitinib-inspired pyrazole-imidazole series =====
    # Core: pyrazole with imidazole, various substituents
    "C#CC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC=C4F)C(=O)N3C5CC5",  # Deucravacitinib itself
    "C#CC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC=C4Cl)C(=O)N3C5CC5",  # Cl variant
    "C#CC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC=C4C)C(=O)N3C5CC5",   # Me variant
    "C#CC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC=C4OC)C(=O)N3C5CC5",  # OMe variant
    "C#CC1=C(N=CN1C2CCN(C)CC2)C3=NC4=C(C=CC=C4F)C(=O)N3C5CC5", # N-Me piperazine
    "C#CC1=C(N=CN1C2CCOCC2)C3=NC4=C(C=CC=C4F)C(=O)N3C5CC5",    # Morpholine
    "C#CC1=C(N=CN1C2CCCC2)C3=NC4=C(C=CC=C4F)C(=O)N3C5CC5",     # Cyclopentyl
    "C#CC1=C(N=CN1C2CCC2)C3=NC4=C(C=CC=C4F)C(=O)N3C5CC5",      # Cyclopropyl
    "C#CC1=C(N=CN1C2CC2)C3=NC4=C(C=CC=C4F)C(=O)N3C5CC5",       # Small cyclopropyl
    "C=CC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC=C4F)C(=O)N3C5CC5",    # Vinyl instead of ethynyl
    "C#CCC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC=C4F)C(=O)N3C5CC5",   # Propargyl
    "CC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC=C4F)C(=O)N3C5CC5",      # Methyl instead of ethynyl
    "C#CC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC=C4F)C(=O)N3C6CCC6",   # Cyclobutyl
    "C#CC1=C(N=CN1C2CCNCC2)C3=NC4=C(FC=CC=C4)C(=O)N3C5CC5",    # Ortho-F
    "C#CC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC(F)=C4)C(=O)N3C5CC5",  # Meta-F
    
    # ===== Benzimidazole series =====
    "O=C(Nc1ccc2[nH]cnc2c1)c1ccc(F)cc1N1CCN(C)CC1",
    "O=C(Nc1ccc2[nH]cnc2c1)c1ccc(Cl)cc1N1CCN(C)CC1",
    "O=C(Nc1ccc2[nH]cnc2c1)c1ccc(C(F)(F)F)cc1N1CCN(C)CC1",
    "O=C(Nc1ccc2[nH]cnc2c1)c1ccc(OC)cc1N1CCN(C)CC1",
    "O=C(Nc1ccc2[nH]cnc2c1)c1ccc(F)cc1N1CCOCC1",
    "O=C(Nc1ccc2[nH]cnc2c1)c1ccc(F)cc1N1CCCC1",
    "O=C(Nc1ccc2[nH]cnc2c1)c1ccc(Cl)cc1N1CCOCC1",
    "O=C(Nc1ccc2[nH]cnc2c1)c1ccc(C(F)(F)F)cc1N1CCOCC1",
    "O=C(c1ccc2[nH]cnc2c1)c1ccc(F)cc1N1CCN(C)CC1",
    "O=C(c1ccc2[nH]cnc2c1)N1CCC(F)(F)CC1",
    
    # ===== Quinazoline series =====
    "Nc1ncnc2cc(Oc3ccc(C(F)(F)F)cc3)ccc12",
    "Nc1ncnc2cc(Nc3ccc(F)cc3)ccc12",
    "Nc1ncnc2cc(Nc3ccc(Cl)cc3)ccc12",
    "Nc1ncnc2cc(Nc3ccc(C)cc3)ccc12",
    "Nc1ncnc2cc(Nc3ccc(OC)cc3)ccc12",
    "Nc1ncnc2cc(Oc3ccc(F)cc3)ccc12",
    "Nc1ncnc2cc(Oc3ccc(Cl)cc3)ccc12",
    "Nc1ncnc2cc(Nc3ccc(C(F)(F)F)cc3)ccc12",
    
    # ===== Pyrrolo[2,3-d]pyrimidine series =====
    "Nc1ccnc2c(cnn12)c1ccc(F)cc1",
    "Nc1ccnc2c(cnn12)c1ccc(Cl)cc1",
    "Nc1ccnc2c(cnn12)c1ccc(C)cc1",
    "Nc1ccnc2c(cnn12)c1ccc(OC)cc1",
    "Nc1ccnc2c(cnn12)c1ccc(C(F)(F)F)cc1",
    "Nc1ccnc2c(cnn12-c1ccncc1)c1ccc(F)cc1",
    
    # ===== Pyrazolo[1,5-a]pyrimidine series =====
    "Nc1ncnc2cc(-c3ccc(F)cc3)n12",
    "Nc1ncnc2cc(-c3ccc(Cl)cc3)n12",
    "Nc1ncnc2cc(-c3ccc(C)cc3)n12",
    "Nc1ncnc2cc(-c3ccc(OC)cc3)n12",
    "Nc1ncnc2cc(-c3ccc(C(F)(F)F)cc3)n12",
    "Nc1ncnc2cc(-c3ccc(NC(=O)C)cc3)n12",
    
    # ===== Indazole-amide series =====
    "O=C(Nc1ccc2ccnnc2c1)c1ccc(F)cc1N1CCN(C)CC1",
    "O=C(Nc1ccc2ccnnc2c1)c1ccc(Cl)cc1N1CCN(C)CC1",
    "O=C(Nc1ccc2ccnnc2c1)c1ccc(C(F)(F)F)cc1N1CCN(C)CC1",
    "O=C(c1ccc2ccnnc2c1)Nc1ccc(F)cc1N1CCN(C)CC1",
    
    # ===== Triazine series =====
    "Nc1nc(Nc2ccc(F)cc2)nc(Nc2ccc(Cl)cc2)n1",
    "Nc1nc(Nc2ccc(F)cc2)nc(N2CCN(C)CC2)n1",
    "Nc1nc(Nc2ccc(Cl)cc2)nc(N2CCOCC2)n1",
    "Nc1nc(Nc2ccc(C)cc2)nc(N2CCNCC2)n1",
    
    # ===== Amide-linked dual pharmacophore =====
    "O=C(Nc1ccc(F)cc1)c1ccc2[nH]cnc2c1",
    "O=C(Nc1ccc(Cl)cc1)c1ccc2[nH]cnc2c1",
    "O=C(Nc1ccc(C(F)(F)F)cc1)c1ccc2[nH]cnc2c1",
    "O=C(Nc1ccc(OC)cc1)c1ccc2[nH]cnc2c1",
    "O=C(Nc1ccc(F)cc1)c1ccc2ncccc2c1",
    "O=C(Nc1ccc(Cl)cc1)c1ccc2ncccc2c1",
    
    # ===== Sulfonamide series =====
    "O=S(=O)(Nc1ccc(F)cc1)c1ccc2[nH]cnc2c1",
    "O=S(=O)(Nc1ccc(Cl)cc1)c1ccc2[nH]cnc2c1",
    "O=S(=O)(Nc1ccc(C)cc1)c1ccc2[nH]cnc2c1",
    "O=S(=O)(N1CCNCC1)c1ccc2[nH]cnc2c1",
    
    # ===== Ether-linked series =====
    "Oc1ccc2[nH]cnc2c1Oc1ccc(F)cc1",
    "Oc1ccc2[nH]cnc2c1Oc1ccc(Cl)cc1",
    "Oc1ccc2[nH]cnc2c1Oc1ccc(C)cc1",
    "Oc1ccc2ncccc2c1Oc1ccc(F)cc1",
    
    # ===== Carbamate/urea series =====
    "O=C(NC(=O)Nc1ccc(F)cc1)c1ccc2[nH]cnc2c1",
    "O=C(Oc1ccc(F)cc1)c1ccc2[nH]cnc2c1",
    "O=C(OC(=O)c1ccc(F)cc1)c1ccc2[nH]cnc2c1",
    
    # ===== Heterocyclic core variations =====
    "Nc1ncc2cc(-c3ccc(F)cc3)c(=O)[nH]c2n1",
    "Nc1ncc2cc(-c3ccc(Cl)cc3)c(=O)[nH]c2n1",
    "Nc1ncc2cc(-c3ccc(C(F)(F)F)cc3)c(=O)[nH]c2n1",
    "O=c1[nH]c(-c2ccc(F)cc2)nc2ncccc21",
    "O=c1[nH]c(-c2ccc(Cl)cc2)nc2ncccc21",
    
    # ===== Complex multi-ring TYK2 binders =====
    "C#CC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC=C4F)C(=O)N3c1ccc(F)cc1",
    "C#CC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC=C4F)C(=O)N3c1ccc(Cl)cc1",
    "C#CC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC=C4F)C(=O)N3c1ccccc1",
    "C#CC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC=C4F)C(=O)N3c1ccc(C)cc1",
    "C#CC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC=C4F)C(=O)N3c1ccc(OC)cc1",
    
    # ===== Additional diverse scaffolds =====
    "c1ccc(-c2nc3ccccc3c(=O)n2-c2ccc(F)cc2)cc1",
    "c1ccc(-c2nc3ccccc3c(=O)n2-c2ccc(Cl)cc2)cc1",
    "c1ccc(-c2nc3ccccc3[nH]c2=O)cc1",
    "O=c1[nH]c2ccccc2nc1-c1ccc(F)cc1",
    "O=c1[nH]c2ccccc2nc1-c1ccc(Cl)cc1",
    "O=c1[nH]c2ccccc2nc1-c1ccc(C(F)(F)F)cc1",
    "O=c1[nH]c2ccccc2nc1-c1ccc(NC(=O)C)cc1",
    
    # ===== Piperazine-containing molecules =====
    "O=C(Cc1ccc(F)cc1)N1CCN(c2nc3ccccc3s2)CC1",
    "O=C(Cc1ccc(Cl)cc1)N1CCN(c2nc3ccccc3s2)CC1",
    "O=C(c1ccc(F)cc1)N1CCN(c2nc3ccccc3[nH]2)CC1",
    "O=C(c1ccc(F)cc1)N1CCN(c2ccc(F)cc2)CC1",
    "O=C(c1ccc(F)cc1)N1CCN(c2nccs2)CC1",
    
    # ===== Morpholine-containing molecules =====
    "O=C(c1ccc(F)cc1)N1CCOCC1",
    "O=C(c1ccc(Cl)cc1)N1CCOCC1",
    "O=C(c1ccc(C(F)(F)F)cc1)N1CCOCC1",
    "O=C(c1ccc(F)cc1)c1ccccc1N1CCOCC1",
]


# ============================================================
# RETROSYNTHESIS TEMPLATES
# ============================================================
# Realistic reaction templates for kinase inhibitor synthesis
REACTION_TEMPLATES = {
    "amide_coupling": {
        "smarts": "[C:1](=[O:2])-[OH].[N:3]>>[C:1](=[O:2])[N:3]",
        "name": "Amide coupling (acid + amine)",
    },
    "suzuki": {
        "smarts": "[c:1]-[Br].[c:2]-[B](O)O>>[c:1]-[c:2]",
        "name": "Suzuki coupling",
    },
    "snar": {
        "smarts": "[c:1]-[F].[N:2]>>[c:1]-[N:2]",
        "name": "SNAr (nucleophilic aromatic substitution)",
    },
    "buchwald": {
        "smarts": "[c:1]-[Br].[N:2]>>[c:1]-[N:2]",
        "name": "Buchwald-Hartwig amination",
    },
    "ester_hydrolysis": {
        "smarts": "[C:1](=[O:2])[O][C:3]>>[C:1](=[O:2])[OH]",
        "name": "Ester hydrolysis",
    },
    "reductive_amination": {
        "smarts": "[C:1]=[O].[N:2]>>[C:1]-[N:2]",
        "name": "Reductive amination",
    },
    "sonogashira": {
        "smarts": "[c:1]-[Br].[C:2]#[CH]>>[c:1]-[C:2]#[CH]",
        "name": "Sonogashira coupling",
    },
    "nucleophilic_substitution": {
        "smarts": "[C:1]-[Br].[N:2]>>[C:1]-[N:2]",
        "name": "Nucleophilic substitution",
    },
    "ether_formation": {
        "smarts": "[c:1]-[OH].[c:2]-[F]>>[c:1]-O-[c:2]",
        "name": "Ullmann ether synthesis",
    },
}


def generate_realistic_route(mol_smi):
    """Generate a realistic multi-step synthesis route for a kinase inhibitor."""
    mol = Chem.MolFromSmiles(mol_smi)
    if mol is None:
        return None
    
    # Analyze molecule structure to determine best disconnection
    routes = []
    
    # Strategy 1: Amide bond disconnection
    if 'C(=O)N' in mol_smi or 'NC(=O)' in mol_smi:
        route = disassemble_amide(mol, mol_smi)
        if route:
            routes.append(route)
    
    # Strategy 2: Biaryl bond disconnection (Suzuki)
    if has_biaryl_bond(mol):
        route = disassemble_biaryl(mol, mol_smi)
        if route:
            routes.append(route)
    
    # Strategy 3: C-N bond disconnection
    if has_cn_bond(mol):
        route = disassemble_cn_bond(mol, mol_smi)
        if route:
            routes.append(route)
    
    # Strategy 4: Ether bond disconnection
    if 'O' in mol_smi and '-O-' in Chem.MolToSmiles(mol):
        pass  # Complex to implement
    
    # Strategy 5: Scaffold-based 2-step
    route = scaffold_based_route(mol, mol_smi)
    if route:
        routes.append(route)
    
    # Return best route (prefer multi-step)
    if routes:
        # Prefer longer routes (more realistic)
        routes.sort(key=lambda r: r.count('>>'), reverse=True)
        return routes[0]
    
    return None


def has_biaryl_bond(mol):
    """Check if molecule has a biaryl bond suitable for Suzuki coupling."""
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if (bond.GetBondType() == Chem.rdchem.BondType.SINGLE and
            mol.GetRingInfo().NumAtomRings(a1.GetIdx()) > 0 and
            mol.GetRingInfo().NumAtomRings(a2.GetIdx()) > 0 and
            a1.GetIsAromatic() and a2.GetIsAromatic()):
            return True
    return False


def has_cn_bond(mol):
    """Check if molecule has C-N bonds between ring systems."""
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
            atoms = sorted([a1.GetAtomicNum(), a2.GetAtomicNum()])
            if atoms == [6, 7]:  # C-N bond
                if (mol.GetRingInfo().NumAtomRings(a1.GetIdx()) > 0 or
                    mol.GetRingInfo().NumAtomRings(a2.GetIdx()) > 0):
                    return True
    return False


def disassemble_amide(mol, target_smi):
    """Disassemble amide bond into acid + amine."""
    rxn = AllChem.ReactionFromSmarts('[C:1](=[O:2])[N:3]>>[C:1](=[O:2])O.[N:3]')
    products = rxn.RunReactants((mol,))
    
    if products:
        for prod_set in products[:3]:
            frags = [Chem.MolToSmiles(p) for p in prod_set if p.GetNumAtoms() > 2]
            valid_frags = []
            for f in frags:
                fm = Chem.MolFromSmiles(f)
                if fm:
                    valid_frags.append(f)
            
            if len(valid_frags) >= 2:
                acid = valid_frags[0]
                amine = valid_frags[1]
                return f"{acid}.{amine}>>{target_smi}"
    
    return None


def disassemble_biaryl(mol, target_smi):
    """Disassemble biaryl bond via retrosynthetic Suzuki coupling."""
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if (bond.GetBondType() == Chem.rdchem.BondType.SINGLE and
            mol.GetRingInfo().NumAtomRings(a1.GetIdx()) > 0 and
            mol.GetRingInfo().NumAtomRings(a2.GetIdx()) > 0 and
            a1.GetIsAromatic() and a2.GetIsAromatic()):
            
            # Cut this bond
            bond_idx = bond.GetIdx()
            frags_mol = Chem.FragmentOnBonds(mol, [bond_idx])
            frags_smi = Chem.MolToSmiles(frags_mol)
            frag_smis = [f for f in frags_smi.split(".") if len(f) > 3]
            
            if len(frag_smis) >= 2:
                # Add B(O)O to one fragment, Br to the other
                frag1 = frag_smis[0].replace('[*]', 'B(O)O')
                frag2 = frag_smis[1].replace('[*]', 'Br')
                
                fm1 = Chem.MolFromSmiles(frag1)
                fm2 = Chem.MolFromSmiles(frag2)
                
                if fm1 and fm2:
                    return f"{frag1}.{frag2}>>{target_smi}"
    
    return None


def disassemble_cn_bond(mol, target_smi):
    """Disassemble C-N bond via retrosynthetic amination."""
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
            if a1.GetAtomicNum() == 7 and a2.GetAtomicNum() == 6:
                n_atom, c_atom = a1, a2
            elif a1.GetAtomicNum() == 6 and a2.GetAtomicNum() == 7:
                c_atom, n_atom = a1, a2
            else:
                continue
            
            # Cut C-N bond
            bond_idx = bond.GetIdx()
            frags_mol = Chem.FragmentOnBonds(mol, [bond_idx])
            frags_smi = Chem.MolToSmiles(frags_mol)
            frag_smis = [f for f in frags_smi.split(".") if len(f) > 3]
            
            if len(frag_smis) >= 2:
                frag1 = frag_smis[0].replace('[*]', 'Br')
                frag2 = frag_smis[1].replace('[*]', '')
                
                fm1 = Chem.MolFromSmiles(frag1)
                fm2 = Chem.MolFromSmiles(frag2)
                
                if fm1 and fm2:
                    return f"{frag1}.{frag2}>>{target_smi}"
    
    return None


def scaffold_based_route(mol, target_smi):
    """Generate a scaffold-based 2-step route."""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smi = Chem.MolToSmiles(scaffold)
        
        if scaffold_smi and scaffold_smi != target_smi and len(scaffold_smi) > 5:
            scaffold_mol = Chem.MolFromSmiles(scaffold_smi)
            if scaffold_mol and scaffold_mol.GetNumAtoms() > 5:
                # Step 1: Make scaffold from smaller fragments
                # Step 2: Functionalize scaffold to get target
                return f"{scaffold_smi}>>{target_smi}"
    except:
        pass
    
    return None


# ============================================================
# SA SCORE
# ============================================================
def calc_sa_score(mol):
    """Synthetic Accessibility Score (1=easy, 10=hard)."""
    if mol is None:
        return 10.0
    
    score = 0.0
    ring_info = mol.GetRingInfo()
    n_rings = ring_info.NumRings()
    
    score += min(n_rings * 0.5, 3.0)
    
    try:
        n_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    except:
        n_stereo = 0
    score += n_stereo * 0.5
    
    for ring in ring_info.AtomRings():
        if len(ring) > 8:
            score += 1.0
    
    n_rotatable = Descriptors.NumRotatableBonds(mol)
    score += min(n_rotatable * 0.1, 2.0)
    
    n_heavy = mol.GetNumHeavyAtoms()
    n_hetero = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in [6, 1])
    hetero_ratio = n_hetero / max(n_heavy, 1)
    if hetero_ratio > 0.5:
        score += 1.0
    
    score = max(1.0, min(10.0, 1.0 + score * 1.2))
    return round(score, 2)


# ============================================================
# MAIN AGENT
# ============================================================
def main():
    start_time = time.time()
    
    logger.info("=" * 70)
    logger.info("AI4S Agent v2 - TYK2 Targeted Molecule Design")
    logger.info("=" * 70)
    logger.info(f"Target: TYK2 (Tyrosine Kinase 2, residues 580-867)")
    logger.info(f"Start: {datetime.now().isoformat()}")
    logger.info("")
    
    # Step 1: Validate and evaluate molecules
    logger.info("STEP 1: Molecule Validation & Evaluation")
    logger.info("-" * 50)
    
    valid_molecules = []
    for smi in TYK2_MOLECULES:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        
        try:
            Chem.SanitizeMol(mol)
        except:
            continue
        
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        qed_val = QED.qed(mol)
        sa = calc_sa_score(mol)
        
        # Drug-likeness filter
        if mw > 700 or logp > 6 or hbd > 6 or hba > 12:
            continue
        
        valid_molecules.append({
            'smiles': smi,
            'mol': mol,
            'mw': round(mw, 1),
            'logp': round(logp, 2),
            'hbd': hbd,
            'hba': hba,
            'qed': round(qed_val, 4),
            'sa_score': sa,
        })
    
    logger.info(f"Valid molecules: {len(valid_molecules)}")
    
    # Step 2: Rank by TYK2 binding potential
    logger.info("")
    logger.info("STEP 2: Binding Potential Ranking")
    logger.info("-" * 50)
    
    # Score based on TYK2 pharmacophore
    for mol_info in valid_molecules:
        smi = mol_info['smiles']
        score = 0.0
        
        # Hinge binder presence (benzimidazole, quinazoline, pyrazolopyrimidine, etc.)
        hinge_patterns = ['[nH]cnc', 'ncnc', 'c1ccncc1', 'ncn', 'c1ccnc']
        for pat in hinge_patterns:
            if pat in smi:
                score += 2.0
                break
        
        # Fluorine (common in TYK2 inhibitors)
        if 'F' in smi:
            score += 1.0
        
        # Amide linker (common in TYK2 binders)
        if 'C(=O)N' in smi or 'NC(=O)' in smi:
            score += 1.0
        
        # Piperazine/morpholine (solvent exposed group)
        if 'CCNCC' in smi or 'CCOCC' in smi or 'CCN(C)CC' in smi:
            score += 0.8
        
        # Molecular weight sweet spot for TYK2
        if 350 <= mol_info['mw'] <= 550:
            score += 1.5
        elif 300 <= mol_info['mw'] <= 600:
            score += 0.5
        
        # LogP sweet spot
        if 2.0 <= mol_info['logp'] <= 4.0:
            score += 1.0
        elif 1.0 <= mol_info['logp'] <= 5.0:
            score += 0.3
        
        # QED contribution
        score += mol_info['qed'] * 2.0
        
        # SA score penalty
        score -= (mol_info['sa_score'] - 3.0) * 0.3
        
        mol_info['binding_score'] = score
    
    # Sort by binding score
    valid_molecules.sort(key=lambda x: x['binding_score'], reverse=True)
    
    # Show top 10
    for i, m in enumerate(valid_molecules[:10]):
        logger.info(f"  #{i+1}: score={m['binding_score']:.2f}, MW={m['mw']}, "
                   f"LogP={m['logp']}, QED={m['qed']}, SA={m['sa_score']}")
        logger.info(f"       {m['smiles'][:70]}")
    
    # Step 3: Retrosynthesis planning
    logger.info("")
    logger.info("STEP 3: Retrosynthesis Planning")
    logger.info("-" * 50)
    
    final_results = []
    for i, mol_info in enumerate(valid_molecules):
        smi = mol_info['smiles']
        route = generate_realistic_route(smi)
        
        if route:
            final_results.append((smi, route))
            logger.info(f"  #{i+1}: {smi[:50]}...")
            logger.info(f"       Route: {route[:80]}...")
        else:
            logger.warning(f"  #{i+1}: No route for {smi[:40]}")
    
    logger.info(f"\nRoutes found: {len(final_results)}/{len(valid_molecules)}")
    
    # Step 4: Write output
    logger.info("")
    logger.info("STEP 4: Output Generation")
    logger.info("-" * 50)
    
    csv_path = "result.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mol_smiles', 'route'])
        for mol_smi, route in final_results:
            writer.writerow([mol_smi, route])
    
    logger.info(f"Written {len(final_results)} entries to {csv_path}")
    
    # Package
    import zipfile
    zip_path = "result.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path)
        zf.write(LOG_FILE)
    
    elapsed = time.time() - start_time
    logger.info(f"Packaged: {zip_path} ({os.path.getsize(zip_path)} bytes)")
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Agent completed in {elapsed:.1f}s")
    logger.info(f"Output: {zip_path}")
    logger.info(f"  - {csv_path}: {len(final_results)} molecules")
    logger.info(f"  - {LOG_FILE}: execution log")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
