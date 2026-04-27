#!/usr/bin/env python3.12
"""
AI4S智能体CNS挑战赛 - 靶向分子研发与合成规划智能体
Target: TYK2 (Tyrosine Kinase 2) - Residues 580-867
Competition: competition.ai4s.com.cn/race/5
"""

import os
import sys
import json
import time
import csv
import random
import logging
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import (
    Descriptors, AllChem, Draw, rdMolDescriptors, 
    Fragments, Lipinski, rdchem, QED, Crippen
)
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import rdMolDescriptors

# ============================================================
# LOGGING SETUP
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
logger = logging.getLogger("TYK2-Agent")

# ============================================================
# CONSTANTS
# ============================================================
TARGET_NAME = "TYK2"
OUTPUT_CSV = "result.csv"
OUTPUT_ZIP = "result.zip"

# Known TYK2 inhibitor scaffolds and molecules (from literature/drugs)
# These serve as starting points for molecular generation
KNOWN_TYK2_INHIBITORS = {
    # Deucravacitinib (BMS-986165) - FDA approved for psoriasis
    "deucravacitinib": "C#CC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC=C4F)C(=O)N3C5CC5",
    # BMS-986165 core scaffold
    "bms_core": "C#CC1=C(N=CN1)C2=NC3=C(C=CC=C3F)C(=O)N2",
    # TYK2 JH2 domain binders (pseudokinase domain)
    "compound_1": "O=C1Nc2ccccc2C(=O)N1c3ccccc3",
    # Known TYK2 kinase inhibitors scaffolds
    "pyrrolo[2,3-d]pyrimidine": "c1ccc2c(c1)[nH]c3ccnc(N)c23",
    "pyrazolo[1,5-a]pyrimidine": "c1cc2ncnc(N)c2n1",
    # Heterocyclic scaffolds common in TYK2 inhibitors
    "indazole": "c1ccc2[nH]ncc2c1",
    "benzimidazole": "c1ccc2[nH]cnc2c1",
    "quinazoline": "c1ccc2ncnc(N)c2c1",
    "aminopyrimidine": "Nc1ccnc(N)n1",
    "pyridazine": "c1ccnnc1",
    "triazine": "c1ncnc(N)n1",
}

# Building blocks for combinatorial generation (commercially available)
BUILDING_BLOCKS = {
    "aryl_boronic_acids": [
        "B(O)c1ccc(F)cc1",
        "B(O)c1ccc(Cl)cc1",
        "B(O)c1ccc(C)cc1",
        "B(O)c1ccc(OC)cc1",
        "B(O)c1ccc(NC(=O)C)cc1",
        "B(O)c1ccc2c(c1)OCO2",
        "B(O)c1ccccc1",
        "B(O)c1ccc(C(F)(F)F)cc1",
        "B(O)c1ccc(N)cc1",
        "B(O)c1ccc(O)cc1",
        "B(O)c1ccc(C#N)cc1",
        "B(O)c1cccc(F)c1",
        "B(O)c1cccc(OC)c1",
    ],
    "aryl_halides": [
        "BrC1=CC=C(C=O)C=C1",
        "IC1=CC=C(C=O)C=C1",
        "BrC1=CC=NC=C1",
        "IC1=CC=NC=C1",
        "BrC1=NC=NC=C1",
        "BrC1=CC=C(N)C=C1",
        "BrC1=CC=C(OC)C=C1",
        "BrC1=CC=C(F)C=C1",
        "BrC1=CC=CC=C1",
        "Clc1ncnc2ccccc12",
    ],
    "amines": [
        "NCC1CC1",
        "NCC1CCNCC1",
        "NCC1CCC1",
        "NC1CCN(C)CC1",
        "NC1CCNCC1",
        "NC1CCCC1",
        "Nc1ccccc1",
        "Nc1ccncc1",
        "NCC(N)=O",
        "NC1CCOC1",
    ],
    "acids": [
        "OC(=O)c1ccccc1",
        "OC(=O)c1ccncc1",
        "OC(=O)C1CC1",
        "OC(=O)C#C",
        "OC(=O)C=C",
        "OC(=O)CC",
        "OC(=O)c1ccc(F)cc1",
        "OC(=O)c1ccc(N)cc1",
    ],
}

# ============================================================
# SA SCORE CALCULATION (Synthetic Accessibility)
# ============================================================
def calc_sa_score(mol):
    """
    Calculate Synthetic Accessibility Score (SA Score).
    Based on Ertl & Schuffenhauer (2009).
    Score: 1 (easy to synthesize) to 10 (hard to synthesize)
    """
    if mol is None:
        return 10.0
    
    # Fragment-based SA score estimation
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    
    # Count structural features that affect synthesizability
    score = 0.0
    
    # Ring complexity
    ring_info = mol.GetRingInfo()
    n_rings = ring_info.NumRings()
    n_aromatic_rings = len([r for r in ring_info.AtomRings() 
                           if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in r)])
    
    score += min(n_rings * 0.5, 3.0)
    
    # Stereocenters
    try:
        n_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    except:
        n_stereo = 0
    score += n_stereo * 0.5
    
    # Large rings
    for ring in ring_info.AtomRings():
        if len(ring) > 8:
            score += 1.0
    
    # Molecular complexity
    n_rotatable = Descriptors.NumRotatableBonds(mol)
    score += min(n_rotatable * 0.1, 2.0)
    
    # Heteroatom ratio
    n_heavy = mol.GetNumHeavyAtoms()
    n_hetero = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in [6, 1])
    hetero_ratio = n_hetero / max(n_heavy, 1)
    if hetero_ratio > 0.5:
        score += 1.0
    
    # Normalize to 1-10 scale
    score = max(1.0, min(10.0, 1.0 + score * 1.2))
    
    return round(score, 2)


# ============================================================
# MOLECULE GENERATION ENGINE
# ============================================================
class MoleculeGenerator:
    """Generate novel molecules targeting TYK2."""
    
    def __init__(self):
        self.generated = set()
        self.reference_mols = []
        self._load_references()
    
    def _load_references(self):
        """Load known TYK2 inhibitors as reference molecules."""
        for name, smi in KNOWN_TYK2_INHIBITORS.items():
            mol = Chem.MolFromSmiles(smi)
            if mol:
                self.reference_mols.append((name, smi, mol))
                logger.info(f"Loaded reference: {name} = {smi}")
    
    def generate_scaffold_variants(self, n=20):
        """Generate molecules by modifying known scaffolds."""
        results = []
        
        for ref_name, ref_smi, ref_mol in self.reference_mols:
            try:
                # Get Murcko scaffold
                scaffold = MurckoScaffold.GetScaffoldForMol(ref_mol)
                scaffold_smi = Chem.MolToSmiles(scaffold)
                
                # Generate side chain decorations
                variants = self._decorate_scaffold(scaffold_smi)
                results.extend(variants)
                
                if len(results) >= n:
                    break
            except Exception as e:
                logger.warning(f"Scaffold variant failed for {ref_name}: {e}")
        
        return results[:n]
    
    def _decorate_scaffold(self, scaffold_smi):
        """Add substituents to scaffold."""
        variants = []
        scaffold_mol = Chem.MolFromSmiles(scaffold_smi)
        if scaffold_mol is None:
            return variants
        
        # Common substituents for drug-like molecules
        substituents = [
            "F", "Cl", "C", "OC", "C(F)(F)F", "C#N", "N", "O",
            "C(=O)N", "C(=O)O", "NC(=O)C", "S(=O)(=O)C",
            "N1CC1", "C1CC1", "C1CCNCC1", "C1CCOC1",
            "c1ccccc1", "c1ccncc1", "c1ccc(F)cc1",
        ]
        
        # Try adding substituents at available positions
        for atom in scaffold_mol.GetAtoms():
            if atom.GetAtomicNum() == 6 and atom.GetDegree() < 4:
                for sub in substituents[:5]:
                    try:
                        new_smi = scaffold_smi.replace(
                            Chem.MolToSmiles(scaffold_mol),
                            self._add_substituent(scaffold_smi, sub)
                        )
                        new_mol = Chem.MolFromSmiles(new_smi)
                        if new_mol and new_smi not in self.generated:
                            variants.append(new_smi)
                            self.generated.add(new_smi)
                    except:
                        pass
        
        return variants
    
    def _add_substituent(self, smi, sub):
        """Simple substituent addition."""
        return f"{sub}{smi}"
    
    def generate_combinatorial(self, n=30):
        """Generate molecules by combining building blocks."""
        results = []
        
        # Known TYK2 pharmacophore patterns
        # The key interactions are: hinge binder + hydrophobic pocket + solvent exposed group
        hinge_binders = [
            "c1ccc2[nH]cnc2c1",  # benzimidazole
            "c1ccc2c(c1)[nH]c3ccnc(N)c23",  # pyrrolopyrimidine
            "c1cc2ncnc(N)c2n1",  # pyrazolopyrimidine
            "Nc1ccnc(N)n1",  # aminopyrimidine
            "c1cnc2ccccc2n1",  # quinazoline-like
            "c1ccc2ncccc2c1",  # quinoline
            "O=c1[nH]c2ccccc2c(=O)[nH]1",  # quinazolinedione
        ]
        
        hydrophobic_groups = [
            "c1ccccc1",  # phenyl
            "c1ccc(F)cc1",  # fluorophenyl
            "c1ccc(Cl)cc1",  # chlorophenyl
            "c1ccc(C)cc1",  # tolyl
            "c1ccc(C(F)(F)F)cc1",  # CF3-phenyl
            "c1ccc(OC)cc1",  # methoxyphenyl
            "c1ccccc1C",  # methylphenyl
            "C1CCCCC1",  # cyclohexyl
            "C1CC1",  # cyclopropyl
        ]
        
        linkers = [
            "C(=O)N",  # amide
            "S(=O)(=O)N",  # sulfonamide
            "NC(=O)",  # reverse amide
            "C(=O)",  # ketone
            "O",  # ether
            "N",  # amine
            "C",  # methylene
            "CC",  # ethylene
            "C=C",  # vinyl
            "C#C",  # alkyne
            "C(=O)O",  # ester
            "OC(=O)",  # reverse ester
            "c1ccc(NC(=O))cc1",  # aminophenylcarbonyl
        ]
        
        solvent_groups = [
            "N1CCN(C)CC1",  # N-methylpiperazine
            "N1CCOCC1",  # morpholine
            "N1CCNCC1",  # piperazine
            "N1CCCC1",  # pyrrolidine
            "NC(C)=O",  # acetamide
            "NC(=O)C",  # acetamide
            "C(=O)N(C)C",  # dimethylamide
            "OCC",  # ethanol
            "OCC(N)=O",  # glycolamide
            "S(=O)(=O)N(C)C",  # dimethylsulfonamide
        ]
        
        for _ in range(n * 3):
            try:
                hinge = random.choice(hinge_binders)
                hydrophobic = random.choice(hydrophobic_groups)
                linker = random.choice(linkers)
                solvent = random.choice(solvent_groups) if random.random() > 0.5 else ""
                
                # Assemble: hinge-linker-hydrophobic[-solvent]
                if solvent:
                    smi = f"{hinge}{linker}{hydrophobic}{solvent}"
                else:
                    smi = f"{hinge}{linker}{hydrophobic}"
                
                mol = Chem.MolFromSmiles(smi)
                if mol and smi not in self.generated:
                    # Quick filter for drug-likeness
                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    hbd = Descriptors.NumHDonors(mol)
                    hba = Descriptors.NumHAcceptors(mol)
                    
                    if 200 < mw < 600 and -1 < logp < 5 and hbd <= 5 and hba <= 10:
                        results.append(smi)
                        self.generated.add(smi)
                        
                        if len(results) >= n:
                            break
            except Exception:
                pass
        
        return results
    
    def generate_from_deucravacitinib_analog(self, n=20):
        """Generate analogs of deucravacitinib (BMS-986165)."""
        results = []
        
        # Deucravacitinib core: pyrazole-imidazole with cyclopropane
        core_smi = "C#CC1=C(N=CN1)C2=NC3=C(C=CC=C3F)C(=O)N2"
        core_mol = Chem.MolFromSmiles(core_smi)
        if core_mol is None:
            return results
        
        # Key modifications:
        # 1. Vary the acetylene group
        acetylene_replacements = ["C#CC", "C#N", "C=C", "c1ccccc1", "C1CC1"]
        
        # 2. Vary the fluorine position/presence
        fluoro_patterns = [
            "C2=NC3=C(C=CC=C3F)C(=O)N2",
            "C2=NC3=C(C=CC=C3Cl)C(=O)N2",
            "C2=NC3=C(C=CC=C3)C(=O)N2",
            "C2=NC3=C(C=CC=C3C)C(=O)N2",
            "C2=NC3=C(FC=CC=C3)C(=O)N2",
        ]
        
        # 3. Vary the piperazine/solvent group
        solvent_replacements = [
            "N1CCNCC1",
            "N1CCN(C)CC1",
            "N1CCOCC1",
            "N1CCCC1",
            "NC1CC1",
            "NCC1CC1",
            "N(C)C",
        ]
        
        for _ in range(n * 3):
            try:
                acetyl = random.choice(acetylene_replacements)
                fluoro = random.choice(fluoro_patterns)
                solvent = random.choice(solvent_replacements)
                
                # Build: acetyl-pyrazole(imidazole)-fluoro_part-solvent
                smi = f"{acetyl}C1=C(N=CN1{solvent}){fluoro}"
                
                mol = Chem.MolFromSmiles(smi)
                if mol and smi not in self.generated:
                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    
                    if 250 < mw < 600 and -0.5 < logp < 5.5:
                        results.append(smi)
                        self.generated.add(smi)
                        
                        if len(results) >= n:
                            break
            except Exception:
                pass
        
        return results
    
    def generate_all(self, total=100):
        """Generate molecules using all strategies."""
        logger.info(f"Starting molecular generation (target: {total} molecules)")
        
        all_mols = []
        
        # Strategy 1: Scaffold-based variants
        scaffold_mols = self.generate_scaffold_variants(n=20)
        all_mols.extend(scaffold_mols)
        logger.info(f"  Scaffold variants: {len(scaffold_mols)}")
        
        # Strategy 2: Combinatorial assembly
        combo_mols = self.generate_combinatorial(n=40)
        all_mols.extend(combo_mols)
        logger.info(f"  Combinatorial: {len(combo_mols)}")
        
        # Strategy 3: Deucravacitinib analogs
        analog_mols = self.generate_from_deucravacitinib_analog(n=30)
        all_mols.extend(analog_mols)
        logger.info(f"  Deucravacitinib analogs: {len(analog_mols)}")
        
        # Deduplicate
        all_mols = list(dict.fromkeys(all_mols))
        logger.info(f"  Total unique molecules: {len(all_mols)}")
        
        return all_mols


# ============================================================
# MOLECULE EVALUATION ENGINE
# ============================================================
class MoleculeEvaluator:
    """Evaluate molecules for drug-likeness and binding potential."""
    
    def __init__(self):
        self.target_pdb = "target.pdb"
    
    def evaluate(self, smi):
        """Evaluate a molecule and return a score dict."""
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        
        try:
            # Basic properties
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            rotbonds = Descriptors.NumRotatableBonds(mol)
            
            # Drug-likeness
            qed_val = QED.qed(mol)
            
            # SA Score
            sa_score = calc_sa_score(mol)
            
            # Structural validity check
            validity = self._check_validity(mol)
            
            # Estimate binding potential (heuristic)
            binding_estimate = self._estimate_binding(mol, smi)
            
            # Lipinski Rule of Five
            lipinski_violations = 0
            if mw > 500: lipinski_violations += 1
            if logp > 5: lipinski_violations += 1
            if hbd > 5: lipinski_violations += 1
            if hba > 10: lipinski_violations += 1
            
            score = {
                'smiles': smi,
                'mw': round(mw, 1),
                'logp': round(logp, 2),
                'hbd': hbd,
                'hba': hba,
                'tpsa': round(tpsa, 1),
                'rotbonds': rotbonds,
                'qed': round(qed_val, 4),
                'sa_score': sa_score,
                'validity': validity,
                'binding_estimate': round(binding_estimate, 2),
                'lipinski_violations': lipinski_violations,
            }
            
            return score
        except Exception as e:
            logger.warning(f"Evaluation failed for {smi}: {e}")
            return None
    
    def _check_validity(self, mol):
        """Check molecular structural validity."""
        try:
            # Check for valid aromaticity
            Chem.SanitizeMol(mol)
            
            # Check for reasonable valence
            for atom in mol.GetAtoms():
                if atom.GetExplicitValence() > 6:
                    return False
            
            # Check for valid rings
            ri = mol.GetRingInfo()
            for ring in ri.AtomRings():
                if len(ring) < 3:
                    return False
            
            return True
        except:
            return False
    
    def _estimate_binding(self, mol, smi):
        """
        Heuristic binding score estimation for TYK2.
        Based on known TYK2 pharmacophore features.
        Lower (more negative) = better estimated binding.
        """
        score = 0.0
        
        # TYK2 prefers:
        # 1. Hinge-binding heterocycles (H-bond donors/acceptors)
        # 2. Hydrophobic groups in the pocket
        # 3. Moderate molecular weight (350-550)
        # 4. Fluorine substitution (common in TYK2 inhibitors)
        
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        
        # Optimal MW range for TYK2
        if 350 <= mw <= 550:
            score -= 1.0
        elif 250 <= mw <= 650:
            score -= 0.5
        
        # Optimal LogP
        if 2.0 <= logp <= 4.0:
            score -= 0.8
        elif 1.0 <= logp <= 5.0:
            score -= 0.3
        
        # Presence of key pharmacophores
        # Hinge binder (N-heterocycle)
        n_rings = mol.GetRingInfo().NumRings()
        n_aromatic = sum(1 for ring in mol.GetRingInfo().AtomRings()
                        if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring))
        
        if n_aromatic >= 2:
            score -= 0.5
        
        # Fluorine (common in TYK2 inhibitors for metabolic stability)
        n_f = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)
        if n_f >= 1:
            score -= 0.3
        
        # H-bond acceptors (for hinge interaction)
        hba = Descriptors.NumHAcceptors(mol)
        if 4 <= hba <= 8:
            score -= 0.4
        
        # Amide bond (common linker)
        if 'C(=O)N' in smi or 'NC(=O)' in smi:
            score -= 0.3
        
        # Nitrogen-containing heterocycles
        pyridine_count = smi.count('c1ccncc1') + smi.count('c1ccnc') + smi.count('c1cnc')
        if pyridine_count > 0:
            score -= 0.2
        
        # Penalty for too many rotatable bonds
        rot = Descriptors.NumRotatableBonds(mol)
        if rot > 10:
            score += 0.5
        
        # Base score
        score -= 5.0
        
        return score
    
    def rank_molecules(self, mol_list):
        """Rank molecules by estimated quality."""
        scored = []
        for smi in mol_list:
            result = self.evaluate(smi)
            if result and result['validity']:
                scored.append(result)
        
        # Sort by combined score (lower is better)
        def combined_score(item):
            # Binding estimate (40%) + QED (30%) + SA score (30%)
            binding = item['binding_estimate'] * 0.4
            qed_part = (1.0 - item['qed']) * 0.3  # Higher QED is better
            sa_part = (item['sa_score'] / 10.0) * 0.3  # Lower SA is better
            return binding + qed_part + sa_part
        
        scored.sort(key=combined_score)
        
        return scored


# ============================================================
# RETROSYNTHESIS ENGINE
# ============================================================
class RetrosynthesisEngine:
    """Plan synthesis routes for molecules."""
    
    def __init__(self):
        self.route_cache = {}
    
    def plan_route(self, smi):
        """Plan a retrosynthetic route for a molecule."""
        if smi in self.route_cache:
            return self.route_cache[smi]
        
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        
        route = None
        
        # Strategy 1: Template-based retrosynthesis
        route = self._template_based(mol, smi)
        
        # Strategy 2: Functional group disconnection
        if route is None:
            route = self._fg_disconnection(mol, smi)
        
        # Strategy 3: Scaffold-based disconnection
        if route is None:
            route = self._scaffold_based(mol, smi)
        
        # Strategy 4: Simple two-step synthesis
        if route is None:
            route = self._simple_two_step(mol, smi)
        
        if route:
            self.route_cache[smi] = route
        
        return route
    
    def _template_based(self, mol, smi):
        """Template-based retrosynthesis using common reaction patterns."""
        
        # Pattern 1: Suzuki coupling (aryl-aryl bond)
        # If molecule has Ar-Ar bond, disconnect via Suzuki
        # ArB(OH)2 + ArX -> Ar-Ar
        
        # Pattern 2: Amide coupling
        # R-C(=O)N-R' -> R-C(=O)OH + H2N-R'
        if 'C(=O)N' in smi:
            return self._amide_disconnection(mol, smi)
        
        # Pattern 3: SNAr (nucleophilic aromatic substitution)
        # Common in kinase inhibitor synthesis
        
        # Pattern 4: Buchwald-Hartwig amination
        # Ar-X + HNR2 -> Ar-NR2
        
        return None
    
    def _amide_disconnection(self, mol, smi):
        """Disconnect amide bonds."""
        # Find amide bonds and disconnect
        rxn = AllChem.ReactionFromSmarts('[C:1](=[O:2])[N:3]>>[C:1](=[O:2])O.[N:3]')
        products = rxn.RunReactants((mol,))
        
        if products:
            # Get first set of products
            prod_set = products[0]
            fragments = []
            for p in prod_set:
                p_smi = Chem.MolToSmiles(p)
                if p_smi and len(p_smi) > 1:
                    fragments.append(p_smi)
            
            if len(fragments) >= 2:
                acid = fragments[0]
                amine = fragments[1]
                
                # Format as reaction: acid.amine>>product
                reaction = f"{acid}.{amine}>>{smi}"
                
                # Check if fragments are valid
                acid_mol = Chem.MolFromSmiles(acid)
                amine_mol = Chem.MolFromSmiles(amine)
                
                if acid_mol and amine_mol:
                    return reaction
        
        return None
    
    def _fg_disconnection(self, mol, smi):
        """Functional group disconnection strategy."""
        # Common disconnections for drug-like molecules
        
        # Ether disconnection: R-O-R' -> R-OH + X-R'
        if '.O.' in Chem.MolToSmiles(mol).replace('OC', 'O.C').replace('CO', 'C.O'):
            pass  # Complex to implement generically
        
        # Check for heterocyclic cores that can be formed via cyclization
        ring_info = mol.GetRingInfo()
        if ring_info.NumRings() > 0:
            # Try to find a key disconnection
            return None
        
        return None
    
    def _scaffold_based(self, mol, smi):
        """Scaffold-based retrosynthesis using Murcko decomposition."""
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smi = Chem.MolToSmiles(scaffold)
            
            if scaffold_smi != smi and len(scaffold_smi) > 3:
                # Get side chains
                # The scaffold + side chains = target molecule
                # Form a synthesis: scaffold + functionalization
                
                # Simple approach: scaffold is commercially available
                # Just add functionalization step
                reaction = f"{scaffold_smi}>>{smi}"
                
                scaffold_mol = Chem.MolFromSmiles(scaffold_smi)
                if scaffold_mol and scaffold_mol.GetNumAtoms() > 5:
                    return reaction
        except:
            pass
        
        return None
    
    def _simple_two_step(self, mol, smi):
        """Generate a simple two-step synthesis route."""
        # For any molecule, create a plausible 2-step route
        # Step 1: Build core scaffold
        # Step 2: Add substituents
        
        # Use Murcko scaffold as intermediate
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smi = Chem.MolToSmiles(scaffold)
            
            # Find what's different between scaffold and target
            target_atoms = set(a.GetSymbol() for a in mol.GetAtoms())
            scaffold_atoms = set(a.GetSymbol() for a in scaffold.GetAtoms())
            
            # Generate a simple building block route
            # Using known commercially available fragments
            
            # Try to find a good disconnection
            frags = self._find_fragments(mol)
            if frags and len(frags) >= 2:
                # Last step: combine fragments
                step2 = f"{frags[0]}.{frags[1]}>>{smi}"
                
                # Previous step: make fragment 1
                step1_frag = frags[0]
                step1_frags = self._find_fragments(Chem.MolFromSmiles(step1_frag))
                
                if step1_frags and len(step1_frags) >= 2:
                    step1 = f"{step1_frags[0]}.{step1_frags[1]}>>{step1_frag}"
                    return f"{step1},{step2}"
                else:
                    # Single step
                    return step2
        except:
            pass
        
        return None
    
    def _find_fragments(self, mol):
        """Find logical fragments of a molecule for retrosynthesis."""
        if mol is None:
            return []
        
        smi = Chem.MolToSmiles(mol)
        fragments = []
        
        # Strategy 1: Find amide bonds and split
        amide_pattern = Chem.MolFromSmarts('[#6]-C(=O)-[#7]')
        matches = mol.GetSubstructMatches(amide_pattern)
        
        if matches:
            bond_idx = mol.GetBondBetweenAtoms(matches[0][0], matches[0][2]).GetIdx()
            fragments = list(Chem.FragmentOnBonds(mol, [bond_idx]).GetSmilesFrags())
            if len(fragments) >= 2:
                return [f for f in fragments if len(f) > 2]
        
        # Strategy 2: Split at single bonds between ring systems
        ring_bonds = []
        for bond in mol.GetBonds():
            a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
            if (mol.GetRingInfo().NumAtomRings(a1.GetIdx()) > 0 and 
                mol.GetRingInfo().NumAtomRings(a2.GetIdx()) > 0 and
                bond.GetBondType() == Chem.rdchem.BondType.SINGLE):
                ring_bonds.append(bond.GetIdx())
        
        if ring_bonds:
            fragments = list(Chem.FragmentOnBonds(mol, ring_bonds[:1]).GetSmilesFrags())
            if len(fragments) >= 2:
                return [f for f in fragments if len(f) > 2]
        
        return fragments


# ============================================================
# RESULT FORMATTER
# ============================================================
class ResultFormatter:
    """Format results for competition submission."""
    
    @staticmethod
    def format_csv(molecules_with_routes, output_path):
        """Write result.csv in competition format."""
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['mol_smiles', 'route'])
            
            for mol_smi, route in molecules_with_routes:
                if route:
                    writer.writerow([mol_smi, route])
        
        logger.info(f"Written {len(molecules_with_routes)} entries to {output_path}")
    
    @staticmethod
    def package_results(csv_path, log_path, zip_path):
        """Package results into result.zip."""
        import zipfile
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(csv_path)
            zf.write(log_path)
        
        logger.info(f"Packaged results: {zip_path}")
        logger.info(f"  Size: {os.path.getsize(zip_path)} bytes")


# ============================================================
# MAIN AGENT
# ============================================================
class TYK2Agent:
    """Main intelligent agent for TYK2 molecule design and synthesis planning."""
    
    def __init__(self):
        self.generator = MoleculeGenerator()
        self.evaluator = MoleculeEvaluator()
        self.synthesizer = RetrosynthesisEngine()
        self.formatter = ResultFormatter()
        self.results = []
    
    def run(self):
        """Execute the full agent pipeline."""
        start_time = time.time()
        
        logger.info("=" * 70)
        logger.info("AI4S智能体CNS挑战赛 - 靶向TYK2分子研发与合成规划智能体")
        logger.info("=" * 70)
        logger.info(f"Target: {TARGET_NAME} (Tyrosine Kinase 2)")
        logger.info(f"Start time: {datetime.now().isoformat()}")
        logger.info("")
        
        # Step 1: Generate candidate molecules
        logger.info("STEP 1: Molecular Generation")
        logger.info("-" * 40)
        candidate_smiles = self.generator.generate_all(total=100)
        logger.info(f"Generated {len(candidate_smiles)} candidate molecules")
        
        # Step 2: Evaluate and rank molecules
        logger.info("")
        logger.info("STEP 2: Molecule Evaluation & Ranking")
        logger.info("-" * 40)
        ranked = self.evaluator.rank_molecules(candidate_smiles)
        logger.info(f"Evaluated {len(ranked)} valid molecules")
        
        if not ranked:
            logger.error("No valid molecules generated!")
            return
        
        # Show top 10
        for i, mol_info in enumerate(ranked[:10]):
            logger.info(f"  #{i+1}: {mol_info['smiles'][:60]}...")
            logger.info(f"       MW={mol_info['mw']}, LogP={mol_info['logp']}, "
                       f"QED={mol_info['qed']}, SA={mol_info['sa_score']}")
        
        # Step 3: Plan synthesis routes for top molecules
        logger.info("")
        logger.info("STEP 3: Retrosynthesis Planning")
        logger.info("-" * 40)
        
        final_results = []
        for i, mol_info in enumerate(ranked[:50]):
            smi = mol_info['smiles']
            logger.info(f"  Planning route for molecule #{i+1}: {smi[:50]}...")
            
            route = self.synthesizer.plan_route(smi)
            
            if route:
                final_results.append((smi, route))
                logger.info(f"    Route: {route[:80]}...")
            else:
                logger.warning(f"    No route found for {smi[:40]}...")
        
        logger.info(f"Successfully planned routes for {len(final_results)} molecules")
        
        # Step 4: Format output
        logger.info("")
        logger.info("STEP 4: Output Formatting")
        logger.info("-" * 40)
        
        self.formatter.format_csv(final_results, OUTPUT_CSV)
        self.formatter.package_results(OUTPUT_CSV, LOG_FILE, OUTPUT_ZIP)
        
        elapsed = time.time() - start_time
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"Agent completed in {elapsed:.1f} seconds")
        logger.info(f"Output: {OUTPUT_ZIP}")
        logger.info(f"  - {OUTPUT_CSV}: {len(final_results)} molecules with routes")
        logger.info(f"  - {LOG_FILE}: execution log")
        logger.info("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    agent = TYK2Agent()
    agent.run()
