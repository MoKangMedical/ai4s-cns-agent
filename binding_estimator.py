#!/usr/bin/env python3.12
"""
基于分子描述符和药效团的结合亲和力估算
替代AutoDock Vina的分子对接
"""

import os, sys, csv, json, logging
from pathlib import Path
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdDistGeom, QED, rdMolDescriptors
from rdkit.Chem import Draw, rdchem

# ============================================================
# TYK2靶点信息
# ============================================================
# TYK2活性位点特征 (基于已知抑制剂)
TYK2_FEATURES = {
    'hinge_region': {
        'description': 'Hinge region (ATP binding)',
        'key_interactions': ['H-bond donor', 'H-bond acceptor'],
        'preferred_groups': ['benzimidazole', 'quinazoline', 'pyrazolopyrimidine', 'pyrrolopyrimidine'],
    },
    'hydrophobic_pocket': {
        'description': 'Hydrophobic pocket',
        'key_interactions': ['Hydrophobic'],
        'preferred_groups': ['phenyl', 'fluorophenyl', 'chlorophenyl', 'trifluoromethyl'],
    },
    'solvent_front': {
        'description': 'Solvent-exposed region',
        'key_interactions': ['H-bond', 'Polar'],
        'preferred_groups': ['piperazine', 'morpholine', 'amide', 'sulfonamide'],
    }
}

# 已知TYK2抑制剂的结合能 (kcal/mol, 越低越好)
KNOWN_INHIBITORS = {
    'Deucravacitinib': -9.5,
    'BMS-986165': -9.5,
    'Tofacitinib': -8.5,
    'Baricitinib': -9.0,
    'Upadacitinib': -9.2,
}


# ============================================================
# 结合亲和力估算函数
# ============================================================
def estimate_binding_affinity(smiles):
    """
    估算分子与TYK2的结合亲和力
    返回: 结合能 (kcal/mol, 越低越好)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    
    # 1. 基础分子描述符
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rotbonds = Descriptors.NumRotatableBonds(mol)
    
    # 2. 基础分数 (基于药物相似性)
    base_score = 0.0
    
    # 分子量惩罚
    if mw < 200:
        base_score += 2.0  # 太小
    elif mw > 600:
        base_score += 1.5  # 太大
    elif 350 <= mw <= 550:
        base_score -= 1.0  # 最优范围
    
    # LogP惩罚
    if logp < 0:
        base_score += 1.0  # 太亲水
    elif logp > 5:
        base_score += 1.5  # 太疏水
    elif 2 <= logp <= 4:
        base_score -= 0.5  # 最优范围
    
    # TPSA惩罚
    if tpsa < 40:
        base_score += 0.5
    elif tpsa > 140:
        base_score += 1.0
    elif 60 <= tpsa <= 120:
        base_score -= 0.5
    
    # 3. 药效团特征匹配
    pharmacophore_score = 0.0
    
    # Hinge binder检测 (关键!)
    hinge_patterns = {
        'benzimidazole': ['[nH]cnc', 'c1ccc2[nH]cnc2c1'],
        'quinazoline': ['ncnc', 'c1ccc2ncnc(N)c2c1'],
        'pyrazolopyrimidine': ['ncn', 'c1cc2ncnc(N)c2n1'],
        'pyrrolopyrimidine': ['c1ccnc', 'Nc1ccnc2c(cnn12)'],
    }
    
    hinge_found = False
    for group_name, patterns in hinge_patterns.items():
        for pattern in patterns:
            if pattern in smiles:
                pharmacophore_score -= 2.0  # 强hinge binder
                hinge_found = True
                break
        if hinge_found:
            break
    
    if not hinge_found:
        pharmacophore_score += 1.0  # 没有hinge binder
    
    # 疏水基团检测
    hydrophobic_groups = ['F', 'Cl', 'C(F)(F)F', 'c1ccccc1']
    for group in hydrophobic_groups:
        if group in smiles:
            pharmacophore_score -= 0.5
    
    # 溶剂暴露基团检测
    solvent_groups = ['CCNCC', 'CCOCC', 'CCN(C)CC', 'C(=O)N', 'S(=O)(=O)']
    for group in solvent_groups:
        if group in smiles:
            pharmacophore_score -= 0.3
    
    # 4. 分子复杂度惩罚
    complexity_score = 0.0
    
    # 旋转键惩罚
    if rotbonds > 10:
        complexity_score += 1.0
    elif rotbonds > 8:
        complexity_score += 0.5
    
    # 环数量惩罚
    ring_count = mol.GetRingInfo().NumRings()
    if ring_count > 5:
        complexity_score += 0.5
    
    # 5. 综合分数
    total_score = base_score + pharmacophore_score + complexity_score
    
    # 转换为结合能 (kcal/mol)
    # 基准: 已知TYK2抑制剂约-8到-10 kcal/mol
    binding_energy = -8.0 + total_score
    
    # 限制范围
    binding_energy = max(-12.0, min(-4.0, binding_energy))
    
    return round(binding_energy, 2)


def calculate_binding_score(binding_energy):
    """
    将结合能转换为0-1的分数
    越低的结合能 -> 越高的分数
    """
    # 已知TYK2抑制剂: -8到-10 kcal/mol
    # 我们假设:
    # -10 kcal/mol -> 1.0 (完美)
    # -8 kcal/mol -> 0.7 (良好)
    # -6 kcal/mol -> 0.3 (一般)
    # -4 kcal/mol -> 0.0 (差)
    
    if binding_energy >= -4.0:
        return 0.0
    elif binding_energy <= -10.0:
        return 1.0
    else:
        # 线性插值
        score = (binding_energy - (-4.0)) / ((-10.0) - (-4.0))
        return round(max(0.0, min(1.0, score)), 4)


# ============================================================
# SA Score计算
# ============================================================
def calc_sa_score(mol):
    """合成可及性评分 (1=易, 10=难)"""
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
# 测试函数
# ============================================================
def test_scoring():
    """测试打分函数"""
    print("=" * 70)
    print("Testing Binding Affinity Estimation")
    print("=" * 70)
    
    test_molecules = [
        ("Deucravacitinib", "C#CC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC=C4F)C(=O)N3C5CC5"),
        ("Benzimidazole-1", "O=C(Nc1ccc2[nH]cnc2c1)c1ccc(F)cc1N1CCN(C)CC1"),
        ("Quinazoline-1", "Nc1ncnc2cc(Nc3ccc(F)cc3)ccc12"),
        ("Simple-1", "c1ccccc1"),
        ("Complex-1", "C#CC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC=C4F)C(=O)N3c1ccc(F)cc1"),
    ]
    
    for name, smi in test_molecules:
        binding_energy = estimate_binding_affinity(smi)
        binding_score = calculate_binding_score(binding_energy)
        
        mol = Chem.MolFromSmiles(smi)
        mw = Descriptors.MolWt(mol)
        qed = QED.qed(mol)
        sa = calc_sa_score(mol)
        
        print(f"\n{name}:")
        print(f"  SMILES: {smi[:50]}...")
        print(f"  MW: {mw:.1f}")
        print(f"  QED: {qed:.4f}")
        print(f"  SA: {sa:.2f}")
        print(f"  Binding Energy: {binding_energy:.2f} kcal/mol")
        print(f"  Binding Score: {binding_score:.4f}")


if __name__ == "__main__":
    test_scoring()
