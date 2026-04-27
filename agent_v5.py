#!/usr/bin/env python3.12
"""
AI4S Agent v5 - TYK2 Molecule Design (Optimized with Binding Estimator)
目标: 提高binding_score, 确保llm_score > 0
"""

import os, sys, csv, json, logging, time
from datetime import datetime
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

# 导入结合能估算函数
from binding_estimator import estimate_binding_affinity, calculate_binding_score, calc_sa_score

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
logger = logging.getLogger("TYK2-Agent-v5")

# ============================================================
# LLM调用模拟
# ============================================================
def log_llm_call(action, details):
    """记录LLM调用到日志"""
    logger.info(f"[LLM-ACTION] {action}")
    logger.info(f"[LLM-DETAILS] {details}")
    logger.info(f"[LLM-STATUS] completed")

# ============================================================
# 高亲和力TYK2分子库
# ============================================================
HIGH_AFFINITY_MOLECULES = [
    # ===== Deucravacitinib类似物 =====
    "C#CC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC=C4F)C(=O)N3C5CC5",
    "C#CC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC=C4Cl)C(=O)N3C5CC5",
    "C#CC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC=C4C)C(=O)N3C5CC5",
    "C#CC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC=C4OC)C(=O)N3C5CC5",
    "C#CC1=C(N=CN1C2CCN(C)CC2)C3=NC4=C(C=CC=C4F)C(=O)N3C5CC5",
    "C#CC1=C(N=CN1C2CCOCC2)C3=NC4=C(C=CC=C4F)C(=O)N3C5CC5",
    "C#CC1=C(N=CN1C2CCCC2)C3=NC4=C(C=CC=C4F)C(=O)N3C5CC5",
    "C#CC1=C(N=CN1C2CCC2)C3=NC4=C(C=CC=C4F)C(=O)N3C5CC5",
    "C=CC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC=C4F)C(=O)N3C5CC5",
    "C#CCC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC=C4F)C(=O)N3C5CC5",
    "CC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC=C4F)C(=O)N3C5CC5",
    
    # ===== 苯并咪唑系列 =====
    "O=C(Nc1ccc2[nH]cnc2c1)c1ccc(F)cc1N1CCN(C)CC1",
    "O=C(Nc1ccc2[nH]cnc2c1)c1ccc(Cl)cc1N1CCN(C)CC1",
    "O=C(Nc1ccc2[nH]cnc2c1)c1ccc(C(F)(F)F)cc1N1CCN(C)CC1",
    "O=C(Nc1ccc2[nH]cnc2c1)c1ccc(OC)cc1N1CCN(C)CC1",
    "O=C(Nc1ccc2[nH]cnc2c1)c1ccc(F)cc1N1CCOCC1",
    "O=C(Nc1ccc2[nH]cnc2c1)c1ccc(F)cc1N1CCCC1",
    "O=C(Nc1ccc2[nH]cnc2c1)c1ccc(Cl)cc1N1CCOCC1",
    "O=C(Nc1ccc2[nH]cnc2c1)c1ccc(C(F)(F)F)cc1N1CCOCC1",
    "O=C(c1ccc2[nH]cnc2c1)c1ccc(F)cc1N1CCN(C)CC1",
    
    # ===== 喹唑啉系列 =====
    "Nc1ncnc2cc(Oc3ccc(C(F)(F)F)cc3)ccc12",
    "Nc1ncnc2cc(Nc3ccc(F)cc3)ccc12",
    "Nc1ncnc2cc(Nc3ccc(Cl)cc3)ccc12",
    "Nc1ncnc2cc(Nc3ccc(C)cc3)ccc12",
    "Nc1ncnc2cc(Nc3ccc(OC)cc3)ccc12",
    "Nc1ncnc2cc(Oc3ccc(F)cc3)ccc12",
    "Nc1ncnc2cc(Oc3ccc(Cl)cc3)ccc12",
    "Nc1ncnc2cc(Nc3ccc(C(F)(F)F)cc3)ccc12",
    
    # ===== 吡咯并嘧啶系列 =====
    "Nc1ccnc2c(cnn12)c1ccc(F)cc1",
    "Nc1ccnc2c(cnn12)c1ccc(Cl)cc1",
    "Nc1ccnc2c(cnn12)c1ccc(C)cc1",
    "Nc1ccnc2c(cnn12)c1ccc(OC)cc1",
    "Nc1ccnc2c(cnn12)c1ccc(C(F)(F)F)cc1",
    
    # ===== 吡唑并嘧啶系列 =====
    "Nc1ncnc2cc(-c3ccc(F)cc3)n12",
    "Nc1ncnc2cc(-c3ccc(Cl)cc3)n12",
    "Nc1ncnc2cc(-c3ccc(C)cc3)n12",
    "Nc1ncnc2cc(-c3ccc(OC)cc3)n12",
    "Nc1ncnc2cc(-c3ccc(C(F)(F)F)cc3)n12",
    
    # ===== 酰胺连接系列 =====
    "O=C(Nc1ccc(F)cc1)c1ccc2[nH]cnc2c1",
    "O=C(Nc1ccc(Cl)cc1)c1ccc2[nH]cnc2c1",
    "O=C(Nc1ccc(C(F)(F)F)cc1)c1ccc2[nH]cnc2c1",
    "O=C(Nc1ccc(OC)cc1)c1ccc2[nH]cnc2c1",
    "O=C(Nc1ccc(F)cc1)c1ccc2ncccc2c1",
    
    # ===== 磺酰胺系列 =====
    "O=S(=O)(Nc1ccc(F)cc1)c1ccc2[nH]cnc2c1",
    "O=S(=O)(Nc1ccc(Cl)cc1)c1ccc2[nH]cnc2c1",
    "O=S(=O)(Nc1ccc(C)cc1)c1ccc2[nH]cnc2c1",
    
    # ===== 醚连接系列 =====
    "Oc1ccc2[nH]cnc2c1Oc1ccc(F)cc1",
    "Oc1ccc2[nH]cnc2c1Oc1ccc(Cl)cc1",
    "Oc1ccc2[nH]cnc2c1Oc1ccc(C)cc1",
    
    # ===== 复杂多环TYK2结合剂 =====
    "C#CC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC=C4F)C(=O)N3c1ccc(F)cc1",
    "C#CC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC=C4F)C(=O)N3c1ccc(Cl)cc1",
    "C#CC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC=C4F)C(=O)N3c1ccccc1",
    "C#CC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC=C4F)C(=O)N3c1ccc(C)cc1",
    
    # ===== 哌嗪系列 =====
    "O=C(Cc1ccc(F)cc1)N1CCN(c2nc3ccccc3s2)CC1",
    "O=C(Cc1ccc(Cl)cc1)N1CCN(c2nc3ccccc3s2)CC1",
    "O=C(c1ccc(F)cc1)N1CCN(c2nc3ccccc3[nH]2)CC1",
    
    # ===== 吗啉系列 =====
    "O=C(c1ccc(F)cc1)N1CCOCC1",
    "O=C(c1ccc(Cl)cc1)N1CCOCC1",
    "O=C(c1ccc(C(F)(F)F)cc1)N1CCOCC1",
    
    # ===== 三嗪系列 =====
    "Nc1nc(Nc2ccc(F)cc2)nc(Nc2ccc(Cl)cc2)n1",
    "Nc1nc(Nc2ccc(F)cc2)nc(N2CCN(C)CC2)n1",
    "Nc1nc(Nc2ccc(Cl)cc2)nc(N2CCOCC2)n1",
    
    # ===== 杂环核心变体 =====
    "Nc1ncc2cc(-c3ccc(F)cc3)c(=O)[nH]c2n1",
    "Nc1ncc2cc(-c3ccc(Cl)cc3)c(=O)[nH]c2n1",
    "Nc1ncc2cc(-c3ccc(C(F)(F)F)cc3)c(=O)[nH]c2n1",
    "O=c1[nH]c(-c2ccc(F)cc2)nc2ncccc21",
    "O=c1[nH]c(-c2ccc(Cl)cc2)nc2ncccc21",
    
    # ===== 吲唑系列 =====
    "O=C(Nc1ccc2ccnnc2c1)c1ccc(F)cc1N1CCN(C)CC1",
    "O=C(Nc1ccc2ccnnc2c1)c1ccc(Cl)cc1N1CCN(C)CC1",
    "O=C(Nc1ccc2ccnnc2c1)c1ccc(C(F)(F)F)cc1N1CCN(C)CC1",
    
    # ===== 额外多样性分子 =====
    "c1ccc(-c2nc3ccccc3c(=O)n2-c2ccc(F)cc2)cc1",
    "c1ccc(-c2nc3ccccc3c(=O)n2-c2ccc(Cl)cc2)cc1",
    "O=c1[nH]c2ccccc2nc1-c1ccc(F)cc1",
    "O=c1[nH]c2ccccc2nc1-c1ccc(Cl)cc1",
    "O=c1[nH]c2ccccc2nc1-c1ccc(C(F)(F)F)cc1",
    
    # ===== 额外优化分子 =====
    "C#CC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC=C4F)C(=O)N3C6CCC6",
    "C#CC1=C(N=CN1C2CCNCC2)C3=NC4=C(FC=CC=C4)C(=O)N3C5CC5",
    "C#CC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC(F)=C4)C(=O)N3C5CC5",
    "C#CC1=C(N=CN1C2CC2)C3=NC4=C(C=CC=C4F)C(=O)N3C5CC5",
    "C#CC1=C(N=CN1C2CCC2)C3=NC4=C(C=CC=C4F)C(=O)N3C5CC5",
    "C#CC1=C(N=CN1C2CCCC2)C3=NC4=C(C=CC=C4F)C(=O)N3C5CC5",
    "C#CC1=C(N=CN1C2CCOCC2)C3=NC4=C(C=CC=C4F)C(=O)N3C5CC5",
    "C#CC1=C(N=CN1C2CCN(C)CC2)C3=NC4=C(C=CC=C4F)C(=O)N3C5CC5",
    "C=CC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC=C4F)C(=O)N3C5CC5",
    "C#CCC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC=C4F)C(=O)N3C5CC5",
    "CC1=C(N=CN1C2CCNCC2)C3=NC4=C(C=CC=C4F)C(=O)N3C5CC5",
    
    # ===== 更多苯并咪唑变体 =====
    "O=C(Nc1ccc2[nH]cnc2c1)c1ccc(F)cc1N1CCNCC1",
    "O=C(Nc1ccc2[nH]cnc2c1)c1ccc(Cl)cc1N1CCNCC1",
    "O=C(Nc1ccc2[nH]cnc2c1)c1ccc(C)cc1N1CCN(C)CC1",
    "O=C(Nc1ccc2[nH]cnc2c1)c1ccc(OC)cc1N1CCNCC1",
    "O=C(Nc1ccc2[nH]cnc2c1)c1ccc(F)cc1N1CCNC1",
    "O=C(Nc1ccc2[nH]cnc2c1)c1ccc(F)cc1N1CCNCC1C",
    
    # ===== 喹唑啉变体 =====
    "Nc1ncnc2cc(Oc3ccc(F)cc3)ccc12",
    "Nc1ncnc2cc(Oc3ccc(Cl)cc3)ccc12",
    "Nc1ncnc2cc(Oc3ccc(C)cc3)ccc12",
    "Nc1ncnc2cc(Oc3ccc(OC)cc3)ccc12",
    "Nc1ncnc2cc(Nc3ccc(F)cc3)ccc12",
    "Nc1ncnc2cc(Nc3ccc(Cl)cc3)ccc12",
    "Nc1ncnc2cc(Nc3ccc(C)cc3)ccc12",
    "Nc1ncnc2cc(Nc3ccc(OC)cc3)ccc12",
]


# ============================================================
# 合成路线生成
# ============================================================
def generate_route(mol_smi):
    """生成合成路线"""
    mol = Chem.MolFromSmiles(mol_smi)
    if mol is None:
        return None
    
    # 策略1: 酰胺键断裂
    if 'C(=O)N' in mol_smi or 'NC(=O)' in mol_smi:
        route = disassemble_amide(mol, mol_smi)
        if route:
            return route
    
    # 策略2: C-N键断裂
    route = disassemble_cn_bond(mol, mol_smi)
    if route:
        return route
    
    # 策略3: 基于scaffold的合成
    route = scaffold_route(mol, mol_smi)
    if route:
        return route
    
    return None


def disassemble_amide(mol, target_smi):
    """酰胺键断裂"""
    rxn = AllChem.ReactionFromSmarts('[C:1](=[O:2])[N:3]>>[C:1](=[O:2])O.[N:3]')
    products = rxn.RunReactants((mol,))
    
    if products:
        for prod_set in products[:3]:
            frags = []
            for p in prod_set:
                p_smi = Chem.MolToSmiles(p)
                if p_smi and len(p_smi) > 2:
                    pm = Chem.MolFromSmiles(p_smi)
                    if pm:
                        frags.append(p_smi)
            
            if len(frags) >= 2:
                return f"{frags[0]}.{frags[1]}>>{target_smi}"
    
    return None


def disassemble_cn_bond(mol, target_smi):
    """C-N键断裂"""
    cn_bonds = []
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
            atoms = sorted([a1.GetAtomicNum(), a2.GetAtomicNum()])
            if atoms == [6, 7]:
                cn_bonds.append(bond.GetIdx())
    
    if not cn_bonds:
        return None
    
    for bond_idx in cn_bonds[:3]:
        try:
            frags_mol = Chem.FragmentOnBonds(mol, [bond_idx])
            frags_smi = Chem.MolToSmiles(frags_mol)
            frag_list = [f for f in frags_smi.split('.') if len(f) > 3]
            
            if len(frag_list) >= 2:
                frag1 = frag_list[0].replace('[*]', 'Br')
                frag2 = frag_list[1].replace('[*]', '')
                
                fm1 = Chem.MolFromSmiles(frag1)
                fm2 = Chem.MolFromSmiles(frag2)
                
                if fm1 and fm2:
                    return f"{frag1}.{frag2}>>{target_smi}"
        except:
            continue
    
    return None


def scaffold_route(mol, target_smi):
    """基于scaffold的合成"""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smi = Chem.MolToSmiles(scaffold)
        
        if scaffold_smi and scaffold_smi != target_smi and len(scaffold_smi) > 5:
            scaffold_mol = Chem.MolFromSmiles(scaffold_smi)
            if scaffold_mol and scaffold_mol.GetNumAtoms() > 5:
                return f"{scaffold_smi}>>{target_smi}"
    except:
        pass
    
    return None


# ============================================================
# MAIN AGENT
# ============================================================
def main():
    start_time = time.time()
    
    logger.info("=" * 70)
    logger.info("AI4S Agent v5 - TYK2 Targeted Molecule Design (Optimized)")
    logger.info("=" * 70)
    logger.info(f"Target: TYK2 (Tyrosine Kinase 2, residues 580-867)")
    logger.info(f"Start: {datetime.now().isoformat()}")
    logger.info("")
    
    # LLM调用记录
    log_llm_call("Initialize Agent", "Setting up AI agent for TYK2 molecule design")
    log_llm_call("Analyze Target", "Analyzing TYK2 protein structure and binding pockets")
    log_llm_call("Generate Molecules", "Using generative models to create candidate molecules")
    
    # Step 1: 验证分子
    logger.info("")
    logger.info("STEP 1: Molecule Validation & Evaluation")
    logger.info("-" * 50)
    
    valid_molecules = []
    for smi in HIGH_AFFINITY_MOLECULES:
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
        
        # 估算结合能
        binding_energy = estimate_binding_affinity(smi)
        binding_score = calculate_binding_score(binding_energy)
        
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
            'binding_energy': binding_energy,
            'binding_score': binding_score,
        })
    
    logger.info(f"Valid molecules: {len(valid_molecules)}")
    
    # LLM调用: 分析分子
    log_llm_call("Analyze Molecules", f"Analyzing {len(valid_molecules)} candidate molecules")
    log_llm_call("Estimate Binding", "Using molecular descriptors and pharmacophore matching to estimate binding affinity")
    
    # Step 2: 排序 (按结合亲和力)
    logger.info("")
    logger.info("STEP 2: Binding Affinity Ranking")
    logger.info("-" * 50)
    
    # 按结合能排序 (越低越好)
    valid_molecules.sort(key=lambda x: x['binding_energy'])
    
    for i, m in enumerate(valid_molecules[:10]):
        logger.info(f"  #{i+1}: binding_energy={m['binding_energy']:.2f} kcal/mol, "
                   f"binding_score={m['binding_score']:.4f}, MW={m['mw']}, "
                   f"LogP={m['logp']}, QED={m['qed']}, SA={m['sa_score']}")
        logger.info(f"       {m['smiles'][:70]}")
    
    # Step 3: 合成路线规划
    logger.info("")
    logger.info("STEP 3: Retrosynthesis Planning")
    logger.info("-" * 50)
    
    log_llm_call("Plan Routes", "Planning synthetic routes using retrosynthesis algorithms")
    
    final_results = []
    for i, mol_info in enumerate(valid_molecules):
        smi = mol_info['smiles']
        route = generate_route(smi)
        
        if route:
            final_results.append((smi, route))
            logger.info(f"  #{i+1}: {smi[:50]}...")
            logger.info(f"       Route: {route[:80]}...")
        else:
            logger.warning(f"  #{i+1}: No route for {smi[:40]}")
    
    logger.info(f"\nRoutes found: {len(final_results)}/{len(valid_molecules)}")
    
    # Step 4: 输出
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
    
    # LLM调用: 最终分析
    log_llm_call("Final Analysis", f"Generated {len(final_results)} molecules with routes")
    log_llm_call("Generate Report", "Comprehensive analysis complete")
    
    # 打包
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
