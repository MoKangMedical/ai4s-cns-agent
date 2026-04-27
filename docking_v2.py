#!/usr/bin/env python3.12
"""
AutoDock Vina 分子对接流程 (使用meeko准备配体)
"""

import os, sys, subprocess, csv, json, logging
from pathlib import Path
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdDistGeom
from meeko import MoleculePreparation, PDBQTWriterLegacy

# ============================================================
# 配置
# ============================================================
VINA_BIN = "./vina"
TARGET_PDB = "target.pdb"
TARGET_PDBQT = "target.pdbqt"
OUTPUT_DIR = "docking_results"

# 对接盒子参数 (TYK2活性位点)
CENTER_X = 25.0
CENTER_Y = 15.0
CENTER_Z = 30.0
BOX_SIZE = 25.0

# ============================================================
# 蛋白准备 (简化版本)
# ============================================================
def prepare_protein(pdb_file, pdbqt_file):
    """将PDB转换为PDBQT格式"""
    print(f"Preparing protein: {pdb_file} -> {pdbqt_file}")
    
    with open(pdb_file, 'r') as f:
        lines = f.readlines()
    
    pdbqt_lines = []
    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain = line[21]
            res_num = line[22:26].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            element = line[76:78].strip()
            
            # 简化的原子类型和电荷
            atom_type = element[0] if element else 'C'
            charge = 0.0
            
            pdbqt_line = f"ATOM  {line[6:11]} {atom_name:<4} {res_name:<3} {chain}{res_num:>4}    {x:8.3f}{y:8.3f}{z:8.3f}{charge:6.2f} {atom_type:<2}"
            pdbqt_lines.append(pdbqt_line)
        elif line.startswith('TER'):
            pdbqt_lines.append("TER")
        elif line.startswith('END'):
            pdbqt_lines.append("END")
    
    with open(pdbqt_file, 'w') as f:
        f.write('\n'.join(pdbqt_lines))
    
    print(f"  Protein prepared: {len(pdbqt_lines)} atoms")
    return True


# ============================================================
# 配体准备 (使用meeko)
# ============================================================
def prepare_ligand_meeko(smiles, output_file):
    """使用meeko准备配体"""
    print(f"Preparing ligand: {smiles[:50]}...")
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"  ERROR: Invalid SMILES")
        return None
    
    # 添加氢原子
    mol = Chem.AddHs(mol)
    
    # 生成3D构象
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 42
    rdDistGeom.EmbedMolecule(mol, params)
    
    # 优化构象
    AllChem.MMFFOptimizeMolecule(mol)
    
    # 使用meeko准备PDBQT
    preparator = MoleculePreparation()
    mol_setups = preparator.prepare(mol)
    
    if not mol_setups:
        print(f"  ERROR: Meeko preparation failed")
        return None
    
    # 写入PDBQT文件
    pdbqt_string, is_ok, err_msg = PDBQTWriterLegacy.write_string(mol_setups[0])
    
    if not is_ok:
        print(f"  ERROR: PDBQT writing failed: {err_msg}")
        return None
    
    with open(output_file, 'w') as f:
        f.write(pdbqt_string)
    
    print(f"  Ligand prepared: {output_file}")
    return output_file


# ============================================================
# 分子对接
# ============================================================
def run_docking(ligand_pdbqt, output_pdbqt):
    """运行AutoDock Vina对接"""
    print(f"Running docking: {ligand_pdbqt}")
    
    cmd = [
        VINA_BIN,
        "--receptor", TARGET_PDBQT,
        "--ligand", ligand_pdbqt,
        "--center_x", str(CENTER_X),
        "--center_y", str(CENTER_Y),
        "--center_z", str(CENTER_Z),
        "--size_x", str(BOX_SIZE),
        "--size_y", str(BOX_SIZE),
        "--size_z", str(BOX_SIZE),
        "--out", output_pdbqt,
        "--num_modes", "1",
        "--exhaustiveness", "8"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            # 解析输出获取结合能
            for line in result.stdout.split('\n'):
                if 'REMARK VINA RESULT' in line:
                    score = float(line.split()[3])
                    print(f"  Binding affinity: {score:.2f} kcal/mol")
                    return score
            
            # 从输出文件解析
            if os.path.exists(output_pdbqt):
                with open(output_pdbqt, 'r') as f:
                    for line in f:
                        if 'REMARK VINA RESULT' in line:
                            score = float(line.split()[3])
                            print(f"  Binding affinity: {score:.2f} kcal/mol")
                            return score
            
            print(f"  WARNING: No score found")
            return None
        else:
            print(f"  ERROR: {result.stderr[:200]}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"  ERROR: Docking timed out")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 70)
    print("AutoDock Vina Molecular Docking Pipeline (with Meeko)")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 准备蛋白
    if not os.path.exists(TARGET_PDBQT):
        prepare_protein(TARGET_PDB, TARGET_PDBQT)
    
    # 读取分子列表
    molecules = []
    with open('result.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                molecules.append((row[0], row[1]))
    
    print(f"\nTotal molecules to dock: {len(molecules)}")
    
    # 对接分子
    results = []
    for i, (smi, route) in enumerate(molecules[:10]):  # 先测试10个
        print(f"\n[{i+1}/{min(10, len(molecules))}] Processing: {smi[:50]}...")
        
        ligand_file = os.path.join(OUTPUT_DIR, f"ligand_{i:03d}.pdbqt")
        output_file = os.path.join(OUTPUT_DIR, f"docked_{i:03d}.pdbqt")
        
        if prepare_ligand_meeko(smi, ligand_file):
            score = run_docking(ligand_file, output_file)
            
            if score is not None:
                results.append({
                    'smiles': smi,
                    'route': route,
                    'binding_score': score,
                    'index': i
                })
    
    # 按结合能排序
    results.sort(key=lambda x: x['binding_score'])
    
    # 保存结果
    print("\n" + "=" * 70)
    print("Top molecules by binding affinity:")
    print("=" * 70)
    
    for i, r in enumerate(results[:10]):
        print(f"  #{i+1}: {r['binding_score']:.2f} kcal/mol, {r['smiles'][:50]}...")
    
    # 保存到CSV
    output_csv = os.path.join(OUTPUT_DIR, "docking_results.csv")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['smiles', 'route', 'binding_score'])
        for r in results:
            writer.writerow([r['smiles'], r['route'], r['binding_score']])
    
    print(f"\nResults saved to: {output_csv}")
    print(f"Total molecules docked: {len(results)}")
    
    return results


if __name__ == "__main__":
    main()
