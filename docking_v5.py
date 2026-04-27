#!/usr/bin/env python3.12
"""
AutoDock Vina 完整对接流程 (修复版)
"""

import os, subprocess, csv, time
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom
from meeko import MoleculePreparation, PDBQTWriterLegacy

# ============================================================
# 配置
# ============================================================
VINA_BIN = "./vina"
RECEPTOR_PDBQT = "tyk2_minimal.pdbqt"
OUTPUT_DIR = "docking_results"
CENTER_X = 18.9
CENTER_Y = 16.8
CENTER_Z = 12.1
BOX_SIZE = 20.0

# ============================================================
# 配体准备
# ============================================================
def prepare_ligand(smiles, output_file):
    """使用meeko准备配体"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    mol = Chem.AddHs(mol)
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 42
    rdDistGeom.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol)
    
    preparator = MoleculePreparation()
    mol_setups = preparator.prepare(mol)
    
    if not mol_setups:
        return None
    
    pdbqt_string, is_ok, err_msg = PDBQTWriterLegacy.write_string(mol_setups[0])
    
    if not is_ok:
        return None
    
    with open(output_file, 'w') as f:
        f.write(pdbqt_string)
    
    return output_file

# ============================================================
# 分子对接
# ============================================================
def run_docking(ligand_pdbqt, output_pdbqt):
    """运行AutoDock Vina对接"""
    cmd = [
        VINA_BIN,
        "--receptor", RECEPTOR_PDBQT,
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        
        if result.returncode == 0:
            # 解析输出获取结合能
            for line in result.stdout.split('\n'):
                # 检查模式行 (格式: "   1       -8.484          0          0")
                if line.strip().startswith('1') and '-' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            score = float(parts[1])
                            return score
                        except:
                            pass
            return None
        else:
            return None
            
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        return None

# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 70)
    print("AutoDock Vina Molecular Docking Pipeline (Fixed)")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
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
    for i, (smi, route) in enumerate(molecules[:30]):  # 对接30个分子
        print(f"\n[{i+1}/{min(30, len(molecules))}] Processing: {smi[:50]}...")
        
        ligand_file = os.path.join(OUTPUT_DIR, f"ligand_{i:03d}.pdbqt")
        output_file = os.path.join(OUTPUT_DIR, f"docked_{i:03d}.pdbqt")
        
        if prepare_ligand(smi, ligand_file):
            score = run_docking(ligand_file, output_file)
            
            if score is not None:
                results.append({
                    'smiles': smi,
                    'route': route,
                    'binding_score': score,
                    'index': i
                })
                print(f"  Binding affinity: {score:.2f} kcal/mol")
            else:
                print(f"  Docking failed")
        else:
            print(f"  Ligand preparation failed")
    
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
