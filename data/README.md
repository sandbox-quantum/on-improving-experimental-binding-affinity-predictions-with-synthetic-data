# Data

Physics-based descriptors and free-energy calculations accompanying the paper
*"On improving experimental binding affinity predictions with synthetic data."*
Both files are zipped CSVs.

| File | Rows | Description |
|------|------|-------------|
| `docking_and_afep_data_sair.csv.zip` | 98,235 | Docking (Gnina/Vina/Vinardo) and absolute free-energy (AQFEP) values for the SAIR co-folded protein–ligand complexes, with experimental affinity labels and co-folding confidence. |
| `PDBBind_AQFEPscore.csv.zip` | 14,984 | AQFEP absolute binding free energies computed on PDBbind complexes (host/guest), with experimental potency/pK. |

The co-folded structures themselves are from SAIR (Lemos et al.); this repository
adds the docking and free-energy calculations described in the paper's Methods.

---

## `docking_and_afep_data_sair.csv`

One row per protein–ligand complex. **AFEP ΔG is available for 81,761 complexes**
(79,143 with a converged bound leg); docking scores are available for essentially
all rows.

### Identifiers and experimental affinity
| Column | Meaning |
|---|---|
| `entry_id`, `IDs` | Complex identifiers (match SAIR). |
| `protein` | UniProt accession. |
| `sequence` | Protein amino-acid sequence. |
| `SMILES_x` | Prepared (protonated, RDKit-standardized) ligand SMILES used for scoring. |
| `srcSMILES` / `SMILES_y` | Original source ligand SMILES. |
| `source` | Affinity data source (ChEMBL, BindingDB). |
| `assay`, `family`, `description` | Assay type, protein family, free-text description. |
| `potency`, `pIC50_mean`, `pIC50_std`, `pIC50_binned` | Experimental potency (−log₁₀ M); mean/std over repeated measurements; coarse bin. |

### Co-folding confidence (Boltz-1x)
`confidence_score` (aggregated pLDDT/ipTM used for quality filtering), `ptm`, `iptm`,
`complex_plddt`, `complex_iplddt`, `chains_ptm`, `complex_pde`, `complex_ipde`,
`interaction_ptm`. `output_path` is the SAIR structure path.

### Docking / empirical scores (Gnina, AutoDock Vina, Vinardo)
| Column | Meaning |
|---|---|
| `minimizedAffinity` | Vinardo affinity (kcal/mol) after local minimization of the co-folded pose (Gnina). |
| `CNNscore`, `CNNaffinity`, `CNN_VS`, `CNNaffinity_variance` | Gnina CNN pose score, predicted affinity, virtual-screening score, and variance. |
| `vina_score`, `vina_score_min`, `vinardo_score` | AutoDock Vina / Vinardo empirical scores. |
| `aevplig_score`, `onionnet_score` | Auxiliary ML scorer outputs (provided for completeness). |

### Absolute free-energy perturbation (AQFEP), kcal/mol
| Column | Meaning |
|---|---|
| `delta_g (kcal/mol)` | Absolute binding free energy (double-decoupling AQFEP). |
| `uncertainty (kcal/mol)` | Statistical uncertainty of `delta_g`. |
| `delta_g_solvation (kcal/mol)`, `solvation_uncertainty (kcal/mol)` | Solvation free energy and uncertainty. |
| `delta_g_bound / _unbound / _vacuum (kcal/mol)` | Free energy of the individual alchemical legs. |
| `bound_converged`, `unbound_converged`, `vacuum_converged` | Per-leg convergence flags; use `bound_converged == True` for the highest-quality AFEP subset. |

### Structure quality control (PoseBusters)
The remaining columns (`mol_pred_loaded`, `sanitization`, `bond_lengths`,
`bond_angles`, `internal_steric_clash`, `number_valid_bonds`, `all_passed`, …) are
PoseBusters validity checks on the co-folded ligand pose, retained for provenance.

---

## `PDBBind_AQFEPscore.csv`

AQFEP absolute binding free energies on PDBbind complexes.

| Column | Meaning |
|---|---|
| `name` | PDBbind complex identifier. |
| `guest_name`, `host_name` | Ligand (guest) and protein (host) identifiers. |
| `delta_g (kcal/mol)`, `uncertainty (kcal/mol)` | AQFEP binding free energy and uncertainty. |
| `delta_g_solvation (kcal/mol)`, `solvation_uncertainty (kcal/mol)` | Solvation free energy and uncertainty. |
| `delta_g_bound / _unbound / _vacuum (kcal/mol)` | Individual alchemical-leg free energies. |
| `bound_converged`, `unbound_converged`, `vacuum_converged` | Per-leg convergence flags. |
| `potency`, `pK` | Experimental potency and −log₁₀ binding constant. |

---

## Methods (summary)

Docking/CNN scores were computed with Gnina on the co-folded poses; Vina/Vinardo are
empirical rescores. AFEP binding free energies were computed with the AQFEP
double-decoupling absolute-FEP protocol (~157,000 NVIDIA T4 GPU-hours in total),
retaining only complexes that satisfied the convergence criteria. See the paper's
Methods for full details.
