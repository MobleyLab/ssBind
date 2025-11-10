# ssBind: Sub-Structure-based alternative BINDing modes generator for protein-ligand systems


ssBind offers different methods for generating multiple conformations by restraining certain sub-structures. This feature enables researchers to systematically explore the effects of various substituents attached to a common scaffold on the binding and to refine the interaction patterns between ligand molecules and their protein targets.

## Acknowledgement
I would like to thank authors of PLANTS molecular docking software for making PLANTS free for academic use.

## Installation (from source)

```bash
$ git clone https://github.com/suleymanselim/ssBind
$ cd ssBind
$ pip install .
```

## Available options

To see all command-line option, run ``run_ssbind.py --help``

### Conformer generation and docking

* Random conformer generation with RDKit 
* Dihedral angle sampling
* PLANTS docking
* rDock docking
* AutoDock-GPU docking

#### Minimization and Scoring Options
* Scoring and minimization with smina, Gromacs and OpenMM
* local minimization of input structures (ligands) with smina, rDock or PLANTS (``--minimize local``)

### Clustering and pose selection options
* Butina or HDBSCAN clustering
* selection based on docking score, shape similarity to input, or mixed

## Examples using the command line scripts



#### 1. Random conformational sampling using RDKit
RDKit (and angle sampling) require minimization to obtain scores (with smina, or alteratively Gromacs/OpenMM)
For smina, the ``smina`` or ``smina.static`` executable needs to be in PATH (``smina.static`` is included in ``util``).

```console
run_ssBind.py --reference reference.mol2 --ligand ligand.mol2 --receptor receptor.pdb --generator rdkit --minimize smina
```
#### 2. Generating conformers using PLANTS docking tool
PLANTS allows the restraint of the position of a ring system or a single non-ring atom in docking. In this case, all the fixed scaffold's translational and rotational degrees of freedom are completely neglected. The code automatically determines the ring system to be fixed in the reference scaffold. If there is no ring system in the reference, only the specific atom is restrained.

PLANTS can only be used on Linux, and the ``PLANTS`` and ``SPORES`` executables included in ``utils`` and need to be made executeable.

```console
run_ssBind.py --reference reference.mol2 --ligand ligand.mol2 --receptor receptor.pdb --generator plants 

```
Some side-chains can also be treated flexibly with PLANTS.
```console
## A subset of the sidechains around the ligand within 5 Å in the binding pocket will be allowed to move.
run_ssBind.py --reference reference.mol2 --ligand ligand.mol2 --receptor receptor.pdb --generator plants --flexDist 5

## You can also determine the flexible sidechains
run_ssBind.py --reference reference.mol2 --ligand ligand.mol2 --receptor receptor.pdb --generator plants --flexList "MET49,MET165"

```
#### 3. Tethered scaffold docking with rDock
In tethered scaffold docking, the ligand position is constrained, ensuring they align with the substructure coordinates of a reference ligand. This involves overlaying a ligand with a corresponding substructure onto the coordinates of the reference substructure. The ligand's degrees of freedom are anchored to their predefined reference position. rDock facilitates this by offering the flexibility to separately adjust the ligand's position, orientation, and dihedral angles.
```console
run_ssBind.py --reference reference.mol2 --ligand ligand.mol2 --receptor receptor.pdb --generator rdock 
```
rDock needs to be installed separately (compiled from [source](https://github.com/CBDD/rDock)) and the executables needs to be in PATH.

#### 4. Tethered docking with Autodock
Autodock restrains the atoms from the substructure by adding a steep linear bias around their reference positions. 
Autodock needs to compiled from [source](https://github.com/ccsb-scripps/AutoDock-GPU), and renamed to an executable ``autodock`` included in PATH.
```console
run_ssBind.py --reference reference.mol2 --ligand ligand.mol2 --receptor receptor.pdb --generator autodock 
```

## Python tutorial

#### 1. Generating conformers using PLANTS

```python
from ssBind import SSBIND
from ssBind.io import MolFromInput

## Input files
reference_substructure = MolFromInput('reference.mol2')
query_molecule = MolFromInput('ligand.mol2')
receptor_file = 'receptor.pdb'

ssbind = SSBIND(reference_substructure = reference_substructure, query_molecule = query_molecule, \
    receptor_file = receptor_file, generator = "plannts", outputPoses = 10)

## PLANTS generates many conformers 'conformers.sdf' and their scores 'Scores.csv'
ssbind.generate_conformers()

## Clustering identifies some binding modes based on binding scores and PCA.
ssbind.clustering(conformers = 'conformers.sdf', scores = 'Scores.csv')
```