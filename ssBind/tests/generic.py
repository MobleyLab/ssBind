import os

import pytest
from rdkit import Chem
from rdkit.Chem.rdchem import Mol


@pytest.fixture
def path() -> str:
    """Directory where this file is located

    Returns:
        str: Absolute path to tests directory
    """
    return os.path.dirname(__file__)


@pytest.fixture
def receptor_file(path: str) -> str:
    """Absolute path to the stored receptor PDB

    Args:
        path (str): Path to tests directory

    Returns:
        str: Path to receptor
    """
    return os.path.join(path, "data/receptor.pdb")


@pytest.fixture
def receptor_file_mol2(path: str) -> str:
    """Absolute path to receptor in mol2 format (for rDock)

    Args:
        path (str): Path to tests directory

    Returns:
        str: Path to mol2 receptor
    """
    return os.path.join(path, "data/receptor.mol2")


@pytest.fixture
def reference_file(path: str) -> str:
    """Reference substructure as Rdkit mol

    Args:
        path (str): Path to tests directory

    Returns:
        str: reference.mol file incl. absolute path
    """
    return os.path.join(path, "data/reference.mol2")


@pytest.fixture
def reference(reference_file: str) -> Mol:
    """Reference substructure as Rdkit mol

    Args:
        path (str): Path to reference.mol

    Returns:
        Mol: Reference substructure
    """
    return Chem.MolFromMol2File(reference_file)


@pytest.fixture
def ligand_file(path: str) -> str:
    """Ligand file as absolute path

    Args:
        path (str): Path to tests directory

    Returns:
        str: Absolute path to ligand
    """
    return os.path.join(path, "data/ligand.mol2")


@pytest.fixture
def ligand(ligand_file: str) -> Mol:
    """Ligand as Rdkit mol

    Args:
        ligand_file (str): Absolute path to ligand

    Returns:
        Mol: Ligand
    """
    return Chem.MolFromMol2File(ligand_file)


def canonSmiles(mol: Mol) -> str:
    """Convert molecule to canonical SMILES

    Args:
        mol (Mol): RDkit ROMol

    Returns:
        str: canonical SMILES
    """
    return Chem.CanonSmiles(Chem.MolToSmiles(mol))


def ligand_to_sdf(ligand: Mol) -> None:
    """Convert ligand from mol2 to sdf, to be used as conformers.sdf substitute

    Args:
        ligand (Mol): ligand structure
    """
    lig = Chem.RemoveHs(ligand)
    lig.SetProp("_Name", "")
    lig.ClearProp("_TriposChargeType")
    with Chem.SDWriter("ligand.sdf") as w:
        w.write(lig)


def files_equal(f1: str, f2: str) -> bool:
    """Check if two files have the same content

    Args:
        f1 (str): filename 1
        f2 (str): filename 2

    Returns:
        bool: true if content is equal
    """
    with open(f1, "r") as f:
        f1_content = f.read()
    with open(f2, "r") as f:
        f2_content = f.read()
    return f1_content == f2_content
