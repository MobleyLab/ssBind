import shutil

from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from ssBind.generator import *
from ssBind.tests.generic import *


def cleanup() -> None:
    """Remove conformers file created during testing"""

    for f in ["conformers.sdf", "Scores.csv"]:
        try:
            os.remove(f)
        except OSError:
            pass


def cleanup_rdock() -> None:
    """Remove all files rdock creates locally during testing"""

    for f in [
        "conformers.sdf",
        "rbcavity.prm",
        "rbcavity.log",
        "rbdock.as",
        "rbdock_cav1.grd",
        "rbdock.log",
        "rbdock.prm",
        "Scores.csv",
    ]:
        try:
            os.remove(f)
            shutil.rmtree("temp_rdock")
        except OSError:
            pass


def cleanup_plants() -> None:
    """Remove all files rdock creates locally during testing"""

    for f in [
        "conformers.sdf",
        "Scores.csv",
        "plants_config",
        "SPORES.log",
        "ligand.mol2",
        "ligand.pdb",
        "ligand.sdf",
        "receptor.mol2",
    ]:
        try:
            os.remove(f)
        except OSError:
            pass


def generate_and_test(generator: ConformerGenerator, expected_num_confs: int) -> None:
    """Generate conformers, check that 2D topology (SMILES) is preserved as well
    as the desired number of conformers being generated.

    Args:
        generator (AbstractConformerGenerator): Conformer generator
        expected_num_confs (int): Expected number of conformers to generate
    """
    generator.generate_conformers()

    suppl = Chem.SDMolSupplier("conformers.sdf")
    assert len(suppl) == expected_num_confs
    mol = next(suppl)
    assert canonSmiles(mol) == canonSmiles(generator._query_molecule)


def test_rdkit_generator(receptor_file: str, reference: Mol, ligand: Mol) -> None:
    """Test RDKitConformerGenerator

    Args:
        receptor_file (str): receptor.pdb protein structure
        reference (Mol): reference.mol2 restrained ligand substructure
        ligand (Mol): ligand.mol2 ligand structure
    """

    cleanup()
    generator = RDkitConformerGenerator(
        receptor_file,
        ligand,
        reference,
        nprocs=2,
        numconf=20,
    )
    generate_and_test(generator, 20)
    cleanup()


def test_angle_generator(receptor_file: str, reference: Mol, ligand: Mol) -> None:
    """Test AngleConformerGenerator

    Args:
        receptor_file (str): receptor.pdb protein structure
        reference (Mol): reference.mol2 restrained ligand substructure
        ligand (Mol): ligand.mol2 ligand structure
    """

    cleanup()
    generator = AngleConformerGenerator(
        receptor_file,
        ligand,
        reference,
        nprocs=1,
        numconf=20,
        degree=120,
    )
    generate_and_test(generator, 3)
    cleanup()


def test_rdock_generator(receptor_file_mol2: str, reference: Mol, ligand: Mol) -> None:
    """Test RdockConformerGenerator

    Args:
        receptor_file_mol2 (str): receptor mol2 structure
        reference (Mol): reference.mol2 restrained ligand substructure
        ligand (Mol): ligand.mol2 ligand structure
    """

    cleanup_rdock()
    generator = RdockConformerGenerator(
        ligand,
        reference,
        receptor_file_mol2,
        numprocs=4,
        numconf=10,
        working_dir="temp_rdock",
    )
    generate_and_test(generator, 10)
    cleanup_rdock()


def test_plants_generator(receptor_file: str, reference: Mol, ligand: Mol) -> None:
    """Test AngleConformerGenerator

    Args:
        receptor_file (str): receptor.pdb protein structure
        reference (Mol): reference.mol2 restrained ligand substructure
        ligand (Mol): ligand.mol2 ligand structure
    """

    cleanup_plants()
    generator = PlantsConformerGenerator(
        ligand, reference, receptor_file, nprocs=2, numconf=10
    )
    generate_and_test(generator, 10)
    cleanup_plants()


@pytest.mark.xfail
def test_autodock_generator(
    receptor_file: str, reference: Mol, ligand: Mol, ligand_file: str
) -> None:
    """Test RDKitConformerGenerator

    Args:
        receptor_file (str): receptor.pdb protein structure
        reference (Mol): reference.mol2 restrained ligand substructure
        ligand (Mol): ligand.mol2 ligand structure
    """

    cleanup()
    generator = AutodockGenerator(
        receptor_file,
        ligand,
        reference,
        ligand=ligand_file,
        nprocs=2,
        numconf=20,
        working_dir="test_autodock",
        autodock_hydrated=False,
    )
    generate_and_test(generator, 20)
    cleanup()
