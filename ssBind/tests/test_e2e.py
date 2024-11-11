import os
from pathlib import Path

from ssBind.tests.generic import *


def test_e2e(receptor_file, reference_file, ligand_file) -> None:

    root_dir = Path(__file__).parents[1]
    script = os.path.join(root_dir, "run_ssBind.py")

    success = os.system(
        f"python {script} --reference {reference_file} --ligand {ligand_file} --receptor {receptor_file} --generator rdkit "
        "--minimize openmm --FF gaff --proteinFF amber14/protein.ff14SB.xml --clustering Torsion",
    )
    assert success == 0
