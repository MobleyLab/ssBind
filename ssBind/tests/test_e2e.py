import os
from pathlib import Path
from typing import List

from ssBind.tests.generic import *


def cleanup(files: List[str]) -> None:
    """Clean up

    Args:
        files (List[str]): Files to remove
    """
    for f in files:
        try:
            os.remove(f)
        except OSError:
            pass


def test_e2e(
    receptor_file, reference_file, receptor_file_mol2, ligand_file, ligand
) -> None:

    files_to_remove = [
        "conformers.sdf",
        "conformers.pdb",
        "ligand.sdf",
        "minimized_conformers.sdf",
        "Scores.csv",
        "complex.pdb",
        "rbcavity.log",
        "rbdock_cav1.grd",
        "rbdock.as",
        "rbdock.log",
        "rbdock.prm",
        "selected_conformers.sdf",
    ]

    root_dir = Path(__file__).parents[1]
    script = os.path.join(root_dir, "run_ssBind.py")

    cleanup(files_to_remove)

    success = 0
    success = success + os.system(
        f"python {script} --reference {reference_file} --ligand {ligand_file} --receptor {receptor_file} --generator autodock-hydrated "
        f"--numconf 200"  # --seeds selected_conformers.sdf",
    )

    assert success == 0
    cleanup(files_to_remove)
