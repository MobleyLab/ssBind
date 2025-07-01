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


def test_e2e(receptor_file, reference_file, ligand_file, ligand) -> None:

    files_to_remove = [
        "conformers.sdf",
        "ligand.sdf",
        "minimized_conformers.sdf",
        "Scores.csv",
        "cluster_info.csv",
        "conf_info.csv",
        "PC1-PC2.svg",
        "PC1-PC3.svg",
        "PC2-PC3.svg",
    ] + [f"model_{i+1}.sdf" for i in range(11)]

    root_dir = Path(__file__).parents[1]
    script = os.path.join(root_dir, "run_ssBind.py")

    cleanup(files_to_remove)
    ligand_to_sdf(ligand)

    success = os.system(
        f"python {script} --reference {reference_file} --ligand {ligand_file} --receptor {receptor_file} --generator rdkit "
        f"--minimize smina --clustering PCA --numconf 20",
    )
    # success = os.system(
    #     f"python {script} --reference {reference_file} --ligand {ligand_file} --receptor {receptor_file} --generator autodock "
    #     f"--clustering PCA --numconf 250",
    # )

    assert success == 0
    cleanup(files_to_remove)
