import multiprocessing as mp
import os
import shutil
import subprocess
import uuid
from contextlib import closing

from rdkit import Chem, RDLogger
from rdkit.Chem.PandasTools import LoadSDF


class SminaMinimizer:
    def __init__(self, receptor_file: str, **kwargs: dict[str, any]) -> None:
        self._receptor_file = receptor_file

        self._working_dir = kwargs.get("working_dir", os.path.join(os.getcwd(), "tmp"))
        self._nprocs = kwargs.get("nprocs", 1)

    def run_minimization(self, conformers: str) -> None:
        os.makedirs(self._working_dir)
        with closing(mp.Pool(processes=self._nprocs)) as pool:
            pool.starmap(
                self._smina_minimize_score,
                [
                    (i, mol)
                    for i, mol in enumerate(
                        Chem.SDMolSupplier(conformers, sanitize=True)
                    )
                ],
            )
        self._combine_sdf_files("minimized_conformers.sdf", "Scores.csv")

    def _smina_minimize_score(self, index: int, ligand: Chem.rdchem.Mol) -> None:
        name = str(uuid.uuid4())

        ligand_file_path = f"{name}.sdf"
        mol_with_h = Chem.AddHs(ligand, addCoords=True)
        with Chem.SDWriter(ligand_file_path) as w:
            w.write(mol_with_h)

        # Finding smina executable
        smina_executable = self._which("smina") or self._which("smina.static")
        if smina_executable is None:
            raise SystemExit("\nERROR! smina path is not found!!\n")

        command = [
            smina_executable,
            "--receptor",
            self._receptor_file,
            "--ligand",
            ligand_file_path,
            "--minimize",
            "--cpu",
            "1",
            "--out",
            os.path.join(self._working_dir, f"{index}_min.sdf"),
        ]

        log_file_path = f"{name}.log"
        # Running smina
        with open(log_file_path, "w") as log_file:
            subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT)

        command = [
            smina_executable,
            "--receptor",
            self._receptor_file,
            "--ligand",
            os.path.join(self._working_dir, f"{index}_min.sdf"),
            "--score_only",
            "--cpu",
            "1",
            "--scoring",
            "vinardo",
        ]

        # Running smina
        with open(log_file_path, "w") as log_file:
            subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT)

        # Clean up temporary files
        for file_to_remove in [ligand_file_path, log_file_path]:
            try:
                os.remove(file_to_remove)
            except FileNotFoundError as e:
                print(f"Error: {e}")

    def _combine_sdf_files(self, output_sdf_path: str, sorted_csv_path: str) -> None:

        RDLogger.DisableLog("rdApp.*")

        files = [f for f in os.listdir(self._working_dir) if f.endswith(".sdf")]
        mols = []

        for i in range(len(files)):
            full_path = os.path.join(self._working_dir, f"{i}_min.sdf")
            suppl = Chem.SDMolSupplier(full_path, sanitize=False)
            for mol in suppl:
                mols.append(mol)

        mols_sorted = sorted(
            mols, key=lambda mol: float(mol.GetProp("minimizedAffinity"))
        )

        with Chem.SDWriter(output_sdf_path) as writer:
            for mol in mols_sorted:
                writer.write(mol)

        df = LoadSDF(output_sdf_path, sanitize=True)[["minimizedAffinity"]]
        df = df.rename(columns={"minimizedAffinity": "Score"})
        df.to_csv(sorted_csv_path, index_label="Index")

        try:
            shutil.rmtree(self._working_dir)
        except:
            print(f"Warning: failed to remove folder {self._working_dir}")

    @staticmethod
    def _which(program: str):
        """
        Search for an executable in the system's PATH.

        Returns:
        - The full path to the executable if found; None otherwise.
        """

        def is_exe(fpath):
            """Check if a given path is an executable file."""
            return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

        fpath, fname = os.path.split(program)
        if fpath:
            if is_exe(program):
                return program
        else:
            for path in os.environ.get("PATH", "").split(os.pathsep):
                exe_file = os.path.join(path, program)
                if is_exe(exe_file):
                    return exe_file
        return None
