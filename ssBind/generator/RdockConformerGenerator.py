import csv
import math
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import uuid
from contextlib import closing

import pandas as pd
from rdkit import Chem, RDLogger

from ssBind.generator import ConformerGenerator


class RdockConformerGenerator(ConformerGenerator):

    def __init__(
        self,
        query_molecule: str,
        reference_substructure: str,
        receptor_file: str,
        **kwargs,
    ) -> None:
        super().__init__(
            receptor_file, query_molecule, reference_substructure, **kwargs
        )

        curdir = kwargs.get("curdir", os.getcwd())
        self._working_dir = kwargs.get(
            "working_dir", os.path.join(curdir, str(uuid.uuid4()))
        )

    def generate_conformers(self) -> None:
        """
        Generates conformers using rDock.

        """
        rdock_random = os.path.basename(self._working_dir)
        os.makedirs(f"{rdock_random}")

        sdwriter = Chem.SDWriter(f"{rdock_random}/ref.sdf")
        sdwriter.write(self._reference_substructure)
        sdwriter.close()

        self._get_tethered(rdock_random)
        self._prepare_receptor(
            RECEPTOR_FILE=self._receptor_file, REF_MOL=f"{self._working_dir}/ref.sdf"
        )

        with closing(mp.Pool(processes=self._nprocs)) as pool:
            pool.starmap(
                self._run_rdock,
                [
                    (i, f"{self._working_dir}/{rdock_random}.sd", f"{rdock_random}")
                    for i in range(math.ceil(self._numconf / 10))
                ],
            )

        self._combine_files(f"{rdock_random}")
        shutil.rmtree(f"{rdock_random}")

    def _get_tethered(
        self, out
    ):  ###https://github.com/Discngine/rdkit_tethered_minimization/blob/master/tetheredMinimization.py
        """
        Tether a ligand to a reference molecule based on the maximum common substructure (MCS)
        and save the tethered ligand to an SD file in the specified output directory.

        Parameters:
        - refmol: RDKit Mol object, the reference molecule.
        - ligand: RDKit Mol object, the ligand to be tethered.
        - out_dir: str, the directory where the output SD file will be saved.

        """

        RDLogger.DisableLog("rdApp.*")

        ligandHs = Chem.AddHs(self._query_molecule, addCoords=True)
        self._alignToRef(ligandHs)
        ligandMappings = list(zip(*self._mappingLigToRef))[0]

        atoms = map(lambda x: x + 1, ligandMappings)
        atoms_string = ",".join(str(el) for el in atoms)

        ligandHs.SetProp("TETHERED ATOMS", atoms_string)

        w = Chem.SDWriter(f"{out}/{out}.sd")
        w.write(ligandHs)
        w.flush()

        return out

    @staticmethod
    def _prepare_receptor(
        RECEPTOR_FILE="receptor.mol2",
        RECEPTOR_FLEX: float = 3.0,
        REF_MOL="ligand.sdf",
        SITE_MAPPER: str = "RbtLigandSiteMapper",
        RADIUS: float = 15.0,
        SMALL_SPHERE: float = 1.0,
        MIN_VOLUME: int = 100,
        MAX_CAVITIES: int = 1,
        VOL_INCR: float = 0.0,
        GRIDSTEP: float = 0.5,
        SCORING_FUNCTION: str = "RbtCavityGridSF",
        WEIGHT: float = 1.0,
        MAX_TRANS: float = 0.0,
        MAX_ROT: float = 0.0,
    ):
        """
        rbcavity – Cavity mapping and preparation of docking site

        Prepares the receptor for docking by creating an rDock parameter file and running rbcavity.

        For a detailed description of the parameters please see rDock documentation https://rdock.github.io/documentation/html_docs/reference-guide/cavity-mapping.html

        """

        params = """RBT_PARAMETER_FILE_V1.00
TITLE rDock

RECEPTOR_FILE {}
RECEPTOR_FLEX {}

##################################################################
### CAVITY DEFINITION: REFERENCE LIGAND METHOD
##################################################################
SECTION MAPPER
    SITE_MAPPER {}
    REF_MOL {}
    RADIUS {}
    SMALL_SPHERE {}
    MIN_VOLUME {}
    MAX_CAVITIES {}
    VOL_INCR {}
    GRIDSTEP {}
END_SECTION

#################################
#CAVITY RESTRAINT PENALTY
#################################
SECTION CAVITY
    SCORING_FUNCTION {}
    WEIGHT {}
END_SECTION
            
""".format(
            RECEPTOR_FILE,
            RECEPTOR_FLEX,
            SITE_MAPPER,
            REF_MOL,
            RADIUS,
            SMALL_SPHERE,
            MIN_VOLUME,
            MAX_CAVITIES,
            VOL_INCR,
            GRIDSTEP,
            SCORING_FUNCTION,
            WEIGHT,
        )

        with open("rbdock.prm", "w") as f:
            f.write(params)

        RC = os.system("""rbcavity -was -d -r rbdock.prm > rbcavity.log""")
        if RC != 0:
            raise SystemExit(
                "\nERROR!\nFailed to run the rbcavity. see the %s for details."
                % os.path.abspath("rbcavity.log\n")
            )

        params = """
################################
# TETHERED SCAFFOLD
################################
SECTION LIGAND
    TRANS_MODE TETHERED
    ROT_MODE TETHERED
    DIHEDRAL_MODE FREE
    MAX_TRANS {}
    MAX_ROT {}
END_SECTION
""".format(
            MAX_TRANS, MAX_ROT
        )

        with open("rbdock.prm") as f:
            rbcavity_params = f.readlines()
        with open("rbdock.prm", "w") as rbdock_params:
            rbdock_params.writelines(rbcavity_params)
            rbdock_params.write(params)

    @staticmethod
    def _run_rdock(i, ligand, output, nruns=10):
        """
        rbdock – the rDock docking engine itself.

        For a detailed description of the tethered parameters please see rDock documentation https://rdock.github.io/documentation/html_docs/reference-guide/cavity-mapping.html

        """

        cmd = [
            "rbdock",
            "-i",
            ligand,
            "-o",
            f"{output}/{i}",
            "-r",
            "rbdock.prm",
            "-p",
            "dock.prm",
            "-n",
            str(nruns),
        ]

        # Execute the command
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Write output and errors to log file
        with open("rbdock.log", "w") as log_file:
            log_file.write(result.stdout)
            log_file.write(result.stderr)

        # Check return code to see if rDock ran successfully
        if result.returncode != 0:
            error_msg = "\nERROR!\nFailed to run the rDock. See {} for details.".format(
                os.path.abspath("rbdock.log")
            )
            raise SystemExit(error_msg)

        scores = []
        with open(f"{output}/{i}.sd", "r") as file:
            file_content = file.read()
            scores = re.findall(r">\s*<SCORE>\s*([-\d.]+)", file_content)

        with open(f"{output}/{i}.csv", "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            for score in scores:
                csvwriter.writerow([score])
        return

    @staticmethod
    def _combine_files(dockdir):
        # Directory containing the csv files
        files = sorted([f for f in os.listdir(dockdir) if f.endswith(".csv")])

        numeric_list = [int(re.findall(r"\d+", s)[0]) for s in files]

        # Create an SDF writer
        sdf_writer = Chem.SDWriter("conformers.sdf")
        combined_df = pd.DataFrame(columns=["Score"])

        # Process each sd file
        for num in numeric_list:
            sd_path = os.path.join(dockdir, f"{num}.sd")
            mols = Chem.SDMolSupplier(sd_path, sanitize=False)
            if mols is not None:
                for i, mol in enumerate(mols):
                    mol.UpdatePropertyCache(strict=False)
                    sdf_writer.write(mol)
            df = pd.read_csv(
                os.path.join(dockdir, f"{num}.csv"), header=None, names=["Score"]
            )
            combined_df = pd.concat([combined_df, df], ignore_index=True)

        combined_df.reset_index(inplace=True)
        combined_df.rename(columns={"index": "Index"}, inplace=True)
        combined_df.to_csv("Scores.csv", index=False)

        sdf_writer.close()
