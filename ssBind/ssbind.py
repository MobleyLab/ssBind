import datetime
import multiprocessing as mp
import os
import time
import uuid

import MDAnalysis as mda

from ssBind.filter import ConformerFilter
from ssBind.generator import *
from ssBind.minimizer import *
from ssBind.posepicker import *

# Substructure-based alternative BINDing modes generator for protein-ligand systems


class SSBIND:

    def __init__(self, **kwargs):

        kwargs.setdefault("nprocs", mp.cpu_count())
        kwargs.setdefault("numconf", 1000)
        kwargs.setdefault("curdir", os.getcwd())
        kwargs.setdefault(
            "working_dir", os.path.join(kwargs["curdir"], str(uuid.uuid4()))
        )
        self._kwargs = kwargs
        self._generator = None
        self._filter = None
        self._minimizer = None
        self._posepicker = None

    def generate_conformers(
        self,
    ):
        """
        Generates conformers using RDKit or dihedral angle sampling.

        """
        # initialize timer
        start = time.time()

        generator_type = self._kwargs.get("generator")

        if generator_type == "rdkit":
            self._generator = RDkitConformerGenerator(
                **self._kwargs,
            )
        elif generator_type == "angle":
            self._generator = AngleConformerGenerator(
                **self._kwargs,
            )
        elif generator_type == "rdock":
            self._generator = RdockConformerGenerator(
                **self._kwargs,
            )
        elif generator_type == "plants":
            self._generator = PlantsConformerGenerator(**self._kwargs)
        elif generator_type == "autodock":
            self._generator = AutodockGenerator(**self._kwargs)
        else:
            raise Exception(f"Invalid conformer generator: {generator_type}")

        self._generator.generate_conformers()

        # print elapsed time
        numconf = self._kwargs.get("numconf")
        self._print_time("Conformer generation", start, numconf)

    def run_minimization(
        self,
        conformers="conformers.sdf",
    ):
        """
        Performs minimization and scoring.

        """

        # initialize timer
        start = time.time()

        minimizer_type = self._kwargs.get("minimize")

        if minimizer_type is None:
            pass
        elif minimizer_type == "gromacs":
            self._minimizer = GromacsMinimizer(**self._kwargs)
        elif minimizer_type == "smina":
            self._minimizer = SminaMinimizer(**self._kwargs)
        elif minimizer_type == "openmm":
            self._minimizer = OpenMMinimizer(**self._kwargs)
        else:
            raise Exception(f"Invalid minimizer: {minimizer_type}")

        self._minimizer.run_minimization(conformers)

        # print elapsed time
        numconf = self._conformer_count(conformers)
        self._print_time("Energy minimization", start, numconf)

    def clustering(
        self,
        conformers: str = "conformers.sdf",
        scores: str = "Scores.csv",
    ):
        """
        Performs clustering based on the conformational distance (RMSD) matrix.

        """

        # initialize timer
        start = time.time()

        posepicker_type = self._kwargs.get("posepicker")

        if posepicker_type == "PCA":
            self._posepicker = PCAPosePicker(**self._kwargs)
        elif posepicker_type == "Torsion":
            self._posepicker = TorsionPosePicker(**self._kwargs)
        else:
            raise Exception(f"Invalid clustering method: {posepicker_type}")

        self._posepicker.pick_poses(conformers, scores)

        # print elapsed time
        numconf = self._conformer_count(conformers)
        self._print_time("Pose clustering and selection", start, numconf)

    @staticmethod
    def _conformer_count(filename: str) -> int:
        """Get number of conformers in a designated file by counting END entries
        (applicable to SDF and PDB). This is only used to calculate timings per
        conformer.

        Args:
            filename (str): PDB or SDF file with conformers

        Returns:
            int: Number of conformers in file
        """

        format = filename.split(".")[-1]

        if format in ["sdf", "pdb"]:
            with open(filename, "r") as f:
                content = f.readlines()

            numconf = len([line for line in content if "END" in line])
            return numconf
        elif format == "dcd":
            try:
                u = mda.Universe("complex.pdb", filename)
                return u.trajectory.n_frames
            except:
                pass

    def _print_time(self, step_name: str, start: float, numconf: int) -> None:
        """Print timing information after every step of the SSBIND pipeline

        Args:
            step_name (str): Name of step (e.g. energy minimization)
            start (float): Timestamp at start
            numconf (int): number of conformers run through this step
        """

        end = time.time()
        diff = end - start
        diff_str = str(datetime.timedelta(seconds=diff))
        n_cpu = self._kwargs.get("nprocs")
        if numconf == 0:
            diff_per_conf = 0
        else:
            diff_per_conf = diff / numconf

        print(
            "----------------------------------------------------------------------------------------"
        )
        print(
            f"Elapsed time for {step_name}:\t {diff_str}\t\t {diff_per_conf} s / conformer ({n_cpu} core(s))"
        )
        print(
            "----------------------------------------------------------------------------------------"
        )
