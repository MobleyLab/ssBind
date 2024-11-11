#!/usr/bin/python
import argparse
import multiprocessing as mp
import os

from ssBind import SSBIND
from ssBind.generator import *
from ssBind.io import MolFromInput

# Substructure-based alternative BINDing modes generator for protein-ligand systems


def ParserOptions():
    parser = argparse.ArgumentParser()

    """Parse command line arguments."""
    parser.add_argument(
        "--reference", dest="reference", help="Referance molecule", required=True
    )
    parser.add_argument(
        "--ligand", dest="ligand", help="Ligand molecule", required=True
    )
    parser.add_argument(
        "--receptor",
        dest="receptor",
        help="PDB file for receptor protein",
        required=True,
    )
    parser.add_argument(
        "--FF",
        dest="FF",
        default="gaff",
        help="Generalized Force Fields GAFF, CGenFF, OpenFF",
        choices=["gaff", "gaff2", "openff", "cgenff"],
    )
    parser.add_argument("--proteinFF", dest="proteinFF", default="amber99sb-ildn")
    parser.add_argument(
        "--degree",
        dest="degree",
        type=float,
        help="Amount, in degrees, to enumerate torsions by (default 60.0)",
        default=60.0,
    )
    parser.add_argument(
        "--cutoff",
        dest="cutoff_dist",
        type=float,
        help="Cutoff for eliminating any conformer close to protein within cutoff by (default 1.5 A)",
        default=1.5,
    )
    parser.add_argument(
        "--rms",
        dest="rms",
        type=float,
        help="Only keep structures with RMS > CUTOFF (default 0.2 A)",
        default=0.2,
    )
    parser.add_argument(
        "--cpu",
        dest="cpu",
        type=int,
        help="Number of CPU. If not set, it uses all available CPUs.",
    )
    parser.add_argument(
        "--generator",
        dest="generator",
        help="Choose a method for the conformer generation.",
        choices=["angle", "rdkit", "plants", "rdock"],
    )
    parser.add_argument(
        "--numconf", dest="numconf", type=int, help="Number of confermers", default=1000
    )
    parser.add_argument(
        "--minimize",
        dest="minimize",
        help="Perform minimization",
        choices=["gromacs", "smina", "openmm"],
    )
    parser.add_argument(
        "--openmm-score",
        dest="openmm_score",
        help="Calculate total or interaction energy upon minimization with OpenMM",
        choices=["interaction", "total"],
        default="interaction",
    )
    parser.add_argument(
        "--openmm-flex",
        dest="openmm_flex",
        help="Treat protein as flexible upon OpenMM minimization",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--clustering",
        dest="posepicker",
        help="Conformer clustering algorithm",
        choices=["PCA", "Torsion"],
        default="PCA",
    )
    parser.add_argument(
        "--flexDist",
        dest="flexDist",
        type=int,
        help="Residues having side-chain flexibility taken into account. Take an interger to calculate closest residues around the ligand",
    )
    parser.add_argument(
        "--flexList",
        dest="flexList",
        type=str,
        help="Residues having side-chain flexibility taken into account. Take a list of residues for flexibility",
    )
    parser.add_argument(
        "--bin",
        dest="bin",
        type=float,
        help="Numeric vector giving bin width in both vertical and horizontal directions in PCA analysis",
        default=0.25,
    )
    parser.add_argument(
        "--distThresh",
        dest="distThresh",
        type=float,
        help="elements within this range of each other are considered to be neighbors during clustering",
        default=0.5,
    )
    parser.add_argument(
        "--numbin",
        dest="numbin",
        type=int,
        help="Number of bins to be extract for clustering conformations",
        default=10,
    )
    args = parser.parse_args()
    return args


def main(args, nprocs):

    reference_substructure = MolFromInput(args.reference)
    query_molecule = MolFromInput(args.ligand)

    receptor_extension = os.path.splitext(args.receptor)[1].lower()
    if args.generator == "rdock" and receptor_extension != ".mol2":
        print(
            f"""Warning: {args.receptor} is not a .mol2 file.
        The receptor “.mol2″ file must be preparated (protonated, charged, etc.)"""
        )

    kwargs = {
        "reference_substructure": reference_substructure,
        "query_molecule": query_molecule,
        "receptor_file": args.receptor,
        "nprocs": nprocs,
        **vars(args),
    }

    ssbind = SSBIND(**kwargs)

    ssbind.generate_conformers()

    if args.minimize is not None:
        ssbind.run_minimization(conformers="conformers.sdf")

    conformers_map = {
        "smina": "minimized_conformers.sdf",
        "gromacs": "minimized_conformers.sdf",
        "openmm": "minimized_conformers.dcd",
    }
    conformers = conformers_map.get(args.minimize, "conformers.sdf")
    ssbind.clustering(
        conformers=conformers,
        scores="Scores.csv",
    )


if __name__ == "__main__":

    args = ParserOptions()

    nprocs = args.cpu if args.cpu is not None else mp.cpu_count()

    print(f"\nNumber of CPU in use for conformer generation: {nprocs}")

    main(args, nprocs)
