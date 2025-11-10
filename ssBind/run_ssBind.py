#!/usr/bin/python
import argparse
import copy
import multiprocessing as mp
import os
import shutil
from typing import List

import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolFromMol2File

from ssBind import SSBIND
from ssBind.generator import *

# Substructure-based alternative BINDing modes generator for protein-ligand systems


def none_or_str(value):
    if value == "None":
        return None
    return value


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
        "--cutoff",
        dest="cutoff_dist",
        type=float,
        help="Cutoff for eliminating any conformer close to protein within cutoff by (default 1.5 A)",
        default=1.5,
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
        choices=[
            "angle",
            "rdkit",
            "plants",
            "rdock",
            "autodock",
            "autodock-hydrated",
            None,
        ],
        type=none_or_str,
    )
    parser.add_argument(
        "--numconf", dest="numconf", type=int, help="Number of confermers", default=1000
    )
    parser.add_argument(
        "--minimize",
        dest="minimize",
        help="Perform minimization (recommended: smina with rdkit). local means minimization of the reference only.",
        choices=["smina", "smina-score", "gromacs", "openmm", "local", None],
        type=none_or_str,
    )
    parser.add_argument(
        "--selectionStrategy",
        dest="selectionStrategy",
        help="Pose selection strategy post clustering",
        choices=["score", "similarity", "mixed"],
        default="mixed",
    )
    parser.add_argument(
        "--outputPoses",
        dest="outputPoses",
        help="Number of poses to select",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--no_selection",
        dest="no_selection",
        help="Omit clustering and selection",
        action="store_true",
    )
    parser.add_argument(
        "--flexDist",
        dest="flexDist",
        type=int,
        help="(PLANTS only) Residues having side-chain flexibility taken into account. Take an interger to calculate closest residues around the ligand",
    )
    parser.add_argument(
        "--flexList",
        dest="flexList",
        type=str,
        help="(PLANTS only) Residues having side-chain flexibility taken into account. Take a list of residues for flexibility",
    )
    parser.add_argument(  # for plants
        "--no_prepare_ligand",
        dest="no_prepare_ligand",
        action="store_true",
        help="(PLANTS only) omit ligand preparation by SPORES",
    )
    parser.add_argument(
        "--seeds",
        dest="seeds",
        help="SDF with ligand structures to seed docking",
    )
    args = parser.parse_args()
    return args


def main(args, nprocs):

    if args.ligand.endswith("mol2"):
        query_molecule = MolFromMol2File(
            args.ligand, cleanupSubstructures=True, removeHs=False
        )
    elif args.ligand.endswith("sdf"):
        query_molecule = next(Chem.SDMolSupplier(args.ligand, removeHs=False))
    else:
        raise Exception("Ligand must be MOL2 or SDF!")

    if args.seeds is None:
        run_ssbind(args, nprocs, query_molecule)
    else:  # batch mode

        seeds = [mol for mol in Chem.SDMolSupplier(args.seeds, removeHs=False)]

        if len(seeds) == 0:
            seeds = [query_molecule]

        args_confgen = copy.deepcopy(args)
        args_confgen.numconf = int(args.numconf / len(seeds))
        args_confgen.minimize = None
        args_confgen.no_selection = True

        for i, seed in enumerate(seeds):
            args_confgen.do_receptor_prep = i == 0
            args_confgen.iseed = i
            run_ssbind(args_confgen, nprocs, seed)
            shutil.move("conformers.sdf", f"conformers_{i}.sdf")
            shutil.move("Scores.csv", f"Scores_{i}.csv")

        combine_confs_and_scores(seeds)

        # 3) Cluster conformers together - take query as reference
        args_cluster = copy.deepcopy(args)
        args_cluster.generator = None
        run_ssbind(args_cluster, nprocs, query_molecule)


def combine_confs_and_scores(seeds: List[Chem.Mol]):
    confs = []
    for i in range(len(seeds)):
        confs = confs + [
            mol for mol in Chem.SDMolSupplier(f"conformers_{i}.sdf", sanitize=False)
        ]
    writer = Chem.SDWriter("conformers.sdf")
    writer.SetKekulize(False)
    for conf in confs:
        writer.write(conf)
    writer.close()

    dfs = [
        pd.read_csv(f"Scores_{i}.csv", index_col=[0], parse_dates=[0])
        for i in range(len(seeds))
    ]
    finaldf = pd.concat(dfs)
    finaldf.to_csv("Scores.csv")

    for i in range(len(seeds)):
        os.remove(f"conformers_{i}.sdf")
        os.remove(f"Scores_{i}.csv")


def run_ssbind(args, nprocs, query_molecule):
    reference_substructure = MolFromMol2File(args.reference, cleanupSubstructures=True)
    reference_substructure = Chem.RemoveAllHs(reference_substructure)

    receptor_extension = os.path.splitext(args.receptor)[1].lower()
    if args.generator == "rdock" and receptor_extension != ".mol2":
        print(
            f"""Warning: {args.receptor} is not a .mol2 file.
        The receptor “.mol2″ file must be preparated (protonated, charged, etc.)"""
        )

    kwargs = {
        **vars(args),
        "reference_substructure": reference_substructure,
        "query_molecule": query_molecule,
        "receptor_file": args.receptor,
        "nprocs": nprocs,
    }

    ssbind = SSBIND(**kwargs)

    if args.generator is not None:
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
