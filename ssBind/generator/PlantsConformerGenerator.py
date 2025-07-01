import csv
import math
import multiprocessing as mp
import os
import re
import shutil
import uuid
from contextlib import closing
from typing import Dict, List, Tuple

import MDAnalysis as mda
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops

from ssBind.generator import ConformerGenerator
from ssBind.io import obabel_convert, parse_pdb_line


class PlantsConformerGenerator(ConformerGenerator):

    def __init__(
        self,
        query_molecule: str,
        reference_substructure: str,
        receptor_file: str,
        flexDist: int = None,
        flexList: str = None,
        no_prepare_ligand=False,
        **kwargs: Dict,
    ) -> None:
        super().__init__(
            receptor_file, query_molecule, reference_substructure, **kwargs
        )

        self._flexDist = flexDist
        self._flexList = flexList
        self._no_prepare_ligand = no_prepare_ligand
        self._working_dir = kwargs.get("working_dir", os.path.join(os.getcwd(), "tmp"))
        self._ligand_file = kwargs.get("ligand", None)

    def generate_conformers(self) -> None:
        """
        Generates conformers using PLANTS.

        """
        output_dir = self._working_dir
        os.makedirs(output_dir)
        self._alignToRef(self._query_molecule)

        xyz = " ".join(
            map(
                str,
                [
                    round(coord, 3)
                    for coord in self._molecule_center(self._query_molecule)
                ],
            )
        )
        fixedAtom = self._getAtomConst() + 1
        # Handle flexibility for PLANTS
        flex_res = self._handle_flexibility()

        self._SPORES(self._receptor_file, "receptor.mol2", "settypes")

        with closing(mp.Pool(processes=self._nprocs)) as pool:
            pool.starmap(
                self._plants_docking,
                [
                    (
                        i,
                        output_dir,
                        15,
                        10,
                        xyz,
                        fixedAtom,
                        flex_res,
                        self._minimize_only,
                    )
                    for i in range(math.ceil(self._numconf / 10))
                ],
            )

        self._combine_files(output_dir)

    @staticmethod
    def _molecule_center(mol: Chem.rdchem.Mol) -> List[float]:
        """Calculate the center of geometry of the molecule

        Args:
            mol (Chem.rdchem.Mol): Rdkit molecule

        Returns:
            List[float]: Center of geometry, list of length 3
        """

        conf = mol.GetConformer()
        center = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
        center = [
            sum(coord.x for coord in center) / len(center),
            sum(coord.y for coord in center) / len(center),
            sum(coord.z for coord in center) / len(center),
        ]

        return center

    def _getAtomConst(self) -> int:

        ligand = self._query_molecule
        reflig = self._reference_substructure

        if not self._no_prepare_ligand:
            pdwriter = Chem.PDBWriter("ligand.pdb", flavor=4)
            pdwriter.write(ligand)
            pdwriter.close()

            self._SPORES("ligand.pdb", "ligand.mol2", "complete")
        else:
            shutil.copyfile(os.path.abspath(self._ligand_file), "ligand.mol2")

        ligcenter = self._molecule_center(reflig)

        ringAtoms = []
        for pair in self._mappingRefToLig:
            if any(pair[1] in sublist for sublist in self._GetRingSystems(ligand)):
                ringAtoms.append(pair[1])

        conf = ligand.GetConformer()
        if not ringAtoms:
            ringAtoms = [t[1] for t in self._mappingRefToLig]

        dis = []
        for atom in ringAtoms:
            dis.append(
                {
                    atom: math.sqrt(
                        (ligcenter[0] - conf.GetAtomPosition(atom).x) ** 2
                        + (ligcenter[1] - conf.GetAtomPosition(atom).y) ** 2
                        + (ligcenter[2] - conf.GetAtomPosition(atom).z) ** 2
                    )
                }
            )
        min_key = min(dis, key=lambda x: list(x.values())[0])

        atomID = list(min_key.keys())[0]
        return atomID

    def _handle_flexibility(self):
        # Handle flexibility for PLANTS
        if self._flexDist is not None:
            return [
                f"flexible_protein_side_chain_string {residue[0]}{residue[1]}"
                for residue in self._get_flex_residues()
            ]
        elif self._flexList is not None:
            residuelist = [
                [item[:3], int(item[3:])] for item in self._flexList.split(",")
            ]
            return [
                f"flexible_protein_side_chain_string {residue[0]}{residue[1]}"
                for residue in residuelist
            ]
        else:
            return []

    @staticmethod
    def _SPORES(inputfile, outputfile, mode):

        RC = os.system(
            f"""SPORES --mode {mode} {inputfile} {outputfile} > SPORES.log 2>&1 """
        )
        if RC != 0:
            raise SystemExit(
                "\nERROR!\nFailed to run the SPORES. See the {} for details.".format(
                    os.path.abspath("SPORES.log\n")
                )
            )

    @staticmethod
    def _plants_docking(
        i,
        dock_dir,
        radius=15,
        cluster_structures=10,
        xyz=None,
        fixedAtom=None,
        flex_res=None,
        min_only=False,
    ):

        template = f"""
#Input Options
protein_file             receptor.mol2

#Binding Site
bindingsite_center {xyz}
bindingsite_radius {radius}

#Cluster Algorithm
cluster_structures {cluster_structures}

#Scoring Functions
scoring_function chemplp
ligand_intra_score lj

#Output Options
write_multi_mol2 0
write_rescored_structures 1

#Fixed Scaffold
ligand_file ligand.mol2 fixed_scaffold_{fixedAtom}

#Flexible Side-chains
"""

        success = False

        while not success:
            f = open("plants_config", "w")
            f.write(template)
            for item in flex_res:
                f.write(item + "\n")
            f.close()

            curdir = os.getcwd()
            output_dir = str(uuid.uuid4())
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy(os.path.join(curdir, "receptor.mol2"), output_dir)
            shutil.copy(os.path.join(curdir, "ligand.mol2"), output_dir)
            shutil.copy(os.path.join(curdir, "plants_config"), output_dir)
            os.chdir(output_dir)

            mode = "rescore" if min_only else "screen"
            os.system(f"""PLANTS --mode {mode} plants_config > PLANTS.log 2>&1 """)

            for filename in os.listdir("."):
                if flex_res == []:
                    if (
                        filename.startswith("ligand_entry_00001_conf_")
                        and "protein" not in filename
                    ):
                        shutil.move(
                            filename,
                            os.path.join(
                                os.path.join("..", str(dock_dir)),
                                filename.replace("_entry_00001", f"_{i}"),
                            ),
                        )

                else:
                    residues = [item.split()[-1] for item in flex_res]
                    residues.append("UNL1")
                    unwanted_atoms = []
                    if (
                        filename.startswith("ligand_entry_00001_conf_")
                        and "protein" not in filename
                    ):
                        filenamepdb = filename.replace("_entry_00001", f"_{i}")
                        obabel_convert(filename, "ligand.pdb")
                        obabel_convert(
                            filename.replace(".mol2", "_protein.mol2"), "protein.pdb"
                        )
                        mol1 = Chem.MolFromPDBFile(
                            "ligand.pdb", removeHs=False, sanitize=False
                        )
                        mol2 = Chem.MolFromPDBFile(
                            "protein.pdb", removeHs=False, sanitize=False
                        )
                        mol = rdmolops.CombineMols(mol1, mol2)
                        Chem.MolToPDBFile(mol, "complex.pdb", flavor=1)
                        with open("complex.pdb", "r") as pdb_in:
                            lines = pdb_in.readlines()
                            atom_lines = [
                                line for line in lines if line.startswith("ATOM")
                            ]
                            conect_lines = [
                                line for line in lines if line.startswith("CONECT")
                            ]
                        atoms = []
                        for line in atom_lines:
                            parser = parse_pdb_line(line)
                            residue = f"{parser['residue_name']}{parser['residue_sequence_number']}"
                            if residue in residues:
                                atoms.append(line)
                            else:
                                unwanted_atoms.append(parser["atom_serial_number"])
                        conects = []
                        for line in conect_lines:
                            atoms_in_line = set(map(int, line[6:].split()))
                            if set(unwanted_atoms).isdisjoint(atoms_in_line):
                                conects.append(line)
                        with open(
                            os.path.join(
                                "..", dock_dir, filenamepdb.replace(".mol2", ".pdb")
                            ),
                            "w",
                        ) as pdb_out:
                            pdb_out.writelines(atoms)
                            pdb_out.writelines(conects)
            # shutil.move("complex.pdb", os.path.join('..', dock_dir, filenamepdb.replace('.mol2', '.pdb')))

            try:
                with open("ranking.csv", "r") as csv_in, open(
                    os.path.join("..", dock_dir, "Scores.csv"), "a"
                ) as csv_out:
                    reader = csv.reader(csv_in)
                    writer = csv.writer(csv_out)

                    # Skip the header
                    next(reader)

                    # Write docking scores
                    for row in reader:
                        ligand_entry = row[0]
                        ligand_last = ligand_entry.replace("_entry_00001", f"_{i}")
                        total_score = row[1]
                        writer.writerow([ligand_last, total_score])
                success = True
            except FileNotFoundError:
                print(f"The docking {i} failed. Running again...")

            os.chdir(curdir)
            shutil.rmtree(output_dir)

    @staticmethod
    def _combine_files(dockdir):

        # Generate a key for sorting strings that contain numbers,
        # ensuring that numerical parts are sorted numerically.
        natural_sort_key = lambda s: [
            int(text) if text.isdigit() else text.lower()
            for text in re.split("([0-9]+)", s)
        ]

        # Directory containing the Mol2 files
        files = sorted(
            [f for f in os.listdir(dockdir) if f.endswith(".mol2")],
            key=natural_sort_key,
        )

        csv_path = f"{dockdir}/Scores.csv"
        df = pd.read_csv(csv_path, header=None)

        flex = False
        if len(files) == 0:
            flex = True
            files = sorted(
                [f for f in os.listdir(dockdir) if f.endswith(".pdb")],
                key=natural_sort_key,
            )
            combined_lines = []
            conect_lines = []
            model_count = 0
            for pdb_file in files:
                with open(f"{dockdir}/{pdb_file}", "r") as file:
                    lines = file.readlines()
                    atom_lines = [line for line in lines if line.startswith("ATOM")]
                    if pdb_file == files[0]:
                        conect_lines = [
                            line for line in lines if line.startswith("CONECT")
                        ]
                        shutil.copy(f"{dockdir}/{pdb_file}", "ref_conf.pdb")
                    combined_lines.append(f"MODEL        {model_count}\n")
                    combined_lines.extend(atom_lines)
                    combined_lines.append(
                        "ENDMDL\n"
                    )  # Append ENDMDL after each file's ATOM lines
                    model_count += 1
            combined_lines.extend(conect_lines)
            combined_lines.append("END\n")
            with open("combined_file.pdb", "w") as combined_file:
                combined_file.writelines(combined_lines)
            df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: x + ".pdb")
        else:
            df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: x + ".mol2")

            # Create an SDF writer
        sdf_writer = Chem.SDWriter("conformers.sdf")
        sdf_writer.SetKekulize(False)

        # Process each Mol2 file
        for f in files:
            path = os.path.join(dockdir, f)
            if flex:
                mol = Chem.MolFromPDBFile(path, sanitize=False)
            else:
                mol = Chem.MolFromMol2File(path, sanitize=False)
                if mol is None:
                    try:
                        obabel_convert(path, path + ".sdf")
                    except:
                        pass
                    mol = next(Chem.SDMolSupplier(path + ".sdf", sanitize=False))
            if mol is not None:
                sdf_writer.write(mol)
        sdf_writer.close()

        # Create a dictionary for mapping filenames to their rows
        filename_to_row = {filename: row for row, filename in enumerate(files)}

        # Sort the dataframe based on the order of the mol2 files
        df["sort_index"] = df.iloc[:, 0].map(filename_to_row)
        df.sort_values(by="sort_index", inplace=True)

        df.drop(columns=["sort_index"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Save the rearranged CSV with index column
        df.columns = ["names", "Score"]
        df["names"] = df["names"].str.replace(r"\.\w+$", "", regex=True)

        df.to_csv("Scores.csv", index=True, index_label="Index")
        shutil.rmtree(dockdir)

    def _get_flex_residues(self, cutoff=3.5) -> List[Tuple[str, int]]:

        protein = mda.Universe(self._receptor_file)
        ligand = mda.Universe(self._query_molecule)

        atom1 = protein.select_atoms("not name H* CA CB C N O")
        atom2 = ligand.select_atoms("not name H*")

        res_list = []
        for res in atom1.residues:
            resi = atom1.select_atoms("resid {} ".format(res.resid))
            distances = mda.analysis.distances.distance_array(
                resi.positions, atom2.positions
            )
            if distances.min() < cutoff:
                res_list.append((res.resname, res.resid))
        return res_list

    @staticmethod
    def _GetRingSystems(
        mol, includeSpiro=False
    ):  # https://gist.github.com/greglandrum/de1751a42b3cae54011041dd67ae7415
        ri = mol.GetRingInfo()
        systems = []
        for ring in ri.AtomRings():
            ringAts = set(ring)
            nSystems = []
            for system in systems:
                nInCommon = len(ringAts.intersection(system))
                if nInCommon and (includeSpiro or nInCommon > 1):
                    ringAts = ringAts.union(system)
                else:
                    nSystems.append(system)
            nSystems.append(ringAts)
            systems = nSystems
        return systems
