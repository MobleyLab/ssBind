import logging
import re
from io import StringIO
from typing import List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import SDWriter
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdDistGeom import GetExperimentalTorsions
from rdkit.Chem.rdMolTransforms import GetDihedralDeg
from spyrmsd.molecule import Molecule
from spyrmsd.rmsd import rmsdwrapper


class TorsionPosePicker:

    kbT = 0.6  # kcal / mol - used for entropy estimation

    def __init__(self, **kwargs) -> None:
        """Initialize pose picker.

        Args:
            receptor_file (str): receptor.pdb filename - we don't need it here
        """
        self._nprocs = kwargs.get("nprocs", 1)

        self._cutoff_angle = kwargs.get("cutoff_angle", 90)
        self._rmsd_symmetry_threshold = kwargs.get("rmsd_symmetry_threshold", 0.3)
        self._rmsd_threshold = kwargs.get("rmsd_threshold", 2.0)
        self._dG_threshold = kwargs.get("dG_threshold", 1.0)

        # initialize logger to capture information about torsions from RDKit output
        # there is probably a better way to do this if we don't need multiplicities
        logger = logging.getLogger("rdkit")
        logger.handlers[0].setLevel(logging.WARN)
        logger.handlers[0].setFormatter(
            logging.Formatter("[RDKit] %(levelname)s:%(message)s")
        )

        rdBase.LogToPythonLogger()
        self._logger = logger

    def pick_poses(
        self, conformers: str = "conformers.sdf", csv_scores: str = "Scores.csv"
    ) -> None:
        """Select poses based on energy-based conformer pooling, and write to model_x.sdf

        Args:
            conformers (str, optional): SD file with (minimized) conformers. Defaults to "conformers.sdf".
            csv_scores (str, optional): CSV with conformer energies in kcal/mol. Defaults to "Scores.csv".
        """

        confs = Chem.SDMolSupplier(conformers)
        mols = [c for c in confs]

        scores = pd.read_csv(csv_scores, index_col="Index")
        scores["Score"] = scores.Score - min(scores.Score)

        G = self._make_graph(mols)
        df_confs_scored = self._estimate_fe(G, scores, mols)

        df_to_write = df_confs_scored[df_confs_scored.delta_G <= self._dG_threshold]
        self._write_models(df_to_write)
        self._write_data(df_confs_scored)

    def _make_graph(self, confs: List[Mol]) -> nx.Graph:
        """Construct a NetworkX graph of conformers, taking into consideration the RMSD and torion-
        based cutoffs for pairwise conformer comparison. Edges in the graph represent conformers
        which are not separated by energy barriers.

        Args:
            confs (List[Mol]): List of conformers

        Returns:
            nx.Graph: Graph with connected subgraphs representing FE basins
        """

        mol_info = []

        for m in confs:
            info = self._getTorsionInfo(m)
            mol_info.append(info)

        G = nx.empty_graph(len(confs))

        for i1, m1 in enumerate(confs):
            for i2, m2 in enumerate(confs[:i1]):

                dangle = self._calcDangle(mol_info[i1], mol_info[i2])
                rmsd_12 = self._rmsd(m1, m2)

                no_torsion_barrier = (np.max(np.abs(dangle)) < self._cutoff_angle) and (
                    rmsd_12 < self._rmsd_threshold
                )
                symmetry_equivalent = rmsd_12 < self._rmsd_symmetry_threshold

                if no_torsion_barrier or symmetry_equivalent:
                    G.add_edge(i1, i2)

        self._mol_info = mol_info
        return G

    @staticmethod
    def _estimate_fe(
        G: nx.Graph, scores: pd.DataFrame, confs: List[Mol]
    ) -> pd.DataFrame:
        """Estimate free energy of each basin from the potential energy of the lowest conformer
        and the number of conformers in each basin.

        Args:
            G (nx.Graph): Graph with conformers
            scores (pd.DataFrame): Potential energy for each conformer
            confs (List[Mol]): conformers

        Returns:
            pd.DataFrame: Representative conformers for each basin, ranked by estimated delta G
        """

        states = pd.DataFrame(
            columns=["ID", "states", "U", "Nstates", "delta_G", "mol"]
        )
        lowest_energy_conf = scores[scores.Score == 0].index[0]

        ref_state = [g for g in nx.connected_components(G) if lowest_energy_conf in g][
            0
        ]
        # ref_Z = len(
        #     ref_state
        # )  # approx partition function by the number of confs in this state
        ref_Z = sum(
            [
                np.exp(-scores.iloc[conf].Score / TorsionPosePicker.kbT)
                for conf in ref_state
            ]
        )

        for g in nx.connected_components(G):
            u = min(
                scores.iloc[list(g)].Score
            )  # approx pot energy of state by minimum of the state (because exp weighting)
            min_conf = scores[scores.Score == u].index[0]
            # Z = len(g)
            Z = sum(
                [np.exp(-scores.iloc[conf].Score / TorsionPosePicker.kbT) for conf in g]
            )
            # delta_G = u - TorsionPosePicker.kbT * np.log(
            #     Z / ref_Z
            # )  # approx FE difference to ref (which has u=0 by definition)
            delta_G = -TorsionPosePicker.kbT * np.log(Z / ref_Z)
            g_dict = {
                "ID": min_conf,
                "states": ",".join([str(x) for x in g]),
                "U": u,
                "Nstates": int(len(g)),
                "delta_G": delta_G,
                "mol": confs[min_conf],
            }
            states = pd.concat(
                [states, pd.DataFrame([g_dict])], axis=0, ignore_index=True
            )

        states = states.sort_values(by="delta_G")
        return states

    @staticmethod
    def _write_models(df_confs_scored: pd.DataFrame) -> None:
        """Write SDF outputs for all representative states in the input df, including
        energy information as SD tags.

        Args:
            df_confs_scored (pd.DataFrame): Subset of representative conformers to write
        """

        for i, row in df_confs_scored.reset_index().iterrows():
            mol = row.mol
            mol.SetProp("U", str(row.U))
            mol.SetProp("delta_G", str(row.delta_G))
            with SDWriter(f"model_{i+1}.sdf") as writer:
                writer.write(row.mol)

    @staticmethod
    def _write_data(df_confs_scored: pd.DataFrame) -> None:
        """Write dataframe without ROMols

        Args:
            df_confs_scored (pd.DataFrame): Dataframe to store as csv
        """
        df_scores = df_confs_scored.drop(columns="mol")
        df_scores.to_csv("cluster_info.csv")

    def _getTorsionInfo(self, mol: Mol) -> Tuple[np.array, np.array]:
        """Helper function to extract torsion indices for each conformer and get dihedral angles.

        Args:
            mol (Mol): conformer

        Returns:
            Tuple[np.array, np.array]: Indices (i,j,k,l) of each torsion, and the associated signed
            angles in the interval [-180,180] degrees.
        """

        text = self._getExpTorsionText(mol)

        lines = text.split("\n")
        torsion_indices = np.array([])

        for line in lines:

            # get indices
            re_match = re.findall(r"\]\:(.*?)\,", line)
            if len(re_match):
                indices = set([int(m) for m in re_match[0].split()])
                torsion_indices = np.append(torsion_indices, indices)

            conf = mol.GetConformer(0)
            angles = np.array(
                [GetDihedralDeg(conf, *indices) for indices in torsion_indices]
            )

        return torsion_indices, angles

    def _getExpTorsionText(self, mol: Mol) -> str:
        """Get RDKit output on experimental torsions for a given molecule from the logger

        Args:
            mol (Mol): molecule

        Returns:
            str: RDKit output with torsion information captured as a string
        """

        logger_sio = StringIO()
        handler = logging.StreamHandler(logger_sio)
        handler.setLevel(logging.INFO)

        self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

        GetExperimentalTorsions(mol, printExpTorsionAngles=True)
        text = logger_sio.getvalue()
        return text

    @staticmethod
    def _calcDangle(
        molinfo: Tuple[np.array, np.array], molinfo_ref: Tuple[np.array, np.array]
    ) -> np.array:
        """Calculate differences in dihedral angles between two conformers

        Args:
            molinfo (Tuple[np.array, np.array]): torsion indices and dihedral angles
            molinfo_ref (Tuple[np.array, np.array]): torsion indices and dihedral angles of reference mol

        Returns:
            np.array: Signed angle difference in degrees, [-180,180]
        """

        indices, angles = molinfo
        indices_ref, angles_ref = molinfo_ref

        index_mapping = [list(indices_ref).index(torsion) for torsion in indices]
        dangle = ((angles[index_mapping] - angles_ref) + 180) % 360 - 180

        return dangle

    @staticmethod
    def _rmsd(mol: Mol, ref: Mol) -> float:
        """Calculate symmetry-corrected RMSD between two conformers

        Args:
            mol (Mol): conformer 1
            ref (Mol): conformer 2

        Returns:
            float: RMSD in A
        """
        return rmsdwrapper(Molecule.from_rdkit(ref), Molecule.from_rdkit(mol))[0]
