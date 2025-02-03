from typing import List

import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdMolTransforms import GetDihedralDeg
from rdkit.Chem.TorsionFingerprints import CalculateTorsionLists

from ssBind.posepicker.PosePicker import PosePicker


class TorsionPosePicker(PosePicker):

    def __init__(self, **kwargs) -> None:
        """Initialize pose picker.

        Args:
            receptor_file (str): receptor.pdb filename - we don't need it here
        """
        super().__init__(**kwargs)

        self._cutoff_angle = kwargs.get("cutoff_angle", 90)
        self._rmsd_symmetry_threshold = kwargs.get("rmsd_symmetry_threshold", 0.3)
        self._rmsd_threshold = kwargs.get("rmsd_threshold", 2.0)
        self._dG_threshold = kwargs.get("dG_threshold", 5)
        self._dG_clustering_threshold = kwargs.get("dG_clustering_threshold", 0)
        self._dG_diff_clustering_threshold = kwargs.get(
            "dG_diff_clustering_threshold", np.inf
        )
        self._refMols = kwargs.get("ref_ligands", [])

    def pick_poses(
        self, conformers: str = "conformers.sdf", csv_scores: str = "Scores.csv"
    ) -> None:
        """Select poses based on energy-based conformer pooling, and write to model_x.sdf

        Args:
            conformers (str, optional): SD file with (minimized) conformers. Defaults to "conformers.sdf".
            csv_scores (str, optional): CSV with conformer energies in kcal/mol. Defaults to "Scores.csv".
        """

        _, confs, _, _ = self._process_inputs(conformers)
        mols = [c for c in confs]

        torsionLists, ringTorsionLists = CalculateTorsionLists(mols[0])
        nonRingTorsions = self._flattenAndUniquifyTorsions(torsionLists)
        ringTorsions = self._flattenAndUniquifyTorsions(ringTorsionLists)
        torsions = nonRingTorsions + ringTorsions

        scores = pd.read_csv(csv_scores, index_col="Index")
        scores["diffToBest"] = scores.Score - min(scores.Score)
        molsBelowThresh = scores[
            (scores.Score < self._dG_clustering_threshold)
            & (scores.diffToBest < self._dG_diff_clustering_threshold)
        ].index

        G = self._make_graph(mols, molsBelowThresh, torsions)
        df_clusters, df_confs = self._getStateData(G, mols, scores)

        df_to_write = df_clusters[df_clusters.dGdiff <= self._dG_threshold]
        self._write_models(df_to_write)

        df_clusters_to_write = df_clusters.drop(columns="mol")
        df_clusters_to_write.to_csv(
            "cluster_info.csv", float_format="%.5f", index=False
        )
        df_confs.to_csv("conf_info.csv", float_format="%.5f", index=False)

    def _make_graph(
        self, mols: List[Mol], molsBelowThresh: List[int], torsions: List[List[int]]
    ) -> nx.Graph:
        """Construct a NetworkX graph of conformers, taking into consideration the RMSD and torion-
        based cutoffs for pairwise conformer comparison. Edges in the graph represent conformers
        which are not separated by energy barriers.

        Args:
            confs (List[Mol]): List of conformers

        Returns:
            nx.Graph: Graph with connected subgraphs representing FE basins
        """

        angles = []

        for m in mols:
            angle = self._getAngles(m, torsions)
            angles.append(angle)

        G = nx.empty_graph(len(mols))

        for i1 in molsBelowThresh:
            for i2 in molsBelowThresh[:i1]:

                m1, m2 = mols[i1], mols[i2]

                dangle = self._getAngleDiff(angles[i1], angles[i2])
                rmsd_12 = self._rmsd(m1, m2)

                no_torsion_barrier = (np.max(np.abs(dangle)) < self._cutoff_angle) and (
                    rmsd_12 < self._rmsd_threshold
                )
                symmetry_equivalent = rmsd_12 < self._rmsd_symmetry_threshold

                if no_torsion_barrier or symmetry_equivalent:
                    G.add_edge(i1, i2)

        return G

    def _getStateData(self, G: nx.Graph, mols: List[Mol], scores: pd.DataFrame):
        states = pd.DataFrame(columns=["ID", "dG", "dGdiff", "Nstates", "mol"])

        refs = self._refMols
        mol_data = pd.DataFrame(
            columns=["ID", "dG", "dGdiff", "clusterID", "cluster_dG", "cluster_dGdiff"]
            + [f"rmsd_{i+1}" for i in range(len(refs))]
        )

        for g in nx.connected_components(G):
            min_conf = scores.iloc[list(g)].nsmallest(1, "Score").index[0]
            dG_cluster = scores.iloc[min_conf].Score
            dGAboveMin_cluster = scores.iloc[min_conf].diffToBest

            g_dict = {
                "ID": min_conf,
                "Nstates": int(len(g)),
                "dG": dG_cluster,
                "dGdiff": dGAboveMin_cluster,
                "mol": mols[min_conf],
            }
            new_df = pd.DataFrame([g_dict])
            states = pd.concat(
                [states.astype(new_df.dtypes), new_df], axis=0, ignore_index=True
            )

            for gg in g:

                gg_dict = {
                    "ID": gg,
                    "dG": scores.iloc[gg].Score,
                    "dGdiff": scores.iloc[gg].diffToBest,
                    "clusterID": min_conf,
                    "cluster_dG": dG_cluster,
                    "cluster_dGdiff": dGAboveMin_cluster,
                }
                rms_dict = {
                    f"rmsd_{i+1}": self._rmsd(mols[gg], ref)
                    for i, ref in enumerate(refs)
                }
                new_df = pd.DataFrame([gg_dict | rms_dict])
                mol_data = pd.concat(
                    [mol_data.astype(new_df.dtypes), new_df], axis=0, ignore_index=True
                )

        clusterRank = mol_data.groupby("clusterID").dG.min().rank().astype("int")
        mol_data["clusterRank"] = mol_data.clusterID.apply(lambda x: clusterRank[x])
        for i in range(len(refs)):
            mol_data[f"rankRmsd_{i+1}"] = mol_data[f"rmsd_{i+1}"].rank().astype("int")

        states = states.sort_values(by="dG")
        mol_data = mol_data.sort_values(by="ID").reset_index(drop=True)
        return states, mol_data

    @staticmethod
    def _write_models(df_confs_scored: pd.DataFrame) -> None:
        """Write SDF outputs for all representative states in the input df, including
        energy information as SD tags.

        Args:
            df_confs_scored (pd.DataFrame): Subset of representative conformers to write
        """

        for i, row in df_confs_scored.reset_index().iterrows():
            mol = row.mol
            mol.SetProp("dG", str(row.dG))
            with Chem.SDWriter(f"model_{i+1}.sdf") as writer:
                writer.write(row.mol)

    @staticmethod
    def _flattenAndUniquifyTorsions(lists: list):
        flatlists = [s for sublist in lists for s in sublist[0]]
        uniqueLists = (
            pd.DataFrame(flatlists).drop_duplicates(subset=[1, 2]).values.tolist()
        )
        return uniqueLists

    @staticmethod
    def _getAngles(mol: Mol, torsions: List[List[int]]):
        conf = mol.GetConformer(0)
        angles = np.array([GetDihedralDeg(conf, *indices) for indices in torsions])
        return angles

    @staticmethod
    def _getAngleDiff(angles1: np.array, angles2: np.array):
        return ((angles2 - angles1) + 180) % 360 - 180
