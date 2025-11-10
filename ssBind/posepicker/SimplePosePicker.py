import itertools
import multiprocessing as mp
import subprocess
from contextlib import closing
from typing import List

import numpy as np
import pandas as pd
import pytraj as pt
from rdkit import Chem
from rdkit.Chem.rdShapeHelpers import ShapeTanimotoDist
from rdkit.ML.Cluster import Butina
from sklearn.cluster import HDBSCAN

from ssBind.posepicker.PosePicker import PosePicker


class SimplePosePicker(PosePicker):

    def __init__(self, **kwargs) -> None:

        super().__init__(**kwargs)

        self._query_molecule = kwargs.get("query_molecule")
        self._cluster = kwargs.get("clusteringMethod", "hdbscan")  # or butina
        self._distThresh = kwargs.get(
            "distThresh", 0.5
        )  # dist threshold for clustering (both)
        self._strategy = kwargs.get(
            "selectionStrategy", "mixed"
        )  # or "score" "similarity"
        self._numPoses = kwargs.get("outputPoses", 5)  # number of poses to output
        self._excludeSingletons = kwargs.get(
            "excludeSignletons", self._strategy == "similarity"
        )  # exclude 1-molecule clusters in hdbscan
        self._nprocs = kwargs.get("nprocs", 1)

    def pick_poses(
        self, conformers: str = "conformers.sdf", csv_scores: str = "Scores.csv"
    ) -> None:
        """Choose poses from generated conformers, by first clustering them, and
        subsequently ranking based on docking scores and/or Tanimoto similarity
        to the query molecule.

        Args:
            conformers (str, optional): File with conformers. Defaults to "conformers.sdf".
            csv_scores (str, optional): Sorted file with scores for each conformer. Defaults to "Scores.csv".
        """

        _, mols, _, _ = self._process_inputs(conformers)
        mols_noh = [Chem.RemoveAllHs(mol, sanitize=False) for mol in mols]

        pdist = self._calc_rmsd(mols_noh)

        # if conformers.endswith("dcd"):
        #     pdist = self._calc_rmsd_pytraj(conformers)
        # else:
        #     cmd = ["obabel", conformers, "-O", "conformers.pdb", "-d"]  # remove Hs
        #     subprocess.run(cmd, check=False)

        #     pdist = self._calc_rmsd_pytraj("conformers.pdb")

        if pdist.shape[0] == 1:
            cluster_ids = [0]
        else:
            if self._cluster == "butina":
                cluster_ids = self._cluster_Butina(pdist)
            elif self._cluster == "hdbscan":
                cluster_ids = self._cluster_HDBSCAN(pdist)
            else:
                raise Exception(f"Unknown clustering method: {self._cluster}")

        # get scores
        if self._strategy != "similarity":
            scores = pd.read_csv(csv_scores)["Score"]
        else:
            scores = None
        if self._strategy in ["similarity", "mixed"]:
            lig_noh = Chem.RemoveHs(self._query_molecule)
            similarity = [ShapeTanimotoDist(mol, lig_noh) for mol in mols_noh]
        else:  # so we don't waste time calculating it
            similarity = [0] * len(mols)

        df = pd.DataFrame.from_dict(
            {
                "mols": mols,
                "cluster": cluster_ids,
                "score": scores,
                "similarity": similarity,
            }
        )

        # pick poses from chosen strategy
        if self._strategy in ["score", "similarity"]:
            topN = self._find_bestscoring_poses(df, self._strategy, self._numPoses)
        elif self._strategy == "mixed":
            top_score = self._find_bestscoring_poses(df, "score", self._numPoses)
            top_sim = self._find_bestscoring_poses(df, "similarity", self._numPoses)

            topN = pd.concat([top_score, top_sim]).sort_index()
            topN["dupe"] = topN.duplicated(subset="cluster")
            topN = topN.sort_values(by="dupe")[
                : min(self._numPoses, topN.shape[0])
            ].reset_index()
        else:
            raise Exception(f"Unknown pose selection method: {self._strategy}")

        self.df = df
        self.topN = topN

        # write poses to sdf
        output_file = "selected_conformers.sdf"
        writer = Chem.SDWriter(output_file)
        writer.SetKekulize(False)
        for mol in topN.mols:
            writer.write(mol)
        writer.close()

    def _find_bestscoring_poses(
        self, df: pd.DataFrame, score: str, num: int
    ) -> pd.DataFrame:
        """Get top N poses according to chosen scoring meric, considering only one
        unique pose per cluster.

        Args:
            df (pd.DataFrame): Dataframe with conformers and their metadata.
            score (str): Scoring metric - score or similarity
            num (int): Number of poses to retain

        Returns:
            pd.DataFrame: Dataframe with N rows for the top poses.
        """
        df_copy = df[df.cluster >= 0]  # exclude hdbscan singletons if needed
        topN = df_copy[
            df_copy.groupby("cluster")[score].transform("min") == df_copy[score]
        ]
        topN = topN.drop_duplicates(subset=["cluster", score])
        top_N = topN.sort_values(by=score)[: min(num, topN.shape[0])].reset_index()
        return top_N

    def _cluster_HDBSCAN(self, pdist: np.ndarray) -> List[int]:
        """Cluster conformer "trajectory" using HDBSCAN

        Args:
            pdist (np.ndarray): RMSD matrix

        Returns:
            List[int]: Cluster IDs
        """

        hdb = HDBSCAN(
            metric="precomputed",
            cluster_selection_epsilon=self._distThresh,
            min_samples=min(5, pdist.shape[0]),
        ).fit(pdist)

        start = len(set(hdb.labels_)) - 1
        c = itertools.count()
        if self._excludeSingletons:
            cluster_ids = hdb.labels_
        else:
            cluster_ids = [l if l >= 0 else start + next(c) for l in hdb.labels_]
        return cluster_ids

    def _cluster_Butina(self, pdist: np.ndarray) -> List[int]:
        """Cluster conformers using the Butina method.

        Args:
            pdist (np.ndarray): RMSD matrix

        Returns:
            List[int]: Cluster IDs
        """

        flat_distances = [pdist[i, j] for i in range(pdist.shape[0]) for j in range(i)]
        clusts_butina = Butina.ClusterData(
            flat_distances,
            pdist.shape[0],
            self._distThresh,
            isDistData=True,
            reordering=True,
        )

        cluster_ids = list(
            list(
                zip(*sorted([(cc, i) for i, c in enumerate(clusts_butina) for cc in c]))
            )[1]
        )
        return cluster_ids

    def _calc_rmsd(self, mols: List[Chem.Mol]) -> np.ndarray:
        """Calculate symmetry-corrected RMSD with the spyrmsd package. This is slow but works.

        Args:
            mols (List[Chem.Mol]): Conformers

        Returns:
            np.ndarray: RMSD matrix
        """
        N = len(mols)
        tasks = [(i, j, mols[i], mols[j]) for i in range(N) for j in range(i)]
        with closing(mp.Pool(processes=self._nprocs)) as pool:
            results = pool.map(self._calculate_rms, tasks)
        dists = np.empty((N, N))
        for i, j, rms in results:
            dists[i, j] = rms
            dists[j, i] = rms
        for i in range(N):
            dists[i, i] = 0
        return dists

    def _calc_rmsd_pytraj(self, conformers: str) -> np.ndarray:
        """DEPRECATED Calculate RMSD matrix using PyTraj. A symmetry-corrected RMSD calculation
        is attmepted first, falling back to the standard RMSD upon failure.
        This is much faster than spyrmsd, but was found to sometimes be quite wrong.

        Args:
            conformers (str): File containing conformers. Must be PDB, as Pytraj does not
            suppoer SDF.

        Returns:
            np.ndarray: RMSD matrix
        """

        if conformers.endswith("pdb"):
            ptraj = pt.load(conformers)
        else:  # dcd
            ptraj = pt.load(conformers, "complex.pdb")["(:UNK)"]

        try:
            pdist = np.empty((ptraj.n_frames, ptraj.n_frames))
            for i in range(ptraj.n_frames):
                pdist[i] = pt.symmrmsd(ptraj, ref=i, fit=False)
            # symmrmsd sometimes gives wrong results, not sure why
            assert (
                max(
                    [
                        (abs(pdist[i, j] - pdist[j, i]))
                        for i in range(ptraj.n_frames)
                        for j in range(ptraj.n_frames)
                    ]
                )
                < 1e-5
            )
        except:
            print(
                f"Warning: Pytraj symmrmsd failed, no symmetry correction for clustering."
            )
            pdist = np.empty((ptraj.n_frames, ptraj.n_frames))
            for i in range(ptraj.n_frames):
                pdist[i] = pt.rmsd(ptraj, ref=i, nofit=True)
        return pdist
