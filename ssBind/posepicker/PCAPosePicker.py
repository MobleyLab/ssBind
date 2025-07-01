import multiprocessing as mp
from contextlib import closing

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from MDAnalysis.analysis import pca
from rdkit import Chem
from rdkit.ML.Cluster import Butina

from ssBind.posepicker.PosePicker import PosePicker


class PCAPosePicker(PosePicker):
    def __init__(self, **kwargs) -> None:

        super().__init__(**kwargs)

        self._binsize = kwargs.get("bin", 0.25)
        self._distThresh = kwargs.get("distThresh", 0.5)
        self._numbin = kwargs.get("numbin", 10)

    def pick_poses(
        self, conformers: str = "conformers.sdf", csv_scores: str = "Scores.csv"
    ) -> None:
        """
        Performs clustering analysis on the PCA-transformed data.

        """

        u, confs, select, flex = self._process_inputs(conformers)

        pc = pca.PCA(u, select=select, align=False, mean=None, n_components=None).run()

        atoms = u.select_atoms(select)

        transformed = pc.transform(atoms, n_components=3)
        transformed.shape

        df = pd.DataFrame(transformed, columns=["PC{}".format(i + 1) for i in range(3)])

        df["Index"] = df.index

        scoredata = pd.read_csv(csv_scores, delimiter=",", header=0)
        PCA_Scores = pd.merge(df, scoredata, on="Index")

        label_data = []
        pcs = ["PC1", "PC2", "PC3"]
        for i in range(len(pcs)):
            for j in range(i + 1, len(pcs)):
                df = PCA_Scores

                # functions to find bin boundaries of each point based on binsize
                xmin = lambda x: x - (x % self._binsize)
                xmax = lambda x: x - (x % self._binsize) + self._binsize

                # set bin boundaries for each point in df (6 values, 2 for each PC)
                df[f"{pcs[i]}_xmin"] = df[f"{pcs[i]}"].apply(xmin)
                df[f"{pcs[i]}_xmax"] = df[f"{pcs[i]}"].apply(xmax)

                df[f"{pcs[j]}_ymin"] = df[f"{pcs[j]}"].apply(xmin)
                df[f"{pcs[j]}_ymax"] = df[f"{pcs[j]}"].apply(xmax)

        # group by pairs of PCs
        groups = df.groupby(["PC1_xmin", "PC2_ymin", "PC1_xmax", "PC2_ymax"])

        # get counts of points in each group, and mean of Score
        scores = groups.Score.mean().reset_index()
        counts = groups.count().reset_index()

        raw_data = PCA_Scores
        scores_sorted = scores.sort_values(by=["Score"])
        top_bins = scores_sorted.head(self._numbin)
        extracted_data = top_bins[["PC1_xmin", "PC2_ymin", "PC1_xmax", "PC2_ymax"]]
        for a, rowa in extracted_data.iterrows():
            for b, rowb in raw_data.iterrows():
                if (
                    rowa["PC1_xmin"] < rowb["PC1"] < rowa["PC1_xmax"]
                    and rowa["PC2_ymin"] < rowb["PC2"] < rowa["PC2_ymax"]
                ):
                    label_data.append(rowb)

        counts_sorted = counts.sort_values(by=["Score"])
        top_bins = counts_sorted.tail(self._numbin)
        extracted_data = top_bins[["PC1_xmin", "PC2_ymin", "PC1_xmax", "PC2_ymax"]]

        for a, rowa in extracted_data.iterrows():
            for b, rowb in raw_data.iterrows():
                if (
                    rowa["PC1_xmin"] < rowb["PC1"] < rowa["PC1_xmax"]
                    and rowa["PC2_ymin"] < rowb["PC2"] < rowa["PC2_ymax"]
                ):
                    if any(row["Index"] == rowb["Index"] for row in label_data):
                        label_data.append(rowb)

        cids = []
        index_dict = []

        for i, entry in enumerate(label_data):
            cids.append(confs[int(entry["Index"])])
            index_dict.append({i: int(entry["Index"])})

        dists = []
        tasks = [(i, j, cids[i], cids[j]) for i in range(len(cids)) for j in range(i)]
        with closing(mp.Pool(processes=self._nprocs)) as pool:
            results = pool.map(self._calculate_rms, tasks)

        dists = [rms for _, _, rms in sorted(results, key=lambda x: (x[0], x[1]))]

        clusts = Butina.ClusterData(
            dists, len(cids), self._distThresh, isDistData=True, reordering=True
        )

        labels = []
        PC1 = []
        PC2 = []
        PC3 = []
        mode = 1
        for i in clusts:
            a = label_data[i[0]]
            PC1.append(a["PC1"])
            PC2.append(a["PC2"])
            PC3.append(a["PC3"])
            dict_i = index_dict[i[0]]
            labels.append(dict_i)

            if flex == True:
                model = int(next(iter(dict_i.values())))
                u.trajectory[model]
                u.atoms.write(f"model_{mode}.pdb")
                # get_model_compex(
                #   conformers, model, self._receptor_file, f"model_{model}.pdb"
                # )
            else:
                sdwriter = Chem.SDWriter(f"model_{mode}.sdf")
                sdwriter.write(confs[int(next(iter(dict_i.values())))])
                sdwriter.close()
            mode += 1

        pcs = ["PC1", "PC2", "PC3"]

        # creating own colormap so the contour plot pops more
        contour_colors = ["#E5E5E5", "#999999", "#4C4C4C", "#000000"]
        cmap1 = LinearSegmentedColormap.from_list("mycmap", contour_colors)

        for i in range(len(pcs)):
            for j in range(i + 1, len(pcs)):
                colors = df["Score"].to_numpy()
                fig, ax = plt.subplots()
                scatter = ax.scatter(
                    df[f"{pcs[i]}"],
                    df[f"{pcs[j]}"],
                    s=0.5,
                    c=colors,
                    cmap="gist_rainbow",
                )
                sns.kdeplot(
                    x=df[f"{pcs[i]}"],
                    y=df[f"{pcs[j]}"],
                    levels=[0.010, 0.050, 0.20, 0.50],
                    bw_adjust=0.78,
                    cmap=cmap1,
                )
                ax.set_xlabel(f"{pcs[i]}")
                ax.set_ylabel(f"{pcs[j]}")
                ax.set(
                    xlim=(df[f"{pcs[i]}"].min(), df[f"{pcs[i]}"].max()),
                    ylim=(df[f"{pcs[j]}"].min(), df[f"{pcs[j]}"].max()),
                )
                fig.colorbar(scatter, cmap="gist_rainbow", label="Score")
                for index, dict_i in enumerate(labels):
                    coord_index = list(dict_i.values())[0]
                x_coord = df.loc[df.Index == coord_index, f"{pcs[i]}"]
                y_coord = df.loc[df.Index == coord_index, f"{pcs[j]}"]
                ax.text(x_coord, y_coord, f"{index+1}")
                fig.savefig(f"{pcs[i]}-{pcs[j]}.svg", format="svg")

    @staticmethod
    def _calculate_rms(params):
        i, j, cid_i, cid_j = params
        rms = PCAPosePicker._rmsd(cid_i, cid_j)
        return i, j, rms
