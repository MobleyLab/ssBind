import multiprocessing as mp
from contextlib import closing
from copy import deepcopy
from typing import Dict

from rdkit import Chem
from rdkit.Chem.AllChem import ConstrainedEmbed, EmbedMolecule
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField
from rdkit.Chem.rdMolAlign import AlignMol

from ssBind.generator import ConformerGenerator


class RDkitConformerGenerator(ConformerGenerator):

    def __init__(
        self,
        receptor_file: str,
        query_molecule: Mol,
        reference_substructure: Mol,
        **kwargs: Dict
    ) -> None:
        super().__init__(
            receptor_file, query_molecule, reference_substructure, **kwargs
        )

    def generate_conformers(self) -> None:
        """Generate conformers using random embeddings via RDKit."""

        maxRepeats = 10
        torsionPrefs = True

        for repeat in range(maxRepeats):

            self._generate_n_conformers(
                self._numconf, repeat * self._numconf, torsionPrefs
            )
            numconf_generated = len(Chem.SDMolSupplier("conformers.sdf"))

            # check if we have the number of conformers desired by the user
            if numconf_generated >= self._numconf:
                break

            if self._numconf > 10 and (numconf_generated < self._numconf / 10):
                print(
                    "Warning: generating conformers without torsion angle preferences\nbecause of too frequent clashes"
                )
                torsionPrefs = False

        ###Filter conformers having stearic clashes, clash with the protein, duplicates.
        print(
            "\n{} conformers have been generated.".format(
                len(Chem.SDMolSupplier("conformers.sdf"))
            )
        )

    def _generate_n_conformers(self, n: int, n_offset: int, torsionPrefs: bool) -> None:
        """Helper function to start multiprocessing and generate n conformers

        Args:
            n (int): Number of conformers to generate
            n_offset (int): Offset for random seed - this is the number of confs already generated
                (incl the clashed ones)
            torsionPrefs (bool): Whether or not to set torsion angle preferences
        """

        with closing(mp.Pool(processes=self._nprocs)) as pool:
            pool.starmap(
                self._gen_conf_rdkit,
                [(j, torsionPrefs) for j in range(n_offset, n + n_offset)],
            )

    def _gen_conf_rdkit(self, seed: int, torsionPrefs: bool = True) -> None:
        """Generate one conformer using RDKit.

        Args:
            seed (int): Random seed for constrained embedding.
        """
        ligand = Chem.AddHs(self._query_molecule)
        ligEmbed = self._embed(ligand, seed + 1, torsionPrefs)
        outmol = Chem.RemoveHs(ligEmbed)

        self._filtering(outmol)

    def _embed2(self, ligand: Mol, seed: int = -1, torsionPrefs: bool = True) -> Mol:
        """Use distance geometry (RDKit EmbedMolecule) to generate a conformer of ligand
        tethering to the reference structure (coordMap).

        Args:
            ligand (Mol): Molecule to generate conformer
            seed (int, optional): Random seed for embedding. Defaults to -1.

        Returns:
            Mol: New conformer of ligand
        """

        coordMap = {}
        ligConf = ligand.GetConformer(0)
        for _, ligIdx in self._mappingRefToLig:
            ligPtI = ligConf.GetAtomPosition(ligIdx)
            coordMap[ligIdx] = ligPtI

        l_embed = deepcopy(ligand)
        EmbedMolecule(
            l_embed,
            coordMap=coordMap,
            randomSeed=seed,
            useExpTorsionAnglePrefs=torsionPrefs,
        )
        self._alignToRef(l_embed)
        return l_embed

    def _embed(
        self,
        ligand: Mol,
        seed: int = -1,
        torsionPrefs: bool = True,
        getForceField=UFFGetMoleculeForceField,
        **kwargs
    ):

        core = self._reference_substructure
        match = list(zip(*self._mappingLigToRef))[0]

        coordMap = {}
        ligConf = ligand.GetConformer(-1)
        for _, ligIdx in self._mappingRefToLig:
            ligPtI = ligConf.GetAtomPosition(ligIdx)
            coordMap[ligIdx] = ligPtI

        ci = EmbedMolecule(
            ligand,
            coordMap=coordMap,
            randomSeed=seed,
            useExpTorsionAnglePrefs=torsionPrefs,
            **kwargs
        )
        if ci < 0:
            raise ValueError("Could not embed molecule.")

        algMap = self._mappingLigToRef

        # rotate the embedded conformation onto the core:
        rms = AlignMol(ligand, core, atomMap=algMap)
        ff = getForceField(ligand, confId=0)
        conf = core.GetConformer()
        for i in range(core.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            pIdx = ff.AddExtraPoint(p.x, p.y, p.z, fixed=True) - 1
            ff.AddDistanceConstraint(pIdx, match[i], 0, 0, 100.0)
        ff.Initialize()
        n = 4
        more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
        while more and n:
            more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
            n -= 1
        # realign
        rms = AlignMol(ligand, core, atomMap=algMap)
        ligand.SetProp("EmbedRMS", str(rms))
        return ligand
