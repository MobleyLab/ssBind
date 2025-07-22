import multiprocessing as mp
from contextlib import closing
from copy import deepcopy
from typing import Dict

from rdkit import Chem
from rdkit.Chem.AllChem import EmbedMolecule
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

        if self._minimize_only:
            # do nothing here, smina will minimize
            with Chem.SDWriter("conformers.sdf") as writer:
                writer.write(self._query_molecule)
            return

        maxRepeats = 100
        torsionPrefs = True

        for repeat in range(maxRepeats):

            self._generate_n_conformers(
                self._numconf, repeat * self._numconf, torsionPrefs
            )
            try:
                numconf_generated = len(Chem.SDMolSupplier("conformers.sdf"))
            except OSError:
                numconf_generated = 0

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
        ligand = Chem.AddHs(self._query_molecule, addCoords=True)
        ligEmbed = self._ConstrainedEmbed(
            ligand, randomseed=seed + 1, torsionPrefs=torsionPrefs
        )
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

    def _ConstrainedEmbed(
        self,
        mol,
        torsionPrefs=True,
        useTethers=True,
        coreConfId=-1,
        randomseed=2342,
        getForceField=UFFGetMoleculeForceField,
        **kwargs
    ):

        core = self._reference_substructure
        mappingLigToRef = self._mappingLigToRef

        outmol = Chem.Mol(mol)
        if mappingLigToRef is None:
            match = outmol.GetSubstructMatch(core)
        else:
            sorted_mapping = sorted(mappingLigToRef, key=lambda x: x[1])
            match = list(zip(*sorted_mapping))[0]
        if not match:
            raise ValueError("molecule doesn't match the core")
        coordMap = {}
        coreConf = core.GetConformer(coreConfId)
        for i, idxI in enumerate(match):
            corePtI = coreConf.GetAtomPosition(i)
            coordMap[idxI] = corePtI

        ci = EmbedMolecule(
            outmol,
            coordMap=coordMap,
            randomSeed=randomseed,
            useExpTorsionAnglePrefs=torsionPrefs,
            **kwargs
        )
        if ci < 0:
            raise ValueError("Could not embed molecule.")

        algMap = [(j, i) for i, j in enumerate(match)]

        if not useTethers:
            # clean up the conformation
            ff = getForceField(outmol, confId=0)
            for i, idxI in enumerate(match):
                for j in range(i + 1, len(match)):
                    idxJ = match[j]
                    d = coordMap[idxI].Distance(coordMap[idxJ])
                    ff.AddDistanceConstraint(idxI, idxJ, d, d, 100.0)
            ff.Initialize()
            n = 4
            more = ff.Minimize()
            while more and n:
                more = ff.Minimize()
                n -= 1
            # rotate the embedded conformation onto the core:
            rms = AlignMol(outmol, core, atomMap=algMap)
        else:
            # rotate the embedded conformation onto the core:
            rms = AlignMol(outmol, core, atomMap=algMap)
            ff = getForceField(outmol, confId=0)
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
            rms = AlignMol(outmol, core, atomMap=algMap)
        outmol.SetProp("EmbedRMS", str(rms))
        return outmol
