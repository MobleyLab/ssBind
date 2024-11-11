import itertools
import multiprocessing as mp
from contextlib import closing
from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem.rdchem import Mol

from ssBind.generator.ConformerGenerator import ConformerGenerator


class AngleConformerGenerator(ConformerGenerator):

    def __init__(
        self,
        receptor_file: str,
        query_molecule: str,
        reference_substructure: str,
        degree=60,
        seed=0,
        **kwargs,
    ) -> None:
        super().__init__(
            receptor_file, query_molecule, reference_substructure, **kwargs
        )

        self._degree = degree
        self._seed = seed

    def generate_conformers(self) -> None:
        """Generate conformers unsing angle sampling"""

        molDihedrals = self._get_uniqueDihedrals()
        inputs = itertools.product(
            self._degreeRange(self._degree), repeat=len(molDihedrals)
        )

        numconf_angle = int((360 / self._degree) ** len(molDihedrals))
        np.random.seed(self._seed)
        if numconf_angle > self._numconf:
            subsample = np.random.choice(
                numconf_angle, size=self._numconf, replace=False, p=None
            )
        else:
            subsample = np.arange(self._numconf)
        new_list = [item for index, item in enumerate(inputs) if index in subsample]
        print(
            f"\nConformational sampling is running for {len(molDihedrals)} dihedrals."
        )

        with closing(mp.Pool(processes=self._nprocs)) as pool:
            pool.starmap(self._gen_conf_angle, [(j, molDihedrals) for j in new_list])

        ###Filter conformers having stearic clashes, clash with the protein, duplicates.
        print(
            "\n{} conformers have been generated.".format(
                len(Chem.SDMolSupplier("conformers.sdf"))
            )
        )

    def _gen_conf_angle(self, j, mol_Dihedrals):
        """
        Generate a conformer using Dihedral angles.
        """
        mol = Chem.Mol(self._query_molecule)
        intD = 0
        for i in mol_Dihedrals:
            rdMolTransforms.SetDihedralRad(mol.GetConformer(), *i, value=j[intD])
            intD += 1
        molH = Chem.AddHs(mol)
        self._alignToRef(molH)
        molMinH = self._minimize(molH)
        molMin = Chem.RemoveHs(molMinH)

        self._filtering(molMin)

        # with open("conformers.sdf", "a") as outf:
        #     with Chem.SDWriter(outf) as sdwriter:
        #         sdwriter.write(molMin)

    def _get_uniqueDihedrals(self):
        """
        Identify unique dihedral matches in a molecule that are not found in the MCS with a reference molecule.

        Parameters:
        - refmol: The reference RDKit molecule object.
        - mol: The RDKit molecule object to search for unique dihedrals.

        Returns:
        - A list of tuples, each representing the atom indices of a dihedral unique to `mol` and not part of its MCS with `refmol`.
        """
        DM_mol = self._getDihedralMatches(self._query_molecule)
        queryMatch = list(zip(*self._mappingRefToLig))[1]

        uniqueDihedrals = []
        for a in DM_mol:
            if any((aa not in queryMatch) for aa in a):
                uniqueDihedrals.append((a))
            else:
                continue
        return uniqueDihedrals

    @staticmethod
    def _getDihedralMatches(
        mol: Mol,
    ):  # The code was adapted from David Koes https://github.com/dkoes/rdkit-scripts/blob/master/rdallconf.py
        """
        Get dihedral matches in a molecule.

        Parameters:
        - mol: An RDKit molecule object.

        Returns:
        - A list of tuples, each representing the atom indices of a dihedral match.
        """
        pattern = r"*~[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])&!$([#7,O,S!D1]-!@[CD3]=[N,O,S])&!$([CD3](=[N+])-!@[#7!D1])&!$([#7!D1]-!@[CD3]=[N+])]-!@[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])]~*"
        qmol = Chem.MolFromSmarts(pattern)
        matches = mol.GetSubstructMatches(qmol)

        uniqmatches = []
        seen = set()
        for a, b, c, d in matches:
            if (b, c) not in seen:
                seen.add((b, c))
                uniqmatches.append((a, b, c, d))
        return uniqmatches

    @staticmethod
    def _degreeRange(inc: float) -> List[float]:
        """
        Produces a list of angle radians for dihedral sampling within a 360-degree range.

        Parameters:
        - inc: The increment in degrees for each step in the range.

        Returns:
        - A list of radians corresponding to the degree increments within a 360-degree range.
        """
        degrees = []
        deg = 0
        while deg < 360.0:
            rad = np.pi * deg / 180.0
            degrees.append(rad)
            deg += inc
        return degrees
