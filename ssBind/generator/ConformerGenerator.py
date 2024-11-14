from abc import abstractmethod
from copy import deepcopy
from typing import List, Tuple

import MDAnalysis as mda
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS, rdmolops
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdchem import Mol


class ConformerGenerator:

    def __init__(
        self,
        receptor_file: str,
        query_molecule: Mol,
        reference_substructure: Mol,
        **kwargs
    ) -> None:
        self._receptor_file = receptor_file
        self._query_molecule = query_molecule
        self._reference_substructure = reference_substructure
        self._nprocs = kwargs.get("nprocs", 1)
        self._numconf = kwargs.get("numconf", 1)
        distTol = kwargs.get("distTol", 1.0)

        # for filtering
        self._cutoff_dist = kwargs.get("cutoff_dist", 1.5)

        self._mappingRefToLig = self._MCS_AtomMap(
            query_molecule, reference_substructure, distTol
        )
        self._mappingLigToRef = [(j, i) for i, j in self._mappingRefToLig]

        fixed_atoms = list(zip(*self._mappingLigToRef))[0]
        self._query_molecule.SetProp("fixed_atoms", str(fixed_atoms))

    @abstractmethod
    def generate_conformers(self) -> None:
        pass

    @staticmethod
    def _MCS_AtomMap(ligand: Mol, ref: Mol, distTol: float = 1.0) -> List[Tuple[int]]:

        mcs = rdFMCS.FindMCS([ligand, ref], completeRingsOnly=True, matchValences=False)
        submol = Chem.MolFromSmarts(mcs.smartsString)

        matches_ref = ref.GetSubstructMatches(submol, uniquify=False)
        matches_lig = ligand.GetSubstructMatches(submol, uniquify=False)

        for match_ref in matches_ref:
            for match_lig in matches_lig:
                dist = [
                    (
                        ref.GetConformer().GetAtomPosition(i0)
                        - ligand.GetConformer().GetAtomPosition(ii)
                    ).Length()
                    for i0, ii in zip(match_ref, match_lig)
                ]

                if all([d < distTol for d in dist]):
                    keepMatches = [match_ref, match_lig]
                    return list(zip(*keepMatches))

        raise Exception("ERROR: No MCS found!")

    def _minimize(self, ligand: Mol) -> Mol:
        mcp = deepcopy(ligand)
        ff = Chem.rdForceFieldHelpers.UFFGetMoleculeForceField(mcp, confId=0)
        for atidx in [i for i, j in self._mappingRefToLig]:
            ff.UFFAddPositionConstraint(atidx, 0, 200)
        maxIters = 10
        while ff.Minimize(maxIts=4) and maxIters > 0:
            maxIters -= 1

        self._alignToRef(mcp)
        return mcp

    def _alignToRef(self, ligand: Mol) -> None:
        AllChem.AlignMol(
            ligand, self._reference_substructure, atomMap=self._mappingLigToRef
        )

    def _filtering(self, mol: Mol) -> int:
        """Check if molecule clashes with itself or with the protein.

        Args:
            mol (Mol): The ligand conformer

        Returns:
            int: 0 - clash, 1 - no clash
        """

        if self._steric_clash(mol):
            return 0
        elif self._distance(self._receptor_file, mol, self._cutoff_dist):
            return 0
        else:
            with open("conformers.sdf", "a") as outf:
                sdwriter = Chem.SDWriter(outf)
                sdwriter.write(mol)
                sdwriter.close()
            return 1

    @staticmethod
    def _steric_clash(mol: Mol):
        """Identify steric clashes based on mean bond length."""

        ##Identify stearic clashes based on mean bond length
        ditancelist = rdmolops.Get3DDistanceMatrix(mol)[0]
        for i in range(1, len(ditancelist)):
            if ditancelist[i] < 0.5 * rdMolDraw2D.MeanBondLength(mol):
                return True
            else:
                continue
        return False

    @staticmethod
    def _distance(receptor: str, ligand: Mol, cutoff: float = 1.5):
        """Calculate the minimum distance between a protein and a ligand,
        excluding hydrogen atoms, and return True if it's below a cutoff."""

        protein = mda.Universe(receptor)
        ligand = mda.Universe(ligand)

        atom1 = protein.select_atoms("not name H*")
        atom2 = ligand.select_atoms("not name H*")

        distances = mda.analysis.distances.distance_array(
            atom1.positions, atom2.positions
        )

        return distances.min() < cutoff
