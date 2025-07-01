from typing import List, Tuple

import MDAnalysis as mda
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from spyrmsd.molecule import Molecule
from spyrmsd.rmsd import rmsdwrapper


class PosePicker:

    def __init__(self, **kwargs) -> None:

        self._nprocs = kwargs.get("nprocs", 1)
        self._complex_topology = kwargs.get("complex_topology", "complex.pdb")

    def pick_poses(
        self, conformers: str = "conformers.sdf", csv_scores: str = "Scores.csv"
    ) -> None:
        pass

    def _process_inputs(
        self, conformers: str = "conformers.sdf"
    ) -> Tuple[mda.Universe, List[Mol], str, bool]:
        """Extract conformers and other information from the inputs, depending on whether
        the protein is flexible or not.

        Args:
            conformers (str, optional): SD file with conformers. Defaults to "conformers.sdf".

        Returns:
            Tuple[mda.Universe, List[Mol], str, bool]: MDA universe, conformers, MDA selection
            for the ligand, and flag for protein flexibility
        """

        input_format = conformers.split(".")[-1].lower()

        if input_format == "dcd":

            u = mda.Universe(self._complex_topology, conformers)
            # elements = mda.topology.guessers.guess_types(u.atoms.names)
            # u.add_TopologyAttr("elements", elements)
            atoms = u.select_atoms("resname UNK")
            confs = [atoms.convert_to("RDKIT") for _ in u.trajectory]

            select = "(resname UNK) and not (name H*)"
            flex = True

        else:
            try:
                confs = Chem.SDMolSupplier(conformers)
            except:
                confs = Chem.SDMolSupplier(conformers, sanitize=False)
            u = mda.Universe(confs[0], confs)
            select = "not (name H*)"
            flex = False

        return u, confs, select, flex

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
