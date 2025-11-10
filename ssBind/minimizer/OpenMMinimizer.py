import ast
import os
import subprocess

from openff.toolkit import Molecule, Topology
from openmm import (
    CustomNonbondedForce,
    LangevinMiddleIntegrator,
    NonbondedForce,
    Platform,
    System,
    app,
)
from openmm.unit import *
from openmmforcefields.generators import (
    GAFFTemplateGenerator,
    SMIRNOFFTemplateGenerator,
)
from rdkit.Chem.rdchem import Mol


class OpenMMinimizer:

    OUTPUT_PDB = "complex.pdb"
    OUTPUT_DCD = "minimized_conformers.dcd"
    SCORE_FILE = "Scores.csv"

    def __init__(
        self,
        query_molecule: Mol,
        receptor_file: str,
        proteinFF: str = "amber14/protein.ff14SB.xml, amber14/tip3p.xml",
        FF: str = "smirnoff",
        **kwargs,
    ) -> None:
        """Initialize OpenMM minimization

        Args:
            receptor_file (str): PDB containing protein
            query_molecule (Mol): Ligand as RDKit molecule
            proteinFF (str, optional): Force field to use for protein. Defaults to "sage_ff14sb".
            FF (str, optional): Ligand force field. Defaults to "nagl".
        """
        self._receptor_file = receptor_file
        self._ligand = query_molecule
        self._proteinFF = proteinFF
        self._FF = FF

        platform = kwargs.get("openmm_platform", None)
        if platform is not None:
            self._platform = Platform.getPlatformByName(platform)
        else:
            self._platform = None

        self._constrain_protein = not kwargs.get("openmm_flex", False)
        self._calc_interaction_energy = (
            kwargs.get("openmm_score", "interaction") == "interaction"
        )

        self._integrator = LangevinMiddleIntegrator(
            300 * kelvin, 1 / picosecond, 0.004 * picoseconds
        )

    def run_minimization(self, conformers: str = "conformers.sdf") -> None:
        """Run OpenMM minimization

        Args:
            conformers (str, optional): SDF with conformers to minimize.
            Defaults to "conformers.sdf".

        Raises:
            Exception: Unknown FF or proteinFF
        """

        # clean up
        for f in [self.OUTPUT_PDB, self.SCORE_FILE]:
            if os.path.isfile(f):
                os.remove(f)

        # read in topologies with coordinates
        top = Topology.from_pdb(self._receptor_file)
        self._n_protein = top.n_atoms

        conf = Molecule.from_file(conformers)
        if type(conf) == Molecule:
            conf = [conf]
        ligand = Molecule.from_rdkit(self._ligand)

        # create simulation object based on FFs
        if self._FF in ["gaff", "smirnoff"]:
            simulation = self._simulation_openmm(top, ligand)
        else:
            raise Exception(f"Unknown ligand FF: {self._FF}")

        self._simulation = simulation

        # minimize energy against reference ligand to speed up subsequent minimizations
        simulation.minimizeEnergy(tolerance=0.1)
        new_positions = simulation.context.getState(getPositions=True).getPositions()
        self._protein_pos = new_positions[0 : self._n_protein]
        state = simulation.context.getState(getPositions=True)

        # write pdb for topology information
        with open(OpenMMinimizer.OUTPUT_PDB, "w") as output:
            app.PDBFile.writeFile(
                simulation.topology,
                state.getPositions(),
                output,
            )

        # energies
        self._potEnergies = []

        # set up DCD file for trajectory
        dcdfile = open(OpenMMinimizer.OUTPUT_DCD, "wb")
        self._dcd = app.DCDFile(dcdfile, simulation.topology, 1)

        # minimize conformers
        for i, conformer in enumerate(conf):
            self._minimize_conformer(i, conformer)

        dcdfile.close()

        # write energy
        with open(OpenMMinimizer.SCORE_FILE, "w") as scores_file:
            scores_file.write("Index,Score\n")
            for index in range(len(self._potEnergies)):
                scores_file.write(f"{index},{self._potEnergies[index]}\n")

    def _simulation_openmm(self, top: Topology, ligand: Molecule) -> app.Simulation:
        """Construct OpenMM simulation combining topologies of PDB (top) and ligand
        Molecule, using native openmm.app.Forcefields

        Args:
            top (Topology): Protein topology, read from PDB
            ligand (Molecule): Ligand

        Returns:
            app.Simulation: OpenMM simulation
        """
        top.add_molecule(ligand)
        self._n_total = top.n_atoms
        forcefield = self._get_protein_ff(self._proteinFF)
        if self._FF == "gaff":
            ff = GAFFTemplateGenerator(molecules=ligand)
        else:
            ff = SMIRNOFFTemplateGenerator(molecules=ligand)
        forcefield.registerTemplateGenerator(ff.generator)

        # create system and charge ligand (this takes a few minutes)
        system = forcefield.createSystem(top.to_openmm())

        # constrain reference substructure
        if self._ligand.HasProp("fixed_atoms"):
            fixed_atoms_string = self._ligand.GetProp("fixed_atoms")
            fixed_atoms = ast.literal_eval(fixed_atoms_string)

            for i in fixed_atoms:
                system.setParticleMass(i + self._n_protein, 0 * amu)

        # if not treated flexible, constrain protein too
        if self._constrain_protein:
            for i in range(self._n_protein):
                system.setParticleMass(i, 0 * amu)

        # create force groups
        self._set_force_parameter_offsets(system)

        # create openmm simulation
        simulation = app.Simulation(top.to_openmm(), system, self._integrator)
        simulation.context.setPositions(top.get_positions().to_openmm())
        return simulation

    @staticmethod
    def _get_protein_ff(force_fields_to_include: str) -> app.ForceField:
        """Construct protein forcefield using native app.Forcefield

        Args:
            force_fields_to_include (str): Comma-separated string of FFs

        Returns:
            app.ForceField: OpenMM compatible FF
        """
        ff_list = force_fields_to_include.split(",")
        ff_list = [item.strip() for item in ff_list]
        ff = app.ForceField(*ff_list)
        return ff

    def _set_force_parameter_offsets(self, system: System) -> None:
        """Define force groups in order to recover the interaction energy later.

        Args:
            system (System): OpenMM system

        Raises:
            Exception: CustomNonbondedForce can only act on atoms belonging to the same group
        """
        protein = set(range(self._n_protein))
        ligand = set(range(self._n_protein, self._n_total))

        for force in system.getForces():
            if isinstance(force, NonbondedForce):
                force.setForceGroup(0)
                force.addGlobalParameter("protein_scale", 1)
                force.addGlobalParameter("ligand_scale", 1)
                for i in range(force.getNumParticles()):
                    charge, sigma, epsilon = force.getParticleParameters(i)
                    # Set the parameters to be 0 when the corresponding parameter is 0,
                    # and to have their normal values when it is 1.
                    param = "protein_scale" if i in protein else "ligand_scale"
                    force.setParticleParameters(i, 0, 0, 0)
                    force.addParticleParameterOffset(param, i, charge, sigma, epsilon)
                for i in range(force.getNumExceptions()):
                    p1, p2, chargeProd, sigma, epsilon = force.getExceptionParameters(i)
                    if ((p1 in protein) and (p2 in ligand)) or (
                        (p1 in ligand) and (p2 in protein)
                    ):
                        raise Exception(
                            "Check your CustomNonbondedForce - covalent ligands not supported"
                        )
                    param = "protein_scale" if p1 in protein else "ligand_scale"
                    force.setExceptionParameters(i, p1, p2, 0, 0, 0)
                    force.addExceptionParameterOffset(
                        param, i, chargeProd, sigma, epsilon
                    )
            elif isinstance(force, CustomNonbondedForce):
                force.setForceGroup(1)
                force.addInteractionGroup(protein, ligand)
            else:
                force.setForceGroup(2)

    def _minimize_conformer(
        self,
        index: int,
        conformer: Molecule,
    ) -> None:
        """Minimize single conformer and write it to PDB and its potential energy
        to csv.

        Args:
            index (int): Index of this conformer
            conformer (Molecule): Conformer to minimize
        """
        simulation = self._simulation
        new_positions = simulation.context.getState(getPositions=True).getPositions()
        new_positions[0 : self._n_protein] = self._protein_pos
        new_positions[self._n_protein : self._n_total] = (
            conformer.to_topology().get_positions().to_openmm()
        )
        simulation.context.setPositions(new_positions)
        simulation.minimizeEnergy()
        state = simulation.context.getState(getPositions=True, getEnergy=True)

        if self._calc_interaction_energy:
            potEnergy = self._interaction_energy()
        else:
            potEnergy = state.getPotentialEnergy().value_in_unit(kilocalories_per_mole)

        # DON'T write conformers and energies if they failed to minimize
        if potEnergy < 0:
            # write minimized complex
            self._dcd.writeModel(state.getPositions())
            self._potEnergies.append(potEnergy)

    def _interaction_energy(self) -> float:
        """Calculate protein-ligand interaction energy

        Returns:
            float: interaction energy
        """
        protein_nb = self._nb_energy(1, 0)
        ligand_nb = self._nb_energy(0, 1)
        # IMPORTANT that this line is last, so final scales are (1,1) i.e.
        # fully interacting system, so minimization doesn't get messed up.
        total_nb = self._nb_energy(1, 1)
        interaction_ene = total_nb - protein_nb - ligand_nb
        return interaction_ene.value_in_unit(kilocalories_per_mole)

    def _nb_energy(self, protein_scale: int, ligand_scale: int) -> Quantity:
        """Calculated nonbonded energy between different interaction groups

        Args:
            protein_scale (int): turn on (1) or off (1) protein nonbonded
            ligand_scale (int): turn on (1) or off (1) ligand nonbonded

        Returns:
            Quantity: Nonbonded p-p/l-l/total potential energy
        """
        simulation = self._simulation
        simulation.context.setParameter("protein_scale", protein_scale)
        simulation.context.setParameter("ligand_scale", ligand_scale)
        return simulation.context.getState(
            getEnergy=True, groups={0}
        ).getPotentialEnergy()
