import csv
import glob
import multiprocessing as mp
import os
import pathlib
import re
import shutil
import subprocess
import sys
import uuid
import warnings
from contextlib import closing
from pathlib import Path

# MDAnalysis
import MDAnalysis as mda

# Data handling
import pandas as pd

# ACPYPE for AMBER topologies
from acpype.topol import AbstractTopol, ACTopol, MolTopol, header
from MDAnalysisTests.datafiles import AUX_EDR
from openff.interchange import Interchange
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField

# RDKit and OpenFF
from rdkit import Chem
from rdkit.Chem import AllChem

from ssBind.io import MolFromInput, obabel_convert

# Suppress warnings if necessary (consider being selective about what to ignore)
warnings.simplefilter("ignore")


class GromacsMinimizer:

    def __init__(
        self,
        receptor_file: str,
        proteinFF: str = "amber99sb-ildn",
        FF: str = "gaff",  # ["gaff", "gaff2", "openff", "cgenff"]
        **kwargs,
    ) -> None:
        self._receptor_file = receptor_file
        self._proteinFF = proteinFF
        self._FF = FF

        self._working_dir = kwargs.get("working_dir", os.path.join(os.getcwd(), "tmp"))
        self._nprocs = kwargs.get("nprocs", 1)

    def run_minimization(self, conformers: str = "conformers.sdf") -> None:
        self._optimize_molecule(input_file=conformers)
        self._system_setup(
            self._receptor_file, "ligand.sdf", proteinFF=self._proteinFF, FF=self._FF
        )
        trjdir = self._working_dir
        os.makedirs(trjdir)
        with closing(mp.Pool(processes=self._nprocs)) as pool:
            pool.starmap(
                self._minimize,
                [
                    (i, mol, trjdir)
                    for i, mol in enumerate(Chem.SDMolSupplier(conformers))
                ],
            )
        self._combine_traj(trjdir)
        self._traj_to_sdf()

    @staticmethod
    def _optimize_molecule(input_file="conformers.sdf", output_file="ligand.sdf"):
        """
        Optimizes the geometry of a molecule loaded from an SDF file and saves the result to another SDF file.

        Parameters:
        - input_file (str): The path to the input SDF file containing many conformers.
        - output_file (str): The path to the output SDF file where the optimized molecule will be saved.
        """
        # Load molecule from SDF file
        sdf = MolFromInput(input_file)

        writer = Chem.SDWriter(output_file)
        mol = Chem.AddHs(sdf, addCoords=True)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
        mol.SetProp("_Name", "LIG")
        writer.write(mol)
        writer.close()

    @staticmethod
    def _system_setup(receptor, ligand, proteinFF="amber99sb-ildn", FF="gaff"):
        """
        Sets up the system for molecular dynamics simulations by processing receptor and ligand files.

        """
        # Setup for OpenFF
        if FF == "openff":
            GromacsMinimizer._gmx_pdb2gmx(
                receptor,
                outcoord="protein.gro",
                outtop="protein.top",
                protein_FF=proteinFF,
                ignh=False,
            )
            GromacsMinimizer._get_openff(ligand, "LIG")
            GromacsMinimizer._get_topol(
                "protein.top", "openff_LIG.itp", ff=FF, protein_FF=proteinFF
            )

        # Setup for GAFF or GAFF2
        elif FF in {"gaff", "gaff2"}:
            GromacsMinimizer._gmx_pdb2gmx(
                receptor,
                outcoord="protein.gro",
                outtop="protein.top",
                protein_FF=proteinFF,
                ignh=False,
            )
            GromacsMinimizer._get_gaff(ligand, "LIG", FF)
            GromacsMinimizer._get_topol(
                "protein.top", "LIG.acpype/LIG_GMX.itp", ff=FF, protein_FF=proteinFF
            )

        # Setup for CGenFF
        elif FF == "cgenff":
            if not os.getenv("SILCSBIODIR"):
                raise ValueError(
                    "The SILCSBIODIR environment variable is required but not set."
                )
            GromacsMinimizer._get_cgenff(ligand, "LIG")
            GromacsMinimizer._gmx_pdb2gmx(
                receptor,
                outcoord="protein.gro",
                outtop="protein.top",
                protein_FF="charmm36",
                ignh=False,
            )
            GromacsMinimizer._get_topol("protein.top", "LIG.top", ff=FF)

    @staticmethod
    def _minimize(i, mol, trjdir):

        md = str(uuid.uuid4())

        sdf_path = f"{md}.sdf"
        with Chem.SDWriter(sdf_path) as writer:
            mol_with_h = Chem.AddHs(mol, addCoords=True)
            writer.write(mol_with_h)

        gro_path = f"{md}.gro"
        obabel_convert(sdf_path, gro_path)

        # Replace GRO coordinates and combine GRO files
        GromacsMinimizer._replace_GROcoor(gro_path, "md_setup/LIG.gro", gro_path)
        combined_gro_path = f"{md}x.gro"
        GromacsMinimizer._combine_gro_files(
            "md_setup/protein.gro", gro_path, combined_gro_path
        )

        # Run molecular dynamics simulation and save trajectory files
        GromacsMinimizer._gmx_mdrun(md, "md_setup")
        GromacsMinimizer._gmx_trjcat(md, i, os.path.join(trjdir, f"{md}.xtc"))

        # Calculate interaction energy
        energy_terms = ["LJ-SR:Protein-LIG", "Coul-SR:Protein-LIG"]
        energy = sum(
            GromacsMinimizer._mda_edr(f"{md}.edr", term) for term in energy_terms
        )
        formatted_energy = "{:0.3f}".format(energy)

        data = pd.DataFrame({"Index": [i], "Score": [formatted_energy]})
        data.to_csv("Scores.csv", mode="a", index=False, header=False)

        # Clean up temporary files
        for f in glob.glob(f"*{md}*"):
            os.remove(f)

    @staticmethod
    def _combine_traj(trjdir):
        cmd = f"{GromacsMinimizer.GMX()} trjcat -f {trjdir}/* -o trjout.xtc"
        result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
        if result.returncode != 0:
            with open("gromacs.log", "w") as log_file:
                log_file.write(result.stdout + result.stderr)
            raise SystemExit(
                f"\nERROR! See the log file for details: {os.path.abspath('gromacs.log')}"
            )

        # Sorting CSV
        try:
            with open("Scores.csv", "r") as unsorted_csv_file:
                csv_reader = csv.reader(unsorted_csv_file)
                sorted_csv_rows = sorted(csv_reader, key=lambda x: int(x[0]))

            with open("Scores.csv", "w", newline="") as sorted_csv_file:
                csv_writer = csv.writer(sorted_csv_file)
                csv_writer.writerow(["Index", "Score"])
                csv_writer.writerows(sorted_csv_rows)
        except Exception as e:
            raise SystemExit(f"Failed to sort CSV: {e}")

        # Deleting temporary trajectory files safely
        try:
            shutil.rmtree(f"{trjdir}")
        except FileNotFoundError:
            print(f"Directory {trjdir} not found for deletion.")

    @staticmethod
    def _traj_to_sdf() -> None:
        """Convert trjout.xtc to minimized_conformers.sdf"""
        GromacsMinimizer._gmx_grompp("md_setup")

        u = mda.Universe("md_setup/complex.tpr", "trjout.xtc")
        elements = mda.topology.guessers.guess_types(u.atoms.names)
        u.add_TopologyAttr("elements", elements)
        atoms = u.select_atoms("resname LIG")

        sdwriter = Chem.SDWriter("minimized_conformers.sdf")
        for ts in u.trajectory:
            sdwriter.write(atoms.convert_to("RDKIT"))

        sdwriter.close()

    @staticmethod
    def _gmx_pdb2gmx(
        pdbfile,
        outcoord="protein.gro",
        outtop="topol.top",
        protein_FF="amber99sb-ildn",
        water="tip3p",
        ignh=True,
    ):
        """
        Build a protein topology and coordinate file from a PDB file using GROMACS pdb2gmx.

        """

        # Format the ignh option
        ignh_cmd = "-ignh" if ignh else ""

        # Construct the command
        cmd = f"{GromacsMinimizer.GMX()} pdb2gmx -f {pdbfile} -o {outcoord} -p {outtop} -ff {protein_FF} -water {water} {ignh_cmd} > gromacs.log 2>&1"

        try:
            # Execute the command
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            with open(os.path.abspath("gromacs.log"), "wb") as log_file:
                log_file.write(e.stderr)
            print(
                f"ERROR: gmx pdb2gmx execution failed. See log file",
                file=sys.stderr,
            )
            raise SystemExit(1)

    @staticmethod
    def _get_topol(pro_itp, lig_itp, ff="gaff", protein_FF="amber99sb-ildn"):

        md_setup_dir = "md_setup"
        os.makedirs(md_setup_dir, exist_ok=True)

        if ff in {"gaff", "gaff2", "openff"}:
            initial = f"""; Include forcefield parameters
#include "{protein_FF}.ff/forcefield.itp"\n
"""
            water = f"{protein_FF}.ff/tip3p.itp"
        elif ff == "cgenff":
            initial = """; Include forcefield parameters
#include "./charmm36.ff/forcefield.itp"
;
; Include ligand specific parameters
#include "./charmm36.ff/lig_ffbonded.itp"
"""
            water = "./charmm36.ff/tip3p.itp"
            shutil.move("charmm36.ff", os.path.join(md_setup_dir, "charmm36.ff"))
        else:
            raise ValueError("Invalid force field specified")

        waternumber = ""
        with open(pro_itp, "r") as file:
            for last_line in file:
                pass
            if last_line.startswith("SOL"):
                elements = last_line.split()
                waternumber = f"SOL {elements[1]}"

        # Construct the topology file content
        template = f"""{initial}
[ atomtypes ]
{GromacsMinimizer._get_atomtypes([lig_itp], ff)}

#include "protein.itp"

#include "LIG.itp"

#include "{water}"

[ system ]
Protein-Ligand

[ molecules ]
protein 1
LIG 1
{waternumber}
"""

        # Writing the topol.top file
        with open(os.path.join(md_setup_dir, "topol.top"), "w") as f:
            f.write(template)

        GromacsMinimizer._mol_itp(
            lig_itp, "LIG", os.path.join(md_setup_dir, "LIG.itp"), ff
        )
        GromacsMinimizer._protein_itp(
            pro_itp, "protein", os.path.join(md_setup_dir, "protein.itp")
        )
        shutil.move("protein.gro", os.path.join(md_setup_dir, "protein.gro"))

        lig_gro_src = "LIG.acpype/LIG_GMX.gro" if ff in {"gaff", "gaff2"} else "LIG.gro"
        shutil.copy(lig_gro_src, os.path.join(md_setup_dir, "LIG.gro"))

        # Combining .gro files for complex setup
        GromacsMinimizer._combine_gro_files(
            os.path.join(md_setup_dir, "protein.gro"),
            os.path.join(md_setup_dir, "LIG.gro"),
            os.path.join(md_setup_dir, "complex.gro"),
        )

    @staticmethod
    def _get_openff(molecule_path, molname: str = "LIG"):
        """
        Generates a ligand topology and coordinate file from a ligand file using the OpenFF Toolkit.

        Note:
            This function requires the OpenFF Toolkit and its dependencies to be installed.
        """
        molecule = Molecule.from_file(molecule_path)

        topology = Topology.from_molecules([molecule])

        # Load the OpenFF force field: version 2.1.0, "Sage"
        forcefield = ForceField("openff-2.1.0.offxml")

        interchange = Interchange.from_smirnoff(
            force_field=forcefield, topology=topology
        )

        # Write to GROMACS files
        interchange.to_gro(f"{molname}.gro")
        interchange.to_top(f"openff_{molname}.itp")

    @staticmethod
    def _get_gaff(ligandfile, molname: str = "LIG", ff: str = "gaff"):
        """
        Builds a ligand topology and coordinate file from a ligand file using acpype for gaff.

        Args:
            ligandfile (str): The path to the input ligand file.
            molname (str): The molecule name to be used for output files.
            ff (str): The force field to be used.

        Raises:
            RuntimeError: If acpype fails to run successfully.
        """

        net_charge = Chem.GetFormalCharge(MolFromInput(ligandfile))
        acpype_cmd = f"acpype -i {ligandfile} -b {molname} -n {net_charge} -a {ff} -f -o gmx >acpype.{molname}.log 2>&1"

        RC = os.system(acpype_cmd)
        if RC != 0:
            pass  # acpype can segfault at the very end
            # log_path = os.path.abspath(f"acpype.{molname}.log")
            # raise RuntimeError(
            #     f"ERROR! Failed to run acpype. See the log at {log_path} for details."
            # )

        shutil.copy("LIG.acpype/LIG_GMX.gro", f"{molname}.gro")

    @staticmethod
    def _get_cgenff(molecule, molname: str = "LIG"):
        """
        Builds a ligand topology and coordinate file from a ligand file using SILCSBIO for cgenff.

        Args:
            molecule (str): Path to the input ligand file.
            molname (str): Molecule name for naming output files.

        Raises:
            RuntimeError: If cgenff processing fails.
        """

        silcsbio_dir = os.environ.get("SILCSBIODIR")
        if not silcsbio_dir:
            raise RuntimeError("SILCSBIODIR environment variable is not set.")

        molecule_path = Path(molecule)
        input_format = molecule_path.suffix[1:].lower()  # Remove dot from suffix

        if input_format != "mol2":
            obabel_convert(molecule, "ligand.mol2", unique_names=True)
            molecule = "ligand.mol2"
            molecule_path = Path(molecule)

        with molecule_path.open("r") as file:
            content = file.read()

        pattern = r"@<TRIPOS>UNITY_ATOM_ATTR.*?(?=@<TRIPOS>|$)"
        updated_content = re.sub(pattern, "", content, flags=re.DOTALL)

        with molecule_path.open("w") as file:
            file.write(updated_content)

        cmd = f"{silcsbio_dir}/cgenff/cgenff_to_gmx.sh mol={molecule_path} > cgenff.log 2>&1"
        RC = os.system(cmd)
        if RC != 0:
            raise RuntimeError(
                f"ERROR! See the log file for details: {Path('cgenff.log').resolve()}"
            )

        shutil.move("posre.itp", f"posre_{molname}.itp")
        shutil.move(f"{molecule_path.stem}_gmx.top", f"{molname}.top")
        obabel_convert(f"{molecule_path.stem}_gmx.pdb", "LIG.gro")

    @staticmethod
    def _replace_GROcoor(gro_file1, gro_file2, output_path):
        # Read lines from both input files
        with open(gro_file1) as file1, open(gro_file2) as file2:
            file1_lines = file1.readlines()
            file2_lines = file2.readlines()

        num_atoms = int(file1_lines[1].strip())

        title = "LIG Gro File"
        header = f"{title}\n{num_atoms}\n"
        box = file2_lines[-1]

        with open(output_path, "w") as output_file:
            output_file.write(header)
            for i in range(2, num_atoms + 2):
                line1 = file1_lines[i]
                line2 = file2_lines[i]
                new_line = "{:5d}{:<5s}{:>5s}{:5d}{:8.3f}{:8.3f}{:8.3f}\n".format(
                    1,
                    "LIG",
                    str(line2[10:15]).strip(),
                    i - 1,
                    float(str(line1[20:27]).strip()),
                    float(str(line1[28:35]).strip()),
                    float(str(line1[36:43]).strip()),
                )
                output_file.write(new_line)
            output_file.write(box)

    @staticmethod
    def _combine_gro_files(file1_path, file2_path, output_path, box="10 10 10"):
        # Read and store lines from both input files
        with open(file1_path) as file1, open(file2_path) as file2:
            file1_lines = file1.readlines()
            file2_lines = file2.readlines()

        # Extract atom counts and calculate total
        num_atoms1 = int(file1_lines[1].strip())
        num_atoms2 = int(file2_lines[1].strip())
        total_atoms = num_atoms1 + num_atoms2

        atoms = []

        atoms.extend(
            line for line in file1_lines[2 : num_atoms1 + 2] if "HOH" not in line
        )
        atoms.extend(
            f"{line[:5]}LIG  {line[10:]}" for line in file2_lines[2 : num_atoms2 + 2]
        )
        atoms.extend(line for line in file1_lines[2 : num_atoms1 + 2] if "HOH" in line)

        # Construct the combined file content
        combined_content = (
            f"Combined Gro File\n{total_atoms}\n" + "".join(atoms) + f"{box}\n"
        )

        # Write the combined content to the output file
        with open(output_path, "w") as output_file:
            output_file.write(combined_content)

    @staticmethod
    def _gmx_mdrun(idx, md_setup):
        """
        Runs the GROMACS energy minimization steps.

        """
        # Preprocess the simulation setup
        grompp_cmd = f"{GromacsMinimizer.GMX()} grompp -f {GromacsMinimizer.MDPFILES()}/em.mdp -o {idx}.tpr -c {idx}x.gro -p {md_setup}/topol.top -maxwarn 1"
        try:
            subprocess.run(grompp_cmd, shell=True, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            with open(os.path.abspath("gromacs.log"), "wb") as log_file:
                log_file.write(e.stderr)
            print(
                f"ERROR: gmx grompp execution failed. See log file",
                file=sys.stderr,
            )
            raise SystemExit(1)

        # Perform energy minimization
        mdrun_cmd = f"{GromacsMinimizer.GMX()} mdrun -deffnm {idx} -nt 1"
        try:
            subprocess.run(mdrun_cmd, shell=True, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            with open(os.path.abspath("gromacs.log"), "wb") as log_file:
                log_file.write(e.stderr)
            print(
                f"ERROR: gmx mdrun execution failed. See log file",
                file=sys.stderr,
            )
            raise SystemExit(1)

    @staticmethod
    def _gmx_trjcat(idx, time, outtrj):
        """
        Processes trajectory files with GROMACS tools, including re-centering, fitting,
        and concatenating trajectories.

        """

        commands = [
            f'echo "Protein System" | {GromacsMinimizer.GMX()} trjconv -s md_setup/complex.gro -f {idx}.trr -o {idx}.xtc -pbc nojump -ur compact -center',
            f'echo "Protein System" | {GromacsMinimizer.GMX()} trjconv -s md_setup/complex.gro -f {idx}.xtc -o {idx}x.xtc -fit rot+trans',
            f'echo "{time}" | {GromacsMinimizer.GMX()} trjcat -f {idx}x.xtc -o {outtrj}.xtc -settime 1',
        ]

        for cmd_idx, cmd in enumerate(commands, 1):
            try:
                subprocess.run(cmd, shell=True, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                with open(os.path.abspath("gromacs.log"), "wb") as log_file:
                    log_file.write(e.stderr)
                print(
                    f"ERROR: gmx trjcat execution failed. See log file",
                    file=sys.stderr,
                )
                raise SystemExit(1)

    @staticmethod
    def _mda_edr(edrfile, term):
        """
        Reads a GROMACS .edr file and returns the average of the last 100 steps in kcal/mol.

        Args:
            edrfile (str): Path to the .edr file.
            term (str): The term to extract and average, e.g., "Potential".

        Returns:
            float: The average value of the specified term over the last 100 steps, converted to kcal/mol.
        """
        # Initialize the EDR reader
        aux = mda.auxiliary.EDR.EDRReader(edrfile)

        # Extract data for the specified term
        data = pd.DataFrame.from_records(aux.get_data(term), columns=[term])

        # Calculate the mean of the last 100 steps and convert units to kcal/mol
        mean_value = data.tail(100).mean()[term] * 0.2390057361

        return mean_value

    @staticmethod
    def _get_atomtypes(itp_files, ff):
        """
        Combine [ atomtypes ] sections from a list of .itp files.

        Returns:
            str: A combined string of unique atom types from all files.
        """
        atom_type_start = re.compile(r"\[ atomtypes \]")
        molecule_type_start = re.compile(r"\[ moleculetype \]")
        unique_atomtypes = set()

        for itp_file in itp_files:
            with open(itp_file, "r") as file:
                capture = False
                for line in file:
                    if atom_type_start.search(line):
                        capture = True
                        continue
                    elif molecule_type_start.search(line) or line.strip() == "":
                        capture = False
                        continue
                    if capture:
                        line = line.replace("MOL0", "") if ff == "openff" else line
                        unique_atomtypes.add(line)

        return "".join(unique_atomtypes)

    @staticmethod
    def _mol_itp(itp_file, resname, output_itp, ff):
        # Compile regular expressions
        moleculetype_re = re.compile(r"\[ moleculetype \]")
        system_re = re.compile(r"\[ system \]")
        atoms_re = re.compile(r"\[ atoms \]")
        bonds_re = re.compile(r"\[ bonds \]")
        pairs_re = re.compile(r"\[ pairs \]")

        # Flags for current section and renaming state
        in_relevant_section = False
        should_rename = False

        with open(output_itp, "w") as output:
            with open(itp_file, "r") as file:
                for line in file:
                    if moleculetype_re.search(line):
                        output.write(line)
                        output.write(f"{resname}              3\n")
                        continue
                    elif line.startswith("#"):
                        in_relevant_section = False
                    elif system_re.search(line):
                        in_relevant_section = False
                        break
                    elif atoms_re.search(line):
                        in_relevant_section = True
                        should_rename = True
                        output.write(line)
                    elif bonds_re.search(line) or pairs_re.search(line):
                        should_rename = False
                        output.write(line)
                    elif in_relevant_section and not should_rename:
                        output.write(line)
                    elif in_relevant_section and should_rename:
                        if not line.startswith((";", "#", "[")) and line.strip():
                            line_list = line.split()
                            res_name = line_list[3]
                            line = line.replace(res_name, resname)
                            output.write(line)

    @staticmethod
    def _protein_itp(itp_file, molname, output_itp):
        # Compile regular expressions
        moleculetype_re = re.compile(r"\[ moleculetype \]")
        atoms_re = re.compile(r"\[ atoms \]")
        system_re = re.compile(r"\[ system \]")

        in_atoms_section = False

        with open(output_itp, "w") as output, open(itp_file, "r") as file:
            for line in file:
                if moleculetype_re.search(line):
                    output.write(line)
                    output.write(f";name            nrexcl\n{molname}              3\n")
                    continue
                elif atoms_re.search(line):
                    in_atoms_section = True
                    output.write(line)
                elif line.startswith("#") or system_re.search(line):
                    in_atoms_section = False
                    if system_re.search(line):
                        break
                elif in_atoms_section:
                    output.write(line)

    @staticmethod
    def _gmx_grompp(md_setup):
        """
        Executes the GROMACS grompp command to preprocess simulation input files.

        """
        complex_tpr_path = os.path.join(md_setup, "complex.tpr")
        complex_gro_path = os.path.join(md_setup, "complex.gro")
        topol_top_path = os.path.join(md_setup, "topol.top")

        cmd = f"{GromacsMinimizer.GMX()} grompp -f {GromacsMinimizer.MDPFILES()}/em.mdp -o {complex_tpr_path} -c {complex_gro_path} -p {topol_top_path} -maxwarn 1"

        try:
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            with open(os.path.abspath("gromacs.log"), "wb") as log_file:
                log_file.write(e.stderr)
            print(
                f"ERROR: gmx grompp execution failed. See log file",
                file=sys.stderr,
            )
            raise SystemExit(1)

    @staticmethod
    def GMX():
        """
        Finds and returns the available GROMACS executable on the system.

        Returns:
            str: The GROMACS executable ('gmx' or 'gmx_mpi').

        Raises:
            SystemExit: If neither 'gmx' nor 'gmx_mpi' can be found on the system.
        """
        for cmd in ["gmx", "gmx_mpi"]:
            if os.system(f"{cmd} -h >/dev/null 2>&1") == 0:
                return cmd
        print("Warning: Not found gmx or gmx_mpi")

    @staticmethod
    def MDPFILES():
        repo_root = pathlib.Path(os.path.abspath(__file__)).parents[1]
        return os.path.join(repo_root, "utils")
