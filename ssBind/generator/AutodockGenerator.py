import os
import shutil
import subprocess
import uuid
from typing import Any, Dict, List, Tuple

import numpy as np
from rdkit.Chem.rdchem import Mol

from ssBind.generator.ConformerGenerator import ConformerGenerator


class AutodockGenerator(ConformerGenerator):

    def __init__(
        self,
        receptor_file: str,
        query_molecule: Mol,
        reference_substructure: Mol,
        **kwargs: Dict,
    ) -> None:
        super().__init__(
            receptor_file, query_molecule, reference_substructure, **kwargs
        )
        self._ligand_file = kwargs.get("ligand")
        self._hydrated = kwargs.get("autodock_hydrated", False)
        self._ligand_padding = kwargs.get("autodock_ligand_padding", 5)  # 5A
        self._rmstol = kwargs.get("autodock_rmstol", 0.1)
        self._curdir = kwargs.get("curdir", os.getcwd())
        self._working_dir = kwargs.get(
            "working_dir", os.path.join(self._curdir, str(uuid.uuid4()))
        )

    def generate_conformers(self) -> None:

        if os.path.exists(self._working_dir):
            shutil.rmtree(self._working_dir)
        os.makedirs(self._working_dir)

        self._mk_prepare_receptor()
        self._run_autogrid4()
        if self._hydrated:
            self._map_water()
        self._mk_prepare_ligand()
        changed_atypes = self._edit_tethered_atomtypes()
        self._add_bias(changed_atypes)
        if self._hydrated:
            self._insert_W_to_fld()
        self._run_docking(changed_atypes)
        self._export_results()

        shutil.move(os.path.join(self._working_dir, "conformers.sdf"), self._curdir)
        try:
            shutil.rmtree(self._working_dir)
        except:
            pass

    def _mk_prepare_receptor(self) -> None:
        prepare_receptor_cmd = [
            "mk_prepare_receptor.py",
            "-i",
            self._receptor_file,
            "-o",
            "receptor",
            "-g",
            "-p",
            "--box_enveloping",
            self._ligand_file,
            "--padding",
            str(self._ligand_padding),
        ]
        subprocess.run(prepare_receptor_cmd, cwd=self._working_dir, check=True)

    def _run_autogrid4(self) -> None:
        autogrid_cmd = ["autogrid4", "-p", "receptor.gpf", "-l", "receptor.glg"]
        subprocess.run(autogrid_cmd, cwd=self._working_dir, check=True)

    def _map_water(self) -> None:

        filedir = os.path.dirname(__file__)
        cmd = [
            os.path.join(filedir, "autodockUtils", "mapwater.py"),
            "-r",
            "receptor.pdbqt",
            "-s",
            "receptor.W.map",
        ]
        subprocess.run(cmd, cwd=self._working_dir, check=True)

    def _mk_prepare_ligand(self) -> None:
        cmd = ["mk_prepare_ligand.py", "-i", self._ligand_file, "-o", "ligand.pdbqt"]
        if self._hydrated:
            cmd.append("-w")
        subprocess.run(
            cmd,
            cwd=self._working_dir,
            check=True,
        )

    def _edit_tethered_atomtypes(self) -> List[Tuple[Any]]:

        atomsToTether = list(zip(*self._mappingLigToRef))[0]
        pos = self._query_molecule.GetConformer(0).GetPositions()

        with open(os.path.join(self._working_dir, "ligand.pdbqt"), "r") as f:
            lines = f.readlines()

        changed_atypes = []

        for iline, line in enumerate(lines):
            if line.startswith("ATOM"):
                r = np.array(
                    [float(line[31:39]), float(line[39:47]), float(line[47:54])]
                )

                match_ligand = [np.allclose(r, p, atol=0.02) for p in pos]
                matching_atoms_ligand = [i for i, x in enumerate(match_ligand) if x]

                if len(matching_atoms_ligand) > 0 and (
                    matching_atoms_ligand[0] in atomsToTether
                ):
                    matching_atom_tether = dict(self._mappingLigToRef)[
                        matching_atoms_ligand[0]
                    ]
                    new_type = "A" + self._base36encode(matching_atom_tether)
                    lines[iline] = line[:77] + new_type + "\n"
                    changed_atypes.append([line[76:].strip(), new_type, r])

        with open(os.path.join(self._working_dir, "ligand_new.pdbqt"), "w") as f:
            f.writelines(lines)

        return changed_atypes

    def _add_bias(self, changed_atypes: List[Tuple[Any]]) -> None:

        filedir = os.path.dirname(__file__)

        for old_atype, new_atype, position in changed_atypes:
            addbias_cmd = [
                os.path.join(filedir, "autodockUtils", "addbias.py"),
                "-i",
                f"receptor.{old_atype}.map",
                "-o",
                f"receptor.{new_atype}.map",
                "-x",
                str(position[0]),
                str(position[1]),
                str(position[2]),
            ]
            subprocess.run(addbias_cmd, cwd=self._working_dir, check=True)
            insert_type_cmd = [
                os.path.join(filedir, "autodockUtils", "insert_type_in_fld.py"),
                "receptor.maps.fld",
                "--newtype",
                new_atype,
            ]
            subprocess.run(insert_type_cmd, cwd=self._working_dir, check=True)

    def _insert_W_to_fld(self) -> None:

        cmd = [
            os.path.join(
                os.path.dirname(__file__), "autodockUtils", "insert_type_in_fld.py"
            ),
            "receptor.maps.fld",
            "--newtype",
            "W",
        ]
        subprocess.run(cmd, cwd=self._working_dir, check=True)

    def _run_docking(self, changed_atypes: List[Tuple[Any]]) -> None:

        t_options_map = {}
        for old_atype, new_atype, _ in changed_atypes:
            t_options_map.setdefault(old_atype, []).append(new_atype)
        t_options = "/".join([f"{','.join(v)}={k}" for k, v in t_options_map.items()])
        print(t_options)

        adgpu_cmd = [
            "autodock_gpu_128wi",
            "-L",
            "ligand_new.pdbqt",
            "-M",
            "receptor.maps.fld",
            "-T",
            t_options,
            "--nrun",
            str(self._numconf),
            "--heuristics",
            "0",
            "--autostop",
            "0",
            "--nev",
            "256000",
            "--rmstol",
            str(self._rmstol),
            # "--seed",
            # "0,1,2",
        ]
        subprocess.run(adgpu_cmd, cwd=self._working_dir, check=True)

    def _export_results(self) -> None:
        export_cmd = ["mk_export.py", "ligand_new.dlg", "-s", "conformers.sdf"]
        subprocess.run(export_cmd, cwd=self._working_dir, check=True)

    @staticmethod
    def _base36encode(number, length=2):
        chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        result = ""
        while number > 0:
            number, remainder = divmod(number, 36)
            result = chars[remainder] + result
        return result.zfill(length)
