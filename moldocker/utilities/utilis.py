import os
import re
import io
import sys
import json
import numpy as np
import pandas as pd

from rdkit import Chem
from io import StringIO
from .pdbfixer.pdbfixer import PDBFixer
from openmm.app import PDBFile

from rdkit.Geometry import Point3D

with open(os.path.join(os.path.dirname(__file__), "data", "residue_params.json")) as f:
    residue_params = json.load(f)

with open(os.path.join(os.path.dirname(__file__), "data", "flexres_templates.json")) as f:
    flexres_templates = json.load(f)

retreive_mdl_compiled = re.compile(r'MODEL\s+[0-9]+\s+((\n|.)*?)ENDMDL')

# substitutions = {
#     '2AS':'ASP', '3AH':'HIS', '5HP':'GLU', '5OW':'LYS', 'ACL':'ARG', 'AGM':'ARG', 'AIB':'ALA', 'ALM':'ALA', 'ALO':'THR', 'ALY':'LYS', 'ARM':'ARG',
#     'ASA':'ASP', 'ASB':'ASP', 'ASK':'ASP', 'ASL':'ASP', 'ASQ':'ASP', 'AYA':'ALA', 'BCS':'CYS', 'BHD':'ASP', 'BMT':'THR', 'BNN':'ALA',
#     'BUC':'CYS', 'BUG':'LEU', 'C5C':'CYS', 'C6C':'CYS', 'CAS':'CYS', 'CCS':'CYS', 'CEA':'CYS', 'CGU':'GLU', 'CHG':'ALA', 'CLE':'LEU', 'CME':'CYS',
#     'CSD':'ALA', 'CSO':'CYS', 'CSP':'CYS', 'CSS':'CYS', 'CSW':'CYS', 'CSX':'CYS', 'CXM':'MET', 'CY1':'CYS', 'CY3':'CYS', 'CYG':'CYS',
#     'CYM':'CYS', 'CYQ':'CYS', 'DAH':'PHE', 'DAL':'ALA', 'DAR':'ARG', 'DAS':'ASP', 'DCY':'CYS', 'DGL':'GLU', 'DGN':'GLN', 'DHA':'ALA',
#     'DHI':'HIS', 'DIL':'ILE', 'DIV':'VAL', 'DLE':'LEU', 'DLY':'LYS', 'DNP':'ALA', 'DPN':'PHE', 'DPR':'PRO', 'DSN':'SER', 'DSP':'ASP',
#     'DTH':'THR', 'DTR':'TRP', 'DTY':'TYR', 'DVA':'VAL', 'EFC':'CYS', 'FLA':'ALA', 'FME':'MET', 'GGL':'GLU', 'GL3':'GLY', 'GLZ':'GLY',
#     'GMA':'GLU', 'GSC':'GLY', 'HAC':'ALA', 'HAR':'ARG', 'HIC':'HIS', 'HIP':'HIS', 'HMR':'ARG', 'HPQ':'PHE', 'HTR':'TRP', 'HYP':'PRO',
#     'IAS':'ASP', 'IIL':'ILE', 'IYR':'TYR', 'KCX':'LYS', 'LLP':'LYS', 'LLY':'LYS', 'LTR':'TRP', 'LYM':'LYS', 'LYZ':'LYS', 'MAA':'ALA', 'MEN':'ASN',
#     'MHS':'HIS', 'MIS':'SER', 'MK8':'LEU', 'MLE':'LEU', 'MPQ':'GLY', 'MSA':'GLY', 'MSE':'MET', 'MVA':'VAL', 'NEM':'HIS', 'NEP':'HIS', 'NLE':'LEU',
#     'NLN':'LEU', 'NLP':'LEU', 'NMC':'GLY', 'OAS':'SER', 'OCS':'CYS', 'OMT':'MET', 'PAQ':'TYR', 'PCA':'GLU', 'PEC':'CYS', 'PHI':'PHE',
#     'PHL':'PHE', 'PR3':'CYS', 'PRR':'ALA', 'PTR':'TYR', 'PYX':'CYS', 'SAC':'SER', 'SAR':'GLY', 'SCH':'CYS', 'SCS':'CYS', 'SCY':'CYS',
#     'SEL':'SER', 'SEP':'SER', 'SET':'SER', 'SHC':'CYS', 'SHR':'LYS', 'SMC':'CYS', 'SOC':'CYS', 'STY':'TYR', 'SVA':'SER', 'TIH':'ALA',
#     'TPL':'TRP', 'TPO':'THR', 'TPQ':'ALA', 'TRG':'LYS', 'TRO':'TRP', 'TYB':'TYR', 'TYI':'TYR', 'TYQ':'TYR', 'TYS':'TYR', 'TYY':'TYR'
# }   # Non-standard amino acids from PDBFixer
# Originally used to exclude non-standard amino acids. But sometimes they are used as ligand, so let's just keep them.

exclude_list = ['HOH', 'SO4', 'GOL', 'ACT']
amino_acids = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLU",
               "GLN", "GLY", "HIS", "ILE", "LEU", "LYS",
               "MET", "PHE", "PRO", "SER", "THR", "TRP",
               "TYR", "VAL"]

protein_format_regexes = {'pdb': {'line'  : re.compile(r'(^|\n)(ATOM..|TER...|HETATM).{11}(?!\s+(DT|DA|DC|DG|DI|A|U|C|G|I)).*'),
                                  'hetatm': re.compile(rf'HETATM.{{11}}(\w{{3}}(?<!{"|".join(exclude_list)})).(\w)(.{{4}}).{{4}}(.{{8}})(.{{8}})(.{{8}})'),
                                  'aa_het': re.compile(rf'HETATM.{{11}}({"|".join(amino_acids)}).*'),   # inverse match!
                                  },
                          'cif': {'line'  : re.compile(r'ATOM\s+\d+\s+\w+\s[\w|\"|\']+\s+\.\s(DT|DA|DC|DG|DI|A\s|T\s|C\s|G\s|I\s).*'),    # inverse match!
                                  'hetatm': re.compile(rf'HETATM\s+\d+\s+\w+\s+\w+\s+\.\s+(\w+(?<!{"|".join(exclude_list)}))\s+(\w+)\s(\d+)\s\.\s+.\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+'),
                                  'aa_het': re.compile(rf'HETATM\s+\d+\s+\w+\s+\w+\s+\.\s+({"|".join(amino_acids)}).*'),    # inverse match!
                                  },
                          }

### PDBQTReceptor / RDKitMolCreate / PDBQTMolecule are copied/modified from meeko so I don't have to package the whole meeko package ###

atom_type_map = {'HD': 'H', 'HS': 'H',
                 'NA': 'N', 'NS': 'N',
                 'A' : 'C', 'G' : 'C', 'CG0': 'C', 'CG1': 'C', 'CG2': 'C', 'CG3': 'C', 'G0': 'C', 'G1': 'C', 'G2': 'C', 'G3': 'C',
                 'OA': 'O', 'OS': 'O',
                 'SA': 'S'}

atom_property_definitions = {'H': 'vdw', 'C': 'vdw', 'A': 'vdw', 'N': 'vdw', 'P': 'vdw', 'S': 'vdw',
                             'Br': 'vdw', 'I': 'vdw', 'F': 'vdw', 'Cl': 'vdw',
                             'NA': 'hb_acc', 'OA': 'hb_acc', 'SA': 'hb_acc', 'OS': 'hb_acc', 'NS': 'hb_acc',
                             'HD': 'hb_don', 'HS': 'hb_don',
                             'Mg': 'metal', 'Ca': 'metal', 'Fe': 'metal', 'Zn': 'metal', 'Mn': 'metal',
                             'MG': 'metal', 'CA': 'metal', 'FE': 'metal', 'ZN': 'metal', 'MN': 'metal',
                             'W': 'water',
                             'G0': 'glue', 'G1': 'glue', 'G2': 'glue', 'G3': 'glue',
                             'CG0': 'glue', 'CG1': 'glue', 'CG2': 'glue', 'CG3': 'glue'}

def _read_receptor_pdbqt_string(pdbqt_string, skip_typing=False):
    atoms = []
    atoms_dtype = [('idx', 'i4'), ('serial', 'i4'), ('name', 'U4'), ('resid', 'i4'),
                   ('resname', 'U3'), ('chain', 'U1'), ("xyz", "f4", (3)),
                   ('partial_charges', 'f4'), ('atom_type', 'U2'),
                   ('alt_id', 'U1'), ('in_code', 'U1'),
                   ('occupancy', 'f4'), ('temp_factor', 'f4'), ('record_type', 'U6')
                  ]
    atom_annotations = {'hb_acc': [], 'hb_don': [],
                        'all': [], 'vdw': [],
                        'metal': []}
    # TZ is a pseudo atom for AutoDock4Zn FF
    pseudo_atom_types = ['TZ']

    idx = 0
    for line in pdbqt_string.split('\n'):
        if line.startswith('ATOM') or line.startswith("HETATM"):
            serial = int(line[6:11].strip())
            name = line[12:16].strip()
            resname = line[17:20].strip()
            chainid = line[21].strip()
            resid = int(line[22:26].strip())
            xyz = np.array([line[30:38].strip(), line[38:46].strip(), line[46:54].strip()], dtype=np.float32)
            try:
                partial_charges = float(line[71:77].strip())
            except:
                partial_charges = None # probably reading a PDB, not PDBQT
            atom_type = line[77:79].strip()
            alt_id = line[16:17].strip()
            in_code = line[26:27].strip()
            try:
                occupancy = float(line[54:60])
            except:
                occupancy = None
            try:
                temp_factor = float(line[60:68])
            except:
                temp_factor = None
            record_type = line[0:6].strip()

            if skip_typing:
                atoms.append((idx, serial, name, resid, resname, chainid, xyz, partial_charges, atom_type,
                              alt_id, in_code, occupancy, temp_factor, record_type))
                continue
            if not atom_type in pseudo_atom_types:
                atom_annotations['all'].append(idx)
                atom_annotations[atom_property_definitions[atom_type]].append(idx)
                atoms.append((idx, serial, name, resid, resname, chainid, xyz, partial_charges, atom_type,
                              alt_id, in_code, occupancy, temp_factor, record_type))

            idx += 1

    atoms = np.array(atoms, dtype=atoms_dtype)

    return atoms, atom_annotations

def _write_pdbqt_line(atomidx, x, y, z, charge, atom_name, res_name, res_num, atom_type, chain,
                      alt_id=" ", in_code="", occupancy=1.0, temp_factor=0.0, record_type="ATOM"):
    if len(atom_name) > 4:
        raise ValueError("max length of atom_name is 4 but atom name is %s" % atom_name)
    atom_name = "%-3s" % atom_name
    line = "{:6s}{:5d} {:4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}    {:6.3f} {:<2s}"
    line += '\n'
    return line.format(record_type, atomidx, atom_name, alt_id, res_name, chain,
                   res_num, in_code, x, y, z,
                   occupancy, temp_factor, charge, atom_type)

class PDBQTReceptor:
    flexres_templates = flexres_templates
    skip_types=("H",)

    def __init__(self, pdbqt_string: str, skip_typing=False):
        self._atoms = None
        self._atom_annotations = None
        
        self._atoms, self._atom_annotations = _read_receptor_pdbqt_string(pdbqt_string, skip_typing)
        self.atom_idxs_by_res = self.get_atom_indices_by_residue(self._atoms)
    
    @staticmethod
    def get_atom_indices_by_residue(atoms):
        """ return a dictionary where residues are keys and
             values are lists of atom indices

            >>> atom_idx_by_res = {("A", "LYS", 417): [0, 1, 2, 3, ..., 8]}
        """

        atom_idx_by_res = {}
        for atom_index, atom in enumerate(atoms):
            res_id = (atom["chain"], atom["resname"], atom["resid"])
            atom_idx_by_res.setdefault(res_id, [])
            atom_idx_by_res[res_id].append(atom_index)
        return atom_idx_by_res
    
    @staticmethod
    def get_params_for_residue(resname, atom_names, residue_params=residue_params):
        excluded_params = ("atom_names", "bond_cut_atoms", "bonds")
        atom_params = {}
        atom_counter = 0
        err = ""
        ok = True
        is_matched = False
        for terminus in ["", "N", "C"]: # e.g. "CTYR" for C-term TYR, hard-coded in residue_params
            r_id = "%s%s" % (terminus, resname)
            if r_id not in residue_params:
                err = "residue %s not in residue_params" % r_id + '\n'
                ok = False
                return atom_params, ok, err
            ref_names = set(residue_params[r_id]["atom_names"])
            query_names = set(atom_names)
            if ref_names == query_names:
                is_matched = True
                break

        if not is_matched:
            ok = False
            return atom_params, ok, err

        for atom_name in atom_names:
            name_index = residue_params[r_id]["atom_names"].index(atom_name)
            for param in residue_params[r_id].keys():
                if param in excluded_params:
                    continue
                if param not in atom_params:
                    atom_params[param] = [None] * atom_counter
                value = residue_params[r_id][param][name_index]
                atom_params[param].append(value)
            atom_counter += 1

        return atom_params, ok, err
    
    def assign_types_charges(self, residue_params=residue_params):
        wanted_params = ("atom_types", "gasteiger")
        atom_params = {key: [] for key in wanted_params}
        ok = True
        err = ""
        for r_id, atom_indices in self.atom_idxs_by_res.items():
            atom_names = tuple(self.atoms(atom_indices)["name"])
            resname = r_id[1]
            params_this_res, ok_, err_ = self.get_params_for_residue(resname, atom_names, residue_params)
            ok &= ok_
            err += err_
            if not ok_:
                print("did not match %s with template" % str(r_id), file=sys.stderr)
                continue
            for key in wanted_params:
                atom_params[key].extend(params_this_res[key])
        if ok:
            self._atoms["partial_charges"] = atom_params["gasteiger"]
            self._atoms["atom_type"] = atom_params["atom_types"]
        return ok, err
    
    def write_flexres_from_template(self, res_id, atom_index=0):
        success = True
        error_msg = ""
        branch_offset = atom_index # templates assume first atom is 1
        output = {"pdbqt": "", "flex_indices": [], "atom_index": atom_index}
        resname = res_id[1]
        if resname not in self.flexres_templates:
            success = False
            error_msg = "no flexible residue template for resname %s, sorry" % resname
            return output, success, error_msg
        if res_id not in self.atom_idxs_by_res:
            success = False
            chains = set(self._atoms["chain"])
            error_msg += "could not find residue with chain='%s', resname=%s, resnum=%d" % res_id + '\n'
            error_msg += "chains in this receptor: %s" % ", ".join("'%s'" % c for c in chains) + '\n'
            if " " in chains: # should not happen because we use strip() when parsing the chain
                error_msg += "use ' ' (a space character) for empty chain" + '\n'
            if "" in chains:
                error_msg += "use '' (empty string) for empty chain" + '\n'
            return output, success, error_msg

        # collect lines of res_id
        atoms_by_name = {}
        for i in self.atom_idxs_by_res[res_id]:
            name = self._atoms[i]["name"]
            if name in ['C', 'N', 'O', 'H', 'H1', 'H2', 'H3', 'OXT']: # skip backbone atoms
                continue
            atype = self._atoms[i]["atom_type"]
            if atype in self.skip_types:
                continue
            output["flex_indices"].append(i)
            atoms_by_name[name] = self.atoms(i)

        # check it was a full match
        template = self.flexres_templates[resname]
        got_atoms = set(atoms_by_name)
        ref_atoms = set()
        for i in range(len(template["is_atom"])):
            if template["is_atom"][i]:
                ref_atoms.add(template["atom_name"][i])
        if got_atoms != ref_atoms:
            success = False
            error_msg += "mismatch in atom names for residue %s" % str(res_id) + '\n'
            error_msg += "names found but not in template: %s" % str(got_atoms.difference(ref_atoms)) + '\n'
            error_msg += "missing names: %s" % str(ref_atoms.difference(got_atoms)) + '\n'
            return output, success, error_msg

        # create output string
        n_lines = len(template['is_atom'])
        for i in range(n_lines):
            if template['is_atom'][i]:
                atom_index += 1
                name = template['atom_name'][i]
                atom = atoms_by_name[name]
                if atom["atom_type"] not in self.skip_types:
                    atom["serial"] = atom_index
                    output["pdbqt"] += self.write_pdbqt_line(atom)
            else:
                line = template['original_line'][i]
                if branch_offset > 0 and (line.startswith("BRANCH") or line.startswith("ENDBRANCH")):
                    keyword, i, j = line.split()
                    i = int(i) + branch_offset
                    j = int(j) + branch_offset
                    line = "%s %3d %3d" % (keyword, i, j)
                output["pdbqt"] += line + '\n' # e.g. BRANCH keywords

        output["atom_index"] = atom_index
        return output, success, error_msg
    
    @staticmethod
    def write_pdbqt_line(atom):
        return _write_pdbqt_line(atom["serial"], atom["xyz"][0], atom["xyz"][1], atom["xyz"][2],
                                 atom["partial_charges"], atom["name"], atom["resname"],
                                 atom["resid"], atom["atom_type"], atom["chain"],
                                 atom["alt_id"], atom["in_code"], atom["occupancy"],
                                 atom["temp_factor"], atom["record_type"])
        
    def write_pdbqt_string(self, flexres=()):
        ok = True
        err = ""
        pdbqt = {"rigid": "",
                 "flex":  {},
                 "flex_indices": []}
        atom_index = 0
        for res_id in set(flexres):
            output, ok_, err_ = self.write_flexres_from_template(res_id, atom_index)
            atom_index = output["atom_index"] # next residue starts here
            ok &= ok_
            err += err_
            pdbqt["flex_indices"].extend(output["flex_indices"])
            pdbqt["flex"][res_id] = ""
            pdbqt["flex"][res_id] += "BEGIN_RES %3s %1s%4d" % (res_id) + '\n'
            pdbqt["flex"][res_id] += output["pdbqt"]
            pdbqt["flex"][res_id] += "END_RES %3s %1s%4d" % (res_id) + '\n'
        all_flex_pdbqt = ""
        for res_id, flexres_pdbqt in pdbqt["flex"].items():
            all_flex_pdbqt += flexres_pdbqt
        pdbqt['flex'] = all_flex_pdbqt
        
        # use non-flex lines for rigid part
        for i, atom in enumerate(self._atoms):
            if i not in pdbqt["flex_indices"] and atom["atom_type"] not in self.skip_types:
                pdbqt["rigid"] += self.write_pdbqt_line(atom)
                
        return pdbqt, ok, err
    
    def atoms(self, atom_idx=None):
        """Return the atom i

        Args:
            atom_idx (int, list): index of one or multiple atoms

        Returns:
            ndarray: 2d ndarray (atom_id, atom_name, resname, resid, chainid, xyz, q, t)

        """
        if atom_idx is not None and self._atoms.size > 1:
            if not isinstance(atom_idx, (list, tuple, np.ndarray)):
                atom_idx = np.array(atom_idx, dtype=int)
            atoms = self._atoms[atom_idx]
        else:
            atoms = self._atoms

        return atoms.copy()

def parse_line(line):
    return {'record_name': line[:6].strip(),
            'atom_pos'   : int(line[6:11].strip()),
            'atom_name'  : line[12:16].strip(),
            'alt_id'     : line[16:17],
            'aa_name'    : line[17:20].strip(),
            'chain'      : line[21].strip(),
            'aa_pos'     : int(line[22:26].strip()),
            'insert'     : line[26],
            'x'          : float(line[30:38].strip()),
            'y'          : float(line[38:46].strip()),
            'z'          : float(line[46:54].strip()),
            'occupency'  : float(line[54:60].strip()),
            'b_factor'   : float(line[60:66].strip()),
            'atom_type'  : line[76:78].strip()}
    
def parse_ter(line):
    return {'record_name': line[:6].strip(),
            'atom_pos'   : int(line[6:11].strip()),
            'atom_name'  : line[12:16].strip(),
            'alt_id'     : line[16:17],
            'aa_name'    : line[17:20].strip(),
            'chain'      : line[21].strip(),
            'aa_pos'     : int(line[22:26].strip()),}

def clean_pdb(protein_str: str, return_hetatm: bool=False, fill_gap: bool=False, format: str=None):
    if format is None:
        if not protein_str.startswith('data_'):
            format = 'pdb'
        else:
            format = 'cif'  # For CIF file directly downloaded from PDB
    retrieve_line_without_na_compiled = protein_format_regexes[format]['line']
    exclude_hetatm_contain_aa_line_compiled = protein_format_regexes[format]['aa_het']
    s = re.search(retreive_mdl_compiled, protein_str)
    if s is not None:
        header_strs = []
        for l in protein_str.splitlines():
            if l.startswith('MODEL        1'):
                break
            header_strs.append(l)
        protein_str = '\n'.join(header_strs) + '\n' + s.group(0)
    if not fill_gap:
        if format == 'pdb':
            final_protein_str = ''.join([l.group(0) for l in re.finditer(retrieve_line_without_na_compiled, protein_str)]).strip()
        elif format == 'cif':
            final_protein_str = '\n'.join(l for l in protein_str.splitlines() if not retrieve_line_without_na_compiled.match(l))
    else:
        final_protein_str = protein_str
    if not return_hetatm:
        final_protein_str = '\n'.join(l for l in final_protein_str.splitlines() if not exclude_hetatm_contain_aa_line_compiled.match(l))
        return final_protein_str
    else:
        retrieve_hetatm_name_chain_xyz_compiled = protein_format_regexes[format]['hetatm']
        hetatm_dict = {}
        # HOH & ions & non-standard amino acids are not matched
        for matched in re.finditer(retrieve_hetatm_name_chain_xyz_compiled, protein_str):
            name, chain, pos, x, y, z = matched.group(1, 2, 3, 4, 5, 6)
            ligand = f'[{name}]{pos.strip()}:{chain}'
            if ligand not in hetatm_dict:
                hetatm_dict[ligand] = []
            hetatm_dict[ligand].append([float(x), float(y), float(z)])
        final_hetatm_dict = {}
        for ligand, xyz in hetatm_dict.items():
            xyz = np.array(xyz)
            max_xyz, min_xyz = xyz.max(0)+1.5, xyz.min(0)-1.5   # 1.5 Å padding on each direction
            box = np.round(max_xyz - min_xyz, 3)
            center = np.round((max_xyz + min_xyz) / 2, 3)
            volume = np.prod(box).round(3)
            final_hetatm_dict[ligand] = {'Center': list(center),
                                         'Box'   : list(box)   ,
                                         'Volume': float(volume),}
        final_protein_str = '\n'.join(l for l in final_protein_str.splitlines() if not exclude_hetatm_contain_aa_line_compiled.match(l))
        return final_protein_str, final_hetatm_dict

def read_pdb_string(pdb_str: str):
    result = []
    conect_record = []
    for l in pdb_str.splitlines():
        if l.startswith('ATOM'):    # HETATM is already cleaned!
            d = parse_line(l)
            result.append(d)
        elif l.startswith('TER'):
            d = parse_ter(l)
            result.append(d)
        elif l.startswith('CONECT'):
            conect_record.append(l)
    return pd.DataFrame(result), conect_record

def read_pdb_file(pdb_pth: str):
    with open(pdb_pth, 'r') as f:
        result = []
        conect_record = []
        for l in f:
            if l.startswith('ATOM'):    # HETATM is already cleaned!
                d = parse_line(l)
                result.append(d)
            elif l.startswith('TER'):
                d = parse_ter(l)
                result.append(d)
            elif l.startswith('CONECT'):
                conect_record.append(l)
    return pd.DataFrame(result), conect_record

def fix_pdb_missing_atoms(pdb_pth: str, out_pth: str | None=None, replace_nonstandard: bool=True,
                          fill_gap: bool=False, ph: float=7.0):
    fixer = PDBFixer(pdb_pth)
    fixer.findNonstandardResidues()
    nonstandards = fixer.nonstandardResidues.copy()
    
    # Sometimes non-standards amino acids are used as ligand in resolved structure, remove them.
    for chain_resname in nonstandards:
        tmp = chain_resname[0]
        chain = tmp.chain
        if tmp.index > len(list(chain.residues())):
            fixer.nonstandardResidues.remove(chain_resname)
    if fill_gap:
        fixer.findMissingResidues()
        
        # Only fill in gaps "within" protein. Missing residues in terminals are ignored.
        chains = list(fixer.topology.chains())
        keys = list(fixer.missingResidues.keys())
        for key in keys:
            chain = chains[key[0]]
            if key[1] == 0 or key[1] == len(list(chain.residues())):
                del fixer.missingResidues[key]
    else:
        fixer.missingResidues = {}
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    if out_pth is not None:
        PDBFile.writeFile(fixer.topology, fixer.positions, out_pth, keepIds=True)
    else:
        sio = io.StringIO()
        PDBFile.writeFile(fixer.topology, fixer.positions, sio, keepIds=True)
        return sio.getvalue()

def retrieve_disulfide_bonds(conect_record: list):
    disulfide_bonds = []
    for l in conect_record:
        pos1, pos2 = int(l[6:11].strip()), int(l[11:16].strip())
        disulfide_bonds.extend([pos1, pos2])
    return set(disulfide_bonds)

def _check_histidine(aa_df: pd.DataFrame, *args):
    chains = list(dict.fromkeys(aa_df['chain']))
    result = []
    for c in chains:
        chain_df = aa_df[aa_df['chain'] == c]
        for aa_pos in set(chain_df[chain_df['aa_name'] == 'HIS']['aa_pos']):
            curr_pos = chain_df['aa_pos'] == aa_pos
            curr_atom_names = set(chain_df[curr_pos]['atom_name'])
            if {'HD1', 'HE2'}.issubset(curr_atom_names):
                chain_df.loc[curr_pos, 'aa_name'] = 'HIP'
            elif {'HD1'}.issubset(curr_atom_names):
                chain_df.loc[curr_pos, 'aa_name'] = 'HID'
            elif {'HE2'}.issubset(curr_atom_names):
                chain_df.loc[curr_pos, 'aa_name'] = 'HIE'
        result.append(chain_df)
    return pd.concat(result, axis=0, ignore_index=True)

def _check_glutmate(aa_df: pd.DataFrame, *args):
    chains = list(dict.fromkeys(aa_df['chain']))
    result = []
    for c in chains:
        chain_df = aa_df[aa_df['chain'] == c]
        for aa_pos in set(chain_df[chain_df['aa_name'] == 'GLU']['aa_pos']):
            curr_pos = chain_df['aa_pos'] == aa_pos
            curr_atom_names = set(chain_df[curr_pos]['atom_name'])
            if {'HE2'}.issubset(curr_atom_names):
                chain_df.loc[curr_pos, 'aa_name'] = 'GLH'
        result.append(chain_df)
    return pd.concat(result, axis=0, ignore_index=True)

def _check_aspartate(aa_df: pd.DataFrame, *args):
    chains = list(dict.fromkeys(aa_df['chain']))
    result = []
    for c in chains:
        chain_df = aa_df[aa_df['chain'] == c]
        for aa_pos in set(chain_df[chain_df['aa_name'] == 'ASP']['aa_pos']):
            curr_pos = chain_df['aa_pos'] == aa_pos
            curr_atom_names = set(chain_df[curr_pos]['atom_name'])
            if {'HD2'}.issubset(curr_atom_names):
                chain_df.loc[curr_pos, 'aa_name'] = 'ASH'
        result.append(chain_df)
    return pd.concat(result, axis=0, ignore_index=True)

def _check_lysine(aa_df: pd.DataFrame, *args):
    chains = list(dict.fromkeys(aa_df['chain']))
    result = []
    for c in chains:
        chain_df = aa_df[aa_df['chain'] == c]
        for aa_pos in set(chain_df[chain_df['aa_name'] == 'LYS']['aa_pos']):
            curr_pos = chain_df['aa_pos'] == aa_pos
            curr_atom_names = set(chain_df[curr_pos]['atom_name'])
            if not {'HZ1'}.issubset(curr_atom_names):
                chain_df.loc[curr_pos, 'aa_name'] = 'LYN'
        result.append(chain_df)
    return pd.concat(result, axis=0, ignore_index=True)

def _check_cysteine(aa_df: pd.DataFrame, conect_record: list):
    # remaining_texts are used to check for CONECT to find recorded disulfide bond.
    disulfide_bonds = None
    chains = list(dict.fromkeys(aa_df['chain']))
    result = []
    for c in chains:
        chain_df = aa_df[aa_df['chain'] == c]
        all_cys_position = set(chain_df[chain_df['aa_name'] == 'CYS']['aa_pos'])
        for aa_pos in all_cys_position:
            curr_pos = chain_df['aa_pos'] == aa_pos
            curr_atom_names = set(chain_df[curr_pos]['atom_name'])
            if not {'HG'}.issubset(curr_atom_names):
                if len(all_cys_position) == 1:
                    chain_df.loc[curr_pos, 'aa_name'] = 'CYM'
                else:
                    if disulfide_bonds is None:
                        disulfide_bonds = retrieve_disulfide_bonds(conect_record)
                    if disulfide_bonds:
                        atom_pos = set(chain_df[curr_pos]['atom_pos'].to_list())
                        intersect = disulfide_bonds.intersection(atom_pos)
                        if intersect:
                            chain_df.loc[curr_pos, 'aa_name'] = 'CYX'
                        else:
                            chain_df.loc[curr_pos, 'aa_name'] = 'CYM'
        result.append(chain_df)
    return pd.concat(result, axis=0, ignore_index=True)

def _check_for_N_terminal_H(aa_df: pd.DataFrame, *args):
    chains = list(dict.fromkeys(aa_df['chain']))
    result = []
    for c in chains:
        chain_df = aa_df[aa_df['chain'] == c]
        first_aa_pos = chain_df.iloc[0]['aa_pos']
        curr_pos = chain_df['aa_pos'] == first_aa_pos
        curr_atom_names = set(chain_df[curr_pos]['atom_name'])
        if {'H2', 'H3'}.issubset(curr_atom_names):
            mask_h = curr_pos & (chain_df['atom_name'] == 'H')
            chain_df.loc[mask_h, 'atom_name'] = 'H1'
        elif {'H3'}.issubset(curr_atom_names):  # Some N-PRO only has H3 and some only H2, so the other "H" is set to "H2"/"H3".
            mask_h = curr_pos & (chain_df['atom_name'] == 'H')
            chain_df.loc[mask_h, 'atom_name'] = 'H2'
        elif {'H2'}.issubset(curr_atom_names):  # Some N-PRO only has H3 and some only H2, so the other "H" is set to "H2"/"H3".
            mask_h = curr_pos & (chain_df['atom_name'] == 'H')
            chain_df.loc[mask_h, 'atom_name'] = 'H3'
        result.append(chain_df)
    aa_df = pd.concat(result, axis=0, ignore_index=True)
    aa_df['atom_pos'] = list(range(1, len(aa_df) + 1))
    return aa_df

def _check_for_inserts(aa_df: pd.DataFrame, *args):
    finalidx_chain_dict = {}
    last_aa_pos = aa_df.loc[0, 'aa_pos']
    last_insert = aa_df.loc[0, 'insert']
    for idx, row in aa_df.iterrows():
        if idx == 0:
            continue
        curr_aa_pos, curr_insert = row.loc['aa_pos'], row['insert']
        if (curr_aa_pos == last_aa_pos) & (curr_insert != last_insert):
            chain = row.loc['chain']
            finalidx_chain_dict[idx] = chain
        last_aa_pos, last_insert = curr_aa_pos, curr_insert
    for idx, chain in finalidx_chain_dict.items():
        mask = (aa_df['chain'] == chain) & (aa_df.index >= idx)
        aa_df.loc[mask, 'aa_pos'] += 1
    aa_df['insert'] = ' '
    return aa_df

def _check_for_aa_count(aa_df: pd.DataFrame):
    chain_unique_cnt = aa_df.groupby('chain')['aa_pos'].nunique()
    single_aa_chain = chain_unique_cnt[chain_unique_cnt == 1]
    if not single_aa_chain.empty:
        for chain in single_aa_chain.index:
            aa_df = aa_df[aa_df['chain'] != chain]
    return aa_df

def check_amino_acids(aa_df: pd.DataFrame, conect_record: list):
    # Using a dict for readability, bc my brain is fried
    aa_func_map = {'INS'  : _check_for_inserts,
                   'HIS'  : _check_histidine,
                   'GLU'  : _check_glutmate ,
                   'ASP'  : _check_aspartate,
                   'LYS'  : _check_lysine   ,
                   'CYS'  : _check_cysteine ,
                   'N_ter': _check_for_N_terminal_H,}
    for aa_func in aa_func_map.values():
        aa_df = aa_func(aa_df, conect_record)
    return aa_df.reset_index(drop=True)

def get_pdb_string(aa_df: pd.DataFrame):
    pdb_format = "{:6s}{:5d} {:4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}           {:<2s}"
    ter_format = "{:6s}{:5d} {:4s}{:1s}{:3s} {:1s}{:4d}"
    pdb_strings = []
    for _, line in aa_df.iterrows():
        if line['record_name'] == 'TER':
            pdb_strings.append(ter_format.format(*line))
        else:
            pdb_strings.append(pdb_format.format(*line))
    return '\n'.join(pdb_strings)

def write_to_pdbqt(output_pth: str | None, aa_df: pd.DataFrame):
    pdb_string = get_pdb_string(aa_df)
    receptor = PDBQTReceptor(pdb_string, skip_typing=True)
    ok, err = receptor.assign_types_charges()
    if ok:
        pdbqt_string, ok, err = receptor.write_pdbqt_string()
        if ok:
            if output_pth is not None:
                with open(output_pth, 'w') as f:
                    f.write(pdbqt_string['rigid'])
            else:
                return pdbqt_string['rigid']
        else:
            return (err,)
    else:
        return (err,)

def pdbqt_to_pdb(pdbqt_str: str):
    meet_end_of_chain = 0
    sub_re_map = {'HIS': r'HID|HIP|HIE',
                  'GLU': r'GLH',
                  'ASP': r'ASH',
                  'LYS': r'LYN',
                  'CYS': r'CYM|CYX',}
    last_chain, last_atomidx, last_resname, last_respos = None, None, None, None
    
    def map_pdbqt_line_to_pdb(line: str, idx: int):
        nonlocal meet_end_of_chain, last_chain, last_resname, last_respos, last_atomidx
        atom_type = line[77:].strip()
        if atom_type in atom_type_map:
            atom_type = atom_type_map[atom_type]
        chain = line[21]
        res_name = line[17:20].strip()
        res_pos = int(line[22:26].strip())
        final = []
        if chain != last_chain and last_chain is not None:
            ter = {'atom_idx': last_atomidx + 1,
                   'res_name': last_resname    ,
                   'chain'   : last_chain      ,
                   'res_pos' : last_respos     ,
                   }
            final.append(ter)
            meet_end_of_chain += 1
        atom_idx = idx + meet_end_of_chain
        line_dict = {'atom_idx'   : atom_idx                  ,
                     'atom_name'  : line[12:16].strip()       ,
                     'alt_id'     : line[16]                  ,
                     'res_name'   : res_name                  ,
                     'chain'      : chain                     ,
                     'res_pos'    : res_pos                   ,
                     'others'     : line[26:66]               ,  # skip partial charge
                     'atom_type'  : atom_type                 ,
                     }
        final.append(line_dict)
        last_chain, last_resname, last_respos, last_atomidx = chain, res_name, res_pos, atom_idx
        return final
    
    def convert_to_pdb_str(atom_data: list):
        lines = []
        format_str = "ATOM  {:5d} {:4s}{:1s}{:3s} {:1s}{:4d}{:40s}          {:>2s}\n"
        term_str   = "TER   {:5d}      {:3s} {:1s}{:4d}\n"
        for entry in atom_data:
            if len(entry) == 8:
                lines.append(format_str.format(
                    entry['atom_idx'], entry['atom_name'], entry['alt_id'], 
                    entry['res_name'], entry['chain'], entry['res_pos'],
                    entry['others'], entry['atom_type'],
                ))
            else:
                lines.append(term_str.format(
                    entry['atom_idx'], entry['res_name'], 
                    entry['chain'], entry['res_pos'],
                ))
        return ''.join(lines)
    
    for aa, re_comp in sub_re_map.items():
        pdbqt_str = re.sub(re_comp, aa, pdbqt_str)
    
    protein_data = [item for idx, line in enumerate(pdbqt_str.strip().splitlines()) if line.startswith('ATOM')
                    for item in map_pdbqt_line_to_pdb(line, idx)]
    final_ter = {'atom_idx': last_atomidx + 1,
                 'res_name': last_resname    ,
                 'chain'   : last_chain      ,
                 'res_pos' : last_respos     ,
                 }
    protein_data.append(final_ter)
    
    return protein_data, convert_to_pdb_str(protein_data)

# TODO: Check why some of these proteins failed.
# 4QM8: ('D', 'CYS', 202), ('F', 'CYS', 202), Amino acids are used as ligand. Cleaning pdb first fix this issue.
# 3RUV: ('F', 'ASN', 493), FIXED. Likely unclean? Cleaning pdb first fix this issue.
# 3KQB: ('A', 'ASH', 185), FIXED. Insertion issue.
# 1TON: ('A', 'GLH', 186), FIXED. Insertion issue.
# 7YCD: ('C', 'PRO', 4), FIXED. N-ter Pro issue.
# 1AVG: ('L', 'GLH', 8), ('L', 'GLH', 34), ('H', 'ASH', 66), FIXED. Insertion issue.
# 3FIN: Ribosome, large complex. Cleaning regex failed.
# 1TMB: ('T', 'PRO', 52), FIXED. Chain T only has 1 amino acid. Chains with only 1 aa are removed by default.
# 1EIV: ('A', 'ASN', 37), PDB obsolete so not gonna fix it~
def fix_and_convert(pdb_pth: str, output_pth: str=None, fill_gap: bool=False, ph: float=7.0):
    try:
        pdb_str = fix_pdb_missing_atoms(pdb_pth, fill_gap=fill_gap, ph=ph)
    except Exception as e:
        return ('PDB file lacks protein!',)
    df, conect_record = read_pdb_string(pdb_str)
    df = _check_for_aa_count(df) # remove chains with only 1 amino acid
    if df.empty:
        return ('PDB file lacks protein!',)
    df = check_amino_acids(df, conect_record)
    return write_to_pdbqt(output_pth, df)

def process_rigid_flex(pdbqt_str: str, rigid_out_pth: str, flex_out_pth: str, flex_res: set):
    receptor = PDBQTReceptor(pdbqt_str, skip_typing=True)
    pdbqt_string, _, _ = receptor.write_pdbqt_string(flex_res)
    with open(rigid_out_pth, 'w') as f:
        f.write(pdbqt_string['rigid'])
    if flex_res and flex_out_pth is not None:
        with open(flex_out_pth, 'w') as f:
            f.write(pdbqt_string['flex'])

class RDKitMolCreate:

    ambiguous_flexres_choices = {
        "HIS": ["HIE", "HID", "HIP"],
        "ASP": ["ASP", "ASH"],
        "GLU": ["GLU", "GLH"],
        "CYS": ["CYS", "CYM"],
        "LYS": ["LYS", "LYN"],
        "ARG": ["ARG", "ARG_mgltools"],
        "ASN": ["ASN", "ASN_mgltools"],
        "GLN": ["GLN", "GLN_mgltools"],
    }

    flexres = {
        "CYS": {
            "smiles": "CCS",
            "atom_names_in_smiles_order": ["CA", "CB", "SG"],
            "h_to_parent_index": {"HG": 2},
        },
        "CYM": {
            "smiles": "CC[S-]",
            "atom_names_in_smiles_order": ["CA", "CB", "SG"],
            "h_to_parent_index": {},
        },
        "ASP": {
            "smiles": "CCC(=O)[O-]",
            "atom_names_in_smiles_order": ["CA", "CB", "CG", "OD1", "OD2"],
            "h_to_parent_index": {},
        },
        "ASH": {
            "smiles": "CCC(=O)O",
            "atom_names_in_smiles_order": ["CA", "CB", "CG", "OD1", "OD2"],
            "h_to_parent_index": {"HD2": 4},
        },
        "GLU": {
            "smiles": "CCCC(=O)[O-]",
            "atom_names_in_smiles_order": ["CA", "CB", "CG", "CD", "OE1", "OE2"],
            "h_to_parent_index": {},
        },
        "GLH": {
            "smiles": "CCCC(=O)O",
            "atom_names_in_smiles_order": ["CA", "CB", "CG", "CD", "OE1", "OE2"],
            "h_to_parent_index": {"HE2": 5},
        },
        "PHE": {
            "smiles": "CCc1ccccc1",
            "atom_names_in_smiles_order": ["CA", "CB", "CG", "CD1", "CE1", "CZ", "CE2", "CD2"],
            "h_to_parent_index": {},
        },
        "HIE" : {
            "smiles": "CCc1c[nH]cn1",
            "atom_names_in_smiles_order": ["CA", "CB", "CG", "CD2", "NE2", "CE1", "ND1"],
            "h_to_parent_index": {"HE2": 4},
        },
        "HID" : {
            "smiles": "CCc1cnc[nH]1",
            "atom_names_in_smiles_order": ["CA", "CB", "CG", "CD2", "NE2", "CE1", "ND1"],
            "h_to_parent_index": {"HD1": 6},
        },
        "HIP" : {
            "smiles": "CCc1c[nH+]c[nH]1",
            "atom_names_in_smiles_order": ["CA", "CB", "CG", "CD2", "NE2", "CE1", "ND1"],
            "h_to_parent_index": {"HE2": 4, "HD1": 6},
        },
        "ILE": {
            "smiles": "CC(C)CC",
            "atom_names_in_smiles_order": ["CA", "CB", "CG2", "CG1", "CD1"],
            "h_to_parent_index": {},
        },
        "LYS": {
            "smiles": "CCCCC[NH3+]",
            "atom_names_in_smiles_order": ["CA", "CB", "CG", "CD", "CE", "NZ"],
            "h_to_parent_index": {"HZ1": 5, "HZ2": 5, "HZ3": 5},
        },
        "LYN": {
            "smiles": "CCCCCN",
            "atom_names_in_smiles_order": ["CA", "CB", "CG", "CD", "CE", "NZ"],
            "h_to_parent_index": {"HZ2": 5, "HZ3": 5},
        },
        "LEU": {
            "smiles": "CCC(C)C",
            "atom_names_in_smiles_order": ["CA", "CB", "CG", "CD1", "CD2"],
            "h_to_parent_index": {},
        },
        "MET": {
            "smiles": "CCCSC",
            "atom_names_in_smiles_order": ["CA", "CB", "CG", "SD", "CE"],
            "h_to_parent_index": {},
        },
        "ASN": {
            "smiles": "CCC(=O)N",
            "atom_names_in_smiles_order": ["CA", "CB", "CG", "OD1", "ND2"],
            "h_to_parent_index": {"HD21": 4, "HD22": 4},
        },
        "ASN_mgltools": {
            "smiles": "CCC(=O)N",
            "atom_names_in_smiles_order": ["CA", "CB", "CG", "OD1", "ND2"],
            "h_to_parent_index": {"1HD2": 4, "2HD2": 4},
        },
        "GLN": {
            "smiles": "CCCC(=O)N",
            "atom_names_in_smiles_order": ["CA", "CB", "CG", "CD", "OE1", "NE2"],
            "h_to_parent_index": {"HE21": 5, "HE22": 5},
        },
        "GLN_mgltools": {
            "smiles": "CCCC(=O)N",
            "atom_names_in_smiles_order": ["CA", "CB", "CG", "CD", "OE1", "NE2"],
            "h_to_parent_index": {"1HE2": 5, "2HE2": 5},
        },
        "ARG": {
            "smiles": "CCCCNC(N)=[NH2+]",
            "atom_names_in_smiles_order": ["CA", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
            "h_to_parent_index": {"HE": 4, "HH11": 6, "HH12": 6, "HH21": 7, "HH22": 7},
        },
        "ARG_mgltools": {
            "smiles": "CCCCNC(N)=[NH2+]",
            "atom_names_in_smiles_order": ["CA", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
            "h_to_parent_index": {"HE": 4, "1HH1": 6, "2HH1": 6, "1HH2": 7, "2HH2": 7},
        },
        "SER": {
            "smiles": "CCO",
            "atom_names_in_smiles_order": ["CA", "CB", "OG"],
            "h_to_parent_index": {"HG": 2},
        },
        "THR": {
            "smiles": "CC(C)O",
            "atom_names_in_smiles_order": ["CA", "CB", "CG2", "OG1"],
            "h_to_parent_index": {"HG1": 3},
        },
        "VAL": {
            "smiles": "CC(C)C",
            "atom_names_in_smiles_order": ["CA", "CB", "CG1", "CG2"],
            "h_to_parent_index": {},
        },
        "TRP": {
            "smiles": "CCc1c[nH]c2c1cccc2",
            "atom_names_in_smiles_order": ["CA", "CB", "CG", "CD1", "NE1", "CE2", "CD2", "CE3", "CZ3", "CH2", "CZ2"],
            "h_to_parent_index": {"HE1": 4},
        },
        "TYR": {
            "smiles": "CCc1ccc(cc1)O",
            "atom_names_in_smiles_order": ["CA", "CB", "CG", "CD1", "CE1", "CZ", "CE2", "CD2", "OH"],
            "h_to_parent_index": {"HH": 8},
        },
    }

    @classmethod
    def from_pdbqt_mol(cls, pdbqt_mol, only_cluster_leads=False, add_Hs=True):
        if only_cluster_leads and len(pdbqt_mol._pose_data["cluster_leads_sorted"]) == 0:
            raise RuntimeError("no cluster_leads in pdbqt_mol but only_cluster_leads=True")
        mol_list = []
        for mol_index in pdbqt_mol._atom_annotations["mol_index"]:
            smiles = pdbqt_mol._pose_data['smiles'][mol_index]
            index_map = pdbqt_mol._pose_data['smiles_index_map'][mol_index]
            h_parent = pdbqt_mol._pose_data['smiles_h_parent'][mol_index]
            atom_idx = pdbqt_mol._atom_annotations["mol_index"][mol_index]

            if smiles is None: # probably a flexible sidechain, but can be another ligand
                residue_names = set()
                atom_names = []
                for atom in pdbqt_mol.atoms(atom_idx):
                    residue_names.add(atom[4])
                    atom_names.append(atom[2])
                if len(residue_names) == 1:
                    resname = residue_names.pop()
                    smiles, index_map, h_parent = cls.guess_flexres_smiles(resname, atom_names)
                    if smiles is None: # failed guessing smiles for possible flexres
                        mol_list.append(None)
                        continue

            if only_cluster_leads:
                pose_ids = pdbqt_mol._pose_data["cluster_leads_sorted"]
            else:
                pose_ids = range(pdbqt_mol._pose_data["n_poses"])
            
            mol = Chem.MolFromSmiles(smiles)
            
            coordinates_all_poses = []
            for i in pose_ids:
                pdbqt_mol._current_pose = i
                coordinates = pdbqt_mol.positions(atom_idx)
                mol = cls.add_pose_to_mol(mol, coordinates, index_map)
                coordinates_all_poses.append(coordinates)

            # add Hs only after all poses are added as conformers
            # because Chem.AddHs() will affect all conformers at once
            if add_Hs:
                mol = cls.add_hydrogens(mol, coordinates_all_poses, h_parent)
        
            mol_list.append(mol)
        return mol_list

    @classmethod
    def guess_flexres_smiles(cls, resname, atom_names):
        """ Determine a SMILES string for flexres based on atom names,
            as well as the equivalent of smile_index_map and smiles_h_parent
            which are written to PDBQT remarks for regular small molecules.

        Args:
            resname (str):
        
        Returns:
            smiles: SMILES string starting at C-alpha (excludes most of the backbone)
            index_map: list of pairs of integers, first in pair is index in the smiles,
                       second is index of corresponding atom in atom_names         
            h_parent: list of pairs of integers, first in pair is index of a heavy atom
                      in the smiles, second is index of a hydrogen in atom_names.
                      The hydrogen is bonded to the heavy atom. 
        """



        if len(set(atom_names)) != len(atom_names):
            return None, None, None
        candidate_resnames = cls.ambiguous_flexres_choices.get(resname, [resname])
        for resname in candidate_resnames:
            is_match = False
            if resname not in cls.flexres[resname]["atom_names_in_smiles_order"]:
                continue
            atom_names_in_smiles_order = cls.flexres[resname]["atom_names_in_smiles_order"]
            h_to_parent_index = cls.flexres[resname]["h_to_parent_index"]
            expected_names = atom_names_in_smiles_order + list(h_to_parent_index.keys())
            if len(atom_names) != len(expected_names):
                continue
            nr_matched_atom_names = sum([int(n in atom_names) for n in expected_names])
            if nr_matched_atom_names == len(expected_names):
                is_match = True
                break
        if not is_match:
            return None, None, None
        else:
            smiles = cls.flexres[resname]["smiles"]
            index_map = []
            for smiles_index, name in enumerate(atom_names_in_smiles_order):
                index_map.append(smiles_index + 1) 
                index_map.append(atom_names.index(name) + 1)
            h_parent = []
            for name, smiles_index in h_to_parent_index.items():
                h_parent.append(smiles_index + 1)
                h_parent.append(atom_names.index(name) + 1)
            return smiles, index_map, h_parent

    @classmethod
    def add_pose_to_mol(cls, mol, ligand_coordinates, index_map):
        """add given coordinates to given molecule as new conformer.
        Index_map maps order of coordinates to order in smile string
        used to generate rdkit mol

        Args:
            ligand_coordinates: 2D array of shape (nr_atom, 3).
            index_map: list of nr_atom pairs of integers, 1-indexed.
                       In each pair, the first int is the index in mol, and
                       the second int is the index in ligand_coordinates

        Raises:
            RuntimeError: Will raise error if number of coordinates provided does not
                match the number of atoms there should be coordinates for.
        """

        n_atoms = mol.GetNumAtoms()
        n_mappings = int(len(index_map) / 2)
        conf = Chem.Conformer(n_atoms)
        if n_atoms < n_mappings:
            raise RuntimeError(
                "Given {n_coords} atom coordinates "
                "but index_map is greater at {n_at} atoms.".format(
                    n_coords=n_atoms, n_at=n_mappings))
        coord_is_set = [False] * n_atoms
        for i in range(n_mappings):
            pdbqt_index = int(index_map[i * 2 + 1]) - 1
            mol_index = int(index_map[i * 2]) - 1
            x, y, z = [float(coord) for coord in ligand_coordinates[pdbqt_index]]
            conf.SetAtomPosition(mol_index, Point3D(x, y, z))
            coord_is_set[mol_index] = True
        mol.AddConformer(conf, assignId=True)
        e_mol = Chem.RWMol(mol)
        for i, is_set in reversed(list(enumerate(coord_is_set))):
            # Meeko will remove "salt" from SMILES when converting to pdbqt.
            # This "salt" coordinate will raise error when trying to convert pdbqt back to sdf,
            # So atoms with unsetted coordinates are removed here.
            # Removal have to be done in reverse since the atom index will change if atoms before it are removed.
            # Also, some SMILES originally have "multiple" ligands (like CHEMBL2448612), 
            # but only one of them is used during docking
            if not is_set:
                e_mol.RemoveAtom(i)
        mol = e_mol.GetMol()
        Chem.SanitizeMol(mol)
        return mol

    @staticmethod
    def add_hydrogens(mol, coordinates_list, h_parent):
        """Add hydrogen atoms to ligand RDKit mol, adjust the positions of
            polar hydrogens to match pdbqt
        """
        mol = Chem.AddHs(mol, addCoords=True)
        conformers = list(mol.GetConformers())
        num_hydrogens = int(len(h_parent) / 2)
        for conformer_idx, atom_coordinates in enumerate(coordinates_list):
            conf = conformers[conformer_idx]
            used_h = []
            for i in range(num_hydrogens):
                parent_rdkit_index = h_parent[2 * i] - 1
                h_pdbqt_index = h_parent[2 * i + 1] - 1
                x, y, z = [
                    float(coord) for coord in atom_coordinates[h_pdbqt_index]
                ]
                parent_atom = mol.GetAtomWithIdx(parent_rdkit_index)
                candidate_hydrogens = [
                    atom.GetIdx() for atom in parent_atom.GetNeighbors()
                    if atom.GetAtomicNum() == 1
                ]
                h_rdkit_index = None
                for h_rdkit_index in candidate_hydrogens:
                    if h_rdkit_index not in used_h:
                        break
                used_h.append(h_rdkit_index)
                if h_rdkit_index is not None:
                    conf.SetAtomPosition(h_rdkit_index, Point3D(x, y, z))
        return mol

    @staticmethod
    def combine_rdkit_mols(mol_list):
        """Combines list of rdkit molecules into a single one
            None's are ignored
            returns None if input is empty list or all molecules are None
        """
        combined_mol = None
        for mol in mol_list:
            if mol is None:
                continue
            if combined_mol is None: # first iteration
                combined_mol = mol
            else:
                combined_mol = Chem.CombineMols(combined_mol, mol)
        return combined_mol

    @classmethod
    def _verify_flexres(cls):
        for resname in cls.flexres:
            atom_names_in_smiles_order = cls.flexres[resname]["atom_names_in_smiles_order"]
            h_to_parent_index = cls.flexres[resname]["h_to_parent_index"]
            expected_names = atom_names_in_smiles_order + list(h_to_parent_index.keys())
            if len(expected_names) != len(set(expected_names)):
                raise RuntimeError("repeated atom names in cls.flexres[%s]" % resname)

    @staticmethod
    def write_sd_string(pdbqt_mol, only_cluster_leads=False):
        sio = StringIO()
        f = Chem.SDWriter(sio)
        mol_list = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol, only_cluster_leads)
        failures = [i for i, mol in enumerate(mol_list) if mol is None]
        combined_mol = RDKitMolCreate.combine_rdkit_mols(mol_list)
        if combined_mol is None:
            return "", failures
        nr_conformers = combined_mol.GetNumConformers()
        property_names = {
            "free_energy": "free_energies",
            "intermolecular_energy": "intermolecular_energies",
            "internal_energy": "internal_energies",
            "cluster_size": "cluster_size",
            "cluster_id": "cluster_id",
            "rank_in_cluster": "rank_in_cluster",
        }
        props = {}
        if only_cluster_leads:
            nr_poses = len(pdbqt_mol._pose_data["cluster_leads_sorted"])
            pose_idxs = pdbqt_mol._pose_data["cluster_leads_sorted"]
        else:
            nr_poses = pdbqt_mol._pose_data["n_poses"]
            pose_idxs = list(range(nr_poses))
        for prop_sdf, prop_pdbqt in property_names.items():
            if nr_conformers == nr_poses:
                props[prop_sdf] = prop_pdbqt
        has_all_data = True
        for _, key in props.items():
            has_all_data &= len(pdbqt_mol._pose_data[key]) == nr_conformers
        for conformer in combined_mol.GetConformers():
            i = conformer.GetId()
            j = pose_idxs[i]
            if has_all_data:
                data = {k: pdbqt_mol._pose_data[v][j] for k, v in props.items()}
                if len(data): combined_mol.SetProp("meeko", json.dumps(data))
            f.write(combined_mol, i)
        f.close()
        output_string = sio.getvalue()
        return output_string, failures

def _read_ligand_pdbqt_file(pdbqt_string, poses_to_read=-1, energy_range=-1, is_dlg=False, skip_typing=False):
    i = 0
    n_poses = 0
    previous_serial = 0
    tmp_positions = []
    tmp_atoms = []
    tmp_actives = []
    tmp_pdbqt_string = ''
    water_indices = {*()}  
    location = 'ligand'
    energy_best_pose = None
    is_first_pose = True
    is_model = False
    mol_index = -1 # incremented for each ROOT keyword
    atoms_dtype = [('idx', 'i4'), ('serial', 'i4'), ('name', 'U4'), ('resid', 'i4'),
                   ('resname', 'U3'), ('chain', 'U1'), ('xyz', 'f4', (3)),
                   ('partial_charges', 'f4'), ('atom_type', 'U3')]

    atoms = None
    positions = []

    # flexible_residue is for atoms between BEGIN_RES and END_RES keywords, ligand otherwise.
    # flexres assigned "ligand" if BEGIN/END RES keywords are missing
    # mol_index distinguishes different ligands and flexres because ROOT keyword increments mol_index  
    atom_annotations = {'ligand': [], 'flexible_residue': [], 'water': [],
                        'hb_acc': [], 'hb_don': [],
                        'all': [], 'vdw': [],
                        'glue': [], 'reactive': [], 'metal': [],
                        'mol_index': {},
                        }
    pose_data = {
        'n_poses': None,
        'active_atoms': [],
        'free_energies': [], 
        'intermolecular_energies': [],
        'internal_energies': [],
        'index_map': {},
        'pdbqt_string': [],
        'smiles': {},
        'smiles_index_map': {},
        'smiles_h_parent': {},
        'cluster_id': [],
        'rank_in_cluster': [],
        'cluster_leads_sorted': [],
        'cluster_size': [],
    }

    tmp_cluster_data = {}

    buffer_index_map = {}
    buffer_smiles = None
    buffer_smiles_index_map = []
    buffer_smiles_h_parent = []

    lines = pdbqt_string.split('\n')
    if len(lines[-1]) == 0: lines = lines[:-1]
    lines = [line + '\n' for line in lines]
    for line in lines:
        if is_dlg:
            if line.startswith('DOCKED'):
                line = line[8:]
            # parse clustering
            elif line.endswith('RANKING\n'):
                fields = line.split()
                cluster_id = int(fields[0])
                subrank = int(fields[1])
                run_id = int(fields[2])
                tmp_cluster_data[run_id] = (cluster_id, subrank)
            else:
                continue

        if not line.startswith(('MODEL', 'ENDMDL')):
            """This is very lazy I know...
            But would you rather spend time on rebuilding the whole torsion tree and stuff
            for writing PDBQT files or drinking margarita? Energy was already spend to build
            that, so let's re-use it!"""
            tmp_pdbqt_string += line

        if line.startswith('MODEL'):
            # Reinitialize variables
            i = 0
            previous_serial = 0
            tmp_positions = []
            tmp_atoms = []
            tmp_actives = []
            tmp_pdbqt_string = ''
            is_model = True
            mol_index = -1 # incremented for each ROOT keyword
        elif line.startswith('ATOM') or line.startswith("HETATM"):
            serial = int(line[6:11].strip())
            name = line[12:16].strip()
            resname = line[17:20].strip()
            chainid = line[21].strip()
            resid = int(line[22:26].strip())
            xyz = np.array([line[30:38].strip(), line[38:46].strip(), line[46:54].strip()], dtype=float)
            try:
                # PDBQT files from dry.py script are stripped from their partial charges. sigh...
                partial_charges = float(line[71:77].strip())
            except:
                partial_charges = 0.0
            atom_type = line[77:-1].strip()

            """ We are looking for gap in the serial atom numbers. Usually if they
            are not following it means that atoms are missing. This will happen with
            water molecules after using dry.py, only non-overlapping water molecules
            are kept. Also if the current serial becomes suddenly inferior than the
            previous and equal to 1, it means that we are now in another molecule/flexible 
            residue. So here we are adding dummy atoms
            """
            if (previous_serial + 1 != serial) and not (serial < previous_serial and serial == 1):
                diff = serial - previous_serial - 1
                for _ in range(diff):
                    xyz_nan = [999.999, 999.999, 999.999]
                    tmp_atoms.append((i, 9999, 'XXXX', 9999, 'XXX', 'X', xyz_nan, 999.999, 'XX'))
                    tmp_positions.append(xyz_nan)
                    i += 1

            # Once it is done, we can return to a normal life... and add existing atoms
            tmp_atoms.append((i, serial, name, resid, resname, chainid, xyz, partial_charges, atom_type))
            tmp_positions.append(xyz)
            tmp_actives.append(i)

            if is_first_pose:
                atom_annotations["mol_index"].setdefault(mol_index, [])
                atom_annotations["mol_index"][mol_index].append(i)
                # We store water idx separately from the rest since their number can be variable
                if atom_type != 'W':
                    atom_annotations[location].append(i)
                    atom_annotations['all'].append(i)
                    if not skip_typing:
                        atom_annotations[atom_property_definitions[atom_type]].append(i)

            if atom_type == 'W':
                water_indices.update([i])

            previous_serial = serial
            i += 1
        elif line.startswith("ROOT") and is_first_pose:
            mol_index += 1
            # buffers needed because REMARKS preceeds ROOT
            pose_data["index_map"][mol_index] = buffer_index_map
            pose_data["smiles"][mol_index] = buffer_smiles
            pose_data["smiles_index_map"][mol_index] = buffer_smiles_index_map
            pose_data["smiles_h_parent"][mol_index] = buffer_smiles_h_parent
            buffer_index_map = {}
            buffer_smiles = None
            buffer_smiles_index_map = []
            buffer_smiles_h_parent = []
        elif line.startswith('REMARK INDEX MAP') and is_first_pose:
            integers = [int(integer) for integer in line.split()[3:]]
            if len(integers) % 2 == 1:
                raise RuntimeError("Number of indices in INDEX MAP is odd")
            for j in range(int(len(integers) / 2)): 
                buffer_index_map[integers[j*2]] = integers[j*2 + 1]
        elif line.startswith('REMARK SMILES IDX') and is_first_pose:
            integers = [int(integer) for integer in line.split()[3:]]
            if len(integers) % 2 == 1:
                raise RuntimeError("Number of indices in SMILES IDX is odd")
            buffer_smiles_index_map.extend(integers)
        elif line.startswith('REMARK H PARENT') and is_first_pose:
            integers = [int(integer) for integer in line.split()[3:]]
            if len(integers) % 2 == 1:
                raise RuntimeError("Number of indices in H PARENT is odd")
            buffer_smiles_h_parent.extend(integers)
        elif line.startswith('REMARK SMILES') and is_first_pose: # must check after SMILES IDX
            buffer_smiles = line.split()[2]
        elif line.startswith('REMARK VINA RESULT') or line.startswith('USER    Estimated Free Energy of Binding    ='):
            # Read free energy from output PDBQT files
            try:
                # Vina
                energy = float(line.split()[3])
            except:
                # AD4
                energy = float(line[45:].split()[0]) # no guarantee of space between = and number

            if energy_best_pose is None:
                energy_best_pose = energy
            energy_current_pose = energy

            # NOTE this assumes poses are sorted by increasing energy
            diff_energy = energy_current_pose - energy_best_pose
            if (energy_range <= diff_energy and energy_range != -1):
                break

            pose_data['free_energies'].append(energy)
        elif not is_dlg and line.startswith('REMARK INTER:'):
            pose_data['intermolecular_energies'].append(float(line.split()[2]))
        elif not is_dlg and line.startswith('REMARK INTRA:'):
            pose_data['internal_energies'].append(float(line.split()[2]))
        elif is_dlg and line.startswith('USER    (1) Final Intermolecular Energy     ='):
            pose_data['intermolecular_energies'].append(float(line[45:].split()[0]))
        elif is_dlg and line.startswith('USER    (2) Final Total Internal Energy     ='):
            pose_data['internal_energies'].append(float(line[45:].split()[0]))
        elif line.startswith('BEGIN_RES'):
            location = 'flexible_residue'
        elif line.startswith('END_RES'):
            # We never know if there is a molecule just after the flexible residue...
            location = 'ligand'
        elif line.startswith('ENDMDL'):
            n_poses += 1
            # After reading the first pose no need to store atom properties
            # anymore, it is the same for every pose
            is_first_pose = False

            tmp_atoms = np.array(tmp_atoms, dtype=atoms_dtype)

            if atoms is None:
                """We store the atoms (topology) only once, since it is supposed to be
                the same for all the molecules in the PDBQT file (except when water molecules
                are involved... classic). But we will continue to compare the topology of
                the current pose with the first one seen in the PDBQT file, to be sure only
                the atom positions are changing."""
                atoms = tmp_atoms.copy()
            else:
                # Check if the molecule topology is the same for each pose
                # We ignore water molecules (W) and atom type XX
                columns = ['idx', 'serial', 'name', 'resid', 'resname', 'chain', 'partial_charges', 'atom_type']
                topology1 = atoms[np.isin(atoms['atom_type'], ['W', 'XX'], invert=True)][columns]
                topology2 = tmp_atoms[np.isin(atoms['atom_type'], ['W', 'XX'], invert=True)][columns]

                if not np.array_equal(topology1, topology2):
                    error_msg = 'molecules have different topologies'
                    raise RuntimeError(error_msg)

                # Update information about water molecules (W) as soon as we find new ones
                tmp_water_molecules_idx = tmp_atoms[tmp_atoms['atom_type'] == 'W']['idx']
                water_molecules_idx = atoms[atoms['atom_type'] == 'XX']['idx']
                new_water_molecules_idx = list(set(tmp_water_molecules_idx).intersection(water_molecules_idx))
                atoms[new_water_molecules_idx] = tmp_atoms[new_water_molecules_idx]

            positions.append(tmp_positions)
            pose_data['active_atoms'].append(tmp_actives)
            pose_data['pdbqt_string'].append(tmp_pdbqt_string)

            if (n_poses >= poses_to_read and poses_to_read != -1):
                break

    """ if there is no model, it means that there is only one molecule
    so when we reach the end of the file, we store the atoms, 
    positions and actives stuff. """
    if not is_model:
        n_poses += 1
        atoms = np.array(tmp_atoms, dtype=atoms_dtype)
        positions.append(tmp_positions)
        pose_data['active_atoms'].append(tmp_actives)
        pose_data['pdbqt_string'].append(tmp_pdbqt_string)

    positions = np.array(positions).reshape((n_poses, atoms.shape[0], 3))

    pose_data['n_poses'] = n_poses

    # We add indices of all the water molecules we saw
    if water_indices:
        atom_annotations['water'] = list(water_indices)

    # clustering        
    if len(tmp_cluster_data) > 0:
        if len(tmp_cluster_data) != n_poses:
            raise RuntimeError("Nr of poses in cluster data (%d) differs from nr of poses (%d)" % (len(tmp_cluster_data, n_poses)))
        pose_data["cluster_id"] = [None] * n_poses
        pose_data["rank_in_cluster"] = [None] * n_poses
        pose_data["cluster_size"] = [None] * n_poses
        cluster_ids = [cluster_id for _, (cluster_id, _) in tmp_cluster_data.items()]
        n_clusters = max(cluster_ids)
        pose_data["cluster_leads_sorted"] = [None] * n_clusters
        for pose_index, (cluster_id, rank_in_cluster) in tmp_cluster_data.items():
            pose_data["cluster_id"][pose_index - 1] = cluster_id
            pose_data["rank_in_cluster"][pose_index - 1] = rank_in_cluster
            pose_data["cluster_size"][pose_index - 1] = cluster_ids.count(cluster_id)
            if rank_in_cluster == 1: # is cluster lead
                pose_data["cluster_leads_sorted"][cluster_id - 1] = pose_index - 1
    return atoms, positions, atom_annotations, pose_data

class PDBQTMolecule:
    def __init__(self, pdbqt_string, name=None, poses_to_read=None, energy_range=None, is_dlg=False, skip_typing=False):
        """PDBQTMolecule class for reading PDBQT (or dlg) files from AutoDock4, AutoDock-GPU or AutoDock-Vina

        Contains both __getitem__ and __iter__ methods, someone might lose his mind because of this.

        Args:
            pdbqt_string (str): pdbqt string
            name (str): name of the molecule (default: None, use filename without pdbqt suffix)
            poses_to_read (int): total number of poses to read (default: None, read all)
            energy_range (float): read docked poses until the maximum energy difference 
                from best pose is reach, for example 2.5 kcal/mol (default: Non, read all)
            is_dlg (bool): input file is in dlg (AutoDock docking log) format (default: False)
            skip_typing (bool, optional): Flag indicating that atomtyping should be skipped
        """
        self._current_pose = 0
        self._pdbqt_filename = None
        self._atoms = None
        self._positions = None
        self._atom_annotations = None
        self._pose_data = None
        self._name = name

        # Juice all the information from that PDBQT file
        poses_to_read = poses_to_read if poses_to_read is not None else -1
        energy_range = energy_range if energy_range is not None else -1
        results = _read_ligand_pdbqt_file(pdbqt_string, poses_to_read, energy_range, is_dlg, skip_typing)
        self._atoms, self._positions, self._atom_annotations, self._pose_data = results

        if self._atoms.shape[0] == 0:
            raise RuntimeError('read 0 atoms. Consider PDBQTMolecule.from_file(fname)')

    @classmethod
    def from_file(cls, pdbqt_filename, name=None, poses_to_read=None, energy_range=None, is_dlg=False, skip_typing=False): 
        if name is None:
            name = os.path.splitext(os.path.basename(pdbqt_filename))[0]
        with open(pdbqt_filename) as f:
            pdbqt_string = f.read()
        instance = cls(pdbqt_string, name, poses_to_read, energy_range, is_dlg, skip_typing)
        instance._pdbqt_filename = pdbqt_filename
        return instance

    def __getitem__(self, value):
        if isinstance(value, int):
            if value < 0 or value >= self._positions.shape[0]:
                raise IndexError('The index (%d) is out of range.' % value)
        elif isinstance(value, slice):
            raise TypeError('Slicing is not implemented for PDBQTMolecule object.')
        else:
            raise TypeError('Invalid argument type.')

        self._current_pose = value
        return self

    def __iter__(self):
        self._current_pose = -1
        return self

    def __next__(self):
        if self._current_pose + 1 >= self._positions.shape[0]:
            raise StopIteration

        self._current_pose += 1

        return self

    def __repr__(self):
        repr_str = '<Molecule named %s containing %d poses of %d atoms>'
        return (repr_str % (self._name, self._pose_data['n_poses'], self._atoms.shape[0]))

    @property
    def name(self):
        """Return the name of the molecule."""
        return self._name

    @property
    def pose_id(self):
        """Return the index of the current pose."""
        return self._current_pose

    @property
    def score(self):
        """Return the score (kcal/mol) of the current pose."""
        return self._pose_data['free_energies'][self._current_pose]

    def available_atom_properties(self, ignore_properties=None):
        """Return all the available atom properties for that molecule.

        The following properties are ignored: ligand and flexible_residue

        """
        if ignore_properties is None:
            ignore_properties = []

        if not isinstance(ignore_properties, (list, tuple)):
            ignore_properties = [ignore_properties]

        ignore_properties += ['ligand', 'flexible_residue', 'water']

        return [k for k, v in self._atom_annotations.items()
                if k not in ignore_properties and len(v) > 0]

    def has_flexible_residues(self):
        """Tell if the molecule contains a flexible residue or not.

        Returns:
            bool: True if contains flexible residues, otherwise False

        """
        if self._atom_annotations['flexible_residue']:
            return True
        else:
            return False

    def has_water_molecules(self):
        """Tell if the molecules contains water molecules or not in the current pose.

        Returns:
            bool: True if contains water molecules in the current pose, otherwise False

        """
        active_atoms_idx = self._pose_data['active_atoms'][self._current_pose]
        if set(self._atom_annotations['water']).intersection(active_atoms_idx):
            return True
        else:
            return False

    def atoms(self, atom_idx=None, only_active=True):
        """Return the atom i

        Args:
            atom_idx (int, list): index of one or multiple atoms (0-based)
            only_active (bool): return only active atoms (default: True, return only active atoms)

        Returns:
            ndarray: 2d ndarray (atom_id, atom_name, resname, resid, chainid, xyz, q, t)

        """
        if atom_idx is not None:
            if not isinstance(atom_idx, (list, tuple, np.ndarray)):
                atom_idx = np.array(atom_idx, dtype=np.int)
        else:
            atom_idx = np.arange(0, self._atoms.shape[0])

        # Get index of only the active atoms
        if only_active:
            active_atoms_idx = self._pose_data['active_atoms'][self._current_pose]
            atom_idx = sorted(list(set(atom_idx).intersection(active_atoms_idx)))

        atoms = self._atoms[atom_idx].copy()
        atoms['xyz'] = self._positions[self._current_pose, atom_idx,:]

        return atoms
    
    def positions(self, atom_idx=None, only_active=True):
        """Return coordinates (xyz) of all atoms or a certain atom

        Args:
            atom_idx (int, list): index of one or multiple atoms (0-based)
            only_active (bool): return only active atoms (default: True, return only active atoms)

        Returns:
            ndarray: 2d ndarray of coordinates (xyz)

        """
        return np.atleast_2d(self.atoms(atom_idx, only_active)['xyz'])
    
    def write_pdbqt_string(self, as_model=True):
        """Write PDBQT output string of the current pose
        
        Args:
            as_model (bool): Qdd MODEL/ENDMDL keywords to the output PDBQT string (default: True)
        
        Returns:
            string: Description
        
        """
        if as_model:
            pdbqt_string = 'MODEL    %5d\n' % (self._current_pose + 1)
            pdbqt_string += self._pose_data['pdbqt_string'][self._current_pose]
            pdbqt_string += 'ENDMDL\n'
            return pdbqt_string
        else: 
            return self._pose_data['pdbqt_string'][self._current_pose]

    def write_pdbqt_file(self, output_pdbqtfilename, overwrite=False, as_model=False):
        """Write PDBQT file of the current pose

        Args:
            output_pdbqtfilename (str): filename of the output PDBQT file
            overwrite (bool): overwrite on existing pdbqt file (default: False)
            as_model (bool): Qdd MODEL/ENDMDL keywords to the output PDBQT string (default: False)

        """
        print(overwrite and os.path.isfile(output_pdbqtfilename))
        if not overwrite and os.path.isfile(output_pdbqtfilename):
            raise RuntimeError('Output PDBQT file %s already exists' % output_pdbqtfilename)

        if as_model:
            pdbqt_string = 'MODEL    %5d\n' % (self._current_pose + 1)
            pdbqt_string += self._pose_data['pdbqt_string'][self._current_pose] 
            pdbqt_string += 'ENDMDL\n'
        else:
            pdbqt_string = self._pose_data['pdbqt_string'][self._current_pose]

        with open(output_pdbqtfilename, 'w') as w:
            w.write(pdbqt_string)
