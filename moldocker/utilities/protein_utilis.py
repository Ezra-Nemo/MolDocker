import re
import lzma
import pickle
import sqlite3

import numpy as np
import pandas as pd

from rdkit import Chem
from openbabel import pybel

from .utilis import RDKitMolCreate, PDBQTMolecule, process_rigid_flex

atom_type_map = {'HD': 'H', 'HS': 'H',
                 'NA': 'N', 'NS': 'N',
                 'A' : 'C', 'G' : 'C', 'CG0': 'C', 'CG1': 'C', 'CG2': 'C', 'CG3': 'C', 'G0': 'C', 'G1': 'C', 'G2': 'C', 'G3': 'C',
                 'OA': 'O', 'OS': 'O',
                 'SA': 'S'}

single_aa_map = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'ASH': 'D', 'ASX': 'D',
                 'CYS': 'C', 'GLU': 'E', 'GLH': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H',
                 'HID': 'H', 'HIP': 'H', 'HIE': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
                 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
                 'TYR': 'Y', 'VAL': 'V', 'CYX': 'C', 'CYM': 'C', 'LYN': 'K',}

atom_term_compiled = re.compile(r'(ATOM|TER).*')
protein_extract_compiled = re.compile(r'(^|\n)(ATOM|TER).*')
aa_chain_pos_compiled = re.compile(r'([A-Z]{3}).([a-zA-Z0-9])\s*(-?\d+)')
flex_compiled = re.compile(r'BEGIN_RES.*+\n((.|\n)*?)END_RES.*+')
flex_atom_compiled = re.compile(r"ATOM.*")
smina_flex_res = re.compile(r'>  <Flex Sidechains PDB>.*+\n((.|\n)*?)\$\$\$\$')

class PDBEditor:
    def __init__(self, pdbqt_str: str | None=None):
        self.pdbqt_str = pdbqt_str
        self.pdbqt_chain_dict = {}
        
    def parse_pdbqt_text_to_dict(self, display_flex_dict = None):
        self.pdbqt_chain_dict = {}
        chain_aa_dict = {}
        for atom_term_line in re.finditer(atom_term_compiled, self.pdbqt_str):
            line = atom_term_line.group(0)
            aa, chain, aa_pos = re.search(aa_chain_pos_compiled, line).group(1,2,3)
            aa_pos = int(aa_pos)
            if chain not in self.pdbqt_chain_dict:
                self.pdbqt_chain_dict[chain] = {}
                chain_aa_dict[chain] = []
            aa_pos_dict = self.pdbqt_chain_dict[chain]
            if aa_pos not in aa_pos_dict:
                aa_pos_dict[aa_pos] = []
                chain_aa_dict[chain].append(aa)
            aa_pos_dict[aa_pos].append(line)
        for chain, aa_pos_dict in self.pdbqt_chain_dict.items():
            aa_cnt = len(aa_pos_dict)
            for pos, text_list in aa_pos_dict.items():
                aa_pos_dict[pos] = '\n'.join(text_list)
            self.pdbqt_chain_dict[chain] = pd.DataFrame.from_dict(aa_pos_dict, 'index')
            df = self.pdbqt_chain_dict[chain]
            if display_flex_dict is None:
                self.pdbqt_chain_dict[chain]['Display'] = pd.Series([True] * aa_cnt, list(df.index))
                self.pdbqt_chain_dict[chain]['Flexible'] = pd.Series([False] * aa_cnt, list(df.index))
            else:
                self.pdbqt_chain_dict[chain]['Display'] = display_flex_dict[chain]['Display']
                self.pdbqt_chain_dict[chain]['Flexible'] = display_flex_dict[chain]['Flexible']
            self.pdbqt_chain_dict[chain]['AA_Name'] = chain_aa_dict[chain]
        
    def parse_logic(self, series: pd.Series, logic: str):
        def replace_expression(match):
            expr = match.group()
            negation = ''
            if expr.startswith('~'):
                negation = '~'
                expr = expr[1:]
            if '-' in expr:  # range
                start, end = map(int, expr.rsplit('-', 1))
                return f"{negation}((series.index >= {start}) & (series.index <= {end}))"
            else:  # single value
                return f"{negation}(series.index == {expr})"
        
        logic = logic.replace(' ', '').replace(',', '|')    # "," is the same as or "|"
        logic_eval = re.sub(r'~?-?\d+-\d+|~?\d+(?:,~?\d+)*', replace_expression, logic)
        
        try:
            result = pd.eval(logic_eval, local_dict={'series': series}, engine='python') # need to pass local_dict or else Nuitka compiled code won't work
            series[result] = True
        except:
            return None
        
        return series
    
    def update_display(self, chain: str, display_str: str | None):
        full_df = self.pdbqt_chain_dict[chain]
        if display_str is None:
            display_series = pd.Series([False] * len(full_df), list(full_df.index))
            full_df['Display'] = display_series
            return
        elif not display_str:
            display_series = pd.Series([True] * len(full_df), list(full_df.index))
            full_df['Display'] = display_series
            return
        display_series = pd.Series([False] * len(full_df), list(full_df.index)) # default to False
        result = self.parse_logic(display_series, display_str)
        if result is None:
            return f'Invalid syntax for chain {chain}.'
        full_df['Display'] = display_series
    
    def _condense_to_range(self, list_of_nums: list[int]):
        final_text = ''
        start = list_of_nums[0]
        end = list_of_nums[0]
        range_cnt = 1
        for num in list_of_nums[1:]:
            if num == start + range_cnt:
                end = num
                range_cnt += 1
            else:
                if start == end:
                    final_text += f'{start},'
                else:
                    final_text += f'{start}-{end},'
                start = num
                end = num
                range_cnt = 1
        if start == end:
            final_text += f'{start}'
        else:
            final_text += f'{start}-{end}'
        return final_text
    
    def convert_to_range_text(self, chain: str):
        s = self.pdbqt_chain_dict[chain]['Display']
        mask = s == True
        displayed_pos = s[mask].index.to_list()
        if not displayed_pos:
            return None # return None if not displayed
        if len(displayed_pos) == len(s):
            return ''   # empty text if everything is displayed
        return self._condense_to_range(displayed_pos)
    
    def convert_to_flex_set(self, get_pdbqt_str: bool=False):
        pdbqt_str = ''
        flex_res = set()
        for chain, df in self.pdbqt_chain_dict.items():
            display_mask = df['Display'] == True
            if get_pdbqt_str:
                pdbqt_str += '\n'.join(df[0][display_mask].to_list())
                pdbqt_str += '\n'
            flex_mask: pd.Series = (df['Flexible'] == True) & display_mask
            if not flex_mask.empty:
                flex_aa_names: pd.Series = df['AA_Name'][flex_mask]
                flex_res.update(set(zip([chain] * len(flex_aa_names), flex_aa_names, flex_aa_names.index)))
        if get_pdbqt_str:
            return flex_res, pdbqt_str
        return flex_res
    
    def convert_full_dict_to_text(self):
        protein_strs = []
        for df in self.pdbqt_chain_dict.values():
            string = '\n'.join(df[0].to_list())
            protein_strs.append(string)
        return '\n'.join(protein_strs)
    
    def convert_dict_to_pdbqt_text(self, return_scheme=False):
        pdbqt_str = ''
        cnt = 0
        for df in self.pdbqt_chain_dict.values():
            mask = df['Display'] == True
            string = '\n'.join(df[0][mask].to_list())
            if string:
                cnt += 1
            pdbqt_str += string
            if string:
                pdbqt_str += '\n'
        if return_scheme:
            if cnt > 1:
                scheme = 'chainindex'
            else:
                scheme = 'residueindex'
            return pdbqt_str, scheme
        return pdbqt_str
    
    def process_and_save_rigid_flex_pdbqt(self, out_rigid_pth: str, out_flex_pth: str):
        flex_res, pdbqt_str = self.convert_to_flex_set(True)
        process_rigid_flex(pdbqt_str, out_rigid_pth, out_flex_pth, flex_res)
        return bool(flex_res)
    
    def check_format_type(self):
        for chain_dict in self.pdbqt_chain_dict.values():
            line = chain_dict.iloc[0, 0]
            if line[70:76].strip():
                return 'pdbqt'
            else:
                return 'pdb'
    
    def _process_string_fn(self, pdbqt_str: str):
        for l in pdbqt_str.split('\n'):
            if l[12:15].strip() == 'CB':
                x, y, z = float(l[30:37].strip()), float(l[38:45].strip()), float(l[46:53].strip())
                break
        return [x, y, z]
    
    def retrieve_CB_coord(self):
        chain_ca_dict = {}
        for chain, df in self.pdbqt_chain_dict.items():
            displayed_text_series: pd.Series = df[(df['Display'] == True) & 
                                                  (df['Flexible'] == False) &   # don't need to check if it is already flexible
                                                  (~df['AA_Name'].isin(['GLY', 'ALA', 'PRO']))][0]  # Non-flexible
            if displayed_text_series.empty:
                ca_df = None
            else:
                ca_df = displayed_text_series.apply(lambda x: pd.Series(self._process_string_fn(x), index=['x', 'y', 'z']))
                ca_df.set_index(displayed_text_series.index)
            chain_ca_dict[chain] = ca_df
        return chain_ca_dict
    
    def search_amino_acids(self, center_coord: tuple | list, box_width: tuple | list):
        x, y, z = center_coord
        x_wid, y_wid, z_wid = map(lambda x: x / 2, box_width)
        x_max, x_min = x + x_wid, x - x_wid
        y_max, y_min = y + y_wid, y - y_wid
        z_max, z_min = z + z_wid, z - z_wid
        chain_ca_df_dict = self.retrieve_CB_coord()
        result = []
        for chain, ca_df in chain_ca_df_dict.items():
            if ca_df is not None:
                mask = ((x_min <= ca_df['x']) & (ca_df['x'] <= x_max)) & \
                        ((y_min <= ca_df['y']) & (ca_df['y'] <= y_max)) & \
                        ((z_min <= ca_df['z']) & (ca_df['z'] <= z_max))
                within_box = ca_df[mask].index.to_list()
                if within_box:
                    result.extend([(chain, i) for i in within_box])
        return result
    
    def calculate_protein_bounding_box(self):
        final = []
        for df in self.pdbqt_chain_dict.values():
            mask = df['Display'] == True
            string = '\n'.join(df[0][mask].to_list())
            xyz = [[l[30:38].strip(), l[38:46].strip(), l[46:54].strip()] for l in string.split('\n') if l]
            final.extend(xyz)
        xyz_coord = np.array(final, float)
        xyz_min = xyz_coord.min(axis=0)
        xyz_max = xyz_coord.max(axis=0)
        xyz_cen = np.round((xyz_max + xyz_min) / 2, 3)
        xyz_wid = np.round(xyz_max - xyz_min+1    , 3)  # add 1 Å padding
        return xyz_cen, xyz_wid
    
    def retrieve_sequence_abbreviation(self, chain: str):
        aa_names: pd.Series= self.pdbqt_chain_dict[chain]['AA_Name']
        aa_idx_name_map = aa_names.to_dict()
        aa_names_index = list(aa_idx_name_map)
        displayed = self.pdbqt_chain_dict[chain]['Display'].to_dict()
        final_idx_seq_map = {}
        starting_pos = aa_names_index[0]
        while starting_pos % 10 != 1:
            starting_pos -= 1
        for aa_idx in range(starting_pos, aa_names_index[-1]+1):
            if aa_idx not in aa_names_index:
                final_idx_seq_map[aa_idx] = '-'
            else:
                final_idx_seq_map[aa_idx] = single_aa_map[aa_idx_name_map[aa_idx]]
        return final_idx_seq_map, displayed
    
    def _total_count(self):
        chain_cnt_dict = {}
        for chain, df in self.pdbqt_chain_dict.items():
            chain_cnt_dict[chain] = len(df)
        return chain_cnt_dict
    
    def _display_count(self):
        display_cnt_dict = {}
        for chain, df in self.pdbqt_chain_dict.items():
            display_cnt_dict[chain] = int((df['Display'] == True).sum())
        return display_cnt_dict
    
    def _flex_count(self):
        flex_res_dict = {}
        for chain, df in self.pdbqt_chain_dict.items():
            flex_res_dict[chain] = None
            flex_mask = (df['Display'] == True) & (df['Flexible'] == True)
            if not df.empty:
                flex_aa_names: pd.Series = df['AA_Name'][flex_mask]
                flex_res_dict[chain] = list(zip(flex_aa_names, flex_aa_names.index))
        return flex_res_dict
    
    def _spaces(self, num: int):
        return '&nbsp;' * num
    
    def _subgraph(self, num: int, start='└'):
        return start + '─' * (num - 1)
    
    def __str__(self):
        chain_cnt_dict = self._total_count()
        display_cnt_dict = self._display_count()
        flex_res_dict = self._flex_count()
        
        chain_str = f'Format: {self.check_format_type()}<br>'
        
        for chain in chain_cnt_dict:
            chain_str += self._spaces(2) + f'Chain {chain} : {chain_cnt_dict[chain]}<br>'
            display_str = f'{display_cnt_dict[chain]}'
            chain_str += self._spaces(2) + self._subgraph(2) + f'• Display : {display_str}<br>'
            flex_str = f'{len(flex_res_dict[chain])}'
            chain_str += self._spaces(4) + self._subgraph(2) + f'• Flexible : {flex_str}<br>'
            if flex_res_dict[chain] is not None:
                for i, tup in enumerate(flex_res_dict[chain], 1):
                    aa, idx = tup[0], tup[1]
                    start = '└' if i == len(flex_res_dict[chain]) else '├'
                    chain_str += self._spaces(6) + self._subgraph(2, start) + f'■ {aa}:{idx}<br>'
                        
            chain_str += '<br>'
        
        return chain_str

class PDBQTCombiner:
    def __init__(self, mdl_name):
        self.interaction_dict = None
        self.cache_pth = None
        self.name = mdl_name
        self.complex = None
        self.protein = None
        self.ligand = None
    
    def process_strings(self, protein_data: list | None=None, ligand_str: str=None, flex_str: str=None):
        # ligand_str can be pdbqt or sdf converted pdb. If pdb then no need to convert to pdb and check flex
        if protein_data is None:
            return None, ligand_str
        else:
            self.interaction_dict = {}
            alphabet_list = [chr(i) for i in range(65, 65 + 26)]
            self.alphabet_order_list = alphabet_list + [alphabet.lower() for alphabet in alphabet_list]
            protein_delta_data, ligand_pdb_str = self.combine_protein_ligand(protein_data, ligand_str, flex_str)
            return protein_delta_data, ligand_pdb_str
        
    def _map_pdbqt_line(self, line: str):
        # as a format record, not used anymore
        return {'record_name': line[:6].strip()          ,
                'atom_idx'   : int(line[6:11].strip())   ,
                'atom_name'  : line[12:16].strip()       ,
                'alt_id'     : line[16]                  ,
                'res_name'   : line[17:20].strip()       ,
                'chain'      : line[21]                  ,
                'res_pos'    : int(line[22:26].strip())  ,
                'insert'     : line[26]                  ,
                'x'          : float(line[30:38].strip()),
                'y'          : float(line[38:46].strip()),
                'z'          : float(line[46:54].strip()),
                'occupency'  : float(line[54:60].strip()),
                'b_factor'   : float(line[60:66].strip()),
                'charge'     : float(line[70:76].strip()),
                'atom_type'  : line[77:  ].strip()       ,
                }
    
    def map_pdbqt_line_to_pdb(self, line: str, idx: int=None):
        atom_type = line[76:].strip()
        if atom_type in atom_type_map:
            atom_type = atom_type_map[atom_type]
        chain = line[21]
        atom_idx = int(line[6:11].strip()) if idx is None else idx
        res_name = line[17:20].strip()
        res_pos = int(line[22:26].strip())
        line_dict = {'atom_idx'   : atom_idx           ,
                     'atom_name'  : line[12:16].strip(),
                     'alt_id'     : line[16]           ,
                     'res_name'   : res_name           ,
                     'chain'      : chain              ,
                     'res_pos'    : res_pos            ,
                     'others'     : line[26:66]        ,  # skip partial charge
                     'atom_type'  : atom_type          ,  # don't strip to keep spaces
                     }
        final = [line_dict]
        if idx is not None:
            ter = {'atom_idx': atom_idx + 1,
                   'res_name': res_name    ,
                   'chain'   : chain       ,
                   'res_pos' : res_pos     ,
                   }
            final.append(ter)
        return final
        
    def map_pdb_conect_line(self, conect_line: str):
        return {'number_1': conect_line[6 :11].strip(),
                'number_2': conect_line[11:16].strip(),
                'number_3': conect_line[16:21].strip(),
                'number_4': conect_line[21:26].strip(),
                'number_5': conect_line[26:31].strip(),
                }
    
    def read_pdbqt_file(self, protein_pth: str):
        with open(protein_pth, 'r') as f:
            pdbqt_str = f.read()
        return pdbqt_str.strip()
    
    def parse_pdbqt(self, pdbqt_str):
        return [item for idx, line in enumerate(pdbqt_str.splitlines()) 
                for item in self.map_pdbqt_line_to_pdb(line, idx)]
    
    def combine_protein_ligand(self, protein_data: list[dict], ligand_str: str, flex_str: str):
        unique_chains = {entry['chain'] for entry in protein_data}
        ligand_chain = next((alphabet for alphabet in self.alphabet_order_list if alphabet not in unique_chains), 'Z')
        if ligand_str.startswith('REMARK'):
            pdbqt_mol = PDBQTMolecule(ligand_str, poses_to_read=1, skip_typing=True)
            mol = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)[0] # fragment cannot have hydrogen
            Chem.RemoveHs(mol)
            ligand_pdb_str = Chem.MolToPDBBlock(mol)
        elif ligand_str.startswith('HETATM'):
            ligand_pdb_str = ligand_str
            # flex_res does not exist in SDF-converted PDB format created by Uni-Dock
        elif ligand_str.startswith('COMPND'):
            ligand_pdb_str = '\n'.join(ligand_str.split('\n')[1:])
            # RDKit will generate COMPND line for PDB format if the original input format has a name
        modified_lines = {}
        
        if flex_str is not None:    # smina/gnina contains flex residues within there PQBQT string.
            for atom_line in re.findall(flex_atom_compiled, flex_str):
                mapped_line = self.map_pdbqt_line_to_pdb(atom_line)[0]
                for idx, protein_line in enumerate(protein_data):
                    if 'atom_name' in protein_line:
                        if (protein_line['atom_name'] == mapped_line['atom_name'] and
                            protein_line['res_name'] == mapped_line['res_name'] and
                            protein_line['chain'] == mapped_line['chain'] and
                            protein_line['res_pos'] == mapped_line['res_pos']):
                            mapped_line['atom_idx'] = protein_line['atom_idx']
                            modified_lines.update({idx: mapped_line})
                            break
        elif 'BEGIN_RES' in ligand_str:   # only run this when there are flex res within pdbqt ligand
            for flex_groups in re.finditer(flex_compiled, ligand_str):
                flex_lines = flex_groups.group(0)
                for atom_line in re.findall(flex_atom_compiled, flex_lines):
                    mapped_line = self.map_pdbqt_line_to_pdb(atom_line)[0]
                    for idx, protein_line in enumerate(protein_data):
                        if 'atom_name' in protein_line:
                            if (protein_line['atom_name'] == mapped_line['atom_name'] and
                                protein_line['res_name'] == mapped_line['res_name'] and
                                protein_line['chain'] == mapped_line['chain'] and
                                protein_line['res_pos'] == mapped_line['res_pos']):
                                mapped_line['atom_idx'] = protein_line['atom_idx']
                                modified_lines.update({idx: mapped_line})
                                break
                            
        final_ligand_strs = []
        for line in ligand_pdb_str.splitlines():
            if line.startswith('HETATM'):
                splitted = list(line)
                splitted[21] = ligand_chain
                final_ligand_strs.append(''.join(splitted))
            else:
                final_ligand_strs.append(line)
        return modified_lines, '\n'.join(final_ligand_strs)
    
    def convert_to_pdb_str(self, atom_data: list, conect_data: list | None=None):
        lines = []
        if conect_data is None: # Protein
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
        else:   # Ligand
            format_str = "HETATM{:5d} {:4s}{:1s}{:3s} {:1s}{:4d}{:40s}          {:>2s}\n"
            conect_str = "CONECT {:5s}{:5s}{:5s}{:5s}{:5s}\n"
            for entry in atom_data:
                lines.append(format_str.format(
                    entry['atom_idx'], entry['atom_name'], entry['alt_id'], 
                    entry['res_name'], entry['chain'], entry['res_pos'],
                    entry['others'], entry['atom_type'],
                ))
            for entry in conect_data:
                lines.append(conect_str.format(
                    entry['number_1'], entry['number_2'], entry['number_3'], 
                    entry['number_4'], entry['number_5'],
                ))
        return ''.join(lines)
    
    def append_ligand_to_protein(self,
                                 protein: str | list,
                                 protein_delta: dict,
                                 ligand: str | None=None):
        if isinstance(protein, str):
            self.protein = protein
        else:
            for line_idx, replaced_line in protein_delta.items():
                protein[line_idx] = replaced_line
            self.protein = self.convert_to_pdb_str(protein)
        if ligand is not None:
            ending_atom_idx = int(self.protein.splitlines()[-1][6:11].strip())
            ligand_data = []
            conect_data = []
            for idx, line in enumerate(ligand.splitlines(), ending_atom_idx + 1):
                if line.startswith('HETATM'):
                    mapped_line = self.map_pdbqt_line_to_pdb(line, idx)[0]
                    ligand_data.append(mapped_line)
                elif line.startswith('CONECT'):
                    conect_line = self.map_pdb_conect_line(line)
                    for k, v in conect_line.items():
                        if v:
                            conect_line[k] = str(int(v) + ending_atom_idx)
                    conect_data.append(conect_line)
            return self.protein + self.convert_to_pdb_str(ligand_data, conect_data)
    
    def get_combined_string(self):
        if self.ligand is None:
            conn = sqlite3.connect(self.cache_pth)
            cur = conn.cursor()
            
            cur.execute("""
                        SELECT name 
                        FROM sqlite_master 
                        WHERE type='table' 
                        AND name=?;
                        """, ('ProteinReference',))
            result = cur.fetchone()
            if not result:
                cur.execute(f"""SELECT * FROM ProLigDB WHERE name = '{self.name}'""")
                _, _, ligand = cur.fetchone()
                self.ligand = lzma.decompress(ligand).decode('utf-8')
                self.protein = None
                self.complex = None
                conn.close()
                return self.ligand
            
            # Get protein delta & ligand
            cur.execute(f"""SELECT * FROM ProLigDB WHERE name = '{self.name}'""")
            _, protein_delta, ligand = cur.fetchone()
            protein_delta = pickle.loads(lzma.decompress(protein_delta))
            self.ligand = lzma.decompress(ligand).decode('utf-8')
            
            # Get reference protein
            cur.execute(f"""SELECT * FROM ProteinReference WHERE Reference = 'ref'""")
            if protein_delta:   # PDB_Data, since protein_delta is not empty
                _, _, protein = cur.fetchone()
                protein = pickle.loads(protein)
            else:   # PDB_String, protein_delta is empty
                _, protein, _ = cur.fetchone()
            conn.close()
            self.complex = self.append_ligand_to_protein(protein, protein_delta, self.ligand)
        return self.complex
    
    def get_protein_only(self):
        if self.protein is None:
            conn = sqlite3.connect(self.cache_pth)
            cur = conn.cursor()
            
            cur.execute("""
                        SELECT name 
                        FROM sqlite_master 
                        WHERE type='table' 
                        AND name=?;
                        """, ('ProteinReference',))
            result = cur.fetchone()
            if not result:
                conn.close()
                return None
            
            # Get protein delta & ligand
            cur.execute(f"""SELECT * FROM ProLigDB WHERE name = '{self.name}'""")
            _, protein_delta, _ = cur.fetchone()
            protein_delta = pickle.loads(lzma.decompress(protein_delta))
            
            # Get reference protein
            cur.execute(f"""SELECT * FROM ProteinReference WHERE Reference = 'ref'""")
            if protein_delta:   # PDB_Data, since protein_delta is not empty
                _, _, protein = cur.fetchone()
                protein = pickle.loads(protein)
            else:   # PDB_String, protein_delta is empty
                _, protein, _ = cur.fetchone()
            conn.close()
            self.append_ligand_to_protein(protein, protein_delta, None)
        return self.protein
    
    def get_all_pdb(self):
        return self.protein, self.ligand, self.complex
        
    def get_mol_from_ligand_pdb(self):
        # Not recommended since it will lose steroechemistry information
        if self.ligand is None:
            conn = sqlite3.connect(self.cache_pth)
            cur = conn.cursor()
            cur.execute(f"""SELECT * FROM ProLigDB WHERE name = '{self.name}'""")
            _, _, ligand = cur.fetchone()
            self.ligand = lzma.decompress(ligand).decode('utf-8')
            conn.close()
        mol = Chem.MolFromPDBBlock(self.ligand)
        if mol is None:
            # Fall back to obabel if it failed
            obabel_mol = pybel.readstring('pdb', self.ligand)
            mol_string = obabel_mol.write('mol')
            mol = Chem.MolFromMolBlock(mol_string)
        mol = Chem.AddHs(mol, addCoords=True)
        mol_block = Chem.MolToMolBlock(mol)
        return mol_block

class MDMFileProcessor:
    def __init__(self, mdl_name):
        self.interaction_dict = None
        self.cache_pth = None
        self.name = mdl_name
        self.complex = None
        self.ligand = None
        self.protein = None
        
    def get_combined_string(self):
        if self.complex is None:
            with lzma.open(self.cache_pth) as f:
                content = pickle.load(f)
            self.complex = content['complex']
        return self.complex
    
    def get_protein_only(self):
        return None
    
    def get_all_pdb(self):
        if self.ligand is None:
            with lzma.open(self.cache_pth) as f:
                content = pickle.load(f)
            if 'rdmol' not in content:
                sdf = content['lig_sdf']
                self.ligand = Chem.MolToPDBBlock(Chem.MolFromMolBlock(sdf))
            else:
                self.ligand = Chem.MolToPDBBlock(content['rdmol'])
        if self.protein is None:
            self.protein = ''.join([g.group(0) for g in re.finditer(atom_term_compiled, self.complex)])
        return self.protein, self.ligand, self.complex
    
    def get_mol_from_ligand_pdb(self):
        sdf, _ = self.get_sdf()
        return sdf
    
    def get_sdf(self):
        with lzma.open(self.cache_pth) as f:
            content = pickle.load(f)
        if 'rdmol' not in content:
            sdf = content['lig_sdf']
        else:
            sdf = Chem.MolToMolBlock(content['rdmol'])
        energy = content['binding_energy']
        return sdf, [energy, 0, 0]
