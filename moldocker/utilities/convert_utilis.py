import os, re, io, csv, copy, time, lzma, json
import signal, psutil, shutil, sqlite3, zipfile, tempfile, requests, platform

import numpy as np
import pandas as pd
import zstandard as zstd
import multiprocessing as mp
from multiprocessing import Manager

from threading import Semaphore
from collections import defaultdict
from PySide6.QtWidgets import (QLabel, QVBoxLayout, QPushButton, QSizePolicy,
                               QDialog, QComboBox, QDialogButtonBox, QFrame,
                               QDoubleSpinBox, QGridLayout, QCheckBox, QSpinBox,
                               QListWidget, QSpacerItem, QHBoxLayout, QToolButton,
                               QListWidgetItem, QApplication, QLineEdit, QAbstractItemView,
                               QTableWidget, QTableWidgetItem, QMessageBox, QTextEdit,
                               QFileDialog, QWidget, QHeaderView, QProgressBar, QPlainTextEdit,
                               QButtonGroup, QRadioButton, QScrollArea)
from PySide6.QtCore import Qt, QObject, Signal, Slot, QSize, QRegularExpression, QThread
from PySide6.QtGui import QShortcut, QRegularExpressionValidator, QColor, QFont
from openbabel import pybel
pybel.ob.obErrorLog.SetOutputLevel(0)

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.DataStructs.cDataStructs import BitVectToBinaryText, CreateFromBinaryText

from .meeko_functions import MoleculePreparation, PDBQTWriterLegacy, PDBQTMolecule, RDKitMolCreate
from .chemprop_filters import retrieve_filter_params
from .fingerprint_utilis import retrieve_fp_generator, retrieve_similarity_method

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from PIL import PngImagePlugin, Image

class BoundedThreadPoolExecutor(ThreadPoolExecutor):
    ### copied from https://stackoverflow.com/a/78071937
    def __init__(self, max_size=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._semaphore = Semaphore(max_size)

    def submit(self, *args, **kwargs):
        self._semaphore.acquire()
        future = super().submit(*args, **kwargs)
        future.add_done_callback(lambda _: self._semaphore.release())
        return future

def mannhold_logp(mol):
    c_count = len(mol.GetSubstructMatch(Chem.MolFromSmarts("[C]")))
    hetatm_cnt = Descriptors.NumHeteroatoms(mol)
    return 1.46 + 0.11 * c_count - 0.11 * hetatm_cnt

property_functions = {'mw'  : Descriptors.MolWt,
                      'hbd' : Descriptors.NumHDonors,
                      'hba' : Descriptors.NumHAcceptors,
                      'logp': Descriptors.MolLogP,
                      'tpsa': Descriptors.TPSA,
                      'rb'  : Descriptors.NumRotatableBonds,
                      'nor' : Descriptors.RingCount,
                      'fc'  : lambda mol: sum([atom.GetFormalCharge() for atom in mol.GetAtoms()]),
                      'nha' : Descriptors.HeavyAtomCount,
                      'mr'  : Descriptors.MolMR,
                      'na'  : lambda mol: mol.GetNumAtoms(),
                      'QED' : Descriptors.qed}

rdkit_default_filters = ['PAINS', 'BRENK', 'NIH', 'ZINC', 'CHEMBL_BMS',
                         'CHEMBL_Dundee', 'CHEMBL_Glaxo', 'CHEMBL_Inpharmatica',
                         'CHEMBL_LINT', 'CHEMBL_MLSMR', 'CHEMBL_SureChEMBL']

class MultiprocessConversion(QObject):
    progress = Signal(int)
    log = Signal(tuple)
    finished = Signal()
    
    def __init__(self, molecules_dict,
                 exact_chemprop_filters, partial_chemprop_filters,
                 exact_rdkit_filters, partial_rdkit_filters,
                 max_name_length, append, add_h, desalt, timeout,
                 chemprop_type, struct_type, additional_params: dict):
        super().__init__()
        self.molecules_dict = molecules_dict
        self.exact_chemprop_filters = exact_chemprop_filters
        self.partial_chemprop_filters = partial_chemprop_filters
        self.exact_rdkit_fitlers = exact_rdkit_filters
        self.partial_rdkit_fitlers = partial_rdkit_filters
        self.max_name_length = max_name_length
        self.append_to_single_file = append
        self.add_h = add_h
        self.desalt = desalt
        self.timeout = timeout
        self.chemprop_type = chemprop_type
        self.struct_type = struct_type
        self.executor = None
        self.force_stopped = False
        self.params = additional_params
        
    @Slot()
    def run(self):
        self.executor = ProcessPoolExecutor()
        self.futures = [self.executor.submit(convert_to_3d_mol_format, lig, o_pth,
                                             self.exact_chemprop_filters, self.partial_chemprop_filters,
                                             self.exact_rdkit_fitlers, self.partial_rdkit_fitlers,
                                             self.max_name_length, self.append_to_single_file, self.add_h,
                                             self.desalt, self.timeout, self.chemprop_type, self.struct_type,
                                             self.params)
                        for o_pth, lig in self.molecules_dict.items()]
        step = 0
        for f in as_completed(self.futures):
            if self.force_stopped:
                break
            try:
                result = f.result()
                if result is not None:
                    self.log.emit(result)   # emit msg for failed/filtered molecules
                step += 1
                self.progress.emit(step)    # emit current step for progress bar
            except:
                pass
        if not self.force_stopped:
            self.finished.emit()    # Done
        self.executor = None
    
    def stop(self):
        # TODO: Need a more elegant shutdown like the docking one
        if self.executor:
            self.force_stopped = True
            self.executor.shutdown(wait=True, cancel_futures=True)
            for p in mp.active_children():
                p.terminate()
                p.join()
            self.executor = None

def compare_single_values(op, thres, target, match_type):
    if match_type == 'Include':
        if   op == '≥':
            return target >= thres
        elif op == '>':
            return target >  thres
        elif op == '<':
            return target <  thres
        elif op == '≤':
            return target <= thres
    else:   # Exclude
        if   op == '≥':
            return target <  thres
        elif op == '>':
            return target <= thres
        elif op == '<':
            return target >= thres
        elif op == '≤':
            return target >  thres

def check_chemprop_matching(list_operations: list[tuple], target, match_type: str):
    if len(list_operations) == 1:
        return compare_single_values(list_operations[0][0], list_operations[0][1], target, match_type)
    else:
        op1, thres1 = list_operations[0]
        op2, thres2 = list_operations[1]
        or_option = False
        if (op1 in ('≥', '>') and op2 in ('≤', '<')):
            if thres1 == thres2 and op1 == '≥' and op2 == '≤':
                pass   # if things like (≤, 10), (≥, 10), use "and" (=10) instead of "or"
            elif thres1 >= thres2:
                or_option = True
        elif (op1 in ('≤', '<') and op2 in ('≥', '>')):
            if thres1 == thres2 and op1 == '≤' and op2 == '≥':
                pass
            elif thres2 >= thres1:
                or_option = True
        op1_result = compare_single_values(op1, thres1, target, match_type)
        op2_result = compare_single_values(op2, thres2, target, match_type)
        if or_option:
            return op1_result | op2_result
        else:
            return op1_result & op2_result

def rdkit_embed_molecule(lig):
    try:
        report = AllChem.EmbedMolecule(lig, useRandomCoords=True)
        if report == -1:
            return None
        else:
            return lig
    except Exception as e:
        return None

def rdkit_embed_with_timeout(lig, timeout):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(rdkit_embed_molecule, lig)
        try:
            result = future.result(timeout=timeout)
            return result
        except Exception as e:
            future.cancel()
            return 'Timeout'

def convert_to_3d_mol_format(smiles_or_mol: str, target_pth: str,
                             exact_chemprop_filters: dict, partial_chemprop_filters: dict,
                             exact_rdkit_filters: dict, partial_rdkit_filters: dict,
                             max_name_length: int, append_to_single_file: str, add_h: bool,
                             desalt: bool, timeout_time: int, chemprop_type: str, rdkit_filter_type: str,
                             params: dict):
    name, file_type = os.path.basename(target_pth).rsplit('.', 1)
    try:
        max_num_length = len(str(max(len(partial_rdkit_filters['catalog']),
                                     len(exact_rdkit_filters['catalog']),
                                     partial_rdkit_filters['partial_filter_threshold'] - 1,
                                     len(exact_chemprop_filters),
                                     len(partial_chemprop_filters),
                                     partial_chemprop_filters['partial_filter_threshold'] - 1)))
        exact_catalog_params   = FilterCatalogParams()
        partial_catalog_params = FilterCatalogParams()
        exact_structure_mols = []
        partial_structure_mols = []
        for filter in exact_rdkit_filters['catalog']:
            if filter in rdkit_default_filters:
                exact_catalog_params.AddCatalog(getattr(exact_catalog_params.FilterCatalogs, filter))
            else:
                exact_structure_mols.append(Chem.MolFromSmarts(filter))
        for filter in partial_rdkit_filters['catalog']:
            if filter in rdkit_default_filters:
                partial_catalog_params.AddCatalog(getattr(partial_catalog_params.FilterCatalogs, filter))
            else:
                partial_structure_mols.append(Chem.MolFromSmarts(filter))
        exact_rdkit_filter_catalog = FilterCatalog(exact_catalog_params)
        partial_rdkit_filter_catalog = FilterCatalog(partial_catalog_params)
        if '\n' in smiles_or_mol:
            is_3d = True
            lig = Chem.MolFromMolBlock(smiles_or_mol)   # mol block with 3D format
        else:
            is_3d = False
            lig = Chem.MolFromSmiles(smiles_or_mol) # only smiles
        if lig is None:
            text = f'{name:{max_name_length}} failed. Error: Failed to read molecule.'
            return text, 'failed', False
        if desalt:
            remover = SaltRemover()
            lig = remover.StripMol(lig)
            # Just remove all. I don't think anyone will want to dock "salt".
            # lig = remover.StripMol(lig, dontRemoveEverything=True)
            # The salt to be kept for the above line will be decided by rdkit,
            # not user. (the "later" salt in the following order will be kept).
            # 
            # Salts:
            #// start with simple inorganics:
            # [Cl,Br,I]
            # [Li,Na,K,Ca,Mg]
            # [O,N]

            # // "complex" inorganics
            # [N](=O)(O)O
            # [P](=O)(O)(O)O
            # [P](F)(F)(F)(F)(F)F
            # [S](=O)(=O)(O)O
            # [CH3][S](=O)(=O)(O)
            # c1cc([CH3])ccc1[S](=O)(=O)(O)	p-Toluene sulfonate

            # // organics
            # [CH3]C(=O)O	  Acetic acid
            # FC(F)(F)C(=O)O	  TFA
            # OC(=O)C=CC(=O)O	  Fumarate/Maleate
            # OC(=O)C(=O)O	  Oxalate
            # OC(=O)C(O)C(O)C(=O)O	  Tartrate
            # C1CCCCC1[NH]C1CCCCC1	  Dicylcohexylammonium
        protonated_lig = Chem.AddHs(lig, addCoords=is_3d)   # add Hs to plausible coord if it is 3D (might be bugged for some mol though)
        if protonated_lig.GetNumAtoms() == 0:
            text = f'{name:{max_name_length}} failed. Error: No atoms after desalting.'
            return text, 'failed', False
        if exact_chemprop_filters:
            matched_nums = sum(check_chemprop_matching(list_operations, property_functions[chem_prop](protonated_lig), chemprop_type) for chem_prop, list_operations in exact_chemprop_filters.items())
            if matched_nums < len(exact_chemprop_filters):
                text = f'{name:{max_name_length}}  matches {matched_nums:{max_num_length}}   Exact   chemical properties. Threshold = {len(exact_chemprop_filters)}'
                return text, 'property', True
        if partial_chemprop_filters['partial_filter_threshold']:
            matched_nums = sum(check_chemprop_matching(list_operations, property_functions[chem_prop](protonated_lig), chemprop_type) for chem_prop, list_operations in partial_chemprop_filters.items() if chem_prop != 'partial_filter_threshold')
            if matched_nums < partial_chemprop_filters['partial_filter_threshold']:
                text = f'{name:{max_name_length}}  matches {matched_nums:{max_num_length}} Partial   chemical properties. Threshold = {partial_chemprop_filters["partial_filter_threshold"]}'
                return text, 'property', False
        if exact_rdkit_filters['catalog']:
            struct_match_nums = len(set([entry.GetProp('FilterSet') for entry in exact_rdkit_filter_catalog.GetMatches(lig)]))
            struct_match_nums += sum(lig.HasSubstructMatch(pat) for pat in exact_structure_mols)
            if rdkit_filter_type == 'Exclude':
                if struct_match_nums >= len(exact_rdkit_filters['catalog']):
                    text = f'{name:{max_name_length}} violates {struct_match_nums:{max_num_length}}   Exact structural    filters. Threshold = {len(exact_rdkit_filters["catalog"])}'
                    return text, 'structure', True
            else:
                if struct_match_nums <  len(exact_rdkit_filters['catalog']):
                    text = f'{name:{max_name_length}}  matches {struct_match_nums:{max_num_length}}   Exact structural    filters. Threshold = {len(exact_rdkit_filters["catalog"])}'
                    return text, 'structure', True
        if partial_rdkit_filters['partial_filter_threshold']:
            struct_match_nums = len(set([entry.GetProp('FilterSet') for entry in partial_rdkit_filter_catalog.GetMatches(lig)]))
            struct_match_nums += sum(lig.HasSubstructMatch(pat) for pat in partial_structure_mols)
            if rdkit_filter_type == 'Exclude':
                if struct_match_nums >= partial_rdkit_filters['partial_filter_threshold']:
                    text = f'{name:{max_name_length}} violates {struct_match_nums:{max_num_length}} Partial structural    filters. Threshold = {partial_rdkit_filters["partial_filter_threshold"]}'
                    return text, 'structure', False
            else:
                if struct_match_nums <  partial_rdkit_filters['partial_filter_threshold']:
                    text = f'{name:{max_name_length}}  matches {struct_match_nums:{max_num_length}} Partial structural    filters. Threshold = {partial_rdkit_filters["partial_filter_threshold"]}'
                    return text, 'structure', False
        if append_to_single_file:
            target_pth = os.path.dirname(target_pth)    # yep, this is the desired path
            file_open_mode = 'a'
            # file_type = append_to_single_file
        else:
            file_open_mode = 'w'
        if file_type == 'smi':
            if not add_h:
                protonated_lig = Chem.RemoveHs(protonated_lig)
            with open(target_pth, file_open_mode) as f:
                f.write(f'{Chem.MolToSmiles(protonated_lig)} {name}\n')
            return None
        if file_type == 'csv (DiffDock)':
            protonated_lig = Chem.RemoveHs(protonated_lig)
            with open(target_pth, file_open_mode) as f:
                f.write(f'{name},{Chem.MolToSmiles(protonated_lig)}\n')
            return None
        if file_type == 'mddb':
            fpgen = retrieve_fp_generator(params['fp_settings'])
            protonated_lig = Chem.RemoveHs(protonated_lig)
            fp = fpgen(protonated_lig)
            cctx = zstd.ZstdCompressor(level=20)
            bytes = sqlite3.Binary(cctx.compress(BitVectToBinaryText(fp)))
            conn = sqlite3.connect(target_pth)
            cur = conn.cursor()
            cur.execute("INSERT OR REPLACE INTO MolDB (name, fp, smi) VALUES (?, ?, ?)", (name, bytes, 
                                                                                          sqlite3.Binary(cctx.compress(Chem.MolToSmiles(protonated_lig).encode('utf-8')))))
            conn.commit()
            conn.close()
            return None
        if file_type == 'png':
            size, dpi = int(params['size']), int(params['dpi'])
            if not add_h:
                protonated_lig = Chem.RemoveHs(protonated_lig)
            img: PngImagePlugin.PngImageFile = Draw.MolToImage(protonated_lig, size=(size, size)).convert("RGBA")
            arr_img = np.array(img, np.uint8)
            mask = (arr_img[:, :, :-1] == (255, 255, 255)).all(axis=-1)
            arr_img[mask, -1] = 0
            img = Image.fromarray(arr_img)
            img.save(target_pth, dpi=(dpi, dpi))
            return None
        # The remaining file types are all 3D format
        if not is_3d:
            if timeout_time:
                protonated_lig = rdkit_embed_with_timeout(protonated_lig, timeout_time)
                if protonated_lig is None:
                    try:
                        obabel_mol = pybel.readstring('smi', Chem.RemoveHs(protonated_lig))
                        obabel_mol.make3D('uff', steps=50)
                        mol_str = obabel_mol.write('mol')
                        protonated_lig = Chem.MolFromMolBlock(mol_str, removeHs=False)
                    except:
                        text = f'{name:{max_name_length}} failed. Error: Failed to embed molecule.'
                        return text, 'failed', False
                elif protonated_lig == 'Timeout':
                    text = f'{name:{max_name_length}} failed. Error: Embedding timeout. Took over {timeout_time} sec.'
                    return text, 'failed', False
                    try:    # Keep this part for now, maybe I just fall back to Obabel if timeout?
                        obabel_mol = pybel.readstring('smi', Chem.MolToSmiles(Chem.RemoveHs(protonated_lig)))
                        obabel_mol.make3D('uff', steps=50)
                        mol_str = obabel_mol.write('mol')
                        protonated_lig = Chem.MolFromMolBlock(mol_str, removeHs=False)
                    except:
                        text = f'{name:{max_name_length}} failed. Error: Failed to embed molecule.'
                        return text, 'failed', False
                    # return text, 'failed', False
            else:
                report = AllChem.EmbedMolecule(protonated_lig, useRandomCoords=True)
                if report == -1:
                    try:
                        obabel_mol = pybel.readstring('smi', Chem.MolToSmiles(Chem.RemoveHs(protonated_lig)))
                        obabel_mol.make3D('uff', steps=50)
                        mol_str = obabel_mol.write('mol')
                        protonated_lig = Chem.MolFromMolBlock(mol_str, removeHs=False)
                    except:
                        text = f'{name:{max_name_length}} failed. Error: Failed to embed molecule.'
                        return text, 'failed', False
            AllChem.UFFOptimizeMolecule(protonated_lig)
        if file_type in ['sdf', 'mol', 'xyz']:
            protonated_lig.SetProp('_Name' , name)
            protonated_lig.SetProp('SMILES', Chem.MolToSmiles(protonated_lig))
            if not append_to_single_file:
                if not add_h:
                    protonated_lig = Chem.RemoveHs(protonated_lig)  # add H before 3D embed & minimization, then remove H
                Chem.MolToMolFile(protonated_lig, target_pth)
            else:
                sio = io.StringIO()
                with Chem.SDWriter(sio) as w:
                    if not add_h:
                        protonated_lig = Chem.RemoveHs(protonated_lig)
                    w.write(protonated_lig)
                with open(target_pth, file_open_mode) as f:
                    f.write(sio.getvalue())
            return None
        if file_type in ['mol2']:
            protonated_lig.SetProp('_Name' , name)
            protonated_lig.SetProp('SMILES', Chem.MolToSmiles(protonated_lig))
            if add_h:
                mol_string = Chem.MolToMolBlock(protonated_lig)
            else:
                mol_string = Chem.MolToMolBlock(Chem.RemoveHs(protonated_lig))
            mol = pybel.readstring('mol', mol_string)
            mol.title = name
            mol.write(file_type, target_pth, True)
            return None
        # pdbqt
        meeko_prep = MoleculePreparation(rigid_macrocycles=not params['macrocycle'])
        mol_setups = meeko_prep.prepare(protonated_lig)
        for setup in mol_setups:
            pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup)
            if is_ok:
                with open(target_pth, 'w') as f:
                    f.write(pdbqt_string)
                return None
            else:
                text = ("Meeko failed to convert {name:{max_name_length}}. ""Error: {error_msg}.").format(name=name, max_name_length=max_name_length, error_msg=error_msg.strip().replace('\n', '. '))
                return text, 'failed', False
    except Exception as e:
        text = f'{name:{max_name_length}} failed. Error: {str(e).strip()}.'
        return text, 'failed', False

def check_smi_title_line(smi_file: str):
    with open(smi_file, 'r') as f:
        for r, l in enumerate(f):
            possible_smiles = l.split(' ')[0]
            if Chem.MolFromSmiles(possible_smiles) is not None:
                return r
        return 0

def retrieve_chemstr_from_file(chem_pth: str, table_file_params: dict, retain_3d: bool,
                               available_formats: tuple, id_list: list):
    def read_chem_to_rdkit_chem(chem_pth: str):
        if chem_pth.endswith(('.csv', '.tsv')):
            warning_strs = []
            result = {}
            if chem_pth not in table_file_params:
                text = 'Please use the "Read Table" function to read csv / tsv file.'
                return text, 'failed', False
            with open(chem_pth, 'r') as f:
                reader = csv.reader(f, delimiter=table_file_params[chem_pth]['sep'])
                all_columns = None
                data_dict = defaultdict(list)
                all_columns = next(reader)
                for row in reader:
                    for i, value in enumerate(row):
                        if i < len(all_columns):
                            data_dict[all_columns[i]].append(value)
                        else:
                            data_dict[f"extra_col_{i}"].append(value)
            chem_str_id, name_id = table_file_params[chem_pth]['chem_str'], table_file_params[chem_pth]['name']
            chem_str_list, name_list = data_dict[chem_str_id], data_dict[name_id]
            original_length = len(chem_str_list)
            chemstr_name_tuple_list = []
            for chemstr, name in zip(chem_str_list, name_list):
                if bool(chemstr.strip()) & bool(name.strip()):
                    chemstr_name_tuple_list.append((chemstr, name))
            diff_length = original_length - len(chemstr_name_tuple_list)
            if diff_length > 0:
                warning_strs = [f'{diff_length} data dropped from "{chem_pth}".']
            for row in chemstr_name_tuple_list:
                chemstr, name = row[0], row[1]
                if chemstr:
                    if chemstr.startswith('InChI='):
                        mol = Chem.MolFromInchi(chemstr)
                        if mol is None:
                            text = f'Failed to read InChI ({chemstr}).'
                            return text, 'failed', False
                        else:
                            result[name] = Chem.MolToSmiles(mol)
                    else:
                        result[name] = chemstr
            return result, warning_strs
        if chem_pth.endswith('zip'):
            with zipfile.ZipFile(chem_pth, 'r') as zip_ref:
                result = []
                for filename in zip_ref.namelist():
                    if filename.endswith(available_formats):
                        with zip_ref.open(filename) as file_in_zip:
                            file_content = file_in_zip.read().decode()
                            with tempfile.NamedTemporaryFile(suffix='.'+filename.rsplit('.', 1)[-1], delete=False) as temp_file:
                                temp_file.write(file_content.encode('utf-8'))
                                temp_file.flush()
                                temp_file_path = temp_file.name
                            result.extend(read_chem_to_rdkit_chem(temp_file_path))
                            os.remove(temp_file_path)
            return result
        if chem_pth.endswith('.smi'):
            n = check_smi_title_line(chem_pth)
            try:
                supp = Chem.MultithreadedSmilesMolSupplier(chem_pth, titleLine=n)
                return [mol for mol in supp if mol is not None]
            except Exception as e:
                return str(e)
        elif chem_pth.endswith('.sdf'):
            try:
                supp = Chem.MultithreadedSDMolSupplier(chem_pth)
                return [mol for mol in supp if mol is not None]
            except Exception as e:
                return str(e)
        elif chem_pth.endswith('.mol2'):
            return [Chem.MolFromMol2File(chem_pth)]
        elif chem_pth.endswith('.mol'):
            return [Chem.MolFromMolFile(chem_pth)]
        elif chem_pth.endswith('.mrv'):
            return [Chem.MolFromMrvFile(chem_pth)]
        elif chem_pth.endswith('.pdb'):
            mol = pybel.readfile('pdb', chem_pth)
            sdf = mol.write('sdf')
            return [Chem.MolFromMolBlock(sdf)]
        elif chem_pth.endswith('.xyz'):
            return [Chem.MolFromXYZFile(chem_pth)]
        elif chem_pth.endswith('.pdbqt'):
            with open(chem_pth) as f:
                pdbqt_str = f.read()
            try:
                pdbqt_mol = PDBQTMolecule(pdbqt_str, skip_typing=True)
                output_string, _ = RDKitMolCreate.write_sd_string(pdbqt_mol)
                supp = Chem.SDMolSupplier()
                supp.SetData(output_string)
            except:
                try:
                    obabel_mol = pybel.readstring('pdbqt', pdbqt_str)   # fall back to openbabel
                    mol_str = obabel_mol.write('mol')
                    supp = [Chem.MolFromMolBlock(mol_str, removeHs=False)]
                except:
                    supp = [None]
            return supp
        elif chem_pth.endswith('.mddb'):
            name_smi_dict = {}
            conn = sqlite3.connect(chem_pth)
            db = pd.read_sql('SELECT name, smi From MolDB', conn, chunksize=1000)
            for row in db:
                name_smi_dict.update(dict(zip(row.name, row.smi)))
            conn.close()
            data = [f'{lzma.decompress(smi).decode()} {name}' for name, smi in name_smi_dict.items()]
            supp = Chem.SmilesMolSupplier()
            supp.SetData('\n'.join(data), titleLine=False)
            return supp
        
    def retrieve_name_from_mol(mol: Chem):
        properties = mol.GetPropsAsDict()
        for id in id_list:
            if id in properties:
                return properties[id]
        if mol.HasProp("_Name"):
            return mol.GetProp("_Name")
        return None
        
    molecules = {}
    try:
        mols = read_chem_to_rdkit_chem(chem_pth)
        if isinstance(mols, str):
            text = f'Failed to read "{chem_pth}".'
            return text, 'failed', False
        elif isinstance(mols, tuple):
            if len(mols) == 2:  # csv/tsv file
                mols, warning_strs = mols
                for name, mol in mols.items():
                    if name in molecules:
                        n = 1
                        while name + f'_{n}' in molecules:
                            n += 1
                        text = f'{name} already exists, renamed to {name}_{n}.'
                        name = f'{name}_{n}'
                        warning_strs.append(text)
                    molecules[name] = mol
                return molecules, '\n'.join(warning_strs)
            elif len(mols) == 3:    # failed text
                return mols
        else:   # as a list or iterator
            duplicate_idx = 1
            warning_strs = []
            failed_strs = []
            for i, mol in enumerate(mols):
                if mol is not None:
                    name = retrieve_name_from_mol(mol)
                    if name is None:
                        if not chem_pth.endswith(('.smi', '.sdf', '.pdbqt')):
                            name = f'{os.path.basename(chem_pth).rsplit(".")[0]}'
                        else:
                            # multi-ligand files, just add number without warning
                            if len(mols) > 1:
                                name = f'{os.path.basename(chem_pth).rsplit(".")[0]}_{duplicate_idx}'
                                duplicate_idx += 1
                            else:
                                name = f'{os.path.basename(chem_pth).rsplit(".")[0]}'
                    if name in molecules:
                        n = 1
                        while name + f'_{n}' in molecules:
                            n += 1
                        text = f'{name} already exists, renamed to {name}_{n}.'
                        warning_strs.append(text)
                        name = f'{name}_{n}'
                    if retain_3d:
                        if mol.GetNumConformers() and mol.GetConformer().Is3D():
                            molecules[name] = Chem.MolToMolBlock(mol)
                        else:
                            molecules[name] = Chem.MolToSmiles(mol)
                    else:
                        molecules[name] = Chem.MolToSmiles(mol)
                else:
                    if len(mols) > 1:
                        text = f'Failed to read {i+1} th molecule from "{chem_pth}".'
                        failed_strs.append(text)
                    else:
                        text = f'Failed to read molecule from "{chem_pth}".'
                        return text, 'failed', False
            return molecules, '\n'.join(warning_strs), '\n'.join(failed_strs)
    except Exception as e:
        return e, 'failed', False

class MultiprocessConvertReader(QObject):
    finished = Signal()
    failedLog = Signal(tuple)
    processed = Signal(dict, str)
    processedWithFaied = Signal(dict, str, str)
    
    def __init__(self, all_files: list[str], table_file_params: dict, retain_3d: bool, available_formats: tuple):
        super().__init__()
        self.all_files = all_files
        self.table_params = table_file_params
        self.retain_3d = retain_3d
        self.available_formats = list(available_formats)
        self.available_formats.remove('.zip')
        self.available_formats = tuple(self.available_formats)
        id_txt = os.path.join(os.path.dirname(__file__), 'sdf_id_names.txt')
        with open(id_txt) as f:
            self.all_ids = [id for id in f.read().strip().split('\n') if id]
    
    @Slot()
    def run(self):
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(retrieve_chemstr_from_file, chem_pth,
                                       self.table_params, self.retain_3d,
                                       self.available_formats, self.all_ids) for chem_pth in self.all_files]
            for f in as_completed(futures):
                result = f.result()
                if len(result) == 2:    # success csv/tsv
                    molecules, warn = result
                    self.processed.emit(molecules, warn)
                elif isinstance(result[-1], str):  # success with failed (other formats)
                    molecules, warn, failed = result
                    self.processedWithFaied.emit(molecules, warn, failed)
                else:   # just failed
                    self.failedLog.emit(result)
        self.finished.emit()

class TwoWayMappingDict(dict):
    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

class LogTextEdit(QTextEdit):
    def __init__(self, parent, curr_mode, fontsize):
        super().__init__(parent)
        self.fontsize = fontsize
        self.setStyleSheet(f'font-family: Courier New, Courier, monospace; font-size: {fontsize}px;')
        self.light_colors = ['color:#b88608;', 'color:#000080;', 'color:#0000cd;', 'color:#b22222;', 'color:#228b22;']
        self.dark_colors  = ['color:#fffacd;', 'color:#afeeee;', 'color:#e0ffff;', 'color:#ffe4e1;', 'color:#32cd32;']
        #                     warning        ,  property       ,  structure      ,  failed         ,  success
        self.color_mapping_dict = TwoWayMappingDict()
        for lc, dc in zip(self.light_colors, self.dark_colors):
            self.color_mapping_dict[lc] = dc
        self.curr_style = self.get_styles(curr_mode, fontsize)
        
    def get_styles(self, curr_mode, fontsize):
        if curr_mode == 'light':
            styles = {
                'warning'        : f'font-family: Courier New, Courier, monospace; {self.light_colors[0]} font-size:{fontsize}px;',
                'property'       : f'font-family: Courier New, Courier, monospace; {self.light_colors[1]} font-size:{fontsize}px;',
                'property_small' : f'font-family: Courier New, Courier, monospace; {self.light_colors[1]} font-size:{fontsize-3}px;',
                'structure'      : f'font-family: Courier New, Courier, monospace; {self.light_colors[2]} font-size:{fontsize}px;',
                'structure_small': f'font-family: Courier New, Courier, monospace; {self.light_colors[2]} font-size:{fontsize-3}px;',
                'failed'         : f'font-family: Courier New, Courier, monospace; {self.light_colors[3]} font-size:{fontsize}px;',
                'done'           : f'font-family: Courier New, Courier, monospace; font-size:{fontsize+5}px;',
                'success'        : f'font-family: Courier New, Courier, monospace; {self.light_colors[4]}'
            }
        else:
            styles = {
                'warning'        : f'font-family: Courier New, Courier, monospace; {self.dark_colors[0]} font-size:{fontsize}px;',
                'property'       : f'font-family: Courier New, Courier, monospace; {self.dark_colors[1]} font-size:{fontsize}px;',
                'property_small' : f'font-family: Courier New, Courier, monospace; {self.dark_colors[1]} font-size:{fontsize-3}px;',
                'structure'      : f'font-family: Courier New, Courier, monospace; {self.dark_colors[2]} font-size:{fontsize}px;',
                'structure_small': f'font-family: Courier New, Courier, monospace; {self.dark_colors[2]} font-size:{fontsize-3}px;',
                'failed'         : f'font-family: Courier New, Courier, monospace; {self.dark_colors[3]} font-size:{fontsize}px;',
                'done'           : f'font-family: Courier New, Courier, monospace; font-size:{fontsize+5}px;',
                'success'        : f'font-family: Courier New, Courier, monospace; {self.dark_colors[4]}'
            }
        return styles
    
    def update_color_style(self, curr_mode, fontsize):
        if self.verticalScrollBar().value() == self.verticalScrollBar().maximum():
            curr_pos = None # at maximum, move to max value when setting new html
        else:
            curr_pos = self.verticalScrollBar().value()
        self.curr_style = self.get_styles(curr_mode, fontsize)
        self.curr_html = self.toHtml()
        if curr_mode == 'light':    # old = 'dark'
            for c in self.dark_colors:
                self.curr_html = self.curr_html.replace(c, self.color_mapping_dict[c])
        else:
            for c in self.light_colors:
                self.curr_html = self.curr_html.replace(c, self.color_mapping_dict[c])
        self.setHtml(self.curr_html)
        if curr_pos is None:
            self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
        else:
            self.verticalScrollBar().setValue(curr_pos)
        
    def update_fontsize_style(self, curr_mode, fontsize):
        if self.verticalScrollBar().value() == self.verticalScrollBar().maximum():
            curr_pos = self.verticalScrollBar().maximum()
        else:
            curr_pos = self.verticalScrollBar().value()
        diff = fontsize - self.fontsize
        self.fontsize = fontsize
        def increment_font_size(match):
            return f'font-size:{int(match.group(1)) + diff}px;'
        self.curr_style = self.get_styles(curr_mode, fontsize)
        self.curr_html = self.toHtml()
        self.curr_html = re.sub(r'font-size:(\d+)px;', increment_font_size, self.curr_html)
        self.setHtml(self.curr_html)
        self.verticalScrollBar().setValue(curr_pos)
        
    def temp_update(self, text: str, type: str, exact: bool=False):
        prepend = ''
        append  = ''
        if type.endswith('_small'):
            prepend += '&nbsp;&nbsp;&nbsp;&nbsp;'
        if exact:
            prepend += '<u>'
            append += '</u>'
        style = self.curr_style[type]
        text += '\n'
        self.curr_html += f"""<pre><span style="{style}">{prepend}{text}{append}</span></pre>"""
        
    def append_to_log(self, text: str, type: str, exact: bool=False):
        # exact = True: exact match, add underline
        prepend = ''
        append  = ''
        if type.endswith('_small'):
            prepend += '&nbsp;&nbsp;&nbsp;&nbsp;'
        if exact:
            prepend += '<u>'
            append += '</u>'
        style = self.curr_style[type]
        self.append(f"""<pre><span style="{style}">{prepend}{text}{append}</span></pre>""")

class ConvertFilterDialog(QDialog):
    def __init__(self, chem_filter_dict: dict, chem_filter_bool: dict, rdkit_filter_dict: dict,
                 sampling_setting_dict: dict, similarity_db_dict: dict):
        super().__init__()
        self.rdkit_filters_names = ['PAINS', 'BRENK', 'NIH', 'ZINC', 'CHEMBL_BMS',
                                    'CHEMBL_Dundee', 'CHEMBL_Glaxo', 'CHEMBL_Inpharmatica',
                                    'CHEMBL_LINT', 'CHEMBL_MLSMR', 'CHEMBL_SureChEMBL']
        self.initUI(chem_filter_dict, chem_filter_bool, rdkit_filter_dict,
                    sampling_setting_dict, similarity_db_dict)
        screen = self.screen()
        screen_size = screen.availableGeometry()
        self.resize(screen_size.width() * 0.3, min(730, screen_size.height() * 0.85))
        
    def initUI(self, chem_filter_dict, chem_filter_bool, rdkit_filter_dict, sampling_setting_dict, similarity_db_dict):
        self.setWindowTitle('Molecular Filters')
        
        self.overall_layout = QVBoxLayout()
        chem_and_rdkit_filter_layout = QHBoxLayout()
        chemprop_rule_scroll_area = QScrollArea()
        chemprop_rule_scroll_area.horizontalScrollBar().setDisabled(True)
        chemprop_rule_scroll_area.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        
        chemprop_rule_widget = QWidget()
        chemprop_rule_layout = QGridLayout(chemprop_rule_widget)
        chemprop_rule_scroll_area.setWidget(chemprop_rule_widget)
        chemprop_rule_scroll_area.setWidgetResizable(True)
        
        rdkit_filter_frame = QFrame()
        rdkit_filter_frame.setFrameShape(QFrame.Shape.StyledPanel)
        rdkit_filter_frame.setLineWidth(2)
        rdkit_filter_layout = QVBoxLayout(rdkit_filter_frame)
        
        similarity_match_frame = QFrame()
        similarity_match_frame.setFrameShape(QFrame.Shape.StyledPanel)
        similarity_match_frame.setLineWidth(2)
        similarity_match_layout = QVBoxLayout()
        similarity_match_frame.setLayout(similarity_match_layout)
        
        right_side_widget = QWidget()
        right_side_layout = QVBoxLayout()
        right_side_layout.setContentsMargins(0, 0, 0, 0)
        right_side_widget.setLayout(right_side_layout)
        
        example_layout = QHBoxLayout()
        example_frame = QFrame()
        example_frame.setFrameShape(QFrame.Shape.StyledPanel)
        example_frame.setLineWidth(2)
        
        rng_layout = QHBoxLayout()
        rng_frame = QFrame()
        rng_frame.setFrameShape(QFrame.Shape.StyledPanel)
        rng_frame.setLineWidth(2)
        
        first_row_layout = QHBoxLayout()
        
        example_label = QLabel('<b>Checkbox Options :</b>')
        self.full_checkbox = QCheckBox()
        self.full_checkbox.setText('Exact')
        self.full_checkbox.setTristate(True)
        self.full_checkbox.setCheckState(Qt.CheckState.Checked)
        self.full_checkbox.checkStateChanged.connect(self.example_set_original)
        self.partial_checkbox = QCheckBox()
        self.partial_checkbox.setText('Partial')
        self.partial_checkbox.setTristate(True)
        self.partial_checkbox.setCheckState(Qt.CheckState.PartiallyChecked)
        self.partial_checkbox.checkStateChanged.connect(self.example_set_original)
        self.no_checkbox = QCheckBox()
        self.no_checkbox.setText('Ignore')
        self.no_checkbox.setTristate(True)
        self.no_checkbox.setCheckState(Qt.CheckState.Unchecked)
        self.no_checkbox.checkStateChanged.connect(self.example_set_original)
        example_layout.addWidget(example_label)
        example_layout.addWidget(self.no_checkbox)
        example_layout.addSpacerItem(QSpacerItem(10, 5, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        example_layout.addWidget(self.partial_checkbox)
        example_layout.addSpacerItem(QSpacerItem(10, 5, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        example_layout.addWidget(self.full_checkbox)
        example_frame.setLayout(example_layout)
        
        self.sampling_checkbox = QCheckBox(self)
        self.sampling_checkbox.setChecked(sampling_setting_dict['ckbox'])
        self.sampling_options = QComboBox(self)
        self.sampling_options.addItems(['Random Sampling', 'MaxMin Picker'])
        self.sampling_options.setMinimumWidth(160)
        self.sampling_options.setEnabled(sampling_setting_dict['ckbox'])
        self.sampling_options.setCurrentText('MaxMin Picker' if sampling_setting_dict['maxmin'] else 'Random Sampling')
        if sampling_setting_dict['maxmin']:
            self.sampling_options.setCurrentText('Maxmin Picker')
        self.sampling_checkbox.stateChanged.connect(self.setup_sampling_options)
        rng_num_regex = QRegularExpression('[0-9]+')
        rng_num_regex_validator = QRegularExpressionValidator(rng_num_regex)
        rng_seed_label = QLabel('Seed :')
        rng_seed_label.setEnabled(sampling_setting_dict['ckbox'])
        self.rng_seed_line = QLineEdit(self)
        self.rng_seed_line.setText(sampling_setting_dict['seed'])
        self.rng_seed_line.setValidator(rng_num_regex_validator)
        self.rng_seed_line.setMaximumWidth(50)
        self.rng_seed_line.setEnabled(sampling_setting_dict['ckbox'])
        rng_num_label = QLabel('Num :')
        rng_num_label.setEnabled(sampling_setting_dict['ckbox'])
        self.rng_num_line = QLineEdit(self)
        self.rng_num_line.setValidator(rng_num_regex_validator)
        self.rng_num_line.setText(sampling_setting_dict['count'])
        self.rng_num_line.setMaximumWidth(70)
        self.rng_num_line.setEnabled(sampling_setting_dict['ckbox'])
        rng_layout.addWidget(self.sampling_checkbox)
        rng_layout.addWidget(self.sampling_options)
        rng_layout.addWidget(rng_seed_label)
        rng_layout.addWidget(self.rng_seed_line)
        rng_layout.addWidget(rng_num_label)
        rng_layout.addWidget(self.rng_num_line)
        self.rng_objects = [self.sampling_options, rng_seed_label, self.rng_seed_line, rng_num_label, self.rng_num_line]
        rng_frame.setLayout(rng_layout)
        self.rng_setting_dict = copy.deepcopy(sampling_setting_dict)
        
        QBtn = QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        self.reset_button = QPushButton('Reset')
        self.reset_button.clicked.connect(self.reset_filters)
        self.import_button = QPushButton('Import')
        self.import_button.clicked.connect(self.import_chemprop_filter_setting)
        self.export_button = QPushButton('Export')
        self.export_button.clicked.connect(self.export_chemprop_filter_setting)
        optional_button_layout = QHBoxLayout()
        optional_button_widget = QWidget()
        optional_button_layout.setContentsMargins(0, 0, 0, 0)
        optional_button_layout.addWidget(self.reset_button)
        optional_button_layout.addWidget(self.import_button)
        optional_button_layout.addWidget(self.export_button)
        optional_button_widget.setLayout(optional_button_layout)
        self.button_box = QDialogButtonBox(QBtn)
        self.button_box.accepted.connect(self.accept_changes)
        self.button_box.rejected.connect(self.reject)
        button_layout = QHBoxLayout()
        button_layout.addWidget(optional_button_widget, alignment=Qt.AlignmentFlag.AlignLeft)
        button_layout.addWidget(self.button_box, alignment=Qt.AlignmentFlag.AlignRight)
        self.chem_filter_dict = copy.deepcopy(chem_filter_dict)
        self.chem_filter_bool = copy.deepcopy(chem_filter_bool)
        self.rdkit_filter_dict = copy.deepcopy(rdkit_filter_dict)
        
        self.config_values = {'mw'  : {'label': 'Molecular Weight'        , 'min': 0   , 'max': 1e6, 'unit': 'Da', 'step': 5., 'spinbox': QDoubleSpinBox},
                              'hbd' : {'label': 'Hydrogen Bond Donors'    , 'min': 0   , 'max': 1e6, 'unit': ''  , 'step': 1 , 'spinbox': QSpinBox      },
                              'hba' : {'label': 'Hydrogen Bond Acceptors' , 'min': 0   , 'max': 1e6, 'unit': ''  , 'step': 1 , 'spinbox': QSpinBox      },
                              'logp': {'label': 'LogP'                    , 'min': -1e6, 'max': 1e6, 'unit': ''  , 'step': 1., 'spinbox': QDoubleSpinBox},
                              'tpsa': {'label': 'Topological Polar Surface Area', 'min': 0   , 'max': 1e6, 'unit': 'Å²', 'step': 5., 'spinbox': QDoubleSpinBox},
                              'rb'  : {'label': 'Rotatable Bonds'         , 'min': 0   , 'max': 1e6, 'unit': ''  , 'step': 1 , 'spinbox': QSpinBox      },
                              'nor' : {'label': 'Number of Rings'         , 'min': 0   , 'max': 1e6, 'unit': ''  , 'step': 1 , 'spinbox': QSpinBox      },
                              'fc'  : {'label': 'Formal Charge'           , 'min': -1e6, 'max': 1e6, 'unit': ''  , 'step': 1 , 'spinbox': QSpinBox      },
                              'nha' : {'label': 'Number of Heavy Atoms'   , 'min': 0   , 'max': 1e6, 'unit': ''  , 'step': 1 , 'spinbox': QSpinBox      },
                              'mr'  : {'label': 'Molar Refractivity'      , 'min': 0   , 'max': 1e6, 'unit': ''  , 'step': 1 , 'spinbox': QDoubleSpinBox},
                              'na'  : {'label': 'Number of Atoms'         , 'min': 0   , 'max': 1e6, 'unit': ''  , 'step': 1 , 'spinbox': QSpinBox      },
                              'QED' : {'label': 'QED'                     , 'min': 0   , 'max': 1  , 'unit': ''  , 'step': .1, 'spinbox': QDoubleSpinBox},}
        
        ### ChemProp Layout ###
        self.widget_mapping = {}
        self.enabled_rdkit_filter = 0
        
        chemprop_label = QLabel('<b>Chemical Property Filters :</b>')
        self.chemprop_buttongroup = QButtonGroup()
        chemprop_rule_layout.addWidget(chemprop_label, 0, 0)
        chemprop_include_exclude_widget = QWidget()
        chemprop_include_exclude_layout = QHBoxLayout()
        chemprop_include_exclude_layout.setContentsMargins(0, 0, 0, 0)
        chemprop_include_exclude_widget.setLayout(chemprop_include_exclude_layout)
        for name in ['Include', 'Exclude']:
            radio = QRadioButton(name)
            if name == chem_filter_bool['match_type']:
                radio.setChecked(True)
            self.chemprop_buttongroup.addButton(radio)
            chemprop_include_exclude_layout.addWidget(radio)
        chemprop_rule_layout.addWidget(chemprop_include_exclude_widget, 0, 1, 1, 2)
        for i, (config_key, config_dict) in enumerate(self.config_values.items(), 1):
            checkbox = QCheckBox()
            checkbox.setText(f'{config_dict["label"]} :')
            checkbox.setTristate(True)
            checkbox.setChecked(False)
            checkbox.stateChanged.connect(self.change_downstream_options)
            if config_key == 'QED':
                checkbox.setToolTip('Quantitative Estimate of Drug-likeness. Range: [0, 1]')
            
            condition_layout = QVBoxLayout()
            
            unit_label = QLabel(config_dict['unit'])
            
            add_button = QToolButton()
            add_button.setText("+")
            add_button.setStyleSheet('font-family: "Courier New", Courier, monospace; font-weight: 1000; font-size: 12px')
            add_button.clicked.connect(lambda _, k=config_key: self.add_condition_row(k))
            add_button.setDisabled(True)
            
            row_n = 2 * i
            chemprop_rule_layout.addWidget(checkbox, row_n, 0)
            chemprop_rule_layout.addLayout(condition_layout, row_n, 1)
            chemprop_rule_layout.addWidget(unit_label, row_n, 2)
            chemprop_rule_layout.addWidget(add_button, row_n, 3)
            
            h_line = QFrame()
            h_line.setFrameShape(QFrame.Shape.HLine)
            chemprop_rule_layout.addWidget(h_line, row_n+1, 0, 1, 4)
            
            self.widget_mapping[config_key] = {'checkbox'          : checkbox,
                                               'unit_label'        : unit_label,
                                               'add_button'        : add_button,
                                               'condition_layout'  : condition_layout,
                                               'conditions'        : [],}
        
        default_filters_label = QLabel('Default Filters')
        default_filters_combobox = QComboBox()
        default_filters_combobox.addItems(['', 'Lipinski’s Filter', 'Veber Filter', 'Egan Filter', 'Palm Filter',
                                           'REOS Filter', 'Ghose Filter', 'Lead-like Filter', 'Van der Waterbeemd Filter',
                                           'Murcko Filter', 'PPI Filter'])
        default_filters_combobox.setMinimumWidth(220)
        default_filters_combobox.adjustSize()
        default_filters_combobox.currentTextChanged.connect(self.update_chemprops_with_default)
        
        chemprop_threshold_label = QLabel('Partial Filter threshold :')
        chemprop_threshold_label.setToolTip('Chemicals with hits "<" threshold will be removed. (0 = no filter)')
        self.chemprop_threshold_spinbox = QSpinBox()
        self.chemprop_threshold_spinbox.setSingleStep(1)
        self.chemprop_threshold_spinbox.setRange(0, self.chem_filter_bool['partial_filter_threshold'])
        chemprop_rule_layout.addWidget(chemprop_threshold_label, 2 * len(self.config_values) + 2, 0)
        chemprop_rule_layout.addWidget(self.chemprop_threshold_spinbox, 2 * len(self.config_values) + 2, 1)
        chemprop_rule_layout.addWidget(default_filters_label, 2 * len(self.config_values) + 3, 0)
        chemprop_rule_layout.addWidget(default_filters_combobox, 2 * len(self.config_values) + 3, 1)
        
        self.update_chemprops_with_dict(self.chem_filter_dict, self.chem_filter_bool)  # Update the chemprop filter to current settings
        self.chemprop_threshold_spinbox.setValue(self.chem_filter_bool['partial_filter_threshold'])
        
        ### Structural Layout ###
        rdkit_title_layout = QHBoxLayout()
        rdkit_title_layout.setContentsMargins(0, 0, 0, 0)
        rdkit_title_widget = QWidget()
        rdkit_title_widget.setLayout(rdkit_title_layout)
        rdkit_filter_label = QLabel('<b>Structural Filters :</b>')
        self.chosen_filter_list = QListWidget()
        self.custom_smarts_items_list = []
        for name, checked in self.rdkit_filter_dict.items():
            if name not in ['partial_filter_threshold', 'match_type']:
                self.add_rdkit_filter(name, checked)
        self.chosen_filter_list.itemClicked.connect(self.keep_smiles_before_changing)
        self.chosen_filter_list.itemChanged.connect(self.update_rdkit_spinbox_value_or_check_smiles)
        
        self.rdkit_filter_buttongroup = QButtonGroup()
        rdkit_filter_include_exclude_widget = QWidget()
        rdkit_filter_include_exclude_layout = QHBoxLayout()
        rdkit_filter_include_exclude_layout.setContentsMargins(0, 0, 0, 0)
        rdkit_filter_include_exclude_widget.setLayout(rdkit_filter_include_exclude_layout)
        for name in ['Include', 'Exclude']:
            radio = QRadioButton(name)
            if name == rdkit_filter_dict['match_type']:
                radio.setChecked(True)
            self.rdkit_filter_buttongroup.addButton(radio)
            rdkit_filter_include_exclude_layout.addWidget(radio)
        rdkit_title_layout.addWidget(rdkit_filter_label)
        rdkit_title_layout.addWidget(rdkit_filter_include_exclude_widget)
        rdkit_filter_layout.addWidget(rdkit_title_widget)
        rdkit_filter_layout.addWidget(self.chosen_filter_list)
        
        rdkit_filter_option_grid = QGridLayout()
        custom_structure_label = QLabel('SMARTS :')
        self.custom_structure_lineedit = QLineEdit()
        rdkit_filter_option_grid.addWidget(custom_structure_label, 0, 0)
        rdkit_filter_option_grid.addWidget(self.custom_structure_lineedit, 0, 1, 1, 3)
        
        rdkit_threshold_label = QLabel('Partial Filter Threshold :')
        rdkit_threshold_label.setToolTip('Chemicals with hits "≥" threshold will be removed. (0 = no filter)')
        self.rdkit_threshold_spinbox = QSpinBox()
        self.rdkit_threshold_spinbox.setSingleStep(1)
        self.rdkit_threshold_spinbox.setRange(0, self.enabled_rdkit_filter)
        self.rdkit_threshold_spinbox.setValue(self.rdkit_filter_dict['partial_filter_threshold'])
        rdkit_filter_option_grid.addWidget(rdkit_threshold_label, 1, 0)
        rdkit_filter_option_grid.addWidget(self.rdkit_threshold_spinbox, 1, 1, 1, 3)
        rdkit_filter_layout.addLayout(rdkit_filter_option_grid)
        
        ### Molecular Similarity Layout ###
        db_sim_label = QLabel('<b>Database Similarity :</b>')
        add_file_button = QPushButton('Add')
        add_file_button.clicked.connect(self.add_existing_files)
        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.addWidget(db_sim_label)
        title_layout.addWidget(add_file_button, alignment=Qt.AlignmentFlag.AlignRight)
        self.input_sim_file_list = InputFileListWidget(self, ('.mddb',))
        self.input_sim_file_list.addItems(similarity_db_dict['db_pth'])
        self.input_sim_file_list.highlight_mddb_differences()
        self.sim_spinbox_list = []
        self.sim_button_group = QButtonGroup()
        sim_option_layout = QHBoxLayout()
        sim_spinbox_layout = QHBoxLayout()
        for name in ['Include', 'Exclude']:
            radio = QRadioButton(name)
            if name == similarity_db_dict['sim_type']:
                radio.setChecked(True)
            self.sim_button_group.addButton(radio)
            sim_option_layout.addWidget(radio)
        for i in range(2):
            spinbox = QDoubleSpinBox()
            spinbox.setValue(similarity_db_dict['db_sim'][i])
            spinbox.valueChanged.connect(self.change_sim_spinbox_range)
            spinbox.setRange(0, 1)
            spinbox.setSingleStep(0.05)
            spinbox.setStyleSheet("QDoubleSpinBox { font-size: 13px; }")
            spinbox.setMinimumWidth(70)
            self.sim_spinbox_list.append(spinbox)
        sim_label = QLabel('<b>Similarity Range: </b>')
        sim_to_label = QLabel('<b>~</b>')
        sim_spinbox_layout.addWidget(sim_label)
        sim_spinbox_layout.addWidget(self.sim_spinbox_list[0])
        sim_spinbox_layout.addWidget(sim_to_label)
        sim_spinbox_layout.addWidget(self.sim_spinbox_list[1])
        similarity_match_layout.addLayout(title_layout)
        similarity_match_layout.addWidget(self.input_sim_file_list)
        similarity_match_layout.addLayout(sim_option_layout)
        similarity_match_layout.addLayout(sim_spinbox_layout)
        self.similarity_setting_dict = copy.deepcopy(similarity_db_dict)
        
        right_side_layout.addWidget(rdkit_filter_frame, 6)
        right_side_layout.addWidget(similarity_match_frame, 4)
        
        chem_and_rdkit_filter_layout.addWidget(chemprop_rule_scroll_area)
        chem_and_rdkit_filter_layout.addWidget(right_side_widget)
        
        first_row_layout.addWidget(example_frame, alignment=Qt.AlignmentFlag.AlignLeft)
        first_row_layout.addWidget(rng_frame, alignment=Qt.AlignmentFlag.AlignRight)
        self.overall_layout.addLayout(first_row_layout)
        self.overall_layout.addLayout(chem_and_rdkit_filter_layout)
        self.overall_layout.addLayout(button_layout)
        self.setLayout(self.overall_layout)
        
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # self.adjustSize()
    
    def add_existing_files(self):
        files, _ = QFileDialog.getOpenFileNames(self,
                                                'Select Input Files',
                                                '',
                                                'MolDocker Database (*.mddb)')
        if files:
            self.input_sim_file_list.addItems(files)
    
    def keep_smiles_before_changing(self, item: QListWidgetItem):
        if item in self.custom_smarts_items_list:
            self.curr_smiles = item.text()
            
    def example_set_original(self, state):
        if state == Qt.CheckState.Unchecked:
            self.full_checkbox.setCheckState(Qt.CheckState.Checked)
        elif state == Qt.CheckState.Checked:
            self.partial_checkbox.setCheckState(Qt.CheckState.PartiallyChecked)
        else:
            self.no_checkbox.setCheckState(Qt.CheckState.Unchecked)
    
    def change_downstream_options(self, state):
        sender = self.sender()
        if isinstance(sender, QCheckBox):
            for widgets in self.widget_mapping.values():
                if sender == widgets['checkbox']:
                    for k in widgets:
                        if k not in ['checkbox', 'conditions']:
                            widgets[k].setEnabled(state)
                        elif k == 'conditions':
                            for v_dict in widgets[k]:
                                for v in v_dict:
                                    if len(widgets[k]) == 2:
                                        if v != 'layout':
                                            v_dict[v].setEnabled(state)
                                        widgets['add_button'].setDisabled(True)
                                    else:
                                        if v not in ['layout', 'rm_button']:
                                            v_dict[v].setEnabled(state)
            self.update_chemprop_spinbox_value()
        
    def add_condition_row(self, key):
        enabled = self.widget_mapping[key]['checkbox'].isChecked()
        condition_layout = self.widget_mapping[key]['condition_layout']
        
        comparison_combobox = QComboBox()
        comparison_combobox.addItems(['≤', '<', '≥', '>'])
        comparison_combobox.setCurrentText('≤')
        comparison_combobox.setEnabled(enabled)
        comparison_combobox.setFixedWidth(70)
        
        value_spinbox = self.config_values[key]['spinbox']()
        if isinstance(value_spinbox, QDoubleSpinBox):
            value_spinbox.setDecimals(3)
        value_spinbox.setRange(self.config_values[key]['min'], self.config_values[key]['max'])
        value_spinbox.setSingleStep(self.config_values[key]['step'])
        value_spinbox.setEnabled(enabled)
        value_spinbox.setStyleSheet("font-size: 13px;")
        
        rm_button = QToolButton()
        rm_button.setText("-")
        rm_button.setStyleSheet('font-family: "Courier New", Courier, monospace; font-weight: 1000; font-size: 20px')
        rm_button.clicked.connect(lambda _, k=key, rl=rm_button: self.rm_condition_row(k, rl))
        rm_button.setMaximumSize(QSize(15, 15))
        if self.widget_mapping[key]['conditions']:
            rm_button.setEnabled(enabled)
            self.widget_mapping[key]['conditions'][0]['rm_button'].setEnabled(enabled)
        else:
            rm_button.setEnabled(False)
        
        row_layout = QHBoxLayout()
        row_layout.addWidget(comparison_combobox)
        row_layout.addWidget(value_spinbox)
        row_layout.addWidget(rm_button)
        
        condition_layout.addLayout(row_layout)
        self.widget_mapping[key]['conditions'].append({'combobox' : comparison_combobox,
                                                       'spinbox'  : value_spinbox,
                                                       'rm_button': rm_button,
                                                       'layout'   : row_layout})
        if len(self.widget_mapping[key]['conditions']) == 2:
            self.widget_mapping[key]['add_button'].setDisabled(True)
    
    def rm_condition_row(self, key, rm_button):
        condition_list = self.widget_mapping[key]['conditions']
        for condition in condition_list:
            if condition['rm_button'] == rm_button:
                for i in reversed(range(condition['layout'].count())):
                    widget = condition['layout'].itemAt(i).widget()
                    if widget is not None:
                        widget.setParent(None)
                self.widget_mapping[key]['condition_layout'].removeItem(condition['layout'])
                condition_list.remove(condition)
                condition_list[0]['rm_button'].setDisabled(True)
                break
            if not self.widget_mapping[key]['add_button'].isEnabled():
                self.widget_mapping[key]['add_button'].setEnabled(True)
        # QApplication.instance().processEvents()
        # self.adjustSize()
    
    def update_chemprop_spinbox_value(self):
        cnt = 0
        curr_spin_value = self.chemprop_threshold_spinbox.value()
        for k in self.widget_mapping:
            if self.widget_mapping[k]['checkbox'].checkState() == Qt.CheckState.PartiallyChecked:
                cnt += 1
        self.chemprop_threshold_spinbox.setRange(0, cnt)
        if curr_spin_value + 1 == cnt:
            self.chemprop_threshold_spinbox.setValue(cnt)
    
    def update_rdkit_spinbox_value_or_check_smiles(self, item: QListWidgetItem):
        if item in self.custom_smarts_items_list:
            new_smiles = item.text()
            if Chem.MolFromSmiles(new_smiles, sanitize=False) is None:
                QMessageBox.critical(self, 'SMILES Error', f'"{new_smiles}" is not a valid SMILES string.')
                item.setText(self.curr_smiles)
            elif not new_smiles:
                item.setText(self.curr_smiles)
        cnt = 0
        curr_spin_value = self.rdkit_threshold_spinbox.value()
        for index in range(self.chosen_filter_list.count()):
            check_state = self.chosen_filter_list.item(index).checkState()
            if check_state == Qt.CheckState.PartiallyChecked:
                cnt += 1
        self.rdkit_threshold_spinbox.setRange(0, cnt)
        if curr_spin_value + 1 == cnt:
            self.rdkit_threshold_spinbox.setValue(cnt)
    
    def parse_checkstate_bool(self, text: str):
        if text == 'partial':
            return Qt.CheckState.PartiallyChecked
        elif text:
            return Qt.CheckState.Checked
        else:
            return Qt.CheckState.Unchecked
    
    def remove_all_conditions(self, key):
        condition_list = self.widget_mapping[key]['conditions']
        for condition in condition_list:
            for i in reversed(range(condition['layout'].count())):
                widget = condition['layout'].itemAt(i).widget()
                if widget is not None:
                    widget.setParent(None)
                self.widget_mapping[key]['condition_layout'].removeItem(condition['layout'])
        self.widget_mapping[key]['conditions'] = []
        
    def update_chemprops_with_dict(self, chemprops_dict: dict, chemprops_bool: dict):
        for chemprop_key, chemprop_value in chemprops_dict.items():
            self.remove_all_conditions(chemprop_key)
            self.add_condition_row(chemprop_key)
            self.widget_mapping[chemprop_key]['checkbox'].setCheckState(self.parse_checkstate_bool(chemprops_bool[chemprop_key]))
            if chemprop_value[0] and len(chemprop_value) == 1:
                comparison, value = chemprop_value[0]
                self.widget_mapping[chemprop_key]['conditions'][0]['combobox'].setCurrentText(comparison)
                self.widget_mapping[chemprop_key]['conditions'][0]['spinbox'].setValue(value)
            elif len(chemprop_value) == 2:
                comparison_1, value_1 = chemprop_value[0]
                comparison_2, value_2 = chemprop_value[1]
                self.add_condition_row(chemprop_key)
                self.widget_mapping[chemprop_key]['conditions'][0]['combobox'].setCurrentText(comparison_1)
                self.widget_mapping[chemprop_key]['conditions'][0]['spinbox'].setValue(value_1)
                self.widget_mapping[chemprop_key]['conditions'][1]['combobox'].setCurrentText(comparison_2)
                self.widget_mapping[chemprop_key]['conditions'][1]['spinbox'].setValue(value_2)
                self.widget_mapping[chemprop_key]['add_button'].setDisabled(True)
    
    def update_chemprops_with_default(self, text: str):
        filter_value_dict = retrieve_filter_params(text)
        if filter_value_dict is None:
            return
        chemprops_bool_dict = {k: ('partial' if bool(v[0]) else False) for k, v in filter_value_dict.items()}
        self.update_chemprops_with_dict(filter_value_dict, chemprops_bool_dict)
        self.update_chemprop_spinbox_value()
        include_btn = next(iter(self.chemprop_buttongroup.buttons()))
        include_btn.setChecked(True)
        # QApplication.instance().processEvents()
        # self.adjustSize()
    
    def reset_filters(self):
        for chemprop_key in self.config_values:
            self.remove_all_conditions(chemprop_key)
            self.add_condition_row(chemprop_key)
            self.widget_mapping[chemprop_key]['checkbox'].setChecked(False)
        self.chemprop_threshold_spinbox.setValue(0)
        self.chemprop_threshold_spinbox.setRange(0, 0)
        
        for index in range(self.chosen_filter_list.count()):
            item = self.chosen_filter_list.item(index)
            item.setCheckState(Qt.CheckState.Unchecked)
        self.rdkit_threshold_spinbox.setValue(0)
        self.rdkit_threshold_spinbox.setRange(0, 0)
        
        # QApplication.instance().processEvents()
        # self.adjustSize()
        
    def update_chem_filter_dict(self):
        for chemprop_key in self.widget_mapping:
            if not self.widget_mapping[chemprop_key]['checkbox'].isChecked():
                self.chem_filter_dict[chemprop_key] = [()]
            elif len(self.widget_mapping[chemprop_key]['conditions']) == 1:
                comp  = self.widget_mapping[chemprop_key]['conditions'][0]['combobox'].currentText()
                value = self.widget_mapping[chemprop_key]['conditions'][0]['spinbox'].value()
                self.chem_filter_dict[chemprop_key] = [(comp, value)]
            else:
                comp_1  = self.widget_mapping[chemprop_key]['conditions'][0]['combobox'].currentText()
                value_1 = self.widget_mapping[chemprop_key]['conditions'][0]['spinbox'].value()
                comp_2  = self.widget_mapping[chemprop_key]['conditions'][1]['combobox'].currentText()
                value_2 = self.widget_mapping[chemprop_key]['conditions'][1]['spinbox'].value()
                self.chem_filter_dict[chemprop_key] = [(comp_1, value_1), (comp_2, value_2)]
            
            state = self.widget_mapping[chemprop_key]['checkbox'].checkState()
            if state == Qt.CheckState.Checked:
                self.chem_filter_bool[chemprop_key] = True
            elif state == Qt.CheckState.PartiallyChecked:
                self.chem_filter_bool[chemprop_key] = 'partial'
            else:
                self.chem_filter_bool[chemprop_key] = False
            
        self.chem_filter_bool['partial_filter_threshold'] = self.chemprop_threshold_spinbox.value()
        for r in self.chemprop_buttongroup.buttons():
            if r.isChecked():
                self.chem_filter_bool['match_type'] = r.text()
                break
    
    def add_rdkit_filter(self, filter_name: str, checked: str | bool):
        item = QListWidgetItem(filter_name)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsUserTristate)
        if checked == 'partial':
            item.setCheckState(Qt.CheckState.PartiallyChecked)
            self.enabled_rdkit_filter += 1
        elif checked:
            item.setCheckState(Qt.CheckState.Checked)
        else:
            item.setCheckState(Qt.CheckState.Unchecked)
        self.chosen_filter_list.addItem(item)
        if filter_name not in self.rdkit_filters_names:
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            self.custom_smarts_items_list.append(item)
        
    def update_selected_rdkit_filters(self):
        for index in range(self.chosen_filter_list.count()):
            item = self.chosen_filter_list.item(index)
            if item.checkState() == Qt.CheckState.Checked:
                self.rdkit_filter_dict[item.text()] = True
            elif item.checkState() == Qt.CheckState.PartiallyChecked:
                self.rdkit_filter_dict[item.text()] = 'partial'
            else:
                self.rdkit_filter_dict[item.text()] = False
        self.rdkit_filter_dict['partial_filter_threshold'] = self.rdkit_threshold_spinbox.value()
        for r in self.rdkit_filter_buttongroup.buttons():
            if r.isChecked():
                self.rdkit_filter_dict['match_type'] = r.text()
                break
    
    def update_rng_sampler(self):
        self.rng_setting_dict['ckbox'] = True if self.sampling_checkbox.checkState() == Qt.CheckState.Checked else False
        if self.rng_setting_dict['ckbox']:
            if self.sampling_options.currentText() == 'Random Sampling':
                self.rng_setting_dict['random'] = True
                self.rng_setting_dict['maxmin'] = False
            else:
                self.rng_setting_dict['random'] = False
                self.rng_setting_dict['maxmin'] = True
        else:
            self.rng_setting_dict['random'] = False
            self.rng_setting_dict['maxmin'] = False
        self.rng_setting_dict['seed'] = self.rng_seed_line.text() if self.rng_seed_line.text() else '0'
        self.rng_setting_dict['count'] = self.rng_num_line.text() if self.rng_num_line.text() else '0'
    
    def setup_sampling_options(self, state):
        for obj in self.rng_objects:
            obj.setEnabled(state)
            
    def import_chemprop_filter_setting(self):
        file, _ = QFileDialog.getOpenFileName(self, 'Select Setting File', '', 'JSON (*.json)')
        if file:
            with open(file, 'r') as f:
                settings = json.load(f)
            self.update_chemprops_with_dict(settings['Decisions'], settings['Checkbox'])
            self.chemprop_threshold_spinbox.setValue(settings['Checkbox']['partial_filter_threshold'])
            if 'match_type' in settings['Checkbox']:
                match_type = settings['Checkbox']['match_type']
                for r in self.chemprop_buttongroup.buttons():
                    if r.text() == match_type:
                        r.setChecked(True)
                        break
    
    def export_chemprop_filter_setting(self):
        save_file, _ = QFileDialog.getSaveFileName(self, 'Save Filter Settings', '', 'JSON (*.json)')
        if save_file:
            original_chem_filter_bool = copy.deepcopy(self.chem_filter_bool)
            original_chem_filter_dict = copy.deepcopy(self.chem_filter_dict)
            self.update_chem_filter_dict()
            combined_dict = {'Checkbox' : self.chem_filter_bool,
                             'Decisions': self.chem_filter_dict,}
            with open(save_file, 'w') as f:
                json.dump(combined_dict, f, indent=4)
            self.chem_filter_bool = original_chem_filter_bool
            self.chem_filter_dict = original_chem_filter_dict
    
    def add_curr_smarts_to_structure(self):
        smarts_str = self.custom_structure_lineedit.text()
        if smarts_str:
            r = Chem.MolFromSmiles(smarts_str, sanitize=False)  # Just trust the users
            if r is not None:
                self.add_rdkit_filter(smarts_str, True)
                self.custom_structure_lineedit.clear()
            else:
                QMessageBox.critical(self, 'SMARTS Error', f'"{smarts_str}" is not a valid SMARTS string.')
    
    def change_sim_spinbox_range(self):
        self.sim_spinbox_list[0].setRange(0, self.sim_spinbox_list[1].value())
        
    def update_similarity_db_dict(self):
        self.similarity_setting_dict['db_pth'] = self.input_sim_file_list.get_all_names()
        for r in self.sim_button_group.buttons():
            if r.isChecked():
                self.similarity_setting_dict['sim_type'] = r.text()
                break
        self.similarity_setting_dict['db_sim'] = (self.sim_spinbox_list[0].value(), self.sim_spinbox_list[1].value())
    
    def accept_changes(self):
        self.update_chem_filter_dict()
        self.update_selected_rdkit_filters()
        self.update_rng_sampler()
        self.update_similarity_db_dict()
        self.accept()
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            if self.custom_structure_lineedit.hasFocus() and self.custom_structure_lineedit.text():
                self.add_curr_smarts_to_structure()
        elif event.key() == Qt.Key.Key_Backspace:
            if self.chosen_filter_list.hasFocus():
                item = self.chosen_filter_list.currentItem()
                if item and item in self.custom_smarts_items_list:
                    self.chosen_filter_list.takeItem(self.chosen_filter_list.currentRow())
                    self.custom_smarts_items_list.remove(item)
                    text = item.text()
                    if text in self.rdkit_filter_dict:
                        del self.rdkit_filter_dict[text]
        else:
            super().keyPressEvent(event)

class InputFileListWidget(QListWidget):
    currCountChanged = Signal(int)
    currNameChanged = Signal(str, str)
    currNameRemoved = Signal(str)
    
    def __init__(self, parent=None, available_formats: tuple=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.model().rowsInserted.connect(self.return_current_count)
        self.model().rowsRemoved.connect(self.return_current_count)
        self.available_formats = ('.sdf', '.smi') if available_formats is None else available_formats
        self.itemDelegate().closeEditor.connect(self.check_edited_value)
        self.doubleClicked.connect(self.save_original_path)
        self.rm_shortcut = QShortcut('Backspace', self)
        self.rm_shortcut.activated.connect(self.remove_current_selected)
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def highlight_differences(self, db_fp_dict, curr_fp_dict):
        highlighted = {}
        for key in db_fp_dict:
            val1 = db_fp_dict[key]
            val2 = curr_fp_dict.get(key, None)
            if isinstance(val1, dict):
                highlighted[key] = self.highlight_differences(val1, val2)
            else:
                if val1 != val2 or val2 is None:
                    highlighted[key] = f'<font color=#E57373>{val1} &#x27F7; {val2}</font>'
                else:
                    highlighted[key] = f'<font color=#4CAF50>{val1}</font>'
        return highlighted
    
    def highlight_mddb_differences(self):
        for i in range(self.count()):
            item = self.item(i)
            if item.text().endswith('.mddb'):
                conn = sqlite3.connect(item.text())
                cur = conn.cursor()
                cur.execute('SELECT format FROM DBInfo WHERE id = ?', (1,))
                row = cur.fetchone()
                db_fp_settings = json.loads(row[0])
                conn.close()
                with open(os.path.join(os.path.dirname(__file__), 'fp_param', 'fp_param.json'), 'r') as f:
                    curr_fp_dict = json.load(f)
                highlighted_db_fp_settings = self.highlight_differences(db_fp_settings, curr_fp_dict)
                db_data = json.dumps(highlighted_db_fp_settings, indent=4)
                item.setToolTip(
                    f"<html><body>"
                    f"<pre style='font-family:\"Courier New\", monospace;'>{db_data}</pre>"
                    f"</body></html>"
                )
    
    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            for url in urls:
                file_path = url.toLocalFile()
                if os.path.exists(file_path) and file_path.endswith(self.available_formats):
                    item = QListWidgetItem(file_path)
                    if file_path.endswith('.mddb'):
                        conn = sqlite3.connect(file_path)
                        cur = conn.cursor()
                        cur.execute('SELECT format FROM DBInfo WHERE id = ?', (1,))
                        row = cur.fetchone()
                        db_fp_settings = json.loads(row[0])
                        conn.close()
                        with open(os.path.join(os.path.dirname(__file__), 'fp_param', 'fp_param.json'), 'r') as f:
                            curr_fp_dict = json.load(f)
                        highlighted_db_fp_settings = self.highlight_differences(db_fp_settings, curr_fp_dict)
                        db_data = json.dumps(highlighted_db_fp_settings, indent=4)
                        item.setToolTip(
                            f"<html><body>"
                            f"<pre style='font-family:\"Courier New\", monospace;'>{db_data}</pre>"
                            f"</body></html>"
                        )
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                    self.addItem(item)
            event.acceptProposedAction()
            
    def __iter__(self):
        for i in range(self.count()):
            yield self.item(i).text()
    
    def get_all_names(self):
        return [self.item(i).text() for i in range(self.count())]
    
    def remove_current_selected(self):
        all_items = self.selectedItems()
        if not all_items:
            return
        else:
            for item in all_items:
                self.takeItem(self.row(item))
            
    def return_current_count(self):
        self.currCountChanged.emit(self.count())
        
    def retrieve_all_texts(self):
        return [self.item(i).text() for i in range(self.count())]
    
    def check_edited_value(self, x: QLineEdit):
        pth = x.text()
        if not pth:
            self.currNameRemoved.emit(self.previous_path)
            self.takeItem(self.row(self.selectedItems()[0]))
            return
        all_curr_text = self.retrieve_all_texts()
        all_curr_text.remove(pth)
        if pth in all_curr_text:
            QMessageBox.critical(self, 'NameError', f'"{pth}" already exist!')
            curr_selected: QListWidgetItem = self.selectedItems()[0]
            curr_selected.setText(self.previous_path)
            return
        self.currNameChanged.emit(self.previous_path, pth)
        # else:
        #     if not os.path.exists(pth):
        #         QMessageBox.critical(self, 'FileNotExist Error', f'"{pth}" does not exist!')
        #         curr_selected: QListWidgetItem = self.selectedItems()[0]
        #         curr_selected.setText(self.previous_path)
    
    def save_original_path(self):
        curr_selected = self.selectedItems()[0]
        self.previous_path = curr_selected.text()

class InputDirListWidget(QListWidget):
    currCountChanged = Signal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.model().rowsInserted.connect(self.return_current_count)
        self.model().rowsRemoved.connect(self.return_current_count)
        self.itemDelegate().closeEditor.connect(self.check_edited_value)
        self.doubleClicked.connect(self.save_original_path)
        self.rm_shortcut = QShortcut('Backspace', self)
        self.rm_shortcut.activated.connect(self.remove_current_selected)
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            
    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            for url in urls:
                file_path = url.toLocalFile()
                if os.path.isdir(file_path):
                    item = QListWidgetItem(file_path)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                    self.addItem(item)
            event.acceptProposedAction()
            
    def __iter__(self):
        for i in range(self.count()):
            yield self.item(i).text()
            
    def remove_current_selected(self):
        all_items = self.selectedItems()
        if not all_items:
            return
        else:
            for item in all_items:
                self.takeItem(self.row(item))
            
    def return_current_count(self):
        self.currCountChanged.emit(self.count())
        
    def check_edited_value(self, x: QLineEdit):
        pth = x.text()
        if not pth:
            self.takeItem(self.row(self.selectedItems()[0]))
        else:
            if not os.path.isdir(pth):
                QMessageBox.critical(self, 'FileNotExist Error', f'"{pth}" does not exist!')
                curr_selected: QListWidgetItem = self.selectedItems()[0]
                curr_selected.setText(self.previous_path)
    
    def save_original_path(self):
        curr_selected = self.selectedItems()[0]
        self.previous_path = curr_selected.text()

class InputFileDirListWidget(QListWidget):
    currCountChanged = Signal(int)
    currNameChanged = Signal(str, str)
    currNameRemoved = Signal(str)
    signalLineName = Signal(str)
    
    def __init__(self, parent=None, available_formats: tuple=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.model().rowsInserted.connect(self.return_current_count)
        self.model().rowsRemoved.connect(self.return_current_count)
        self.available_formats = ('.zip') if available_formats is None else available_formats
        self.itemDelegate().closeEditor.connect(self.check_edited_value)
        self.doubleClicked.connect(self.save_original_path)
        self.rm_shortcut = QShortcut('Backspace', self)
        self.rm_shortcut.activated.connect(self.remove_current_selected)
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            
    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            for url in urls:
                file_path = url.toLocalFile()
                if os.path.exists(file_path) and (file_path.endswith(self.available_formats) or os.path.isdir(file_path)):
                    item = QListWidgetItem(file_path)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                    self.addItem(item)
            event.acceptProposedAction()
            
    def __iter__(self):
        for i in range(self.count()):
            yield self.item(i).text()
    
    def get_all_names(self):
        return [self.item(i).text() for i in range(self.count())]
            
    def remove_current_selected(self):
        all_items = self.selectedItems()
        if not all_items:
            return
        else:
            for item in all_items:
                self.takeItem(self.row(item))
            
    def return_current_count(self):
        self.currCountChanged.emit(self.count())
        
    def retrieve_all_texts(self):
        return [self.item(i).text() for i in range(self.count())]
    
    def check_edited_value(self, x: QLineEdit):
        pth = x.text()
        if not pth:
            self.currNameRemoved.emit(self.previous_path)
            self.takeItem(self.row(self.selectedItems()[0]))
            return
        else:
            if not os.path.exists(pth):
                self.signalLineName.emit(pth)
    
    def save_original_path(self):
        curr_selected = self.selectedItems()[0]
        self.previous_path = curr_selected.text()

class OutputDirLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            
    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            for url in urls:
                file_path = url.toLocalFile()
                if file_path and os.path.exists(file_path):
                    self.setText(file_path)
            event.acceptProposedAction()

class DirSplitterDialog(QDialog):
    def __init__(self, curr_mode: str):
        super().__init__()
        self.total_files = None
        self.failed_success_color_map = {True: '#4CAF50', False: '#E57373'}
        self.curr_mode = curr_mode
        self.initUI()
        self.setWindowTitle('Directory Splitter / Zipper')
        
    def initUI(self):
        overall_layout = QVBoxLayout()
        
        input_layout = QVBoxLayout()
        output_layout = QVBoxLayout()
        output_line_layout = QHBoxLayout()
        input_line_layout = QHBoxLayout()
        setting_layout = QHBoxLayout()
        table_header_layout = QHBoxLayout()
        file_layout = QVBoxLayout()
        processing_layout = QHBoxLayout()
        
        input_label = QLabel('<b>Input Directory :</b>')
        self.input_dir_list = InputDirListWidget(self)
        self.input_dir_list.setMinimumWidth(550)
        self.input_dir_list.setMinimumHeight(100)
        self.input_dir_list.currCountChanged.connect(self.check_curr_input_list)
        input_btns_widget = QWidget()
        input_btns_layout = QVBoxLayout()
        input_btns_layout.setContentsMargins(0, 0, 0, 0)
        input_btns_widget.setLayout(input_btns_layout)
        self.browse_input_dir_btn = QPushButton('Browse')
        self.browse_input_dir_btn.clicked.connect(self.select_input_directory)
        self.read_input_btn = QPushButton('Read')
        self.read_input_btn.setDisabled(True)
        self.read_input_btn.clicked.connect(self.read_input_files)
        input_btns_layout.addWidget(self.browse_input_dir_btn)
        input_btns_layout.addWidget(self.read_input_btn)
        input_layout.addWidget(input_label)
        input_line_layout.addWidget(self.input_dir_list)
        input_line_layout.addWidget(input_btns_widget)
        input_layout.addLayout(input_line_layout)
        
        output_label = QLabel('<b>Output Parent Directory :</b>')
        self.output_lineedit = QLineEdit()
        self.output_lineedit.setMinimumWidth(550)
        self.output_lineedit.textChanged.connect(lambda _, x=1: self.calculate_total_sum(None, x))
        self.browse_output_dir_btn = QPushButton('Browse')
        self.browse_output_dir_btn.clicked.connect(self.select_output_directory)
        
        num_widget = QWidget()
        num_layout = QHBoxLayout()
        num_layout.setContentsMargins(0, 0, 0, 0)
        num_widget.setLayout(num_layout)
        num_label = QLabel('<b>Num :</b>')
        self.num_spinbox = QSpinBox()
        self.num_spinbox.setRange(1, 1_000_000)  # Should be enough
        self.num_spinbox.valueChanged.connect(self.update_table_row)
        num_layout.addWidget(num_label)
        num_layout.addWidget(self.num_spinbox)
        setting_right_layout = QHBoxLayout()
        setting_right_layout.setContentsMargins(0, 0, 0, 0)
        setting_right_widget = QWidget()
        setting_right_widget.setLayout(setting_right_layout)
        self.zipped_checkbox = QCheckBox('Zip')
        self.zipped_checkbox.setStyleSheet('QCheckBox {font-weight: bold}')
        self.zipped_checkbox.setToolTip('".zip" will be automatically added for output file')
        self.copy_checkbox = QCheckBox('Copy')
        self.copy_checkbox.setStyleSheet('QCheckBox {font-weight: bold}')
        self.copy_checkbox.setToolTip('Copy files (keep original files)')
        self.copy_checkbox.setChecked(True)
        setting_right_layout.addWidget(self.zipped_checkbox)
        setting_right_layout.setSpacing(30)
        setting_right_layout.addWidget(self.copy_checkbox)
        setting_layout.addWidget(num_widget, alignment=Qt.AlignmentFlag.AlignLeft)
        setting_layout.addWidget(setting_right_widget, alignment=Qt.AlignmentFlag.AlignRight)
        
        output_layout.addWidget(output_label)
        output_line_layout.addWidget(self.output_lineedit)
        output_line_layout.addWidget(self.browse_output_dir_btn)
        output_layout.addLayout(output_line_layout)
        
        output_table_label = QLabel('<b>Output Table :</b>')
        self.file_output_table = QTableWidget()
        self.file_output_table.verticalHeader().setVisible(False)
        self.file_output_table.setColumnCount(2)
        self.file_output_table.setHorizontalHeaderLabels(['File Name', 'Count'])
        self.file_output_table.setRowCount(1)
        self.file_output_table.setItem(0, 0, QTableWidgetItem())
        self.num_label = QLabel('<b>Sum: 0</b>')
        self.num_label.setStyleSheet(f"color: {self.failed_success_color_map[True]}")
        cnt_item = QTableWidgetItem()
        self.file_output_table.setItem(0, 1, cnt_item)
        self.file_output_table.setMinimumHeight(300)
        self.file_output_table.cellChanged.connect(self.calculate_total_sum)
        self.file_output_table.cellClicked.connect(self.keep_current_value_as_reference)
        header = self.file_output_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        
        self.process_progress = QProgressBar()
        self.start_saving_btn = QPushButton('Process')
        self.start_saving_btn.setDisabled(True)
        self.start_saving_btn.clicked.connect(self.start_processing_files)
        table_header_layout.addWidget(output_table_label, alignment=Qt.AlignmentFlag.AlignLeft)
        table_header_layout.addWidget(self.num_label, alignment=Qt.AlignmentFlag.AlignRight)
        file_layout.addLayout(table_header_layout)
        file_layout.addWidget(self.file_output_table)
        processing_layout.addWidget(self.process_progress)
        processing_layout.addWidget(self.start_saving_btn, alignment=Qt.AlignmentFlag.AlignRight)
        
        overall_layout.addLayout(input_layout)
        overall_layout.addLayout(output_layout)
        overall_layout.addLayout(setting_layout)
        overall_layout.addLayout(file_layout)
        overall_layout.addLayout(processing_layout)
        
        self.num_regex = re.compile(r'\d+')
        self.count_widget_unmod = {0: True}
        self.setLayout(overall_layout)
    
    def select_input_directory(self):
        dir = QFileDialog.getExistingDirectory(self, 'Select Input Directory', '')
        if dir:
            self.input_dir_list.addItem(dir)
            
    def select_output_directory(self):
        dir = QFileDialog.getExistingDirectory(self, 'Select Output Directory', '')
        if dir:
            self.output_lineedit.setText(dir)
            
    def check_curr_input_list(self, num_row: int):
        self.read_input_btn.setEnabled(bool(num_row))
        
    def determine_next_text(self, all_text_list: list):
        all_nums = [re.findall(self.num_regex, string_with_num) for string_with_num in all_text_list]
        final_string = all_text_list[-1]
        all_pos = [pos.span() for pos in re.finditer(self.num_regex, final_string)][::-1]    # reverse it
        nums = []
        if not all_nums:
            return final_string + '_1'
        for group_nums in zip(*all_nums):
            g = np.array(group_nums, int)
            correct = False
            for step in range(1, len(g)):
                if not np.diff(g, step).sum():
                    correct = True
                    break
            if correct:
                n = []
                for i in range(step-1, 0, -1):
                    n.append(np.diff(g, i)[-1])
                if n:
                    nums.append(int(g[-1]+np.sum(n)))
                else:
                    nums.append(int(g[-1]))
            else:
                nums.append(int(g[-1]) + 1)
        nums = nums[::-1]   # reverse it so index won't change
        for pos, num in zip(all_pos, nums):
            final_string = final_string[:pos[0]] + str(num) + final_string[pos[1]:]
        return final_string
    
    def update_table_row(self, value: int):
        self.file_output_table.blockSignals(True)
        curr_row_num = self.file_output_table.rowCount()
        if value < curr_row_num:
            for rm_target_row in range(value, curr_row_num):
                self.file_output_table.removeRow(rm_target_row)
                if rm_target_row in self.count_widget_unmod:
                    del self.count_widget_unmod[rm_target_row]
            self.file_output_table.setRowCount(value)
        else:   # value > curr_row_num
            self.file_output_table.setRowCount(value)
            for row in range(curr_row_num, value):
                self.file_output_table.setItem(row, 0, QTableWidgetItem())
                self.file_output_table.setItem(row, 1, QTableWidgetItem())
                self.count_widget_unmod[row] = True
        if self.total_files is not None and value > 1:
            for v in range(curr_row_num, value):
                all_texts = [self.file_output_table.item(row, 0).text() for row in range(v)]
                item = self.file_output_table.item(v, 0)
                item.setText(self.determine_next_text(all_texts))
                count_item = self.file_output_table.item(v, 1)
                count_item.setData(Qt.ItemDataRole.DisplayRole, 0)
            unmodded_cnt = sum(self.count_widget_unmod.values())
            sum_of_modded = sum([self.file_output_table.item(row, 1).data(Qt.ItemDataRole.DisplayRole) for row, b in self.count_widget_unmod.items() if not b])
            numerator = self.total_files - sum_of_modded
            if unmodded_cnt:
                single_cnt = numerator // unmodded_cnt
                for i, b in self.count_widget_unmod.items():
                    if b:
                        self.file_output_table.item(i, 1).setData(Qt.ItemDataRole.DisplayRole, single_cnt)
                self.file_output_table.item(i, 1).setData(Qt.ItemDataRole.DisplayRole, int(numerator - single_cnt * (unmodded_cnt - 1)))
        elif self.total_files is not None and value == 1:
            self.file_output_table.item(0, 1).setData(Qt.ItemDataRole.DisplayRole, self.total_files)
        self.calculate_total_sum(None, 1)
        self.file_output_table.blockSignals(False)
        
    def read_input_files(self):
        self.file_output_table.blockSignals(True)
        self.all_files = {}
        self.parent_files_map = {}
        unique_name = 'Group' if self.input_dir_list.count() > 1 else os.path.basename(self.input_dir_list.item(0).text())
        for dir in self.input_dir_list:
            all_files = [os.path.join(dir, f_or_d) for f_or_d in os.listdir(dir) if not f_or_d.endswith('.')]
            self.parent_files_map.update({f: dir for f in all_files})
            total_rows = self.num_spinbox.value()
            curr_cnt = len(self.parent_files_map)
            single_cnt = curr_cnt // total_rows
            for row in range(total_rows - 1):
                file_item = self.file_output_table.item(row, 0)
                count_item = self.file_output_table.item(row, 1)
                file_item.setText(f'{unique_name}_{row+1}')
                count_item.setData(Qt.ItemDataRole.DisplayRole, single_cnt)
            file_item = self.file_output_table.item(total_rows - 1, 0)
            count_item = self.file_output_table.item(total_rows - 1, 1)
            file_item.setText(f'{unique_name}_{total_rows}')
            count_item.setData(Qt.ItemDataRole.DisplayRole, int(curr_cnt - single_cnt * (total_rows - 1)))
        self.all_files = list(self.parent_files_map)
        self.total_files = len(self.all_files)
        self.calculate_total_sum(None, 1)
        self.file_output_table.blockSignals(False)
    
    def convert_num_to_string_format(self, num: int):
        num_str = str(num)[::-1]
        return ','.join([num_str[i:i+3] for i in range(0, len(num_str), 3)])[::-1]
        
    def keep_current_value_as_reference(self, row, column):
        if column == 1:
            self.curr_num = self.file_output_table.item(row, column).data(Qt.ItemDataRole.DisplayRole)
        else:
            self.curr_text = self.file_output_table.item(row, column).text()
    
    def calculate_total_sum(self, row: int|None = None, column: int|None = None):
        if column == 1:
            if self.total_files:
                if row is not None:
                    item = self.file_output_table.item(row, column)
                    num = item.data(Qt.ItemDataRole.DisplayRole)
                    unmodded_cnt = sum(self.count_widget_unmod.values())
                    sum_of_modded = sum([self.file_output_table.item(row, 1).data(Qt.ItemDataRole.DisplayRole) for row, b in self.count_widget_unmod.items() if not b])
                    numerator = self.total_files - sum_of_modded
                    single_cnt = numerator // unmodded_cnt
                    if num != self.curr_num:
                        if num == single_cnt:
                            self.count_widget_unmod[row] = True
                        else:
                            self.count_widget_unmod[row] = False
                summed = sum([self.file_output_table.item(r, 1).data(Qt.ItemDataRole.DisplayRole) for r in range(self.num_spinbox.value())])
                self.num_label.setText(f'<b>Sum: {self.convert_num_to_string_format(summed)} '
                                    f'/ {self.convert_num_to_string_format(self.total_files)}</b>')
                self.num_label.setStyleSheet(f"color: {self.failed_success_color_map[summed == self.total_files]}")
                self.start_saving_btn.setEnabled((summed == self.total_files) & bool(self.output_lineedit.text()))
        else:   # check if name already exist
            if column is not None:
                item = self.file_output_table.item(row, column)
                item_text = item.text()
                all_texts = [self.file_output_table.item(r, 0).text() for r in range(self.file_output_table.rowCount()) if r != row]
                if item_text in all_texts:
                    QMessageBox.critical(self, 'ExistingNameError', f'"{item_text}" already exist!')
                    self.file_output_table.blockSignals(True)
                    item.setText(self.curr_text)
                    self.file_output_table.blockSignals(False)
                    return
                if self.check_filename(item_text):
                    QMessageBox.critical(self, 'InvalidNameError', f'"{item_text}" contains invalid special character(s)!')
                    self.file_output_table.blockSignals(True)
                    item.setText(self.curr_text)
                    self.file_output_table.blockSignals(False)
                    return
    
    def start_processing_files(self):
        self.process_progress.setValue(0)
        self.process_progress.setMaximum(self.total_files)
        is_zipped = self.zipped_checkbox.isChecked()
        is_copy = self.copy_checkbox.isChecked()
        row_nums = self.file_output_table.rowCount()
        name_file_dict = {}
        last_cnt = 0
        for row in range(row_nums):
            cnt = self.file_output_table.item(row, 1).data(Qt.ItemDataRole.DisplayRole)
            name = self.file_output_table.item(row, 0).text()
            name_file_dict[name] = self.all_files[last_cnt:last_cnt+cnt]
            last_cnt += cnt
        parent_dir = self.output_lineedit.text()
        os.makedirs(parent_dir, exist_ok=True)
        self.curr_progress = 0
        
        self.start_saving_btn.setDisabled(True)
        self.worker = MultithreadSplitter(name_file_dict, parent_dir, self.parent_files_map, is_copy, is_zipped)
        self.thread = QThread(self)
        self.worker.moveToThread(self.thread)
        self.worker.currDoneSignal.connect(self.update_progress_bar)
        self.worker.finishedSignal.connect(self.thread.quit)
        self.thread.finished.connect(self.cleanup_thread)
        self.thread.started.connect(self.worker.run)
        self.thread.start()
        
    def update_progress_bar(self):
        self.curr_progress += 1
        self.process_progress.setValue(self.curr_progress)
        
    def cleanup_thread(self):
        self.thread.wait()
        self.worker.deleteLater()
        self.thread.deleteLater()
        self.start_saving_btn.setEnabled(True)
    
    def check_filename(self, filename):
        current_os = platform.system()
        if current_os == 'Windows':
            forbidden_chars = r'[<>:"/\\|?*]'
        elif current_os == 'Darwin':
            forbidden_chars = r'[:]'
        else:
            forbidden_chars = r'[/\0]'
        if re.match(forbidden_chars, filename):
            return True
        return False

class DirCombinerDialog(QDialog):
    def __init__(self, curr_mode: str):
        super().__init__()
        self.total_files = None
        self.failed_success_color_map = {True: '#4CAF50', False: '#E57373'}
        self.curr_mode = curr_mode
        self.initUI()
        self.setWindowTitle('Directory / Zipped Files Combiner')
    
    def initUI(self):
        overall_layout = QVBoxLayout()
        
        input_layout = QHBoxLayout()
        output_layout = QVBoxLayout()
        output_line_layout = QHBoxLayout()
        setting_layout = QHBoxLayout()
        processing_layout = QHBoxLayout()
        
        input_label = QLabel('<b>Input Directories / Zipped Files :</b>')
        input_label.setToolTip('Zip files within directory will NOT be unzipped!')
        input_list_layout = QVBoxLayout()
        self.input_dir_zipped_list = InputFileDirListWidget(self)
        self.input_dir_zipped_list.setMinimumWidth(550)
        self.input_dir_zipped_list.setMinimumHeight(25)
        self.input_dir_zipped_list.currCountChanged.connect(self.check_curr_input_list)
        input_list_layout.addWidget(self.input_dir_zipped_list)
        input_btn_layout = QVBoxLayout()
        self.browse_input_dir_btn = QPushButton('Browse Dir.')
        self.browse_input_dir_btn.clicked.connect(self.select_input_directory)
        self.browse_input_file_btn = QPushButton('Browse File')
        self.browse_input_file_btn.clicked.connect(self.select_input_files)
        input_btn_layout.addWidget(self.browse_input_dir_btn)
        input_btn_layout.addWidget(self.browse_input_file_btn)
        input_layout.addLayout(input_list_layout)
        input_layout.addLayout(input_btn_layout)
        
        output_label = QLabel('<b>Output Directory :</b>')
        self.output_lineedit = OutputDirLineEdit()
        self.output_lineedit.setMinimumWidth(550)
        self.output_lineedit.textChanged.connect(self.check_curr_input_list)
        self.browse_output_dir_btn = QPushButton('Browse')
        self.browse_output_dir_btn.clicked.connect(self.select_output_dir_or_file)
        
        setting_right_layout = QHBoxLayout()
        setting_right_layout.setContentsMargins(0, 0, 0, 0)
        setting_right_widget = QWidget()
        setting_right_widget.setLayout(setting_right_layout)
        self.copy_checkbox = QCheckBox('Copy')
        self.copy_checkbox.setStyleSheet('QCheckBox {font-weight: bold}')
        self.copy_checkbox.setToolTip('Copy files (keep original files)')
        self.copy_checkbox.setChecked(True)
        setting_right_layout.setSpacing(30)
        setting_right_layout.addWidget(self.copy_checkbox)
        setting_layout.addWidget(setting_right_widget, alignment=Qt.AlignmentFlag.AlignRight)
        
        output_layout.addWidget(output_label)
        output_line_layout.addWidget(self.output_lineedit)
        output_line_layout.addWidget(self.browse_output_dir_btn)
        output_layout.addLayout(output_line_layout)
        
        self.process_progress = QProgressBar()
        self.start_processing_btn = QPushButton('Process')
        self.start_processing_btn.setDisabled(True)
        self.start_processing_btn.clicked.connect(self.start_processing_files)
        processing_layout.addWidget(self.process_progress)
        processing_layout.addWidget(self.start_processing_btn, alignment=Qt.AlignmentFlag.AlignRight)
        
        overall_layout.addWidget(input_label)
        overall_layout.addLayout(input_layout)
        overall_layout.addLayout(output_layout)
        overall_layout.addLayout(setting_layout)
        overall_layout.addSpacerItem(QSpacerItem(0, 25))
        overall_layout.addLayout(processing_layout)
        
        self.setLayout(overall_layout)
    
    def select_input_directory(self):
        dir = QFileDialog.getExistingDirectory(self, 'Select Input Directory', '')
        if dir:
            self.input_dir_zipped_list.addItem(dir)
            
    def select_input_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, 'Select Zipped Files', '', 'Zipped File (*.zip)')
        if files:
            self.input_dir_zipped_list.addItems(files)
            
    def select_output_dir_or_file(self):
        dir = QFileDialog.getExistingDirectory(self, 'Select Output Directory', '')
        if dir:
            self.output_lineedit.setText(dir)
            
    def check_curr_input_list(self):
        num_row = self.input_dir_zipped_list.count()
        status = bool(num_row) & bool(self.output_lineedit.text())
        self.start_processing_btn.setEnabled(status)
        
    def recursive_read_all_files(self, input_dir: str, parent_inp_dir: str):
        result = []
        for base_name in os.listdir(input_dir):
            f_or_d = os.path.join(input_dir, base_name)
            if os.path.isdir(f_or_d):   # dir
                files = self.recursive_read_all_files(f_or_d, parent_inp_dir)
                result += files
            else:   # file
                if not base_name.startswith('.'):
                    result.append((parent_inp_dir, f_or_d))
        return result
    
    def start_processing_files(self):
        all_files = []
        for zf_or_dir in self.input_dir_zipped_list:
            if os.path.isdir(zf_or_dir):
                all_files.extend(self.recursive_read_all_files(zf_or_dir, zf_or_dir))
            elif os.path.isfile(zf_or_dir) and zf_or_dir.endswith('.zip'):
                all_files.append(('.zip', zf_or_dir))
        self.process_progress.setValue(0)
        self.process_progress.setMaximum(len(all_files))
        is_copy = self.copy_checkbox.isChecked()
        parent_dir = self.output_lineedit.text()
        os.makedirs(parent_dir, exist_ok=True)
        self.curr_progress = 0
        
        self.start_processing_btn.setEnabled(False)
        self.worker = MultithreadCombiner(all_files, parent_dir, is_copy)
        self.thread = QThread(self)
        self.worker.moveToThread(self.thread)
        self.worker.currDoneSignal.connect(self.update_progress_bar)
        self.worker.finishedSignal.connect(self.thread.quit)
        self.thread.finished.connect(self.cleanup_thread)
        self.thread.started.connect(self.worker.run)
        self.thread.start()
        
    def update_progress_bar(self):
        self.curr_progress += 1
        self.process_progress.setValue(self.curr_progress)
        
    def cleanup_thread(self):
        self.thread.wait()
        self.worker.deleteLater()
        self.thread.deleteLater()
        self.start_processing_btn.setEnabled(True)

class MDDBSplitterDialog(QDialog):  # TODO: NOT DONE YET, FINISH IT IN THE FUTURE!
    def __init__(self, curr_mode: str):
        super().__init__()
        self.total_files = None
        self.failed_success_color_map = {True: '#4CAF50', False: '#E57373'}
        self.curr_mode = curr_mode
        self.initUI()
        self.setWindowTitle('Database Splitter')
        
    def initUI(self):
        overall_layout = QVBoxLayout()
        
        input_layout = QVBoxLayout()
        output_layout = QVBoxLayout()
        output_line_layout = QHBoxLayout()
        input_line_layout = QHBoxLayout()
        setting_layout = QHBoxLayout()
        table_header_layout = QHBoxLayout()
        file_layout = QVBoxLayout()
        processing_layout = QHBoxLayout()
        
        input_label = QLabel('<b>Input Directory :</b>')
        self.input_dir_list = InputFileListWidget(self, ('.mddb',))
        self.input_dir_list.setMinimumWidth(550)
        self.input_dir_list.setMinimumHeight(100)
        self.input_dir_list.currCountChanged.connect(self.check_curr_input_list)
        input_btns_widget = QWidget()
        input_btns_layout = QVBoxLayout()
        input_btns_layout.setContentsMargins(0, 0, 0, 0)
        input_btns_widget.setLayout(input_btns_layout)
        self.browse_input_dir_btn = QPushButton('Browse')
        self.browse_input_dir_btn.clicked.connect(self.select_input_directory)
        self.read_input_btn = QPushButton('Read')
        self.read_input_btn.setDisabled(True)
        self.read_input_btn.clicked.connect(self.read_input_files)
        input_btns_layout.addWidget(self.browse_input_dir_btn)
        input_btns_layout.addWidget(self.read_input_btn)
        input_layout.addWidget(input_label)
        input_line_layout.addWidget(self.input_dir_list)
        input_line_layout.addWidget(input_btns_widget)
        input_layout.addLayout(input_line_layout)
        
        output_label = QLabel('<b>Output Parent Directory :</b>')
        self.output_lineedit = QLineEdit()
        self.output_lineedit.setMinimumWidth(550)
        self.output_lineedit.textChanged.connect(lambda _, x=1: self.calculate_total_sum(None, x))
        self.browse_output_dir_btn = QPushButton('Browse')
        self.browse_output_dir_btn.clicked.connect(self.select_output_directory)
        
        num_widget = QWidget()
        num_layout = QHBoxLayout()
        num_layout.setContentsMargins(0, 0, 0, 0)
        num_widget.setLayout(num_layout)
        num_label = QLabel('<b>Num :</b>')
        self.num_spinbox = QSpinBox()
        self.num_spinbox.setRange(1, 1_000_000)  # Should be enough
        self.num_spinbox.valueChanged.connect(self.update_table_row)
        num_layout.addWidget(num_label)
        num_layout.addWidget(self.num_spinbox)
        setting_right_layout = QHBoxLayout()
        setting_right_layout.setContentsMargins(0, 0, 0, 0)
        setting_right_widget = QWidget()
        setting_right_widget.setLayout(setting_right_layout)
        self.zipped_checkbox = QCheckBox('Zip')
        self.zipped_checkbox.setStyleSheet('QCheckBox {font-weight: bold}')
        self.zipped_checkbox.setToolTip('".zip" will be automatically added for output file')
        self.copy_checkbox = QCheckBox('Copy')
        self.copy_checkbox.setStyleSheet('QCheckBox {font-weight: bold}')
        self.copy_checkbox.setToolTip('Copy files (keep original files)')
        self.copy_checkbox.setChecked(True)
        setting_right_layout.addWidget(self.zipped_checkbox)
        setting_right_layout.setSpacing(30)
        setting_right_layout.addWidget(self.copy_checkbox)
        setting_layout.addWidget(num_widget, alignment=Qt.AlignmentFlag.AlignLeft)
        setting_layout.addWidget(setting_right_widget, alignment=Qt.AlignmentFlag.AlignRight)
        
        output_layout.addWidget(output_label)
        output_line_layout.addWidget(self.output_lineedit)
        output_line_layout.addWidget(self.browse_output_dir_btn)
        output_layout.addLayout(output_line_layout)
        
        output_table_label = QLabel('<b>Output Table :</b>')
        self.file_output_table = QTableWidget()
        self.file_output_table.verticalHeader().setVisible(False)
        self.file_output_table.setColumnCount(2)
        self.file_output_table.setHorizontalHeaderLabels(['File Name', 'Count'])
        self.file_output_table.setRowCount(1)
        self.file_output_table.setItem(0, 0, QTableWidgetItem())
        self.num_label = QLabel('<b>Sum: 0</b>')
        self.num_label.setStyleSheet(f"color: {self.failed_success_color_map[True]}")
        cnt_item = QTableWidgetItem()
        self.file_output_table.setItem(0, 1, cnt_item)
        self.file_output_table.setMinimumHeight(300)
        self.file_output_table.cellChanged.connect(self.calculate_total_sum)
        self.file_output_table.cellClicked.connect(self.keep_current_value_as_reference)
        header = self.file_output_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        
        self.process_progress = QProgressBar()
        self.start_saving_btn = QPushButton('Process')
        self.start_saving_btn.setDisabled(True)
        self.start_saving_btn.clicked.connect(self.start_processing_files)
        table_header_layout.addWidget(output_table_label, alignment=Qt.AlignmentFlag.AlignLeft)
        table_header_layout.addWidget(self.num_label, alignment=Qt.AlignmentFlag.AlignRight)
        file_layout.addLayout(table_header_layout)
        file_layout.addWidget(self.file_output_table)
        processing_layout.addWidget(self.process_progress)
        processing_layout.addWidget(self.start_saving_btn, alignment=Qt.AlignmentFlag.AlignRight)
        
        overall_layout.addLayout(input_layout)
        overall_layout.addLayout(output_layout)
        overall_layout.addLayout(setting_layout)
        overall_layout.addLayout(file_layout)
        overall_layout.addLayout(processing_layout)
        
        self.num_regex = re.compile(r'\d+')
        self.count_widget_unmod = {0: True}
        self.setLayout(overall_layout)
    
    def select_input_directory(self):
        dir = QFileDialog.getExistingDirectory(self, 'Select Input Directory', '')
        if dir:
            self.input_dir_list.addItem(dir)
            
    def select_output_directory(self):
        dir = QFileDialog.getExistingDirectory(self, 'Select Output Directory', '')
        if dir:
            self.output_lineedit.setText(dir)
            
    def check_curr_input_list(self, num_row: int):
        self.read_input_btn.setEnabled(bool(num_row))
        
    def determine_next_text(self, all_text_list: list):
        all_nums = [re.findall(self.num_regex, string_with_num) for string_with_num in all_text_list]
        final_string = all_text_list[-1]
        all_pos = [pos.span() for pos in re.finditer(self.num_regex, final_string)][::-1]    # reverse it
        nums = []
        if not all_nums:
            return final_string + '_1'
        for group_nums in zip(*all_nums):
            g = np.array(group_nums, int)
            correct = False
            for step in range(1, len(g)):
                if not np.diff(g, step).sum():
                    correct = True
                    break
            if correct:
                n = []
                for i in range(step-1, 0, -1):
                    n.append(np.diff(g, i)[-1])
                if n:
                    nums.append(int(g[-1]+np.sum(n)))
                else:
                    nums.append(int(g[-1]))
            else:
                nums.append(int(g[-1]) + 1)
        nums = nums[::-1]   # reverse it so index won't change
        for pos, num in zip(all_pos, nums):
            final_string = final_string[:pos[0]] + str(num) + final_string[pos[1]:]
        return final_string
    
    def update_table_row(self, value: int):
        self.file_output_table.blockSignals(True)
        curr_row_num = self.file_output_table.rowCount()
        if value < curr_row_num:
            for rm_target_row in range(value, curr_row_num):
                self.file_output_table.removeRow(rm_target_row)
                if rm_target_row in self.count_widget_unmod:
                    del self.count_widget_unmod[rm_target_row]
            self.file_output_table.setRowCount(value)
        else:   # value > curr_row_num
            self.file_output_table.setRowCount(value)
            for row in range(curr_row_num, value):
                self.file_output_table.setItem(row, 0, QTableWidgetItem())
                self.file_output_table.setItem(row, 1, QTableWidgetItem())
                self.count_widget_unmod[row] = True
        if self.total_files is not None and value > 1:
            for v in range(curr_row_num, value):
                all_texts = [self.file_output_table.item(row, 0).text() for row in range(v)]
                item = self.file_output_table.item(v, 0)
                item.setText(self.determine_next_text(all_texts))
                count_item = self.file_output_table.item(v, 1)
                count_item.setData(Qt.ItemDataRole.DisplayRole, 0)
            unmodded_cnt = sum(self.count_widget_unmod.values())
            sum_of_modded = sum([self.file_output_table.item(row, 1).data(Qt.ItemDataRole.DisplayRole) for row, b in self.count_widget_unmod.items() if not b])
            numerator = self.total_files - sum_of_modded
            if unmodded_cnt:
                single_cnt = numerator // unmodded_cnt
                for i, b in self.count_widget_unmod.items():
                    if b:
                        self.file_output_table.item(i, 1).setData(Qt.ItemDataRole.DisplayRole, single_cnt)
                self.file_output_table.item(i, 1).setData(Qt.ItemDataRole.DisplayRole, int(numerator - single_cnt * (unmodded_cnt - 1)))
        elif self.total_files is not None and value == 1:
            self.file_output_table.item(0, 1).setData(Qt.ItemDataRole.DisplayRole, self.total_files)
        self.calculate_total_sum(None, 1)
        self.file_output_table.blockSignals(False)
        
    def read_input_files(self):
        self.file_output_table.blockSignals(True)
        self.all_files = {}
        self.parent_files_map = {}
        unique_name = 'Group' if self.input_dir_list.count() > 1 else os.path.basename(self.input_dir_list.item(0).text())
        for dir in self.input_dir_list:
            all_files = [os.path.join(dir, f_or_d) for f_or_d in os.listdir(dir) if not f_or_d.endswith('.')]
            self.parent_files_map.update({f: dir for f in all_files})
            total_rows = self.num_spinbox.value()
            curr_cnt = len(self.parent_files_map)
            single_cnt = curr_cnt // total_rows
            for row in range(total_rows - 1):
                file_item = self.file_output_table.item(row, 0)
                count_item = self.file_output_table.item(row, 1)
                file_item.setText(f'{unique_name}_{row+1}')
                count_item.setData(Qt.ItemDataRole.DisplayRole, single_cnt)
            file_item = self.file_output_table.item(total_rows - 1, 0)
            count_item = self.file_output_table.item(total_rows - 1, 1)
            file_item.setText(f'{unique_name}_{total_rows}')
            count_item.setData(Qt.ItemDataRole.DisplayRole, int(curr_cnt - single_cnt * (total_rows - 1)))
        self.all_files = list(self.parent_files_map)
        self.total_files = len(self.all_files)
        self.calculate_total_sum(None, 1)
        self.file_output_table.blockSignals(False)
    
    def convert_num_to_string_format(self, num: int):
        num_str = str(num)[::-1]
        return ','.join([num_str[i:i+3] for i in range(0, len(num_str), 3)])[::-1]
        
    def keep_current_value_as_reference(self, row, column):
        if column == 1:
            self.curr_num = self.file_output_table.item(row, column).data(Qt.ItemDataRole.DisplayRole)
        else:
            self.curr_text = self.file_output_table.item(row, column).text()
    
    def calculate_total_sum(self, row: int|None = None, column: int|None = None):
        if column == 1:
            if self.total_files:
                if row is not None:
                    item = self.file_output_table.item(row, column)
                    num = item.data(Qt.ItemDataRole.DisplayRole)
                    unmodded_cnt = sum(self.count_widget_unmod.values())
                    sum_of_modded = sum([self.file_output_table.item(row, 1).data(Qt.ItemDataRole.DisplayRole) for row, b in self.count_widget_unmod.items() if not b])
                    numerator = self.total_files - sum_of_modded
                    single_cnt = numerator // unmodded_cnt
                    if num != self.curr_num:
                        if num == single_cnt:
                            self.count_widget_unmod[row] = True
                        else:
                            self.count_widget_unmod[row] = False
                summed = sum([self.file_output_table.item(r, 1).data(Qt.ItemDataRole.DisplayRole) for r in range(self.num_spinbox.value())])
                self.num_label.setText(f'<b>Sum: {self.convert_num_to_string_format(summed)} '
                                    f'/ {self.convert_num_to_string_format(self.total_files)}</b>')
                self.num_label.setStyleSheet(f"color: {self.failed_success_color_map[summed == self.total_files]}")
                self.start_saving_btn.setEnabled((summed == self.total_files) & bool(self.output_lineedit.text()))
        else:   # check if name already exist
            if column is not None:
                item = self.file_output_table.item(row, column)
                item_text = item.text()
                all_texts = [self.file_output_table.item(r, 0).text() for r in range(self.file_output_table.rowCount()) if r != row]
                if item_text in all_texts:
                    QMessageBox.critical(self, 'ExistingNameError', f'"{item_text}" already exist!')
                    self.file_output_table.blockSignals(True)
                    item.setText(self.curr_text)
                    self.file_output_table.blockSignals(False)
                    return
                if self.check_filename(item_text):
                    QMessageBox.critical(self, 'InvalidNameError', f'"{item_text}" contains invalid special character(s)!')
                    self.file_output_table.blockSignals(True)
                    item.setText(self.curr_text)
                    self.file_output_table.blockSignals(False)
                    return
    
    def start_processing_files(self):
        self.process_progress.setValue(0)
        self.process_progress.setMaximum(self.total_files)
        is_zipped = self.zipped_checkbox.isChecked()
        is_copy = self.copy_checkbox.isChecked()
        row_nums = self.file_output_table.rowCount()
        name_file_dict = {}
        last_cnt = 0
        for row in range(row_nums):
            cnt = self.file_output_table.item(row, 1).data(Qt.ItemDataRole.DisplayRole)
            name = self.file_output_table.item(row, 0).text()
            name_file_dict[name] = self.all_files[last_cnt:last_cnt+cnt]
            last_cnt += cnt
        parent_dir = self.output_lineedit.text()
        os.makedirs(parent_dir, exist_ok=True)
        self.curr_progress = 0
        
        self.start_saving_btn.setDisabled(True)
        self.worker = MultithreadSplitter(name_file_dict, parent_dir, self.parent_files_map, is_copy, is_zipped)
        self.thread = QThread(self)
        self.worker.moveToThread(self.thread)
        self.worker.currDoneSignal.connect(self.update_progress_bar)
        self.worker.finishedSignal.connect(self.thread.quit)
        self.thread.finished.connect(self.cleanup_thread)
        self.thread.started.connect(self.worker.run)
        self.thread.start()
        
    def update_progress_bar(self):
        self.curr_progress += 1
        self.process_progress.setValue(self.curr_progress)
        
    def cleanup_thread(self):
        self.thread.wait()
        self.worker.deleteLater()
        self.thread.deleteLater()
        self.start_saving_btn.setEnabled(True)
    
    def check_filename(self, filename):
        current_os = platform.system()
        if current_os == 'Windows':
            forbidden_chars = r'[<>:"/\\|?*]'
        elif current_os == 'Darwin':
            forbidden_chars = r'[:]'
        else:
            forbidden_chars = r'[/\0]'
        if re.match(forbidden_chars, filename):
            return True
        return False

class SDFIDDialog(QDialog):
    def __init__(self, id_strs: str):
        super().__init__()
        overall_layout = QVBoxLayout()
        
        self.text_edit = QPlainTextEdit(id_strs, self)
        
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.button_box = QDialogButtonBox(QBtn)
        self.button_box.accepted.connect(self.accept_changes)
        self.button_box.rejected.connect(self.reject)
        
        overall_layout.addWidget(self.text_edit)
        overall_layout.addWidget(self.button_box)
        
        self.setLayout(overall_layout)
        self.setWindowTitle('IDs to be used for file name')
        self.adjustSize()
    
    def accept_changes(self):
        self.text = self.text_edit.toPlainText()
        self.accept()

class MultithreadSplitter(QObject):
    currDoneSignal = Signal()
    finishedSignal = Signal()
    
    def __init__(self, name_file_dict: dict, parent_dir: str, parent_map: dict, is_copy: bool, is_zipped: bool):
        super().__init__()
        self.name_file_dict = name_file_dict
        self.parent_dir = parent_dir
        self.parent_map = parent_map
        self.is_copy = is_copy
        self.is_zipped = is_zipped
        
    @Slot()
    def run(self):
        def move_single_grouped_files(name: str, files: list):
            target_name = os.path.join(self.parent_dir, name)
            # remove all existing file if they exist
            if not self.is_zipped:
                if os.path.isdir(target_name):
                    shutil.rmtree(target_name)
                os.mkdir(target_name)
            else:
                if os.path.isfile(target_name + '.zip'):
                    os.remove(target_name + '.zip')
            
            if self.is_copy:
                if not self.is_zipped:
                    for f_or_d in files:
                        if os.path.isdir(f_or_d):
                            all_files = recursive_read_all_files(f_or_d)
                            for f in all_files:
                                target_file = os.path.join(target_name, os.path.relpath(f, self.parent_map[f_or_d]))
                                os.makedirs(os.path.dirname(target_file), exist_ok=True)
                                shutil.copy2(f, target_file)
                                self.currDoneSignal.emit()
                        else:
                            target_file = os.path.join(target_name, os.path.relpath(f_or_d, self.parent_map[f_or_d]))
                            os.makedirs(os.path.dirname(target_file), exist_ok=True)
                            shutil.copy2(f_or_d, target_file)
                            self.currDoneSignal.emit()
                else:
                    with zipfile.ZipFile(target_name + '.zip', 'w', zipfile.ZIP_LZMA) as zf:
                        for f_or_d in files:
                            if os.path.isdir(f_or_d):
                                all_files = recursive_read_all_files(f_or_d)
                                for f in all_files:
                                    zf.write(f, os.path.relpath(f, self.parent_map[f_or_d]))
                                    self.currDoneSignal.emit()
                            else:
                                zf.write(f_or_d, os.path.relpath(f_or_d, self.parent_map[f_or_d]))
                                self.currDoneSignal.emit()
            else:
                if not self.is_zipped:
                    for f_or_d in files:
                        if os.path.isdir(f_or_d):
                            all_files = recursive_read_all_files(f_or_d)
                            for f in all_files:
                                target_file = os.path.join(target_name, os.path.relpath(f, self.parent_map[f_or_d]))
                                os.makedirs(os.path.dirname(target_file), exist_ok=True)
                                shutil.move(f, target_file)
                                self.currDoneSignal.emit()
                        else:
                            target_file = os.path.join(target_name, os.path.relpath(f_or_d, self.parent_map[f_or_d]))
                            os.makedirs(os.path.dirname(target_file), exist_ok=True)
                            shutil.move(f_or_d, target_file)
                            self.currDoneSignal.emit()
                else:
                    with zipfile.ZipFile(target_name + '.zip', 'w', zipfile.ZIP_LZMA) as zf:
                        for f_or_d in files:
                            if os.path.isdir(f_or_d):
                                all_files = recursive_read_all_files(f_or_d)
                                for f in all_files:
                                    zf.write(f, os.path.relpath(f, self.parent_map[f_or_d]))
                                    os.remvoe(f)
                                    self.currDoneSignal.emit()
                            else:
                                zf.write(f_or_d, os.path.relpath(f_or_d, self.parent_map[f_or_d]))
                                os.remvoe(f_or_d)
                                self.currDoneSignal.emit()
            
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(move_single_grouped_files, name, files) 
                       for name, files in self.name_file_dict.items()]
            for f in as_completed(futures):
                _ = f.result()
        self.finishedSignal.emit()

class MultithreadCombiner(QObject):
    currDoneSignal = Signal()
    finishedSignal = Signal()
    
    def __init__(self, all_files: list[tuple], target_dir: str, is_copy: bool):
        super().__init__()
        self.all_files = all_files
        self.target_dir = target_dir
        self.is_copy = is_copy
        
    @Slot()
    def run(self):
        def process_single_file(format_file: tuple, index: int):
            format, file = format_file
            if format != '.zip':
                if file == 'minimize.csv':
                    file = f'{index}_minimize.csv'
                target_pth = os.path.join(self.target_dir, os.path.relpath(file, format))
                os.makedirs(os.path.dirname(target_pth), exist_ok=True)
                if self.is_copy:
                    shutil.copy2(file, target_pth)
                else:
                    shutil.move(file, target_pth)
            else:   # zip
                with zipfile.ZipFile(file) as z_f:
                    if 'minimize.csv' in z_f.namelist():
                        for f in z_f.namelist():
                            if f == 'minimize.csv':
                                output_csv_name = f"{index}_minimize.csv"
                                output_csv_path = os.path.join(self.target_dir, output_csv_name)
                                with z_f.open('minimize.csv') as source_file, open(output_csv_path, 'wb') as target_file:
                                    target_file.write(source_file.read())
                            else:
                                z_f.extract(f, self.target_dir)
                    else:
                        z_f.extractall(self.target_dir)
                if not self.is_copy:
                    os.remove(file)
            self.currDoneSignal.emit()
            
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_single_file, format_file, idx) 
                       for idx, format_file in enumerate(self.all_files, start=1)]
            for f in as_completed(futures):
                _ = f.result()
            overall_minimize_str = None
            for file in os.listdir(self.target_dir):
                if file.endswith('_minimize.csv') or file == 'minimize.csv':
                    csv_pth = os.path.join(self.target_dir, file)
                    with open(csv_pth) as f:
                        if overall_minimize_str is None:
                            overall_minimize_str = f.read()
                        else:
                            overall_minimize_str += ''.join(f.readlines()[1:])
                    os.remove(csv_pth)
            if overall_minimize_str is not None:
                with open(os.path.join(self.target_dir, 'minimize.csv'), 'w') as f:
                    f.write(overall_minimize_str)
        self.finishedSignal.emit()

def recursive_read_all_files(input_dir: str):
        result = []
        for base_name in os.listdir(input_dir):
            f_or_d = os.path.join(input_dir, base_name)
            if os.path.isdir(f_or_d):   # dir
                files = recursive_read_all_files(f_or_d)
                result += files
            else:   # file
                if not base_name.startswith('.'):
                    result.append(f_or_d)
        return result

class CSVTSVColumnDialog(QDialog):
    def __init__(self, csv_tsv_pth: str):
        self.file_pth = csv_tsv_pth
        super().__init__()
        self.initUI()
    
    def initUI(self):
        screen = self.screen()
        screen_size = screen.availableGeometry()
        self.resize(screen_size.width() * 0.4, screen_size.height() * 0.7)
        end_map_dict = {'csv': ',', 'tsv': '\t'}
        self.overall_layout = QVBoxLayout()
        self.separator_layout = QHBoxLayout()
        self.table_layout = QVBoxLayout()
        self.selection_layout = QHBoxLayout()
        self.button_layout = QHBoxLayout()
        
        self.file_name_label = QLabel('File Path : ' + f'<b>{self.file_pth}</b>')
        self.overall_layout.addWidget(self.file_name_label)
        
        self.separator_label = QLabel('Separator :')
        self.separator_text = QLineEdit(end_map_dict.get(self.file_pth.rsplit('.', 1)[-1], ';'))
        self.separator_num_label = QLabel('Preview Rows :')
        self.separator_num_spinbox = QSpinBox()
        self.separator_num_spinbox.setValue(10)
        self.separator_num_spinbox.setRange(1, 9999)
        self.apply_separator_button = QPushButton(text='Apply')
        self.apply_separator_button.clicked.connect(self.apply_separator)
        self.separator_layout.addWidget(self.separator_label)
        self.separator_layout.addWidget(self.separator_text)
        self.separator_layout.addWidget(self.separator_num_label)
        self.separator_layout.addWidget(self.separator_num_spinbox)
        self.separator_layout.addWidget(self.apply_separator_button)
        
        self.preview_label = QLabel('Preview :')
        self.preview_table = QTableWidget()
        self.warning_label = QLabel('')
        self.warning_label.setStyleSheet('font-family: "Courier New", Courier, monospace; font-weight: 500; font-size: 13px; color: PaleVioletRed')
        self.table_layout.addWidget(self.preview_label)
        self.table_layout.addWidget(self.preview_table)
        self.table_layout.addWidget(self.warning_label)
        
        self.name_column_label = QLabel('Name Column :')
        self.name_column_combobox = QComboBox()
        self.name_column_combobox.setDisabled(True)
        self.name_column_combobox.currentTextChanged.connect(self.check_name_warning)
        self.name_column_combobox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.chem_string_column_label = QLabel('SMILES/InChI Column :')
        self.chem_string_column_combobox = QComboBox()
        self.chem_string_column_combobox.setDisabled(True)
        self.chem_string_column_combobox.currentTextChanged.connect(self.check_chem_str_warning)
        self.chem_string_column_combobox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.selection_layout.addWidget(self.name_column_label)
        self.selection_layout.addWidget(self.name_column_combobox)
        self.selection_layout.addWidget(self.chem_string_column_label)
        self.selection_layout.addWidget(self.chem_string_column_combobox)
        
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.button_box = QDialogButtonBox(QBtn)
        self.button_box.accepted.connect(self.accept_name_chemstr)
        self.button_box.rejected.connect(self.reject)
        self.button_layout.addWidget(self.button_box)
        
        self.overall_layout.addLayout(self.separator_layout)
        self.overall_layout.addLayout(self.table_layout)
        self.overall_layout.addLayout(self.selection_layout)
        self.overall_layout.addLayout(self.button_layout)
        
        self.setLayout(self.overall_layout)
        self.warning_text_dict = {'Name         ': '',
                                  'SMILES/InChI' : ''}
        
    def apply_separator(self):
        with open(self.file_pth, 'r') as f:
            try:
                reader = csv.reader(f, delimiter=self.separator_text.text())
                all_columns = None
                self.data_dict = defaultdict(list)
                all_columns = next(reader)  # Read header row
                for row in reader:
                    for i, value in enumerate(row):
                        if i < len(all_columns):
                            self.data_dict[all_columns[i]].append(value)
                        else:
                            self.data_dict[f"extra_col_{i}"].append(value)  # let's hope this does not happen, but just in case
            except TypeError as e:
                QMessageBox.critical(self,
                                    f'Parse Error',
                                    f'Failed to parse file with separator "{self.separator_text.text()}".')
                return
            
        if all_columns is not None:
            self.name_column_combobox.setEnabled(True)
            self.chem_string_column_combobox.setEnabled(True)
            self.name_column_combobox.clear()
            self.chem_string_column_combobox.clear()
            self.preview_table.clear()
            self.name_column_combobox.addItems(all_columns)
            self.chem_string_column_combobox.addItems(all_columns)
            
            for col in all_columns:
                if col.lower() in ['smiles', 'inchi']:
                    self.chem_string_column_combobox.setCurrentText(col)
                    break
                
            self.preview_table.setColumnCount(len(all_columns))
            self.preview_table.setRowCount(min(len(self.data_dict[all_columns[0]]), self.separator_num_spinbox.value()))
            self.preview_table.setHorizontalHeaderLabels(all_columns)
            
            for row in range(min(len(self.data_dict[all_columns[0]]), self.separator_num_spinbox.value())):
                for col, name in enumerate(all_columns):
                    t = self.data_dict[name][row] if row < len(self.data_dict[name]) else ''
                    item = QTableWidgetItem(t)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.preview_table.setItem(row, col, item)
    
    def _auto_decide_separator(self):   # TODO: FIX IT!
        with open(self.file_pth) as f:
            possible_chars = {}
            for row, l in enumerate(f):
                if row == 0:
                    for char in l:
                        if char != '\n':
                            if char in possible_chars:
                                possible_chars[char] += 1
                            else:
                                possible_chars[char]  = 1
                    print(possible_chars)
                elif row <= 7:
                    line_chars = {}
                    for char in l:
                        if char != '\n':
                            if char in line_chars:
                                line_chars[char] += 1
                            else:
                                line_chars[char]  = 1
                    print(line_chars)
                    for c in list(possible_chars.keys()):
                        if c in line_chars:
                            if possible_chars[c] != line_chars[c]:
                                del possible_chars[c]
                else:
                    break
        return possible_chars
          
    def check_warning(self, whose, col_name):
        if col_name:
            col_data = self.data_dict.get(col_name, [])
            nan_num = sum(1 for value in col_data if not value.strip())
            
            if nan_num:
                self.warning_text_dict[whose] = f'"{col_name}" misses {nan_num} values (Total: {len(col_data)}). Missing rows will be ignored.'
            else:
                self.warning_text_dict[whose] = ''
            
            text = ''
            for w, n in self.warning_text_dict.items():
                if n:
                    text += f'\n{w} Warning: {n}' if text else f'{w} Warning: {n}'
            
            self.warning_label.setText(text)
    
    def check_chem_str_warning(self, col_name):
        self.check_warning('SMILES/InChI', col_name)
        
    def check_name_warning(self, col_name):
        self.check_warning('Name        ', col_name)
    
    def accept_name_chemstr(self):
        if not self.name_column_combobox.currentText():
            QMessageBox.critical(self,
                                 'ID Error',
                                 'Column ID for Name and/or SMILES/InChI not selected.')
        else:
            self.table_params = {self.file_pth: {'name': self.name_column_combobox.currentText(),
                                                'chem_str': self.chem_string_column_combobox.currentText(),
                                                'sep': self.separator_text.text()}}
            self.accept()
            
    def keyPressEvent(self, event): # apply separator when return is pressed and DOES NOT trigger dialog's OK button.
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if self.separator_text.hasFocus():
                self.apply_separator()
                return
        super().keyPressEvent(event)

class DBSearchDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.db_names = ['CCD', 'BIRD', 'PubChem', 'ChEMBL', 'ChEBI', 'DrugBank', 'ZINC22', 'NPASS', 'SuperNatural 3.0']
        # all non-word and non-numbers, EXCLUDING "_" and ":". since BIRD and CHEBI use these two respectively
        self.split_regex = re.compile(r'(?>\W)(?!\:)')
        self.regex_matching = {'CCD'     : re.compile(r'[A-Z0-9]{3}'),
                               'BIRD'    : re.compile(r'PRD_\d{6}'),
                               'PubChem' : re.compile(r'\d+'),
                               'ChEMBL'  : re.compile(r'^CHEMBL[1-9][0-9]{0,6}'),
                               'ChEBI'   : re.compile(r'^CHEBI:\d{4,6}'),
                               'DrugBank': re.compile(r'^DB\d{5}'),
                               'ZINC22'  : re.compile(r'^ZINC\w{1,12}'),
                               'NPASS'   : re.compile(r'^NPC[1-9][0-9]{3,5}'),
                               'SuperNatural 3.0': re.compile(r'^SN\d{7}'),}
        matched_color = QColor()
        matched_color.setRgb(76, 175, 80)
        unmatched_color = QColor()
        unmatched_color.setRgb(229, 115, 115)
        self.regex_color = {True : matched_color,
                            False: unmatched_color}
        self.all_table_db_combo = []
        self.all_table_search_ckbox = []
        self.db_id_struct_map = {}
        self.current_row = None
        self.searching = False
        self.searched_font = QFont()
        self.searched_font.setBold(True)
        self.initUI()
        
    def initUI(self):
        overall_layout = QVBoxLayout()
        widget_layout = QHBoxLayout()
        
        table_layout = QVBoxLayout()
        table_frame = QFrame()
        table_frame.setFrameShape(QFrame.Shape.StyledPanel)
        table_frame.setLineWidth(2)
        table_frame.setLayout(table_layout)
        
        table_label = QLabel('<b>Search Table :</b>')
        self.db_table = QTableWidget()
        self.db_table.setColumnCount(5)
        self.db_table.setHorizontalHeaderLabels(['Database', 'ID', 'Name', 'Search', 'Found'])
        self.db_table.cellChanged.connect(self.check_table_new_value)
        self.db_table.cellClicked.connect(self.save_current_row)
        self.db_table.verticalHeader().setVisible(False)
        header = self.db_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        
        table_search_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.search_button = QPushButton('Search')
        self.search_button.clicked.connect(self.search_db)
        table_search_layout.addWidget(self.progress_bar)
        table_search_layout.addWidget(self.search_button, alignment=Qt.AlignmentFlag.AlignRight)
        
        table_layout.addWidget(table_label)
        table_layout.addWidget(self.db_table)
        table_layout.addLayout(table_search_layout)
        
        input_layout = QVBoxLayout()
        input_setting_layout = QHBoxLayout()
        input_frame = QFrame()
        input_frame.setFrameShape(QFrame.Shape.StyledPanel)
        input_frame.setLineWidth(2)
        input_frame.setLayout(input_layout)
        database_label = QLabel('<b>Input ID :</b>')
        self.database_combo = QComboBox()
        self.database_combo.addItems(['Automatic'] + self.db_names)
        self.database_combo.setCurrentText('Automatic')
        self.database_id_input = QTextEdit()
        self.database_id_input.setAcceptRichText(False)
        add_button = QPushButton('Add')
        add_button.clicked.connect(self.add_current_list_to_table)
        self.autocheck_checkbox = QCheckBox('Validate ID')
        self.autocheck_checkbox.setChecked(True)
        self.autocheck_checkbox.setStyleSheet('font-weight: bold')
        
        input_layout.addWidget(database_label)
        input_layout.addWidget(self.database_combo)
        input_layout.addWidget(self.database_id_input)
        input_setting_layout.addWidget(add_button, alignment=Qt.AlignmentFlag.AlignLeft)
        input_setting_layout.addWidget(self.autocheck_checkbox, alignment=Qt.AlignmentFlag.AlignLeft)
        input_layout.addLayout(input_setting_layout)
        
        widget_layout.addWidget(input_frame, 2)
        widget_layout.addWidget(table_frame, 3)
        
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        button_box = QDialogButtonBox(QBtn)
        button_box.accepted.connect(self.accept_changes)
        button_box.rejected.connect(self.reject_and_close)
        
        overall_layout.addLayout(widget_layout)
        overall_layout.addWidget(button_box)
        
        self.setLayout(overall_layout)
        self.setWindowTitle('Search Database')
        screen = self.screen()
        screen_size = screen.availableGeometry()
        self.resize(screen_size.width() * 0.5, screen_size.height() * 0.6)
        
    def add_single_item_to_table(self, db_name: str, id: str, row: int):
        db_combo = QComboBox()
        db_combo.addItems(self.db_names)
        db_combo.setCurrentText(db_name)
        db_combo.setFixedWidth(120)
        db_combo.currentTextChanged.connect(lambda t, x=db_combo: self.check_table_new_db(t, x))
        self.all_table_db_combo.append(db_combo)
        
        db_item = QTableWidgetItem(id)
        db_item.setForeground(self.regex_color[bool(re.fullmatch(self.regex_matching[db_name], id))])
        
        if db_name in ['PubChem', 'CCD']:
            db_name = QTableWidgetItem(f'{db_name}_{id}')
        else:
            db_name = QTableWidgetItem(f'{id}')
        
        cell_widget = QWidget()
        cell_layout = QHBoxLayout()
        cell_layout.setContentsMargins(0, 0, 0, 0)
        cell_widget.setLayout(cell_layout)
        db_ckbox = QCheckBox()
        db_ckbox.setChecked(True)
        db_ckbox.setStyleSheet('QCheckBox { spacing: 0px; }')
        cell_layout.addWidget(db_ckbox, alignment=Qt.AlignmentFlag.AlignCenter)
        self.all_table_search_ckbox.append(db_ckbox)
        
        db_state = QTableWidgetItem('X')
        db_state.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        db_state.setFont(self.searched_font)
        db_state.setForeground(self.regex_color[False])
        db_state.setFlags(db_state.flags() & ~Qt.ItemFlag.ItemIsEditable)
        
        self.db_table.setCellWidget(row, 0, db_combo)
        self.db_table.setItem(row, 1, db_item)
        self.db_table.setItem(row, 2, db_name)
        self.db_table.setCellWidget(row, 3, cell_widget)
        self.db_table.setItem(row, 4, db_state)
        
    def add_current_list_to_table(self):
        db_name = self.database_combo.currentText()
        if db_name != 'Automatic':
            if self.autocheck_checkbox.isChecked():
                regex = self.regex_matching[db_name]
                all_ids = [s for s in re.split(self.split_regex, self.database_id_input.toPlainText()) if re.fullmatch(regex, s)]
            else:
                all_ids = [s for s in re.split(self.split_regex, self.database_id_input.toPlainText()) if s]
            
            curr_row_cnt = self.db_table.rowCount()
            self.db_table.blockSignals(True)
            self.db_table.setRowCount(curr_row_cnt + len(all_ids))
            for row, id in enumerate(all_ids, curr_row_cnt):
                self.add_single_item_to_table(db_name, id, row)
        else:
            id_dbname_dict = {}
            all_ids = [s for s in re.split(self.split_regex, self.database_id_input.toPlainText()) if s]
            for id in all_ids:
                for dbname, regex in self.regex_matching.items():
                    if re.fullmatch(regex, id):
                        id_dbname_dict[id] = dbname
                        break
            curr_row_cnt = self.db_table.rowCount()
            self.db_table.blockSignals(True)
            self.db_table.setRowCount(curr_row_cnt + len(id_dbname_dict))
            for row, (id, db_name) in enumerate(id_dbname_dict.items(), curr_row_cnt):
                self.add_single_item_to_table(db_name, id, row)
        
        header = self.db_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.db_table.blockSignals(False)
        self.database_id_input.clear()
        
    def check_table_new_value(self, row: int, col: int):
        if col == 1:
            self.db_table.blockSignals(True)
            item = self.db_table.item(row, 1)
            item_text = item.text()
            if not item_text:
                self.db_table.removeRow(row)
                self.all_table_db_combo.pop(row)
                self.all_table_search_ckbox.pop(row)
                self.db_table.blockSignals(False)
                self.db_table.clearFocus()
                self.current_row = None
                return
            db_name = self.db_table.cellWidget(row, 0).currentText()
            item.setForeground(self.regex_color[bool(re.fullmatch(self.regex_matching[db_name], item_text))])
            self.current_row = None
            self.db_table.blockSignals(False)
            self.db_table.clearFocus()
        elif col == 2:
            item = self.db_table.item(row, 2)
            new_name = item.text()
            all_curr_names = [self.db_table.item(i, 2).text() for i in range(self.db_table.rowCount()) if i != row]
            if new_name in all_curr_names:
                QMessageBox.critical(self, 'NameError', f'"{new_name}" already exists!')
                item.setText(self.current_name)
                return
            if self.current_name in self.db_id_struct_map:
                self.db_id_struct_map[new_name] = self.db_id_struct_map.pop(self.current_name)
    
    def check_table_new_db(self, new_db: str, combobox: QComboBox):
        curr_row = self.all_table_db_combo.index(combobox)
        item = self.db_table.item(curr_row, 1)
        self.db_table.blockSignals(True)
        item.setForeground(self.regex_color[bool(re.fullmatch(self.regex_matching[new_db], item.text()))])
        self.db_table.blockSignals(False)
    
    def save_current_row(self, row: int, col: int):
        self.current_row = row
        if col == 2:
            self.current_name = self.db_table.item(row, 2).text()
        
    def search_db(self):
        if not self.searching:
            self.searching = True
            self.rows_to_search = []
            self.row_dbname_id_map = {}
            for idx, ckbox in enumerate(self.all_table_search_ckbox):
                if ckbox.isChecked():
                    self.rows_to_search.append(idx)
                    # ckbox.setDisabled(True)
            for row in self.rows_to_search:
                self.all_table_db_combo[row].setDisabled(True)
                db_name = self.all_table_db_combo[row].currentText()
                id_item = self.db_table.item(row, 1)
                flags = id_item.flags()
                id_item.setFlags(flags & ~Qt.ItemFlag.ItemIsEditable)
                # db_name = self.db_table.item(row, 2)
                # db_name.setFlags(db_name.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.db_table.item(row, 4).setText('…')
                self.row_dbname_id_map[row] = {'db_name': db_name,
                                               'id'     : id_item.text()}
            self.search_button.setText('Stop')
            self.progress_bar.setMaximum(len(self.row_dbname_id_map))
            self.progress_bar.setValue(0)
            self.value = 0
            
            self.thread = QThread()
            self.worker = MultithreadDatabaseExtractor(self.row_dbname_id_map)
            self.worker.moveToThread(self.thread)
            self.worker.searched.connect(self.update_searched_result)
            self.worker.finished.connect(self.thread.quit)
            self.thread.finished.connect(self.finish_searching)
            self.thread.started.connect(self.worker.run)
            self.thread.start()
        else:
            self.worker.stop()
            self.search_button.setDisabled(True)
            self.search_button.setText('Stopping')
            # kill_child_processes(os.getpid())
            self.thread.quit()
            self.finish_searching()
            self.worker.deleteLater()
            
    def update_searched_result(self, result: str, row: int):
        dbname, _ = self.row_dbname_id_map[row].values()
        if dbname == 'ZINC22':
            splitted = result.strip().split('\n')
            if len(splitted) == 3:
                result = splitted[2].split(',')[1]
            else:
                result = ''
        if not result:
            self.all_table_db_combo[row].setEnabled(True)
            self.db_table.item(row, 4).setText('X')
            db_item = self.db_table.item(row, 1)
            db_item.setFlags(db_item.flags() | Qt.ItemFlag.ItemIsEditable)
            # db_name = self.db_table.item(row, 2)
            # db_name.setFlags(db_name.flags() | Qt.ItemFlag.ItemIsEditable)
            db_item.setForeground(self.regex_color[False])
        else:
            dbname, _ = self.row_dbname_id_map[row].values()
            name = self.db_table.item(row, 2).text()
            self.db_id_struct_map[name] = result
            self.db_table.item(row, 4).setText('✓')
            self.db_table.item(row, 4).setForeground(self.regex_color[True])
            self.all_table_search_ckbox[row].setChecked(False)
            del self.row_dbname_id_map[row]
        self.value += 1
        self.progress_bar.setValue(self.value)
        
    def finish_searching(self):
        self.progress_bar.setValue(self.progress_bar.maximum())
        if self.row_dbname_id_map:
            for row in self.row_dbname_id_map:
                self.all_table_db_combo[row].setEnabled(True)
                self.db_table.item(row, 4).setText('X')
                db_item = self.db_table.item(row, 1)
                db_item.setFlags(db_item.flags() | Qt.ItemFlag.ItemIsEditable)
                self.all_table_search_ckbox[row].setChecked(True)
        self.row_dbname_id_map = {}
        self.searching = False
        self.thread.wait()
        self.worker.deleteLater()
        self.thread.deleteLater()
        self.search_button.setText('Search')
        self.search_button.setEnabled(True)
        
    def accept_changes(self):
        if self.searching:
            self.worker.stop()
            self.thread.quit()
            self.thread.wait()
            self.worker.deleteLater()
            self.thread.deleteLater()
        self.final_dict = {}
        for row in range(self.db_table.rowCount()):
            if self.db_table.item(row, 4).text() == '✓':
                name = self.db_table.item(row, 2).text()
                self.final_dict[name] = self.db_id_struct_map[name]
        self.accept()
        
    def reject_and_close(self):
        if self.searching:
            self.worker.stop()
            # kill_child_processes(os.getpid())
            self.thread.quit()
            self.thread.wait()
            self.worker.deleteLater()
        self.reject()
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Backspace:
            if self.db_table.hasFocus() and self.current_row is not None and not self.searching:
                self.db_table.removeRow(self.current_row)
                self.all_table_db_combo.pop(self.current_row)
                self.all_table_search_ckbox.pop(self.current_row)
                self.current_row = None
        else:
            super().keyPressEvent(event)

class MultithreadDatabaseExtractor(QObject):
    searched = Signal(str, int)
    finished = Signal()
    
    def __init__(self, row_dbname_id_dict: dict):
        super().__init__()
        self.row_dbname_id_dict = row_dbname_id_dict
        self.headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36',
                        'Accept': 'text/html'}
        self.url_map = {'CCD'     : 'https://files.rcsb.org/ligands/download/{id}_ideal.sdf',
                        'BIRD'    : 'https://files.rcsb.org/birds/download/{id}.cif',
                        'PubChem' : 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{id}/SDF?record_type=3d',
                        'ChEBI'   : 'https://www.ebi.ac.uk/chebi/saveStructure.do?defaultImage=true&chebiId={id}&imageId=0',
                        'ChEMBL'  : 'https://www.ebi.ac.uk/chembl/api/data/molecule.sdf?chembl_id={id}',
                        'DrugBank': 'https://go.drugbank.com/structures/small_molecule_drugs/{id}.sdf?type=3d',
                        'ZINC22'  : 'https://cartblanche22.docking.org/substances.csv',
                        'NPASS'   : 'https://bidd.group/NPASS/compound.php?compoundID={id}',
                        'SuperNatural 3.0': 'https://bioinf-applied.charite.de/supernatural_3/molfiles/{id}.mol'}
        self.html_regex = {'NPASS': re.compile(r'<div id="load_molFile" style="display:none">(.*?)</div>', re.DOTALL),
                           'BIRD' : r'{id} InChI.*"(.*)"'}
        self.process_to_input_dict()
        self.is_running = True
        
    def process_to_input_dict(self):
        self.list_of_input_dicts = []
        for row, dbname_id_dict in self.row_dbname_id_dict.items():
            dbname, id = dbname_id_dict.values()
            if dbname == 'ZINC22':
                file = io.BytesIO(('ZINCms000002NiP3\n' + id).encode('utf8'))
                files = {'zinc_ids'     : file,
                         'output_fields': (None, 'zinc_id,smiles'),}
            else:
                files = None
            if dbname == 'ChEBI':
                id = id.split(':')[-1]
                
            url = self.url_map[dbname].format(id=id)
            if dbname == 'BIRD':
                cif_id = 'PRDCC_' + id.split('_')[-1]
                url = self.url_map['BIRD'].format(id=cif_id)
                
            regex = self.html_regex.get(dbname, None)
            if dbname == 'BIRD':
                regex = regex.format(id=id)
            input_dict = {'url'           : url,
                          'row'           : row,
                          'files'         : files,
                          'compiled_regex': regex,}
            self.list_of_input_dicts.append(input_dict)
    
    def run(self):
        def retrieve_chem_from_db(url: str, session: requests.Session, headers: dict, row: int,
                                  files: dict | None=None, retries: int=3, compiled_regex=None):
            for _ in range(retries):
                if not self.is_running:
                    return '', row
                try:
                    response = session.get(url, files=files, headers=headers)
                    if not self.is_running:
                        return '', row, None
                    if response.status_code == 200:
                        structure = response.text
                        if structure:
                            return structure, row, compiled_regex
                        return '', row, None
                    elif response.status_code == 404:
                        return '', row, None
                    else:
                        if files is not None:
                            files['zinc_ids'].seek(0)
                        time.sleep(2)   # wait then retry
                except requests.RequestException as e:
                    if files is not None:
                        files['zinc_ids'].seek(0)
                    time.sleep(2)
            return '', row, None
        
        with BoundedThreadPoolExecutor(max_size=5) as self.executor:
            with requests.Session() as session:
                futures = [self.executor.submit(retrieve_chem_from_db, 
                                                session=session, 
                                                headers=self.headers, **d) for d in self.list_of_input_dicts]
                for f in as_completed(futures):
                    structure, row, regex = f.result()
                    if regex is not None:
                        matched = re.search(regex, structure)
                        if matched is None:
                            structure = ''
                        else:
                            structure = matched.group(1)
                        if structure.startswith('InChI='):
                            structure = Chem.MolToSmiles(Chem.MolFromInchi(structure))
                    if not self.is_running:
                        self.searched.emit('', row)
                        break
                    self.searched.emit(structure, row)
        self.finished.emit()
        
    def stop(self):
        self.is_running = False
        self.executor.shutdown(wait=False, cancel_futures=True)

def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in children:
        process.send_signal(sig)

class ThreadedMaxMinPicker(QObject):
    pickedResult = Signal(dict)
    
    def __init__(self, mol_dict: dict, sample_num: int, seed: int, fp_settings: dict):
        super().__init__()
        self.mol_dict = mol_dict
        self.sample_num = sample_num
        self.seed = seed
        self.fp_settings = fp_settings
    
    def run(self):
        def rdkit_minmax_finder():
            fpgen = retrieve_fp_generator(self.fp_settings)
            mol_fps = [fpgen(Chem.MolFromSmiles(smi_or_sdf) if not '\n' in smi_or_sdf else Chem.MolFromMolBlock(smi_or_sdf)) 
                       for smi_or_sdf in self.mol_dict.values()]
            picker = MaxMinPicker()
            indices = picker.LazyBitVectorPick(mol_fps, len(mol_fps), self.sample_num, seed=self.seed)
            indices = list(indices)
            r = {k: self.mol_dict[k] for i, k in enumerate(self.mol_dict) if i in indices}
            self.pickedResult.emit(r)
        
        rdkit_minmax_finder()

def exclude_thres(sim, left, right):
    return (sim > right) | (sim < left)

def include_thres(sim, left, right):
    return (sim >= left) & (sim <= right)

def read_chunk(db_pth: str, chunk_size: int, dctx: zstd.ZstdDecompressor):
    conn = sqlite3.connect(db_pth)
    db = pd.read_sql('SELECT fp From MolDB', conn, chunksize=chunk_size)
    for row in db:
        yield [CreateFromBinaryText(dctx.decompress(fp)) for fp in row.fp]

def calculate_db_similarity(target_name: str, target_smi_or_molblock: str, db_pths: list,
                            chunk_size: int, desalt_bool: bool, fp_settings: dict,
                            is_picking):
    if not is_picking.value:
        return None, None, None
    fpgen = retrieve_fp_generator(fp_settings)
    sim_func = retrieve_similarity_method(fp_settings['sim'], True)
    dctx = zstd.ZstdDecompressor()
    if '\n' in target_smi_or_molblock:
        mol = Chem.MolFromMolBlock(target_smi_or_molblock)
    else:
        mol = Chem.MolFromSmiles(target_smi_or_molblock)
    if mol is None:
        return None, target_name, target_smi_or_molblock
    if desalt_bool:
        remover = SaltRemover()
        mol = remover.StripMol(mol)
    if mol.GetNumAtoms() == 0:
        return None, target_name, target_smi_or_molblock
    target_fps = fpgen(mol)
    max_sim = -1
    with ThreadPoolExecutor() as executor:
        future_chunks = [executor.submit(read_chunk, pth, chunk_size, dctx) for pth in db_pths]
        for future in as_completed(future_chunks):
            for chunk in future.result():
                # only store the maximum of each chunk to save memory
                max_sim = max(max_sim, max(sim_func(target_fps, chunk)))
    return max_sim, target_name, target_smi_or_molblock

class MultiProcessDBSimilarityPicker(QObject):
    pickedResult = Signal(dict)
    pickingSuccess = Signal(str)
    pickingFail = Signal(str)
    pickingStopped = Signal()
    
    def __init__(self, mol_dict: dict, db_list: str, sim_type: str,
                 sim_thres: tuple, desalt_bool: bool, fp_settings: dict):
        super().__init__()
        self.mol_dict = mol_dict
        self.db_list = db_list
        if sim_type == 'Include':
            self.condition_func = include_thres
        else:
            self.condition_func = exclude_thres
        self.left_num = round(sim_thres[0], 2)
        self.right_num = round(sim_thres[1], 2)
        self.chunk_size = 2_000
        self.final_dict = {}
        self.desalt_bool = desalt_bool
        self.total_mols = len(self.mol_dict)
        self.max_len = len(str(self.total_mols))
        self.curr_mol = 0
        self.max_name_length = max(len(i) for i in mol_dict)
        self.futures = []
        self.fp_settings = fp_settings
    
    def run(self):
        self.manager = Manager()
        self.is_picking = self.manager.Value('b', True)
        self.curr_picking = True
        self.executor = ProcessPoolExecutor()
        
        for name, smi_or_molblock in self.mol_dict.items():
            future = self.executor.submit(calculate_db_similarity,
                                          name,
                                          smi_or_molblock,
                                          self.db_list,
                                          self.chunk_size,
                                          self.desalt_bool,
                                          self.fp_settings,
                                          self.is_picking)
            future.add_done_callback(self.process_future)
            self.futures.append(future)
    
    def process_future(self, future):
        try:
            sim, name, smi = future.result()
            self.curr_mol += 1
            if sim is not None:
                if self.curr_picking:
                    if self.condition_func(sim, self.left_num, self.right_num):
                        self.final_dict[name] = smi
                        self.pickingSuccess.emit(f'{name:{self.max_name_length}}    kept.   Similarity = {sim:.2f} ({self.curr_mol:{self.max_len}} / {self.total_mols})')
                    else:
                        self.pickingFail.emit(f'{name:{self.max_name_length}} removed.   Similarity = {sim:.2f} ({self.curr_mol:{self.max_len}} / {self.total_mols})')
            else:
                if name is not None:
                    self.pickingFail.emit(f'{name:{self.max_name_length}} removed. Failed to read mol. ({self.curr_mol:{self.max_len}} / {self.total_mols})')
        except Exception as e:
            pass
            
        if all(f.done() for f in self.futures):
            if self.curr_picking:
                self.pickedResult.emit(self.final_dict)
    
    @Slot()
    def stop(self):
        self.curr_picking = False
        self.is_picking.value = False
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.manager.shutdown()
        self.pickingStopped.emit()
