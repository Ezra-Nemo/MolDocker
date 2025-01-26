import json
from functools import partial
from PySide6.QtWidgets import (QLabel, QVBoxLayout, QDialog, QComboBox,
                               QDialogButtonBox, QGridLayout, QCheckBox, QSpinBox,
                               QTableWidget, QTableWidgetItem, QHeaderView, QHBoxLayout,
                               QPushButton, QWidget, QFileDialog)
from PySide6.QtCore import Qt

from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import cDataStructs
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.rdmolops import LayeredFingerprint, PatternFingerprint
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint

functions = {'Morgan'             : rdFingerprintGenerator.GetMorganGenerator,
             'RDKit'              : rdFingerprintGenerator.GetRDKitFPGenerator,
             'Topological Torsion': rdFingerprintGenerator.GetTopologicalTorsionGenerator,
             'Atom Pair'          : rdFingerprintGenerator.GetAtomPairGenerator,
             'Layered'            : LayeredFingerprint,
             'Pattern'            : PatternFingerprint,
             'Avalon'             : pyAvalonTools.GetAvalonFP,
             'MACCS'              : GetMACCSKeysFingerprint,}
atom_bond_gen_map = {'Morgan atom generator'        : rdFingerprintGenerator.GetMorganAtomInvGen,
                     'Morgan Feature atom generator': rdFingerprintGenerator.GetMorganFeatureAtomInvGen,
                     'Morgan bond generator'        : rdFingerprintGenerator.GetMorganBondInvGen,
                     'Atom Pair atom generator'     : rdFingerprintGenerator.GetAtomPairAtomInvGen}
atom_bond_gen = {
    rdFingerprintGenerator.GetMorganAtomInvGen  : {'includeRingMembership': True,},
    rdFingerprintGenerator.GetMorganBondInvGen  : {'useBondTypes'    : True ,
                                                   'includeChirality': False,}, # remember to change it to "useChirality"!
    rdFingerprintGenerator.GetAtomPairAtomInvGen: {'includeChirality': False,},
    }

generator_types = ['Morgan', 'RDKit', 'Topological Torsion', 'Atom Pair']

def retrieve_similarity_method(sim_name: str, is_bulk: bool=False):
    sim_name = sim_name.replace('-', '')
    if is_bulk:
        return getattr(cDataStructs, f'Bulk{sim_name}Similarity')
    else:
        return getattr(cDataStructs, f'{sim_name}Similarity')

def retrieve_fp_generator(fp_setting_dict: dict):
    m = fp_setting_dict['method']
    f = functions[m]
    final_params = {}
    for k, v in fp_setting_dict['params'].items():
        if isinstance(v, str):
            v_f = atom_bond_gen_map[v]
            allowed_params = dict(atom_bond_gen[v_f])
            for gen_param in list(allowed_params):
                allowed_params[gen_param] = fp_setting_dict['params'][gen_param]
            if v == 'Morgan bond generator':
                allowed_params['useChirality'] = allowed_params.pop('includeChirality')
            final_params[k] = v_f(**allowed_params)
        else:
            final_params[k] = v
    if m in generator_types:
        return lambda m: f(**final_params).GetFingerprint(m)
    else:
        final = partial(f, **final_params)
        # just so I don't need to use f(mol=m) when calling this function, because my old code is f(m)
        return lambda m: final(mol=m)

class FingerprintSettingDialog(QDialog):
    def __init__(self, fp_settings: dict):
        super().__init__()
        self.fp_settings = fp_settings
        self.fp_name_list = ['Morgan', 'RDKit', 'Topological Torsion', 'Atom Pair', 'Layered', 'Pattern', 'Avalon', 'MACCS']
        self.sim_method_list = ['Tanimoto', 'Dice', 'Braun-Blanquet', 'Cosine',
                                'Kulczynski', 'McConnaughey', 'Rogot-Goldberg',
                                'Russel', 'Sokal', 'Tversky']
        self.name_to_func_map = {'Morgan'             : rdFingerprintGenerator.GetMorganGenerator,
                                 'RDKit'              : rdFingerprintGenerator.GetRDKitFPGenerator,
                                 'Topological Torsion': rdFingerprintGenerator.GetTopologicalTorsionGenerator,
                                 'Atom Pair'          : rdFingerprintGenerator.GetAtomPairGenerator,
                                 'Layered'            : LayeredFingerprint,
                                 'Pattern'            : PatternFingerprint,
                                 'Avalon'             : pyAvalonTools.GetAvalonFP,
                                 'MACCS'              : GetMACCSKeysFingerprint,}
        self.atom_bond_gen_name_map = {'Morgan atom generator'        : rdFingerprintGenerator.GetMorganAtomInvGen,
                                       'Morgan Feature atom generator': rdFingerprintGenerator.GetMorganFeatureAtomInvGen,
                                       'Morgan bond generator'        : rdFingerprintGenerator.GetMorganBondInvGen,
                                       'Atom Pair atom generator'     : rdFingerprintGenerator.GetAtomPairAtomInvGen,}
        self.fp_param_map = {
            rdFingerprintGenerator.GetMorganGenerator            : {'fpSize'                 : 16384,
                                                                    'radius'                 : 3,
                                                                    'countSimulation'        : False,
                                                                    'includeChirality'       : False,
                                                                    'useBondTypes'           : True,
                                                                    'onlyNonzeroInvariants'  : False,
                                                                    'includeRingMembership'  : True,
                                                                    'atomInvariantsGenerator': ['Morgan atom generator',
                                                                                                'Morgan Feature atom generator'],
                                                                    'bondInvariantsGenerator': ['Morgan bond generator'],},
            rdFingerprintGenerator.GetRDKitFPGenerator           : {'fpSize'           : 2048,
                                                                    'minPath'          : 1,
                                                                    'maxPath'          : 7,
                                                                    'useHs'            : True,
                                                                    'branchedPaths'    : True,
                                                                    'useBondOrder'     : True,
                                                                    'countSimulation'  : False,
                                                                    'numBitsPerFeature': 2,},
            rdFingerprintGenerator.GetTopologicalTorsionGenerator: {'fpSize'          : 2048,
                                                                    'includeChirality': False,
                                                                    'torsionAtomCount': 4,
                                                                    'countSimulation' : True,},
            rdFingerprintGenerator.GetAtomPairGenerator          : {'fpSize'                 : 2048,
                                                                    'minDistance'            : 1,
                                                                    'maxDistance'            : 30,
                                                                    'includeChirality'       : False,
                                                                    'use2D'                  : True,
                                                                    'countSimulation'        : True,
                                                                    'atomInvariantsGenerator': ['Atom Pair atom generator'],},
            LayeredFingerprint                                   : {'fpSize'       : 2048,
                                                                    'minPath'      : 1,
                                                                    'maxPath'      : 7,
                                                                    'branchedPaths': True,},
            PatternFingerprint                                   : {'fpSize'              : 2048,
                                                                    'tautomerFingerprints': False,},
            pyAvalonTools.GetAvalonFP                            : {'nBits'    : 512,
                                                                    'isQuery'  : False,
                                                                    'resetVect': False},
            GetMACCSKeysFingerprint: {},
            }
        self.atom_bond_gen_param_map = {
            rdFingerprintGenerator.GetMorganAtomInvGen  : {'includeRingMembership': True,},
            rdFingerprintGenerator.GetMorganBondInvGen  : {'useBondTypes'    : True ,
                                                           'includeChirality': False,}, # remember to change it to "useChirality"!
            rdFingerprintGenerator.GetAtomPairAtomInvGen: {'includeChirality': False,},
            }
        self.setup_ui()
        screen = self.screen()
        screen_size = screen.availableGeometry()
        self.resize(screen_size.width() * 0.3, screen_size.height() * 0.6)
        
    def setup_ui(self):
        overall_layout = QVBoxLayout()
        
        fp_name_layout = QGridLayout()
        sim_label = QLabel('<b>Similarity Metric :</b>')
        self.sim_combobox = QComboBox()
        self.sim_combobox.addItems(self.sim_method_list)
        self.sim_combobox.setCurrentText(self.fp_settings['sim'])
        fp_label = QLabel('<b>Fingerprint Method :</b>')
        self.fp_combo_box = QComboBox()
        self.fp_combo_box.addItems(self.fp_name_list)
        self.fp_combo_box.setCurrentText(self.fp_settings['method'])
        self.fp_combo_box.currentTextChanged.connect(self.fill_table_with_params)
        fp_name_layout.addWidget(sim_label, 0, 0)
        fp_name_layout.addWidget(self.sim_combobox, 0, 1)
        fp_name_layout.addWidget(fp_label, 1, 0)
        fp_name_layout.addWidget(self.fp_combo_box, 1, 1)
        
        setting_layout = QVBoxLayout()
        setting_label = QLabel('<b>Fingerprint Parameters :</b>')
        self.setting_table = QTableWidget()
        self.setting_table.setColumnCount(2)
        self.setting_table.setHorizontalHeaderLabels(['Parameter', 'Value'])
        self.setting_table.verticalHeader().setHidden(True)
        self.fill_table_with_params()
        setting_layout.addWidget(setting_label)
        setting_layout.addWidget(self.setting_table)
        
        button_layout = QHBoxLayout()
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        button_box = QDialogButtonBox(QBtn)
        button_box.accepted.connect(self.accept_changes)
        button_box.rejected.connect(self.reject)
        left_button_layout = QHBoxLayout()
        left_button_widget = QWidget()
        left_button_layout.setContentsMargins(0, 0, 0, 0)
        left_button_widget.setLayout(left_button_layout)
        export_button = QPushButton('Export')
        export_button.clicked.connect(self.export_curr_dict)
        import_button = QPushButton('Import')
        import_button.clicked.connect(self.import_json_dict)
        left_button_layout.addWidget(export_button)
        left_button_layout.addWidget(import_button)
        button_layout.addWidget(left_button_widget, alignment=Qt.AlignmentFlag.AlignLeft)
        button_layout.addWidget(button_box, alignment=Qt.AlignmentFlag.AlignRight)
        
        overall_layout.addLayout(fp_name_layout)
        overall_layout.addSpacing(15)
        overall_layout.addLayout(setting_layout)
        overall_layout.addLayout(button_layout)
        self.setLayout(overall_layout)
    
    def export_curr_dict(self):
        save_file, _ = QFileDialog.getSaveFileName(self, 'Export Fingerprint & Similarity Settings', '', 'JSON (*.json)')
        if save_file:
            params = {}
            for row in range(self.setting_table.rowCount()):
                param_name = self.setting_table.item(row, 0).text()
                widget = self.setting_table.cellWidget(row, 1)
                if isinstance(widget, QComboBox):
                    params[param_name] = widget.currentText()
                elif isinstance(widget, QCheckBox):
                    params[param_name] = widget.isChecked()
                elif isinstance(widget, QSpinBox):
                    params[param_name] = widget.value()
            curr_settings = {'method': self.fp_combo_box.currentText(),
                             'params': params,
                             'sim'   : self.sim_combobox.currentText()}
            with open(save_file, 'w') as f:
                json.dump(curr_settings, f, indent=4)
    
    def import_json_dict(self):
        file, _ = QFileDialog.getOpenFileName(self, 'Import Fingerprint & Similarity Settings', '', 'JSON (*.json)')
        if file:
            with open(file, 'r') as f:
                fp_settings = json.load(f)
            self.fp_settings = fp_settings
            self.fp_combo_box.setCurrentText(fp_settings['method'])
            self.sim_combobox.setCurrentText(fp_settings['sim'])
            self.fill_table_with_params()
    
    def fill_table_with_params(self):
        fp_method = self.fp_combo_box.currentText()
        fp_func = self.name_to_func_map[fp_method]
        if fp_method == self.fp_settings['method']:
            fp_params = self.fp_settings['params']
        else:
            fp_params = self.fp_param_map[fp_func]
        self.setting_table.clearContents()
        self.setting_table.setRowCount(len(fp_params))
        
        for row, (param_name, value) in enumerate(fp_params.items()):
            name_item = QTableWidgetItem(param_name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.setting_table.setItem(row, 0, name_item)
            if isinstance(value, str):
                widget = QComboBox()
                all_options = self.fp_param_map[fp_func][param_name]
                widget.addItems(all_options)
                widget.setCurrentText(value)
            elif isinstance(value, list):
                widget = QComboBox()
                widget.addItems(value)
                widget.setCurrentIndex(0)
            elif isinstance(value, bool):
                widget = QCheckBox()
                widget.setChecked(value)
            elif isinstance(value, int):
                widget = QSpinBox()
                widget.setRange(1, 2 ** 16)
                widget.setValue(value)
            self.setting_table.setCellWidget(row, 1, widget)
        
        self.setting_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
    def accept_changes(self):
        params = {}
        for row in range(self.setting_table.rowCount()):
            param_name = self.setting_table.item(row, 0).text()
            widget = self.setting_table.cellWidget(row, 1)
            if isinstance(widget, QComboBox):
                params[param_name] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                params[param_name] = widget.isChecked()
            elif isinstance(widget, QSpinBox):
                params[param_name] = widget.value()
        self.new_settings = {'method': self.fp_combo_box.currentText(),
                             'params': params,
                             'sim'   : self.sim_combobox.currentText()}
        self.accept()
