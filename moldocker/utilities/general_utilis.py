import os, io, re, copy, time, lzma, json
import pickle, signal, psutil, aiohttp, asyncio, sqlite3
import tempfile, requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import zstandard as zstd

from urllib import parse
from threading import Semaphore
from multiprocessing import Manager
from PySide6.QtWidgets import (QLabel, QVBoxLayout, QPushButton, QSizePolicy,
                               QDialog, QComboBox, QDialogButtonBox, QFrame,
                               QDoubleSpinBox, QGridLayout, QCheckBox, QSpinBox,
                               QListWidget, QHBoxLayout, QToolButton,
                               QListWidgetItem, QApplication, QLineEdit,
                               QTableWidget, QTableWidgetItem, QHeaderView,
                               QSpacerItem, QWidget, QProgressBar, QMenu,
                               QFileDialog, QMessageBox, QTreeWidget, QTreeWidgetItem,
                               QProgressDialog, QColorDialog, QSplitter,
                               QScrollArea, QRadioButton, QTabBar,
                               QStyle, QStyleOptionTab, QStylePainter, QToolBox,
                               QAbstractButton, QTableView, QTreeView, QStackedWidget,
                               QAbstractItemView, QButtonGroup, QStyledItemDelegate, QTextEdit, QToolTip)
from PySide6.QtCore import (Qt, QObject, Signal, Slot, QSize, QThread, QRect, QPoint, QAbstractTableModel,
                            QModelIndex, QEasingCurve, QParallelAnimationGroup, QPropertyAnimation,
                            QAbstractAnimation, QSortFilterProxyModel, QRegularExpression, QEvent)
from PySide6.QtGui import (QColor, QKeyEvent, QPixmap, QAction, QShortcut, QIcon,
                           QRegularExpressionValidator, QCursor, QFont,
                           QStandardItem, QStandardItemModel, QPainter,
                           QFontMetrics, QKeySequence)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtSvgWidgets import QSvgWidget

from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, QED
from rdkit.Chem.Draw.rdMolDraw2D import SetDarkMode
from rdkit.DataStructs.cDataStructs import CreateFromBinaryText

from .utilis import RDKitMolCreate, PDBQTMolecule, clean_pdb, fix_and_convert, fix_pdb_missing_atoms, process_rigid_flex
from .protein_utilis import PDBQTCombiner, PDBEditor, MDMFileProcessor
from .browser_utilis import (ProteinCoordinateBrowserWidget, FragmentPlotBrowserWidget,
                             SearchDBBrowserWindow, BrowserWithTabs)
from .fingerprint_utilis import retrieve_similarity_method, retrieve_fp_generator
from .MacFrag import MacFrag
from PIL import Image
from PIL.ImageQt import ImageQt

from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor

vina_eng_compiled = re.compile(r'REMARK VINA RESULT:.*')
vina_score_only_compiled = re.compile(r'REMARK VINA RESULT:\s+(-?\d+\.\d+)')
sdf_match_eng_rmsd_compiled = re.compile(r'-?\d+\.\d+')
vina_mdl_compiled = re.compile(r'MODEL [0-9]+\n((.|\n)*?)ENDMDL\n')
sdf_regex_list = [re.compile(r'ENERGY=.*'), re.compile(r'Score:.*')]
smina_eng_compiled = re.compile(r'>\s*<minimizedAffinity>(?:\s*\(\d+\))?\s?\n\s?(-?\d+\.\d+)')
gnina_eng_compiled = re.compile(r'REMARK VINA RESULT:(.*)\nREMARK CNNscore\s(-?\d+\.\d+)\nREMARK CNNaffinity\s(-?\d+\.\d+)')

chem_prop_to_full_name_map = {'mw'  : 'Molecular Weight'        ,
                              'hbd' : 'Hydrogen Bond Donors', 'hba' : 'Hydrogen Bond Acceptors' ,
                              'logp': 'LogP'                , 'tpsa': 'Topological Polar Surface Area',
                              'rb'  : 'Rotatable Bonds'     , 'nor' : 'Number of Rings'         ,
                              'fc'  : 'Formal Charge'       , 'nha' : 'Number of Heavy Atoms'   ,
                              'mr'  : 'Molar Refractivity'  , 'na'  : 'Number of Atoms'         ,
                              'QED' : 'QED'}

def svgtopng(svg_bytes: str, save_pth: str):
    renderer = QSvgRenderer(svg_bytes)
    pixmap = QPixmap(600, 600)  # TODO: Allow custom size & DPI
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    pixmap.save(save_pth, 'PNG')

class SettingDialog(QDialog):
    def __init__(self, param_dict: dict, mode: str, pdbqt_editor: PDBEditor):
        super().__init__()
        self.pdbqt_editor = pdbqt_editor
        self.fpocket_dialog = None
        self.hetatm_dialog = None
        self.chain_seq_dialog = {}
        self.chain_highlight_text = {}
        self.initUI(param_dict, mode)
        
    def initUI(self, param_dict: dict, mode: str):
        self.setWindowTitle('MolDocker Settings')
        self.overall_layout = QHBoxLayout()
        radio_layout = QVBoxLayout()
        pocket_layout = QGridLayout()
        setting_layout = QVBoxLayout()
        setting_widget = QWidget()
        setting_layout.setContentsMargins(0, 0, 0, 0)
        self.color_theme = mode
        
        radio_frame = QFrame()
        radio_frame.setFrameShape(QFrame.Shape.StyledPanel)
        radio_frame.setLineWidth(4)
        radio_frame.setLayout(radio_layout)
        
        pocket_frame = QFrame()
        pocket_frame.setFrameShape(QFrame.Shape.StyledPanel)
        pocket_frame.setLineWidth(4)
        pocket_frame.setLayout(pocket_layout)
        
        center_xyz_frame = QFrame()
        center_xyz_frame.setFrameShape(QFrame.Shape.StyledPanel)
        center_xyz_frame.setLineWidth(4)
        center_xyz_layout = QGridLayout(center_xyz_frame)
        
        width_xyz_frame = QFrame()
        width_xyz_frame.setFrameShape(QFrame.Shape.StyledPanel)
        width_xyz_frame.setLineWidth(4)
        width_xyz_layout = QGridLayout(width_xyz_frame)
        
        other_settings_frame = QFrame()
        other_settings_frame.setFrameShape(QFrame.Shape.StyledPanel)
        other_settings_frame.setLineWidth(4)
        other_settings_layout = QGridLayout(other_settings_frame)
        
        ### Radio for click swapping ###
        click_label = QLabel('<b>Select Mode :</b>')
        radio_btn_layout = QHBoxLayout()
        radio_btn_layout.setContentsMargins(0, 0, 0, 0)
        self.center_radio = QRadioButton('Center')
        self.flex_radio = QRadioButton('Flexible')
        self.disable_radio = QRadioButton('Disable')
        self.disable_radio.setChecked(True)
        radio_btn_layout.addWidget(self.center_radio)
        radio_btn_layout.addWidget(self.flex_radio)
        radio_btn_layout.addWidget(self.disable_radio)
        radio_layout.addWidget(click_label)
        radio_layout.addLayout(radio_btn_layout)
        
        ### FPocket Dialog ###
        fpocket_label = QLabel('<b>FPocket Table :</b>')
        fpocket_button = QPushButton('Show')
        fpocket_button.setStyleSheet(f'font-weight: bold')
        fpocket_button.setEnabled(bool(param_dict['fpocket']))
        fpocket_button.clicked.connect(self.show_fpocket_dialog)
        pocket_layout.addWidget(fpocket_label, 0, 0)
        pocket_layout.addWidget(fpocket_button, 0, 1)
        
        ### HETATM Dialog ###
        hetatm_label = QLabel('<b>Heteroatom Table :</b>')
        hetatm_button = QPushButton('Show')
        hetatm_button.setStyleSheet(f'font-weight: bold')
        hetatm_button.setEnabled(bool(param_dict['hetatm']))
        hetatm_button.clicked.connect(self.show_hetatm_dialog)
        pocket_layout.addWidget(hetatm_label, 1, 0)
        pocket_layout.addWidget(hetatm_button, 1, 1)
        
        ### Proten Bounding Box ###
        protein_box_label = QLabel('<b>Protein Box :</b>')
        protein_box_button = QPushButton('Apply')
        protein_box_button.setStyleSheet(f'font-weight: bold')
        protein_box_button.setEnabled(bool(self.pdbqt_editor))
        protein_box_button.clicked.connect(self.apply_blind_docking_box)
        pocket_layout.addWidget(protein_box_label, 2, 0)
        pocket_layout.addWidget(protein_box_button, 2, 1)
        
        ### XYZ Center ###
        self.xyz_center_widgets = {}
        center_label = QLabel('<b>Center :<b/>')
        center_setting_layout = QHBoxLayout()
        center_setting_layout.setContentsMargins(0, 0, 0, 0)
        center_color = QPushButton('Color')
        center_color.clicked.connect(self.change_center_color)
        self.center_visibility = QCheckBox('Visible')
        self.center_visibility.setChecked(True)
        self.center_visibility.stateChanged.connect(self.set_center_visibility)
        center_setting_layout.addWidget(center_color)
        center_setting_layout.addWidget(self.center_visibility)
        self.center_color = param_dict['center_color']
        center_xyz_layout.addWidget(center_label, 0, 0)
        center_xyz_layout.addLayout(center_setting_layout, 0, 1)
        center_regex_validator = QRegularExpressionValidator('[-]?([0-9]*[.])?[0-9]+')
        xyz_color_map = {'x': '#E57373', 'y': '#4CAF50', 'z': '#42A5F5'}
        for idx, name in enumerate(['x', 'y', 'z'], 1):
            pos_label = QLabel(f'{name.upper()} :')
            pos_label.setStyleSheet(f'color: {xyz_color_map[name]}; font-weight: bold')
            pos_lineedit = ChangeValueLineEdit()
            pos_lineedit.setValidator(center_regex_validator)
            value = param_dict['dock_center'][name]
            if value is not None:
                pos_lineedit.setText(str(value))
            pos_lineedit.textChanged.connect(self.check_center_value)
            center_xyz_layout.addWidget(pos_label   , idx, 0, Qt.AlignmentFlag.AlignRight)
            center_xyz_layout.addWidget(pos_lineedit, idx, 1, Qt.AlignmentFlag.AlignRight)
            self.xyz_center_widgets[name] = {'LineEdit' : pos_lineedit}
        
        ### XYZ Width ###
        self.xyz_width_widgets = {}
        width_label = QLabel('<b>Box :<b/>')
        width_color = QPushButton('Color')
        width_setting_layout = QHBoxLayout()
        width_setting_layout.setContentsMargins(0, 0, 0, 0)
        self.width_visibility = QCheckBox('Visible')
        self.width_visibility.setChecked(True)
        self.width_visibility.stateChanged.connect(self.set_box_visibility)
        width_color.clicked.connect(self.change_width_color)
        self.width_color = param_dict['width_color']
        width_setting_layout.addWidget(width_color)
        width_setting_layout.addWidget(self.width_visibility)
        width_xyz_layout.addWidget(width_label, 0, 0, Qt.AlignmentFlag.AlignLeft)
        width_xyz_layout.addLayout(width_setting_layout, 0, 1)
        width_regex_validator = QRegularExpressionValidator('([0-9]*[.])?[0-9]+')
        for idx, name in enumerate(['x', 'y', 'z'], 1):
            pos_label = QLabel(f'{name.upper()} :')
            pos_label.setStyleSheet(f'color: {xyz_color_map[name]}; font-weight: bold')
            pos_lineedit = ChangeValueLineEdit(None, False)
            pos_lineedit.setValidator(width_regex_validator)
            value = param_dict['dock_width'][name]
            if value is not None:
                pos_lineedit.setText(str(value))
            pos_lineedit.textChanged.connect(self.check_center_value)
            width_xyz_layout.addWidget(pos_label   , idx, 0, Qt.AlignmentFlag.AlignRight)
            width_xyz_layout.addWidget(pos_lineedit, idx, 1, Qt.AlignmentFlag.AlignRight)
            self.xyz_width_widgets[name] = {'LineEdit' : pos_lineedit}
        
        ### Other Options ###
        other_settings_label = QLabel('<b>Others :</b>')
        exhaustiveness_label = QLabel('Exhaustiveness :')
        self.exhaustiveness_spinbox = QSpinBox()
        self.exhaustiveness_spinbox.setValue(param_dict['exhaustiveness'])
        self.exhaustiveness_spinbox.setRange(1, int(1e6))
        eval_pose_label = QLabel('Evaluated Poses :')
        self.eval_pose_spinbox = QSpinBox()
        self.eval_pose_spinbox.setValue(param_dict['eval_poses'])
        self.eval_pose_spinbox.setRange(1, int(1e6))
        
        other_settings_layout.addWidget(other_settings_label, 0, 0)
        other_settings_layout.addWidget(exhaustiveness_label, 1, 0, Qt.AlignmentFlag.AlignRight)
        other_settings_layout.addWidget(self.exhaustiveness_spinbox, 1, 1, Qt.AlignmentFlag.AlignRight)
        other_settings_layout.addWidget(eval_pose_label, 2, 0, Qt.AlignmentFlag.AlignRight)
        other_settings_layout.addWidget(self.eval_pose_spinbox, 2, 1, Qt.AlignmentFlag.AlignRight)
        
        ### Button Layout ###
        button_layout = QGridLayout()
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.button_box = QDialogButtonBox(QBtn)
        self.button_box.accepted.connect(self.accept_changes)
        self.button_box.rejected.connect(self.reject_changes)
        self.export_setting_btn = QPushButton('Export')
        self.export_setting_btn.clicked.connect(self.export_settings_with_pickle)
        
        button_layout.addWidget(self.export_setting_btn, 0, 0)
        button_layout.addWidget(self.button_box, 0, 1, Qt.AlignmentFlag.AlignRight)
        
        setting_layout.addWidget(radio_frame, 1)
        setting_layout.addWidget(pocket_frame, 1)
        setting_layout.addWidget(center_xyz_frame, 4)
        setting_layout.addWidget(width_xyz_frame, 4)
        setting_layout.addWidget(other_settings_frame, 2)
        setting_layout.addLayout(button_layout)
        
        ### Browser Layout ###
        self.browser_widget = ProteinCoordinateBrowserWidget(mode)
        self.browser_widget.browser.loadFinished.connect(self.load_pdbqt_string_into_browser)
        self.browser_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.browser_widget.signals.send_coordinates.connect(self.set_clicked_position)
        
        setting_widget.setLayout(setting_layout)
        setting_widget.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        
        ### Protein Editing Layout ###
        edit_label = QLabel('<b>Protein Editing :</b>')
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_area.setWidget(scroll_content)
        reorient_btn = QPushButton('Reorient')
        reorient_btn.setEnabled(bool(self.pdbqt_editor))
        reorient_btn.clicked.connect(self.reorient_to_boundingbox)
        update_btn = QPushButton('Update')
        update_btn.setEnabled(bool(self.pdbqt_editor))
        update_btn.clicked.connect(self.update_protein_display)
        
        edit_label_button_layout = QHBoxLayout()
        edit_label_button_layout.addWidget(edit_label)
        edit_label_button_layout.addWidget(reorient_btn, alignment=Qt.AlignmentFlag.AlignRight)
        edit_label_button_layout.addWidget(update_btn, alignment=Qt.AlignmentFlag.AlignRight)
        edit_label_button_layout.setContentsMargins(0, 0, 0, 0)
        
        edit_layout = QVBoxLayout()
        edit_layout.setContentsMargins(0, 0, 0, 0)
        edit_widget = QWidget()
        edit_widget.setLayout(edit_layout)
        display_layout = QVBoxLayout(scroll_content)
        self.protein_editing_dict = {}
        if self.pdbqt_editor:
            for chain in self.pdbqt_editor.pdbqt_chain_dict:
                layout = QGridLayout()
                frame = QFrame()
                frame.setFrameShape(QFrame.Shape.StyledPanel)
                frame.setLineWidth(4)
                frame.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
                frame.setLayout(layout)
                
                curr_displayed = self.pdbqt_editor.convert_to_range_text(chain)
                ck_state = False if curr_displayed is None else True
                
                ckbox_label = QLabel('<b>Chain '+chain+'</b>')
                ckbox = QCheckBox()
                ckbox.setChecked(ck_state)
                ckbox.stateChanged.connect(lambda state, n=chain: self.update_edit_lineedit(state, n))
                
                display_label = QLabel('<b>Position</b>')
                display_lineedit = QLineEdit()
                display_lineedit.setText('' if not ck_state else curr_displayed)
                display_lineedit.setEnabled(ck_state)
                display_lineedit.setMinimumWidth(100)
                
                display_seq_btn = QPushButton('>>')
                display_seq_btn.setStyleSheet('QPushButton { font-weight: bold; }')
                display_seq_btn.clicked.connect(lambda _, c=chain: self.show_sequence_dialog(c))
                self.chain_seq_dialog[chain] = None
                self.chain_highlight_text[chain] = None
                
                layout.addWidget(ckbox_label, 0, 0, Qt.AlignmentFlag.AlignLeft)
                layout.addWidget(ckbox, 0, 1, Qt.AlignmentFlag.AlignLeft)
                layout.addWidget(display_seq_btn, 0, 2, Qt.AlignmentFlag.AlignRight)
                layout.addWidget(display_label, 1, 0, Qt.AlignmentFlag.AlignLeft)
                layout.addWidget(display_lineedit, 1, 1, 1, 2, Qt.AlignmentFlag.AlignLeft)
                
                display_layout.addWidget(frame, alignment=Qt.AlignmentFlag.AlignTop)
                
                self.protein_editing_dict[chain] = {'chain_checkbox'  : ckbox,
                                                    'display_label'   : display_label,
                                                    'display_lineedit': display_lineedit}
        display_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        
        edit_layout.addLayout(edit_label_button_layout)
        edit_layout.addWidget(scroll_area)
        
        ### Flexible layout ###
        flex_layout = QVBoxLayout()
        flex_layout.setContentsMargins(0, 5, 0, 0)
        flex_widget = QWidget()
        flex_widget.setLayout(flex_layout)
        flex_label = QLabel('<b>Flexible Position :</b>')
        self.flex_scroll_area = QScrollArea()
        self.flex_scroll_area.setWidgetResizable(True)
        flex_scroll_content = QWidget()
        self.flex_scroll_area.setWidget(flex_scroll_content)
        self.flex_scroll_area.verticalScrollBar().rangeChanged.connect(self.update_flex_scrollbar_position)
        flex_add_button = QPushButton('Add')
        flex_add_button.clicked.connect(self.add_flex_options)
        flex_add_button.setEnabled(bool(self.pdbqt_editor))
        self.flex_autosearch_btn = QPushButton('Auto')
        self.flex_autosearch_btn.clicked.connect(self.auto_search_flexible)
        self.flex_autosearch_btn.setEnabled(self.check_center_and_box())
        
        label_button_layout = QHBoxLayout()
        label_button_layout.addWidget(flex_label)
        label_button_layout.addWidget(self.flex_autosearch_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        label_button_layout.addWidget(flex_add_button, alignment=Qt.AlignmentFlag.AlignRight)
        label_button_layout.setContentsMargins(0, 0, 0, 0)
        
        self.flex_select_layout = QVBoxLayout(flex_scroll_content)
        self.protein_flex_dict = {}
        has_flex = False
        if self.pdbqt_editor:
            for chain, df in self.pdbqt_editor.pdbqt_chain_dict.items():
                if df['Flexible'].sum():
                    has_flex = True
                    all_flex_res = df.index[df['Flexible'] == True].to_list()
                    for flex_res in all_flex_res:
                        self.add_flex_options(chain, str(flex_res))
        if has_flex:
            self.browser_widget.browser.loadFinished.connect(self.update_protein_sidechain)
        self.flex_select_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        flex_layout.addLayout(label_button_layout)
        flex_layout.addWidget(self.flex_scroll_area)
        
        ### Combine Edit & Flex widget ###
        edit_flex_splitter = QSplitter(Qt.Orientation.Vertical)
        edit_flex_splitter.addWidget(edit_widget)
        edit_flex_splitter.addWidget(flex_widget)
        edit_flex_splitter.setStretchFactor(0, 3)
        edit_flex_splitter.setStretchFactor(1, 2)
        
        ### Overall ###
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.browser_widget)
        splitter.addWidget(edit_flex_splitter)
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)
        
        self.overall_layout.addWidget(setting_widget)
        self.overall_layout.addWidget(splitter, 1)
        
        self.setLayout(self.overall_layout)
        
        self.vina_param = copy.deepcopy(param_dict)
        
        screen = self.screen()
        screen_size = screen.availableGeometry()
        self.resize(screen_size.width() * 0.85, screen_size.height() * 0.7)
        
    def update_edit_lineedit(self, state: bool, chain: str):
        for name, widget in self.protein_editing_dict[chain].items():
            if name != 'chain_checkbox':
                widget.setEnabled(state)
        self.update_protein_display()
    
    def apply_blind_docking_box(self):
        xyz_cen, xyz_wid = self.pdbqt_editor.calculate_protein_bounding_box()
        for i, ax in enumerate(self.xyz_center_widgets):
                w = self.xyz_center_widgets[ax]['LineEdit']
                w.blockSignals(True)
                w.setText(f'{xyz_cen[i]}')
                w.blockSignals(False)
                w = self.xyz_width_widgets[ax]['LineEdit']
                w.blockSignals(True)
                w.setText(f'{xyz_wid[i]}')
                w.blockSignals(False)
        self.check_center_value()
    
    def load_pdbqt_string_into_browser(self):
        if self.pdbqt_editor:
            pdbqt_str, scheme = self.pdbqt_editor.convert_dict_to_pdbqt_text(True)
            self.browser_widget.load_protein_string(pdbqt_str, scheme)
            self.check_center_value()
    
    def change_center_color(self):
        dialog = QColorDialog(self.center_color)
        if dialog.exec():
            self.center_color = dialog.currentColor()
            self.check_center_value()
    
    def change_width_color(self):
        dialog = QColorDialog(self.width_color)
        dialog.setOption(QColorDialog.ShowAlphaChannel)
        if dialog.exec():
            self.width_color = dialog.currentColor()
            self.check_box_value()
    
    def reorient_to_boundingbox(self):
        self.browser_widget.reorient_to_boundingbox()
    
    def set_box_visibility(self, bool_state: bool):
        self.browser_widget.set_box_visibility(bool(bool_state))
        
    def set_center_visibility(self, bool_state: bool):
        self.browser_widget.set_center_visibility(bool(bool_state))
    
    def update_vina_param(self):
        for title, widget_dict in [('dock_center', self.xyz_center_widgets), ('dock_width', self.xyz_width_widgets)]:
            for name in ['x', 'y', 'z']:
                text = widget_dict[name]['LineEdit'].text()
                num = float(text) if text else None
                box_zero_bool = False if title == 'dock_width' and num == 0 else True
                self.vina_param[title][name] = num if bool(num) & box_zero_bool else None
        self.vina_param['exhaustiveness'] = self.exhaustiveness_spinbox.value()
        self.vina_param['eval_poses'] = self.eval_pose_spinbox.value()
        self.vina_param['center_color'] = self.center_color
        self.vina_param['width_color'] = self.width_color
        
    def set_clicked_position(self, coord_dict: dict):
        if self.center_radio.isChecked():
            for ax in coord_dict:
                if ax in ['x', 'y', 'z']:
                    le = self.xyz_center_widgets[ax]['LineEdit']
                    le.blockSignals(True)
                    le.setText(coord_dict[ax])
                    le.blockSignals(False)
            self.check_center_value()
            self.disable_radio.setChecked(True)
        elif self.flex_radio.isChecked():
            name = coord_dict['name']
            res, chain = name.split(']')[1].split('.')[0].split(':')
            exist = False
            for i, d in self.protein_flex_dict.items():
                c = d['chain_combo'].currentText()
                r = d['residue_combo'].currentText()
                if (c == chain) & (r == res):
                    self.remove_curr_flex_idx(i)
                    exist = True
                    break
            if not exist:
                self.add_flex_options(chain, res)
                self.update_protein_sidechain()
        
    def check_center_value(self):
        if self.pdbqt_editor:
            v = []
            for ax in self.xyz_center_widgets:
                sb = self.xyz_center_widgets[ax]['LineEdit']
                if not sb.text():
                    self.browser_widget.remove_sphere()
                    self.browser_widget.remove_box()
                    self.flex_autosearch_btn.setEnabled(False)
                    return
                else:
                    v.append(sb.text())
            vec = f'[{v[0]}, {v[1]}, {v[2]}]'
            colors = self.center_color.getRgbF()
            self.browser_widget.create_center_sphere(f'{vec}', f'[{colors[0]}, {colors[1]}, {colors[2]}]')
            self.set_center_visibility(self.center_visibility.isChecked())
            self.check_box_value()
            
    def check_box_value(self):
        if self.pdbqt_editor:
            center = []
            box    = []
            for ax in self.xyz_center_widgets:
                sb = self.xyz_center_widgets[ax]['LineEdit']
                if not sb.text():
                    return
                else:
                    center.append(sb.text())
            for ax in self.xyz_width_widgets:
                sb = self.xyz_width_widgets[ax]['LineEdit']
                if not sb.text() or float(sb.text()) == 0.:
                    self.browser_widget.remove_box()
                    self.flex_autosearch_btn.setEnabled(False)
                    return
                else:
                    box.append(sb.text())
            colors = self.width_color.getRgbF()
            self.browser_widget.create_bounding_box(center, f'[{colors[0]}, {colors[1]}, {colors[2]}]', box, colors[3])
            self.set_box_visibility(self.width_visibility.isChecked())
            self.flex_autosearch_btn.setEnabled(True)
    
    def update_protein_display(self):
        for chain in self.protein_editing_dict:
            d = self.protein_editing_dict[chain]
            checked = d['chain_checkbox'].isChecked()
            if not checked:
                self.pdbqt_editor.update_display(chain, None)
            else:
                display_text = d['display_lineedit'].text()
                r = self.pdbqt_editor.update_display(chain, display_text)
                if r is not None:
                    QMessageBox.critical(self, 'SyntaxError', r)
                    return
        self.browser_widget.clear_stage()
        self.load_pdbqt_string_into_browser()
        for idx, d in self.protein_flex_dict.items():
            chain = d['chain_combo'].currentText()
            if chain:
                self.update_flex_res_combobox(chain, idx)
        self.update_protein_sidechain()
    
    def direct_update_protein(self):
        self.browser_widget.clear_stage()
        self.load_pdbqt_string_into_browser()
        for idx, d in self.protein_flex_dict.items():
            chain = d['chain_combo'].currentText()
            if chain:
                self.update_flex_res_combobox(chain, idx)
        self.update_protein_sidechain()
        
    def check_available_chains_and_residues(self, chain: str | None=None):
        if chain is not None:
            display: pd.Series = self.pdbqt_editor.pdbqt_chain_dict[chain]['Display']
            available_res = display.index[display == True]
            return list(map(str, available_res.to_list()))
        chain_res_dict = {}
        for chain, chain_df in self.pdbqt_editor.pdbqt_chain_dict.items():
            display: pd.Series = chain_df['Display']
            available_res = display.index[display == True]
            if not available_res.empty:
                chain_res_dict[chain] = list(map(str, available_res.to_list()))
        return chain_res_dict
        
    def add_flex_options(self, chain: str|None=None, res: str|None=None):
        k = 0
        while self.protein_flex_dict.get(k, False):
            k += 1
        layout = QGridLayout()
        
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setLineWidth(4)
        frame.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        frame.setLayout(layout)
        
        chain_label = QLabel('<b>Chain :</b>')
        if chain:
            chain_label.setStyleSheet('color: SeaGreen')
        else:
            chain_label.setStyleSheet('color: #f03939')
        chain_combo = QComboBox()
        chain_combo.setMinimumWidth(80)
        chain_combo.addItems([''] + list(self.pdbqt_editor.pdbqt_chain_dict))
        if chain:
            chain_combo.setCurrentText(chain)
        chain_combo.currentTextChanged.connect(lambda chain, idx=k: self.update_flex_res_combobox(chain, idx))
        residue_label = QLabel('<b>Residue :</b>')
        if res:
            residue_label.setStyleSheet('color: SeaGreen')
        else:
            residue_label.setStyleSheet('color: #f03939')
        residue_combo = QComboBox()
        residue_combo.addItem('')
        if res:
            residue_combo.addItems(self.check_available_chains_and_residues(chain))
            residue_combo.setCurrentText(res)
        residue_combo.setMinimumWidth(80)
        residue_combo.currentTextChanged.connect(self.update_protein_sidechain)
        remove_button = QPushButton('-')
        remove_button.setStyleSheet('QPushButton { font-weight: bold; color: #E57373; font-size: 18px; padding: 0px; }')
        remove_button.clicked.connect(lambda _, idx=k: self.remove_curr_flex_idx(idx))
        aa_label = QLabel()
        aa_label.setStyleSheet('color: SeaGreen')
        
        layout.addWidget(chain_label, 0, 0)
        layout.addWidget(chain_combo, 0, 1)
        layout.addWidget(remove_button, 0, 2)
        layout.addWidget(residue_label, 1, 0)
        layout.addWidget(residue_combo, 1, 1)
        layout.addWidget(aa_label, 1, 2)
        
        self.flex_select_layout.insertWidget(self.flex_select_layout.count()-1, frame)
        self.protein_flex_dict[k] = {'frame'        : frame,
                                     'layout'       : layout,
                                     'chain_label'  : chain_label,
                                     'chain_combo'  : chain_combo,
                                     'residue_label': residue_label,
                                     'residue_combo': residue_combo,
                                     'aa_label'     : aa_label,}
        QApplication.instance().processEvents()
        
    def update_flex_scrollbar_position(self):
        vertitcal_scroll_bar = self.flex_scroll_area.verticalScrollBar()
        vertitcal_scroll_bar.setValue(vertitcal_scroll_bar.maximum())
    
    def update_flex_res_combobox(self, chain, idx):
        available_chain_res = self.check_available_chains_and_residues()
        if chain in available_chain_res:
            self.protein_flex_dict[idx]['chain_label'].setStyleSheet('color:SeaGreen')
        else:
            self.protein_flex_dict[idx]['chain_label'].setStyleSheet('color: #f03939')
        combo: QComboBox = self.protein_flex_dict[idx]['residue_combo']
        if chain and chain in available_chain_res:
            curr_text = combo.currentText()
            combo.clear()
            combo.addItems([''] + [str(x) for x in available_chain_res[chain]])
            combo.setCurrentText(curr_text)
        else:
            combo.clear()
            self.protein_flex_dict[idx]['residue_label'].setStyleSheet('color: #f03939')
    
    def remove_curr_flex_idx(self, idx):
        layout: QHBoxLayout = self.protein_flex_dict[idx]['layout']
        for i in reversed(range(layout.count())):
            widget = layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
        self.protein_flex_dict[idx]['frame'].setParent(None)
        self.flex_select_layout.removeItem(layout)
        del self.protein_flex_dict[idx]
        self.update_protein_sidechain()
        
    def update_protein_sidechain(self):
        available_chain_res = self.check_available_chains_and_residues()
        text_1_list = []
        text_2_list = []
        for chain in self.pdbqt_editor.pdbqt_chain_dict:
            df = self.pdbqt_editor.pdbqt_chain_dict[chain]
            df['Flexible'] = [False] * len(df)
        for i in self.protein_flex_dict:
            d = self.protein_flex_dict[i]
            chain = d['chain_combo'].currentText()
            res = d['residue_combo'].currentText()
            if chain in available_chain_res and res in available_chain_res[chain]:
                text_1_list.append(f'{res}:{chain}.CA')
                text_2_list.append(f'{res}:{chain} and sidechain')
                d['chain_label'].setStyleSheet('color:SeaGreen')
                d['residue_label'].setStyleSheet('color:SeaGreen')
                self.pdbqt_editor.pdbqt_chain_dict[chain].loc[int(res), 'Flexible'] = True
                d['aa_label'].setText('<b>' + self.pdbqt_editor.pdbqt_chain_dict[chain].loc[int(res), 'AA_Name'] + '</b>')
            else:
                d['chain_label'].setStyleSheet('color: #f03939')
                d['residue_label'].setStyleSheet('color: #f03939')
                d['aa_label'].clear()
        if bool(text_1_list) & bool(text_2_list):
            self.browser_widget.show_sidechains(' OR '.join(text_1_list) + ' OR (' + ' OR '.join(text_2_list) + ')')
        else:
            self.browser_widget.show_sidechains('none')
    
    def export_settings_with_pickle(self):
        if self.pdbqt_editor is not None:
            filter = 'Molecule Docker Settings (*.mds)' + f';;Protein (*.{self.pdbqt_editor.check_format_type()})'
        else:
            filter = 'Molecule Docker Settings (*.mds)'
        save_file, _ = QFileDialog.getSaveFileName(self, 'Export settings / protein structure', '', filter)
        if save_file:
            if save_file.endswith('.mds'):
                self.update_vina_param()
                export_dict = {'dock_center'   : self.vina_param['dock_center'],
                               'dock_width'    : self.vina_param['dock_width'],
                               'exhaustiveness': self.vina_param['exhaustiveness'],
                               'eval_poses'    : self.vina_param['eval_poses'],
                               'pdbqt_editor'  : self.pdbqt_editor.pdbqt_chain_dict}
                with open(save_file, 'wb') as f:
                    pickle.dump(export_dict, f)
            elif save_file.endswith('.pdbqt'):  # PDBQT
                flex_res, pdbqt_str = self.pdbqt_editor.convert_to_flex_set(True)
                if flex_res:
                    flex_pth, _ = QFileDialog.getSaveFileName(self, 'Export flexible PDBQT', '', 'AutoDock PDBQT (*.pdbqt)')
                    if flex_pth:
                        process_rigid_flex(pdbqt_str, save_file, flex_pth, flex_res)
                    else:
                        process_rigid_flex(pdbqt_str, save_file, None, flex_res)
                else:
                    process_rigid_flex(pdbqt_str, save_file, None, flex_res)
            else:   # PDB
                protein_str = self.pdbqt_editor.convert_dict_to_pdbqt_text()
                with open(save_file, 'w') as f:
                    f.write(protein_str)
    
    def check_center_and_box(self, get_value: bool=False):
        center_xyz = []
        box_xyz = []
        for ax in self.xyz_center_widgets:
            sb = self.xyz_center_widgets[ax]['LineEdit']
            if not sb.text():
                return False
            else:
                center_xyz.append(float(sb.text()))
        for ax in self.xyz_width_widgets:
            sb = self.xyz_width_widgets[ax]['LineEdit']
            if not sb.text():
                return False
            else:
                box_xyz.append(float(sb.text()))
        if get_value:
            return center_xyz, box_xyz
        return True
    
    def auto_search_flexible(self):
        center, box = self.check_center_and_box(True)
        chain_flex_res_tuples = self.pdbqt_editor.search_amino_acids(center, box)
        if chain_flex_res_tuples:
            for chain_flex_tup in chain_flex_res_tuples:
                self.add_flex_options(chain_flex_tup[0], str(chain_flex_tup[1]))
            self.update_protein_sidechain()
    
    def show_fpocket_dialog(self):
        pos = QCursor().pos()
        if self.fpocket_dialog is None:
            self.fpocket_dialog = FPocketTable(self, self.vina_param['fpocket'], pos)
        else:
            self.fpocket_dialog.raise_()
            self.fpocket_dialog.activateWindow()
            
    def show_hetatm_dialog(self):
        pos = QCursor().pos()
        if self.hetatm_dialog is None:
            self.hetatm_dialog = HetatmTable(self, self.vina_param['hetatm'], pos)
        else:
            self.hetatm_dialog.raise_()
            self.hetatm_dialog.activateWindow()
    
    def show_sequence_dialog(self, chain: str):
        if self.chain_seq_dialog[chain] is None:
            pos = QCursor().pos()
            seqs, display_dict = self.pdbqt_editor.retrieve_sequence_abbreviation(chain)
            dialog = SequenceDialog(self, seqs, display_dict, chain, self.color_theme, pos)
            self.chain_seq_dialog[chain] = dialog
        else:
            dialog = self.chain_seq_dialog[chain]
            dialog.raise_()
            dialog.activateWindow()
    
    def update_hightlight_text(self):
        selections = []
        for chain, sel in self.chain_highlight_text.items():
            if sel is not None:
                selections.append(f'(:{chain} AND {sel})')
        if selections:
            hl_string = ' OR '.join(selections)
            self.browser_widget.set_highlight(hl_string)
        else:
            self.browser_widget.set_highlight('none')
    
    def set_coord_from_record(self, center: list, box: list):
        if not center:
            for ax in ['x', 'y', 'z']:
                center_le = self.xyz_center_widgets[ax]['LineEdit']
                box_le = self.xyz_width_widgets[ax]['LineEdit']
                center_le.blockSignals(True)
                box_le.blockSignals(True)
                center_le.setText('')
                box_le.setText('')
                center_le.blockSignals(False)
                box_le.blockSignals(False)
        else:
            for i ,ax in enumerate(['x', 'y', 'z']):
                center_le = self.xyz_center_widgets[ax]['LineEdit']
                box_le = self.xyz_width_widgets[ax]['LineEdit']
                center_le.blockSignals(True)
                box_le.blockSignals(True)
                center_le.setText(str(center[i]))
                box_le.setText(str(box[i]))
                center_le.blockSignals(False)
                box_le.blockSignals(False)
        self.check_center_value()
        self.check_box_value()
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            update_chain = False
            for chain in self.protein_editing_dict:
                le = self.protein_editing_dict[chain]['display_lineedit']
                if le.hasFocus():
                    self.update_protein_display()
                    update_chain = True
                    break
            if not update_chain:
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)
    
    def accept_changes(self):
        self.update_vina_param()
        if self.fpocket_dialog is not None:
            self.fpocket_dialog.close()
        if self.hetatm_dialog is not None:
            self.hetatm_dialog.close()
        for seq_dialog in self.chain_seq_dialog.values():
            if seq_dialog is not None:
                seq_dialog.close()
        self.accept()
    
    def reject_changes(self):
        if self.fpocket_dialog is not None:
            self.fpocket_dialog.close()
        if self.hetatm_dialog is not None:
            self.hetatm_dialog.close()
        for seq_dialog in self.chain_seq_dialog.values():
            if seq_dialog is not None:
                seq_dialog.close()
        self.reject()
        
    def closeEvent(self, event):
        if self.fpocket_dialog is not None:
            self.fpocket_dialog.close()
        if self.hetatm_dialog is not None:
            self.hetatm_dialog.close()
        for seq_dialog in self.chain_seq_dialog.values():
            if seq_dialog is not None:
                seq_dialog.close()
        super().closeEvent(event)

class FPocketTable(QDialog):
    def __init__(self, parent, fpocket_dict: dict, pos):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.fpocket_dict = fpocket_dict
        overall_layout = QVBoxLayout()
        headers = ['Pocket #', 'Score', 'Druggability', 'Alpha Spheres', 'Volume']
        
        display_table = QTableWidget(len(fpocket_dict), 5)
        display_table.setHorizontalHeaderLabels(headers)
        display_table.verticalHeader().hide()
        self.fpocket_ckboxes = {}
        
        for row, (mdl, fpocket_params) in enumerate(fpocket_dict.items()):
            for col, h in enumerate(headers):
                if h not in fpocket_params:
                    pocket_checkbox = QCheckBox(f'# {mdl}')
                    self.fpocket_ckboxes[mdl] = pocket_checkbox
                    pocket_checkbox.clicked.connect(lambda c, x=mdl: self.update_fpocket_box(c, x))
                    display_table.setCellWidget(row, col, pocket_checkbox)
                else:
                    item = QTableWidgetItem()
                    item.setData(Qt.ItemDataRole.DisplayRole, fpocket_params[h])
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    display_table.setItem(row, col, item)
        display_table.setSortingEnabled(True)
        header = display_table.horizontalHeader()
        for i in range(len(headers)):
            # header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
            if i == 0:
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
            else:
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        
        overall_layout.addWidget(display_table)
        self.setLayout(overall_layout)
        self.setWindowTitle('FPocket Result')
        
        screen = self.screen()
        screen_size = screen.availableGeometry()
        self.resize(screen_size.width() * 0.35, screen_size.height() * 0.3)
        
        self.move(pos)
        self.show()
    
    def update_fpocket_box(self, checked: bool, mdl_num: int):
        if checked:
            for n, ckbox in self.fpocket_ckboxes.items():
                if n != mdl_num:
                    ckbox.setChecked(False)
            self.parent().set_coord_from_record(self.fpocket_dict[mdl_num]['Center'],
                                                self.fpocket_dict[mdl_num]['Box'])
        else:
            self.parent().set_coord_from_record([], [])
    
    def closeEvent(self, event):
        self.parent().fpocket_dialog = None
        super().closeEvent(event)

class HetatmTable(QDialog):
    def __init__(self, parent, hetatm_dict: dict, pos):
        super().__init__(parent)
        self.web_browser = None
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.hetatm_dict = hetatm_dict
        overall_layout = QVBoxLayout()
        headers = ['Ligand', 'Chain', 'Residue', 'Volume', 'PDB']
        
        display_table = QTableWidget(len(hetatm_dict), 5)
        display_table.setHorizontalHeaderLabels(headers)
        display_table.verticalHeader().hide()
        self.hetatm_ckboxes = {}
        
        for row, (ligand_res_chain, hetatm_params) in enumerate(hetatm_dict.items()):
            tmp = ligand_res_chain.split(']')
            ligand = tmp[0][1:]
            res, chain = tmp[1].split(':')
            ligand_ckbox = QCheckBox(ligand)
            self.hetatm_ckboxes[ligand_res_chain] = ligand_ckbox
            ligand_ckbox.clicked.connect(lambda c, x=ligand_res_chain: self.update_hetatm_box(c, x))
            
            chain_item = QTableWidgetItem(chain)
            flags = chain_item.flags()
            chain_item.setFlags(flags & ~Qt.ItemFlag.ItemIsEditable)
            
            res_item = QTableWidgetItem()
            res_item.setData(Qt.ItemDataRole.DisplayRole, int(res))
            res_item.setFlags(flags & ~Qt.ItemFlag.ItemIsEditable)
            
            volume_item = QTableWidgetItem()
            volume_item.setData(Qt.ItemDataRole.DisplayRole, hetatm_params['Volume'])
            volume_item.setFlags(flags & ~Qt.ItemFlag.ItemIsEditable)
            
            ref_label = QLabel(f'<a href="https://www.rcsb.org/ligand/{ligand}">Ref</a>')
            ref_label.setOpenExternalLinks(False)
            ref_label.linkActivated.connect(self.show_ref_browser)
            
            display_table.setCellWidget(row, 0, ligand_ckbox)
            display_table.setItem(row, 1, chain_item)
            display_table.setItem(row, 2, res_item)
            display_table.setItem(row, 3, volume_item)
            display_table.setCellWidget(row, 4, ref_label)
        display_table.setSortingEnabled(True)
        header = display_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for i in range(1, len(headers)):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        
        overall_layout.addWidget(display_table)
        self.setLayout(overall_layout)
        self.setWindowTitle('Heteroatom Table')
        
        screen = self.screen()
        screen_size = screen.availableGeometry()
        self.resize(screen_size.width() * 0.3, screen_size.height() * 0.3)
        
        self.move(pos)
        self.show()
    
    def update_hetatm_box(self, checked: bool, chain_ligand: str):
        if checked:
            for n, ckbox in self.hetatm_ckboxes.items():
                if n != chain_ligand:
                    ckbox.setChecked(False)
            self.parent().set_coord_from_record(self.hetatm_dict[chain_ligand]['Center'],
                                                self.hetatm_dict[chain_ligand]['Box'])
        else:
            self.parent().set_coord_from_record([], [])
    
    def show_ref_browser(self, url: str):
        if self.web_browser is None:
            self.web_browser = BrowserWithTabs(self, url)
            self.web_browser.closed.connect(lambda: setattr(self, 'web_browser', None))
        else:
            self.web_browser.tab_browser.add_new_tab(url)
    
    def closeEvent(self, event):
        self.parent().hetatm_dialog = None
        super().closeEvent(event)

class SequenceDialog(QDialog):
    def __init__(self, parent,
                 idx_seq_map: dict, display_dict: dict,
                 chain: str, color_theme: str, pos):
        super().__init__(parent)
        if color_theme == 'light':
            self.bg_color = '#b6fcb8'
        else:
            self.bg_color = '#175418'
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        overall_layout = QVBoxLayout()
        
        self.chain = chain
        self.idx_seq_map = idx_seq_map
        self.seq_textedit = QTextEdit()
        self.seq_textedit.setReadOnly(True)
        self.seq_textedit.setMouseTracking(True)
        self.seq_textedit.selectionChanged.connect(self.highlight_selection)
        self.seq_textedit.setStyleSheet('font-family: "Courier New", Courier, monospace;')
        self.seq_textedit.setHtml(self.format_sequence(display_dict))
        
        self.update_display_btn = QPushButton('Update')
        self.update_display_btn.clicked.connect(self.update_display)
        
        overall_layout.addWidget(self.seq_textedit)
        overall_layout.addWidget(self.update_display_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        
        self.setLayout(overall_layout)
        self.setWindowTitle(f'Chain {chain}')
        
        self.move(pos)
        self.show()
    
    def format_sequence(self, display: dict):
        formatted_seq = ""
        for aa_pos, aa in self.idx_seq_map.items():
            display_stat = display.get(aa_pos, None)
            if display_stat is not None and display_stat:
                formatted_seq += f"<span style='background-color: {self.bg_color};'>" + aa + "</span>"
            else:
                formatted_seq += aa
            if aa_pos % 10 == 0:
                formatted_seq += ' '
        return formatted_seq
    
    def retrieve_start_and_end_of_selection(self):
        cursor = self.seq_textedit.textCursor()
        start, end = cursor.selectionStart(), cursor.selectionEnd()
        plain_seq = self.seq_textedit.toPlainText()
        num_spaces = plain_seq[:start].count(' ')
        seq_len = (end - start) - plain_seq[start:end].count(' ')
        start = start - num_spaces + next(iter(self.idx_seq_map))
        end = start + seq_len
        return start, end-1
    
    def highlight_selection(self):
        start, end = self.retrieve_start_and_end_of_selection()
        if start != end:
            self.setToolTip(f'{start}-{end}')
        else:
            self.setToolTip('')
        self.parent().chain_highlight_text[self.chain] = f'{start}-{end}'
        self.parent().update_hightlight_text()
    
    def update_display(self):
        start, end = self.retrieve_start_and_end_of_selection()
        display_series: pd.Series = self.parent().pdbqt_editor.pdbqt_chain_dict[self.chain]['Display']
        matching_idx = display_series.index.isin(list(range(start, end+1)))
        inv_series = display_series[matching_idx].astype(bool)
        display_series[matching_idx] = ~inv_series
        self.seq_textedit.blockSignals(True)
        self.seq_textedit.setHtml(self.format_sequence(display_series.to_dict()))
        self.seq_textedit.blockSignals(False)
        self.parent().pdbqt_editor.pdbqt_chain_dict[self.chain]['Display'] = display_series
        curr_displayed = self.parent().pdbqt_editor.convert_to_range_text(self.chain)
        self.parent().protein_editing_dict[self.chain]['chain_checkbox'].setChecked(False if curr_displayed is None else True)
        self.parent().protein_editing_dict[self.chain]['display_lineedit'].setText(curr_displayed)
        self.parent().direct_update_protein()
        self.parent().chain_highlight_text[self.chain] = None
    
    def closeEvent(self, event):
        self.parent().chain_seq_dialog[self.chain] = None
        self.parent().chain_highlight_text[self.chain] = None
        self.parent().update_hightlight_text()
        super().closeEvent(event)

class DockingTextExtractor(QObject):
    update_text = Signal(str, bool)
    update_progress = Signal()
    docking_complete = Signal(bool)
    docking_cancel = Signal()
    process = None
    multiprocesses = []

class DropDirLineEdit(QLineEdit):
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
                if file_path and os.path.isdir(file_path):
                    self.setText(file_path)
            event.acceptProposedAction()

class DropFileLineEdit(QLineEdit):
    def __init__(self, parent=None, accepted_extension=('.pdb', '.pdbqt')):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.accepted_ext = accepted_extension
        
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
                if file_path and os.path.isfile(file_path) and file_path.endswith(self.accepted_ext):
                    self.setText(file_path)
            event.acceptProposedAction()

class DropFileDirListWidget(QListWidget):
    supplierdictSignal = Signal(dict)
    currCountChanged = Signal(int)
    itemRemovedSignal = Signal(str)
    
    def __init__(self, parent=None, accepted_extension=('.smi', '.sdf', '.mol2', '.mol', '.mrv', '.pdb', '.xyz', '.pdbqt', '.mddb')):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.accepted_ext = accepted_extension
        self.rm_shortcut = QShortcut('Backspace', self)
        self.rm_shortcut.activated.connect(self.remove_current_selected)
        self.model().rowsInserted.connect(self.return_current_count)
        self.model().rowsRemoved.connect(self.return_current_count)
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    @staticmethod
    def check_smi_title_line(smi_file: str):
        with open(smi_file, 'r') as f:
            for r, l in enumerate(f):
                possible_smiles = l.split(' ')[0]
                if Chem.MolFromSmiles(possible_smiles) is not None:
                    return r
            
    def retrieve_name_smiles_for_mols(self, all_input_files: list[str]):
        id_txt = os.path.join(os.path.dirname(__file__), 'sdf_id_names.txt')
        with open(id_txt) as f:
            all_ids = [id for id in f.read().strip().split('\n') if id]
        
        def read_chem_to_rdkit_chem(chem_pth: str):
            lower_chem_pth = chem_pth.lower()
            if lower_chem_pth.endswith('.smi'):
                n = self.check_smi_title_line(chem_pth)
                try:
                    return [mol for mol in Chem.MultithreadedSmilesMolSupplier(chem_pth, titleLine=n) if mol is not None]
                except Exception as e:
                    return str(e)
            elif lower_chem_pth.endswith('.sdf'):
                try:
                    return [mol for mol in Chem.MultithreadedSDMolSupplier(chem_pth) if mol is not None]
                except Exception as e:
                    return str(e)
            elif lower_chem_pth.endswith('.mol2'):
                return [Chem.MolFromMol2File(chem_pth)]
            elif lower_chem_pth.endswith('.mol'):
                return [Chem.MolFromMolFile(chem_pth)]
            elif lower_chem_pth.endswith('.mrv'):
                return [Chem.MolFromMrvFile(chem_pth)]
            elif lower_chem_pth.endswith('.pdb'):
                return [Chem.MolFromPDBFile(chem_pth)]
            elif lower_chem_pth.endswith('.xyz'):
                return [Chem.MolFromXYZFile(chem_pth)]
            elif lower_chem_pth.endswith('.pdbqt'):
                with open(chem_pth, 'r') as f:
                    for i, l in enumerate(f):
                        if i == 0 or i == 6:
                            smiles = l[14:]
                            mol = Chem.MolFromSmiles(smiles)
                            if mol is not None or i == 6:
                                return [mol]
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
            for id in all_ids:
                if id in properties:
                    return properties[id]
            if mol.HasProp("_Name"):
                return mol.GetProp("_Name")
            return None
            
        molecules = {}
        for chem_pth in all_input_files:
            mols = read_chem_to_rdkit_chem(chem_pth)
            if isinstance(mols, str):
                QMessageBox.critical(self, 'FileReadError', f'Failed to parse {chem_pth}.')
            else:
                single_mol = True if len(mols) == 1 else False
                for i, mol in enumerate(mols):
                    if mol is not None:
                        name = retrieve_name_from_mol(mol)
                        if name is None:
                            if not chem_pth.endswith(('.smi', '.sdf')):
                                base_name = os.path.basename(chem_pth)
                                if os.path.basename(chem_pth).endswith('_docked.pdbqt'):
                                    base_name = base_name.rsplit('_docked.pdbqt', 1)[0]
                                else:
                                    base_name = base_name.rsplit('.', 1)[0]
                                name = f'{base_name}'
                            else:
                                if single_mol:
                                    name = f'{os.path.basename(chem_pth).rsplit(".")[0]}'
                                else:
                                    name = f'{os.path.basename(chem_pth).rsplit(".")[0]}_{i+1}'
                        if name in molecules:
                            n = 1
                            while name + f'_{n}' in molecules:
                                n += 1
                            name = f'{name}_{n}'
                        molecules[name] = Chem.MolToSmiles(mol)
                    else:
                        QMessageBox.critical(self, 'FileReadError', f'Failed to parse {chem_pth}.')
        return molecules
    
    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            for url in urls:
                file_or_dir_path = url.toLocalFile()
                if file_or_dir_path:
                    if (os.path.isfile(file_or_dir_path) and file_or_dir_path.endswith(self.accepted_ext)) or os.path.isdir(file_or_dir_path):
                        if os.path.isdir(file_or_dir_path):
                            file_paths = [f for f in os.listdir(file_or_dir_path) if f.endswith(self.accepted_ext)]
                        else:
                            file_paths = [file_or_dir_path]
                        name_smiles_dict = self.retrieve_name_smiles_for_mols(file_paths)
                        if name_smiles_dict:
                            supplier_dict = {'Name': [], 'SMILES': []}
                            supplier_dict.update({prop_name: [] for prop_name in property_functions})
                            for name, smiles in name_smiles_dict.items():
                                supplier_dict['Name'].append(name)
                                supplier_dict['SMILES'].append(smiles)
                                mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                                for prop_name in property_functions:
                                    supplier_dict[prop_name].append(property_functions[prop_name](mol))
                            self.supplierdictSignal.emit(supplier_dict)
            event.acceptProposedAction()
            
    def remove_current_selected(self):
        all_items = self.selectedItems()
        if not all_items:
            return
        else:
            for item in all_items:
                self.itemRemovedSignal.emit(item.text())
                self.takeItem(self.row(item))
                
    def return_current_count(self):
        self.currCountChanged.emit(self.count())
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Backspace:
            self.remove_current_selected()
        else:
            super().keyPressEvent(event)

class DropDBTableWidget(QTableWidget):
    chemDBSignal = Signal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.accepted_ext = '.mddb'
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.rm_shortcut = QShortcut(QKeySequence('Backspace'), self)
        self.rm_shortcut.activated.connect(self.remove_current_selected)
        self.pth_record_dict = {}
    
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def reset_all_highlight(self):
        with open(os.path.join(os.path.dirname(__file__), 'fp_param', 'fp_param.json'), 'r') as f:
            curr_fp_dict = json.load(f)
        del curr_fp_dict['sim']
        for i, pth in enumerate(list(self.pth_record_dict)):
            conn = sqlite3.connect(pth)
            cur = conn.cursor()
            cur.execute('SELECT format FROM DBInfo WHERE id = ?', (1,))
            row = cur.fetchone()
            db_fp_settings = json.loads(row[0])
            conn.close()
            highlighted_db_fp_settings = self.highlight_differences(db_fp_settings, curr_fp_dict)
            db_data = json.dumps(highlighted_db_fp_settings, indent=4)
            self.item(i, 1).setToolTip(
                f"<html><body>"
                f"<pre style='font-family:\"Courier New\", monospace;'>{db_data}</pre>"
                f"</body></html>"
            )
    
    def highlight_differences(self, db_fp_dict, curr_fp_dict):
        highlighted = {}
        same = True
        for key in db_fp_dict:
            val1 = db_fp_dict[key]
            val2 = curr_fp_dict.get(key, None)
            if isinstance(val1, dict):
                highlighted[key], _ = self.highlight_differences(val1, val2)
            else:
                if val1 != val2 or val2 is None:
                    highlighted[key] = f'<font color=#E57373>{val1} &#x27F7; {val2}</font>'
                    same = False
                else:
                    highlighted[key] = f'<font color=#4CAF50>{val1}</font>'
        return highlighted, same
    
    def add_file_to_dict(self, db_pth: str, template_str: str='', checked: bool=True):
        if os.path.isfile(db_pth) and db_pth.endswith(self.accepted_ext):
            name = os.path.basename(db_pth).rsplit('.', 1)[0]
            conn = sqlite3.connect(db_pth)
            cur = conn.cursor()
            cur.execute(f"SELECT COUNT(*) FROM MolDB;")
            row_cnt = cur.fetchone()[0]
            cur.execute('SELECT format FROM DBInfo WHERE id = ?', (1,))
            row = cur.fetchone()
            db_fp_settings = json.loads(row[0])
            conn.close()
            with open(os.path.join(os.path.dirname(__file__), 'fp_param', 'fp_param.json'), 'r') as f:
                curr_fp_dict = json.load(f)
            del curr_fp_dict['sim']
            
            cur_cnt = max(0, self.rowCount())
            self.setRowCount(cur_cnt + 1)
            
            cell_widget = QWidget()
            cell_layout = QHBoxLayout()
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_widget.setLayout(cell_layout)
            db_ckbox = QCheckBox()
            db_ckbox.setChecked(checked)
            db_ckbox.setStyleSheet('QCheckBox { spacing: 0px; }')
            cell_layout.addWidget(db_ckbox, alignment=Qt.AlignmentFlag.AlignCenter)
            db_name = QTableWidgetItem(name)
            db_name.setFlags(db_name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            highlighted_db_fp_settings, _ = self.highlight_differences(db_fp_settings, curr_fp_dict)
            db_data = json.dumps(highlighted_db_fp_settings, indent=4)
            db_name.setToolTip(
                f"<html><body>"
                f"<pre style='font-family:\"Courier New\", monospace;'>{db_data}</pre>"
                f"</body></html>"
                )
            db_cnt = QTableWidgetItem(str(row_cnt))
            db_cnt.setFlags(db_name.flags())
            db_template = QTableWidgetItem(template_str)
            
            db_ckbox.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            cell_widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            
            self.setCellWidget(cur_cnt, 0, cell_widget)
            self.setItem(cur_cnt, 1, db_name)
            self.setItem(cur_cnt, 2, db_cnt)
            self.setItem(cur_cnt, 3, db_template)
            
            self.pth_record_dict[db_pth] = db_ckbox
            
        self.chemDBSignal.emit(len(self.pth_record_dict))
        
    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            _temp_dict = {}
            for url in urls:
                file_pth = url.toLocalFile()
                if file_pth:
                    if os.path.isfile(file_pth) and file_pth.endswith(self.accepted_ext):
                        name = os.path.basename(file_pth).rsplit('.', 1)[0]
                        conn = sqlite3.connect(file_pth)
                        cur = conn.cursor()
                        cur.execute(f"SELECT COUNT(*) FROM MolDB;")
                        row_cnt = cur.fetchone()[0]
                        cur.execute('SELECT format FROM DBInfo WHERE id = ?', (1,))
                        row = cur.fetchone()
                        db_fp_settings = json.loads(row[0])
                        conn.close()
                        _temp_dict[file_pth] = {'name' : name,
                                                'count': row_cnt,
                                                'fp_p' : db_fp_settings}
            if _temp_dict:
                with open(os.path.join(os.path.dirname(__file__), 'fp_param', 'fp_param.json'), 'r') as f:
                    curr_fp_dict = json.load(f)
                del curr_fp_dict['sim']
                name_url_map = {}
                with open(os.path.join(os.path.dirname(__file__), 'database', 'url_template.txt'), 'r') as f:
                    for l in f:
                        name, template = l.strip().split()
                        name_url_map[name] = template
                cur_cnt = self.rowCount()
                self.setRowCount(len(_temp_dict) + cur_cnt)
                i = 0
                
                for pth, map_dict in _temp_dict.items():
                    name = map_dict['name']
                    row_cnt = map_dict['count']
                    db_fp_settings = map_dict['fp_p']
                    
                    cell_widget = QWidget()
                    cell_layout = QHBoxLayout()
                    cell_layout.setContentsMargins(0, 0, 0, 0)
                    cell_widget.setLayout(cell_layout)
                    db_ckbox = QCheckBox()
                    db_ckbox.setStyleSheet('QCheckBox { spacing: 0px; }')
                    cell_layout.addWidget(db_ckbox, alignment=Qt.AlignmentFlag.AlignCenter)
                    db_name = QTableWidgetItem(name)
                    db_name.setFlags(db_name.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    highlighted_db_fp_settings, same = self.highlight_differences(db_fp_settings, curr_fp_dict)
                    db_ckbox.setChecked(same)
                    db_data = json.dumps(highlighted_db_fp_settings, indent=4)
                    db_name.setToolTip(
                        f"<html><body>"
                        f"<pre style='font-family:\"Courier New\", monospace;'>{db_data}</pre>"
                        f"</body></html>"
                        )
                    db_cnt = QTableWidgetItem(str(row_cnt))
                    db_cnt.setFlags(db_name.flags())
                    db_template = QTableWidgetItem()
                    if name in name_url_map:
                        db_template.setText(name_url_map[name])
                    
                    db_ckbox.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                    cell_widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                    
                    self.setCellWidget(cur_cnt + i, 0, cell_widget)
                    self.setItem(cur_cnt + i, 1, db_name)
                    self.setItem(cur_cnt + i, 2, db_cnt)
                    self.setItem(cur_cnt + i, 3, db_template)
                    
                    self.pth_record_dict[pth] = db_ckbox
                    
                    i += 1
            self.chemDBSignal.emit(len(self.pth_record_dict))
            event.acceptProposedAction()
    
    def remove_current_selected(self):
        all_items = self.selectedItems()
        rows_to_remove = {item.row() for item in all_items}
        for row in sorted(rows_to_remove, reverse=True):
            self.removeRow(row)
            
            for i, rec in enumerate(list(self.pth_record_dict)):
                if row == i:
                    self.pth_record_dict[rec].deleteLater()
                    del self.pth_record_dict[rec]
                    break
        self.chemDBSignal.emit(len(self.pth_record_dict))
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Backspace:
            self.remove_current_selected()
        else:
            super().keyPressEvent(event)

class RemovableTableWidget(QTableWidget):
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.rm_shortcut = QShortcut(QKeySequence('Backspace'), self)
        self.rm_shortcut.activated.connect(self.remove_current_selected)
        h = self.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    
    def remove_current_selected(self):
        all_items = self.selectedItems()
        rows_to_remove = {item.row() for item in all_items}
        for row in sorted(rows_to_remove, reverse=True):
            self.removeRow(row)
    
    def get_all_names(self):
        names = []
        for r in range(self.rowCount()):
            names.append(self.item(r, 0).text())
        return names
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Backspace:
            self.remove_current_selected()
        else:
            super().keyPressEvent(event)
            
    def resizeEvent(self, event):
        h = self.horizontalHeader()
        for i in range(h.count() - 1):
            curr_size = h.sectionSize(i)
            h.resizeSection(i, curr_size)
        h.setSectionResizeMode(i + 1, QHeaderView.ResizeMode.Stretch)
        super().resizeEvent(event)

class CopyLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setToolTip('Click to copy')
        self.setStyleSheet('color: #42A5F5')
    
    def enterEvent(self, event):
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        self.unsetCursor()
        super().leaveEvent(event)
        
    def setText(self, text):
        self.setStyleSheet('color: #4CAF50')
        super().setText(text)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            clipboard = QApplication.clipboard()
            clipboard.setText(self.text())
            self.setStyleSheet('color: #42A5F5')
            self.setToolTip('Copied!')
            self.unsetCursor()
        super().mousePressEvent(event)

class CopyLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setToolTip('Click to copy')
    
    def enterEvent(self, event):
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        self.unsetCursor()
        super().leaveEvent(event)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            clipboard = QApplication.clipboard()
            clipboard.setText(self.text())
            self.unsetCursor()
        super().mousePressEvent(event)

class HiddenExpandLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_menu)
        
        copy_regex_action = QAction('Copy Regex', self)
        copy_regex_action.triggered.connect(self.copy_regex)
        self.expand_hidden_action = QAction('Expand', self)
        self.expand_hidden_action.triggered.connect(self.change_label_text)
        
        self.menu = QMenu()
        self.menu.addAction(copy_regex_action)
        self.menu.addAction(self.expand_hidden_action)
        
    def set_initial_text(self, text_list: list):
        text = '\n'.join(text_list)
        self.regex = '(^' + '$|^'.join(text_list) + '$)'
        self.total_num = len(text_list)
        if len(text_list) > 3:
            self.full_text = text
            self.displayed_text = '\n'.join(text_list[:3] + ['...'])
            self.setToolTip(f'Total: {self.total_num}')
            self.curr_mode = 'Hidden'
        else:
            self.full_text = None
            self.displayed_text = text
            self.expand_hidden_action.setDisabled(True)
        self.setText(self.displayed_text)
        
    def copy_regex(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.regex)
        
    def change_label_text(self):
        if self.full_text is not None:
            if self.curr_mode == 'Hidden':
                self.setText(self.full_text)
                self.curr_mode = 'Full'
                self.expand_hidden_action.setText('Hide')
            else:
                self.setText(self.displayed_text)
                self.curr_mode = 'Hidden'
                self.expand_hidden_action.setText('Expand')
                
    def show_menu(self, pos):
        self.menu.exec(self.mapToGlobal(pos))

class ShowPlotLabel(QLabel):
    def __init__(self, tmp_class_with_df: pd.DataFrame, names: list, parent=None):
        super().__init__(parent)
        self.df = tmp_class_with_df
        self.names = names
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_menu)
        
        plot_action = QAction('Show Figure', self)
        plot_action.triggered.connect(self.show_plot)
        
        self.menu = QMenu()
        self.menu.addAction(plot_action)
        
    def show_plot(self):
        parent_dir = os.path.dirname(os.path.dirname(__file__))
        html_pth = os.path.join(parent_dir, 'plot_empty', '_tmp_fragment.html')
        plot_energy_fragment(self.df, self.names, html_pth)
        self.browser_widget = FragmentPlotBrowserWidget(html_pth)
        self.browser_widget.show()
        
    def show_menu(self, pos):
        self.menu.exec(self.mapToGlobal(pos))

class ChangeValueLineEdit(QLineEdit):
    def __init__(self, parent=None, is_center: bool=True):
        super().__init__(parent)
        self.step = 0.1
        self.scroll_step = 0.05
        self.is_center = is_center  # set minimum to 0 if not is_center
        
    def validate_num(self, num: float):
        if not self.is_center:
            return f'{max(0., num):.3f}'
        return f'{num:.3f}'
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Up:
            if self.text():
                v = float(self.text()) + self.step
            else:
                v = self.step
            self.setText(self.validate_num(v))
        elif event.key() == Qt.Key.Key_Down:
            if self.text():
                v = float(self.text()) - self.step
            else:
                v = -self.step
            self.setText(self.validate_num(v))
        else:
            super().keyPressEvent(event)
    
    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            if self.text():
                v = float(self.text()) + self.step
            else:
                v = self.step
            self.setText(self.validate_num(v))
        elif event.angleDelta().y() < 0:
            if self.text():
                v = float(self.text()) - self.step
            else:
                v = -self.step
            self.setText(self.validate_num(v))
        else:
            super().wheelEvent(event)

class _DockResultTable(QTableWidget):
    # Depracated, slower than QTableView
    updateTreeSignal = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.property_full_name = ['Molecular Weight'  , 'Hydrogen Bond Donors'    , 'Hydrogen Bond Acceptors',
                                   'LogP'              , 'Topological Polar Surface Area', 'Rotatable Bonds'        ,
                                   'Number of Rings'   , 'Formal Charge'           , 'Number of Heavy Atoms'  ,
                                   'Molar Refractivity', 'Number of Atoms', 'QED']
        self.chem_prop_to_full_name_map = {'eng' : 'Score'              , 'mw'  : 'Molecular Weight'        ,
                                           'hbd' : 'Hydrogen Bond Donors', 'hba' : 'Hydrogen Bond Acceptors' ,
                                           'logp': 'LogP'                , 'tpsa': 'Topological Polar Surface Area',
                                           'rb'  : 'Rotatable Bonds'     , 'nor' : 'Number of Rings'         ,
                                           'fc'  : 'Formal Charge'       , 'nha' : 'Number of Heavy Atoms'   ,
                                           'mr'  : 'Molar Refractivity'  , 'na'  : 'Number of Atoms'         ,
                                           'engr': 'Score Rank'         , 'QED' : 'QED'}
        self.chem_filter_dict = {'Name': [],
                                 'eng' : [('≤', -7.5)],
                                 'engr': [('≤', 1. )],
                                 'mw'  : [()], 'hbd' : [()], 'hba' : [()], 'logp': [()],
                                 'tpsa': [()], 'rb'  : [()], 'nor' : [()], 'fc'  : [()], 'nha' : [()],
                                 'mr'  : [()], 'na'  : [()], 'QED' : [()]}
        self.chem_column_dict = {'Name'                   : True, 'SMILES'                  : False,
                                 'Score'                 : True, 'Molecular Weight'        : True,
                                 'Hydrogen Bond Donors'   : True, 'Hydrogen Bond Acceptors' : True,
                                 'LogP'                   : True, 'Topological Polar Surface Area': True,
                                 'Rotatable Bonds'        : True, 'Number of Rings'         : True,
                                 'Formal Charge'          : True, 'Number of Heavy Atoms'   : True,
                                 'Molar Refractivity'     : True, 'Number of Atoms'         : True,
                                 'QED'                    : True}
        self.df = None
        self.bool_filter = None
        self.processing_df = None
        self.filtered_df = None
        
        self.horizontalHeader().sortIndicatorChanged.connect(self.update_processing_and_filter_df)
        
    def retrieve_empty_dict(self, fragment: bool, chemprop: bool):
        if fragment:
            d = {'Name': [], 'Score': [], 'File Path': [], 'Fragments': []}
        else:
            d = {'Name': [], 'Score': [], 'File Path': []}
        if chemprop:
            d.update({k: [] for k in self.property_full_name})
            d['SMILES'] = []
        return d
    
    def setup_table(self):
        columns = ['Name', 'Best Score', 'SMILES'] + self.property_full_name
        self.setColumnCount(len(columns))
        self.setHorizontalHeaderLabels(columns)
        header = self.horizontalHeader()
        for i in range(len(columns)):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
    
    def populate_table(self, result_dict: dict):
        self.clearContents()
        self.df = pd.DataFrame(data=result_dict)
        self.reset_current_table()
    
    def update_processing_and_filter_df(self, column, order, signal=True):
        if self.filtered_df is not None:
            ascending = order == Qt.SortOrder.AscendingOrder
            col = self.horizontalHeaderItem(column)
            if col is not None:
                sorted_column_name = col.text()
                sorted_column_name = sorted_column_name if sorted_column_name != 'Best Score' else 'Score'
                self.filtered_df = self.filtered_df.sort_values(by=sorted_column_name, ascending=ascending).reset_index(drop=True)
                self.processing_df = self.processing_df.sort_values(by=sorted_column_name, ascending=ascending).reset_index(drop=True)
                if signal:
                    self.updateTreeSignal.emit()
    
    def reset_current_table(self):
        header = self.horizontalHeader()
        if header.isSortIndicatorShown():
            sorted_section = header.sortIndicatorSection()
        else:
            sorted_section = None
        self.setSortingEnabled(False)
        self.clearContents()
        self.bool_filter = self.filter_dataframe()
        dropped_cols = self.filter_column()
        self.processing_df = self.df[self.bool_filter].reset_index(drop=True)
        self.filtered_df = self.processing_df.drop(columns=dropped_cols)
        if sorted_section is not None:
            self.update_processing_and_filter_df(sorted_section,
                                                 header.sortIndicatorOrder(),
                                                 False)
        columns = self.filtered_df.columns
        header_columns = [c if c != 'Score' else 'Best Score' for c in columns]
        self.setColumnCount(len(columns))
        self.setHorizontalHeaderLabels(header_columns)
        for i in range(len(columns)):
            if columns[i] == 'SMILES':
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.Interactive)
            else:
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        filtered_num = len(self.filtered_df)
        self.setRowCount(filtered_num)
        if filtered_num > 0:
            populate_progress_dialog = QProgressDialog('Populating Table...', 'Cancel', 0, filtered_num, self)
            populate_progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
            cancel_button = QPushButton("Cancel")
            cancel_button.setFixedSize(60, 23)
            populate_progress_dialog.setCancelButton(cancel_button)
            populate_progress_dialog.show()
            for row in range(filtered_num):
                for col, name in enumerate(columns):
                    v = self.filtered_df.iloc[row][name]
                    item = QTableWidgetItem()
                    if name not in ['Name', 'SMILES']:
                        item.setData(Qt.ItemDataRole.DisplayRole, float(v))
                    else:
                        item.setText(v)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.setItem(row, col, item)
                populate_progress_dialog.setValue(row+1)
                if populate_progress_dialog.wasCanceled():
                    break
        self.setSortingEnabled(True)
        
    def _compare_single_values(self, op, thres, target):
        if   op == '≥':
            return target >= thres
        elif op == '>':
            return target >  thres
        elif op == '<':
            return target <  thres
        elif op == '≤':
            return target <= thres
    
    def check_energy_ranking(self, list_operations: list[tuple], target: pd.Series):
        length = len(target)
        if len(list_operations) == 1:
            percentage = list_operations[0][1]
            partition_kth = max(0, int(length * percentage / 100) - 1)
            threshold = np.partition(target, partition_kth)[partition_kth]
            return self._compare_single_values(list_operations[0][0], threshold, target)
        else:
            op1, percentage1 = list_operations[0]
            op2, percentage2 = list_operations[1]
            partition_kth1 = max(0, int(length * percentage1 / 100) - 1)
            partition_kth2 = max(0, int(length * percentage2 / 100) - 1)
            thres1 = np.partition(target, partition_kth1)[partition_kth1]
            thres2 = np.partition(target, partition_kth2)[partition_kth2]
            or_option = False
            if (op1 in ('≥', '>') and op2 in ('≤', '<')):
                if thres1 >= thres2:
                    or_option = True
            elif (op1 in ('≤', '<') and op2 in ('≥', '>')):
                if thres1 >= thres2:
                    or_option = True
            op1_result = self._compare_single_values(op1, thres1, target)
            op2_result = self._compare_single_values(op2, thres2, target)
            if or_option:
                return op1_result | op2_result
            else:
                return op1_result & op2_result
    
    def check_chemprop_matching(self, list_operations: list[tuple], target: pd.Series):
        if len(list_operations) == 1:
            return self._compare_single_values(list_operations[0][0], list_operations[0][1], target)
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
            op1_result = self._compare_single_values(op1, thres1, target)
            op2_result = self._compare_single_values(op2, thres2, target)
            if or_option:
                return op1_result | op2_result
            else:
                return op1_result & op2_result
            
    def filter_dataframe(self):
        final_filter_dict = {}
        existing_columns = self.df.columns.to_list() + ['Score Rank']
        for prop, ops in self.chem_filter_dict.items():
            if prop != 'Name':
                if ops[0]:
                    full_name = self.chem_prop_to_full_name_map[prop]
                    if full_name in existing_columns:
                        final_filter_dict[full_name] = ops
            else:
                if ops:
                    final_filter_dict['Name'] = ops
        bool_column = pd.Series([True] * len(self.df))
        for full_prop, ops in final_filter_dict.items():
            if full_prop == 'Score Rank':
                target = self.df['Score']
                bool_column &= self.check_energy_ranking(ops, target)
            elif full_prop == 'Name':
                text_match_bool_column = pd.Series([False] * len(self.df))
                name_series = self.df['Name']
                for match_txt in ops:
                    text_match_bool_column |= name_series.str.contains(match_txt, case=True, regex=True)
                bool_column &= text_match_bool_column
            else:
                target = self.df[full_prop]
                bool_column &= self.check_chemprop_matching(ops, target)    # currently set to "AND" between each chemical properties
        return bool_column
    
    def filter_column(self):
        rm_cols = ['File Path']
        existing_columns = self.df.columns.to_list()
        if 'Fragments' in existing_columns:
            rm_cols.append('Fragments')
        dropped_columns = [c for c in existing_columns if c not in rm_cols and not self.chem_column_dict[c]] + rm_cols
        return dropped_columns

class TableModel(QAbstractTableModel):
    def __init__(self, data, header_labels):
        super().__init__()
        self._data = data
        self._header_labels = header_labels
        
    def rowCount(self, parent=QModelIndex()):
        return len(self._data)
    
    def columnCount(self, parent=QModelIndex()):
        return len(self._header_labels)
    
    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return self._data[index.row()][index.column()]
        return None
    
    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return self._header_labels[section]
            elif orientation == Qt.Orientation.Vertical:
                return str(section + 1)
        return None
    
    def sort(self, column, order):
        self.layoutAboutToBeChanged.emit()
        self._data.sort(key=lambda x: x[column], reverse=(order == Qt.SortOrder.DescendingOrder))
        self.layoutChanged.emit()

class DockResultTable(QTableView):
    updateTreeSignal = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.property_full_name = ['Molecular Weight'  , 'Hydrogen Bond Donors'    , 'Hydrogen Bond Acceptors',
                                   'LogP'              , 'Topological Polar Surface Area', 'Rotatable Bonds'        ,
                                   'Number of Rings'   , 'Formal Charge'           , 'Number of Heavy Atoms'  ,
                                   'Molar Refractivity', 'Number of Atoms', 'QED']
        self.chem_prop_to_full_name_map = {'eng' : 'Score'               , 'mw'  : 'Molecular Weight'        ,
                                           'hbd' : 'Hydrogen Bond Donors', 'hba' : 'Hydrogen Bond Acceptors' ,
                                           'logp': 'LogP'                , 'tpsa': 'Topological Polar Surface Area',
                                           'rb'  : 'Rotatable Bonds'     , 'nor' : 'Number of Rings'         ,
                                           'fc'  : 'Formal Charge'       , 'nha' : 'Number of Heavy Atoms'   ,
                                           'mr'  : 'Molar Refractivity'  , 'na'  : 'Number of Atoms'         ,
                                           'engr': 'Score Rank'          , 'QED' : 'QED'                     ,
                                           'olds': 'Old Score'           , 'cnns': 'CNN Score'               ,
                                           'cnna': 'CNN Affinity'}
        self.chem_filter_dict = {'Name': [],
                                 'eng' : [()],
                                 'engr': [('≤', 1. )],
                                 'olds': [()],
                                 'cnns': [()],
                                 'cnna': [()],
                                 'mw'  : [()], 'hbd' : [()], 'hba' : [()], 'logp': [()],
                                 'tpsa': [()], 'rb'  : [()], 'nor' : [()], 'fc'  : [()], 'nha' : [()],
                                 'mr'  : [()], 'na'  : [()], 'QED' : [()]}
        self.chem_column_dict = {'Name'                   : True, 'SMILES'                  : False,
                                 'Score'                  : True, 'Molecular Weight'        : True,
                                 'Hydrogen Bond Donors'   : True, 'Hydrogen Bond Acceptors' : True,
                                 'LogP'                   : True, 'Topological Polar Surface Area': True,
                                 'Rotatable Bonds'        : True, 'Number of Rings'         : True,
                                 'Formal Charge'          : True, 'Number of Heavy Atoms'   : True,
                                 'Molar Refractivity'     : True, 'Number of Atoms'         : True,
                                 'QED'                    : True, 'Old Score'               : True,
                                 'CNN Score'              : True, 'CNN Affinity'            : True,}
        self.between_chem_ops_dict = {'And': True, 'Or': False}
        self.df = None
        self.bool_filter = None
        self.processing_df = None
        self.filtered_df = None
        
        self.horizontalHeader().sortIndicatorChanged.connect(self.update_processing_and_filter_df)
        
    def retrieve_empty_dict(self, fragment: bool, chemprop: bool, is_min: bool):
        if fragment:
            d = {'Name': [], 'Score': [], 'File Path': [], 'Fragments': []}
        else:
            d = {'Name': [], 'Score': [], 'File Path': []}
        if chemprop:
            d.update({k: [] for k in self.property_full_name})
            d['SMILES'] = []
        if is_min:
            d.update({'Old Score': []})
        # if is_gnina:
        #     d.update({'CNN Score': [], 'CNN Affinity': []})
        return d
    
    def setup_table(self):
        columns = ['Name', 'Best Score', 'SMILES'] + self.property_full_name
        self.model = TableModel([], columns)  # Initialize an empty model
        self.setModel(self.model)
        
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        # for i in range(len(columns)):
        #     if columns[i] == 'SMILES':
        #         header.setSectionResizeMode(i, QHeaderView.ResizeMode.Interactive)
        #     else:
        #         header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
    
    def populate_table(self, result_dict: dict):
        self.df = pd.DataFrame(data=result_dict)
        self.reset_current_table()
    
    def update_processing_and_filter_df(self, column, order, signal=True):
        if self.filtered_df is not None:
            ascending = order == Qt.SortOrder.AscendingOrder
            sorted_column_name = self.model.headerData(column, Qt.Orientation.Horizontal)
            if sorted_column_name == 'Best Score':
                sorted_column_name = 'Score'
                
            self.filtered_df = self.filtered_df.sort_values(by=sorted_column_name, ascending=ascending).reset_index(drop=True)
            self.processing_df = self.processing_df.sort_values(by=sorted_column_name, ascending=ascending).reset_index(drop=True)
            
            if signal & (len(self.filtered_df) <= 500):
                self.updateTreeSignal.emit()
    
    def reset_current_table(self):
        header = self.horizontalHeader()
        if header.isSortIndicatorShown():
            sorted_section = header.sortIndicatorSection()
        else:
            sorted_section = None
            
        self.bool_filter = self.filter_dataframe()
        dropped_cols = self.filter_column()
        self.processing_df = self.df[self.bool_filter].reset_index(drop=True)
        self.filtered_df = self.processing_df.drop(columns=dropped_cols)
        
        if sorted_section is not None:
            self.update_processing_and_filter_df(sorted_section,
                                                header.sortIndicatorOrder(),
                                                False)
            
        columns = self.filtered_df.columns
        header_columns = [c if c != 'Score' else 'Best Score' for c in columns]
        
        # Populate model data
        data = self.filtered_df.values.tolist()  # Convert the dataframe to a list of rows
        self.model = TableModel(data, header_columns)
        self.setModel(self.model)
        
        self.setSortingEnabled(True)
    
    def _compare_single_values(self, op, thres, target):
        if   op == '≥':
            return target >= thres
        elif op == '>':
            return target >  thres
        elif op == '<':
            return target <  thres
        elif op == '≤':
            return target <= thres
    
    def check_energy_ranking(self, list_operations: list[tuple], target: pd.Series):
        length = len(target)
        if len(list_operations) == 1:
            percentage = list_operations[0][1]
            partition_kth = max(0, int(length * percentage / 100) - 1)
            threshold = np.partition(target, partition_kth)[partition_kth]
            return self._compare_single_values(list_operations[0][0], threshold, target)
        else:
            op1, percentage1 = list_operations[0]
            op2, percentage2 = list_operations[1]
            partition_kth1 = max(0, int(length * percentage1 / 100) - 1)
            partition_kth2 = max(0, int(length * percentage2 / 100) - 1)
            thres1 = np.partition(target, partition_kth1)[partition_kth1]
            thres2 = np.partition(target, partition_kth2)[partition_kth2]
            or_option = False
            if (op1 in ('≥', '>') and op2 in ('≤', '<')):
                if thres1 >= thres2:
                    or_option = True
            elif (op1 in ('≤', '<') and op2 in ('≥', '>')):
                if thres1 >= thres2:
                    or_option = True
            op1_result = self._compare_single_values(op1, thres1, target)
            op2_result = self._compare_single_values(op2, thres2, target)
            if or_option:
                return op1_result | op2_result
            else:
                return op1_result & op2_result
    
    def check_chemprop_matching(self, list_operations: list[tuple], target: pd.Series):
        if len(list_operations) == 1:
            return self._compare_single_values(list_operations[0][0], list_operations[0][1], target)
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
            op1_result = self._compare_single_values(op1, thres1, target)
            op2_result = self._compare_single_values(op2, thres2, target)
            if or_option:
                return op1_result | op2_result
            else:
                return op1_result & op2_result
            
    def filter_dataframe(self):
        final_filter_dict = {}
        existing_columns = self.df.columns.to_list() + ['Score Rank']
        for prop, ops in self.chem_filter_dict.items():
            if prop != 'Name':
                if ops[0]:
                    full_name = self.chem_prop_to_full_name_map[prop]
                    if full_name in existing_columns:
                        final_filter_dict[full_name] = ops
            else:
                if ops:
                    final_filter_dict['Name'] = ops
        is_and = True if self.between_chem_ops_dict['And'] else False
        if is_and:
            bool_column = pd.Series([True] * len(self.df))
        else:
            bool_column = pd.Series([False] * len(self.df))
        for full_prop, ops in final_filter_dict.items():
            if full_prop == 'Score Rank':
                target = self.df['Score']
                if is_and:
                    bool_column &= self.check_energy_ranking(ops, target)
                else:
                    bool_column |= self.check_energy_ranking(ops, target)
            elif full_prop == 'Name':
                text_match_bool_column = pd.Series([False] * len(self.df))
                name_series = self.df['Name']
                for match_txt in ops:
                    text_match_bool_column |= name_series.str.contains(match_txt, case=True, regex=True)
                if is_and:
                    bool_column &= text_match_bool_column
                else:
                    bool_column |= text_match_bool_column
            else:
                target = self.df[full_prop]
                if is_and:
                    bool_column &= self.check_chemprop_matching(ops, target)
                else:
                    bool_column |= self.check_chemprop_matching(ops, target)
        return bool_column
    
    def filter_column(self):
        rm_cols = ['File Path']
        existing_columns = self.df.columns.to_list()
        if 'Fragments' in existing_columns:
            rm_cols.append('Fragments')
        dropped_columns = [c for c in existing_columns if c not in rm_cols and not self.chem_column_dict[c]] + rm_cols
        return dropped_columns
    
    def clear_everything(self):
        self.df = None
        self.bool_filter = None
        self.processing_df = None
        self.filtered_df = None
        self.setup_table()

class TableFilterDialog(QDialog):
    def __init__(self, chem_filter_dict: dict, chem_column_dict: dict, between_chem_ops_dict: dict):
        super().__init__()
        self.chem_filter_dict = copy.deepcopy(chem_filter_dict)
        self.chem_column_dict = copy.deepcopy(chem_column_dict)
        self.between_chem_ops_dict = copy.deepcopy(between_chem_ops_dict)
        self.initUI()
        screen = self.screen()
        screen_size = screen.availableGeometry()
        self.resize(screen_size.width() * 0.6, min(730, screen_size.height() * 0.85))
        
    def initUI(self):
        self.setWindowTitle('Filter')
        
        overall_layout = QVBoxLayout()
        chemprop_rule_name_layout = QHBoxLayout()
        between_logic_layout = QHBoxLayout()
        chemprop_rule_scroll_area = QScrollArea()
        chemprop_rule_scroll_area.horizontalScrollBar().setDisabled(True)
        chemprop_rule_scroll_area.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        
        chemprop_rule_widget = QWidget()
        chemprop_rule_layout = QGridLayout(chemprop_rule_widget)
        chemprop_rule_scroll_area.setWidget(chemprop_rule_widget)
        chemprop_rule_scroll_area.setWidgetResizable(True)
        
        chemprop_name_frame = QFrame()
        chemprop_name_frame.setFrameShape(QFrame.Shape.StyledPanel)
        chemprop_name_frame.setLineWidth(2)
        chemprop_name_layout = QGridLayout(chemprop_name_frame)
        
        between_logic_frame = QFrame()
        between_logic_frame.setFrameShape(QFrame.Shape.StyledPanel)
        between_logic_frame.setLineWidth(2)
        between_logic_layout = QGridLayout(between_logic_frame)
        
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_filters)
        self.button_box = QDialogButtonBox(QBtn)
        self.button_box.accepted.connect(self.accept_changes)
        self.button_box.rejected.connect(self.reject)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.reset_button, alignment=Qt.AlignmentFlag.AlignLeft)
        button_layout.addWidget(self.button_box  , alignment=Qt.AlignmentFlag.AlignRight)
        
        self.config_values = {'eng' : {'label': 'Score'                   , 'min': -1e6, 'max': 1e6, 'unit': ''        , 'step': .5, 'spinbox': QDoubleSpinBox},
                              'engr': {'label': 'Score Rank'              , 'min': 0   , 'max': 100, 'unit': '%'       , 'step': 1 , 'spinbox': QDoubleSpinBox},
                              'cnna': {'label': 'CNN Affinity'            , 'min': -1e6, 'max': 1e6, 'unit': ''        , 'step': .5, 'spinbox': QDoubleSpinBox},
                              'cnns': {'label': 'CNN Score'               , 'min': 0   , 'max': 1  , 'unit': ''        , 'step': .1, 'spinbox': QDoubleSpinBox},
                              'olds': {'label': 'Old Score'               , 'min': -1e6, 'max': 1e6, 'unit': ''        , 'step': .5, 'spinbox': QDoubleSpinBox},
                              'mw'  : {'label': 'Molecular Weight'        , 'min': 0   , 'max': 1e6, 'unit': 'Da'      , 'step': 5., 'spinbox': QDoubleSpinBox},
                              'hbd' : {'label': 'Hydrogen Bond Donors'    , 'min': 0   , 'max': 1e6, 'unit': ''        , 'step': 1 , 'spinbox': QSpinBox      },
                              'hba' : {'label': 'Hydrogen Bond Acceptors' , 'min': 0   , 'max': 1e6, 'unit': ''        , 'step': 1 , 'spinbox': QSpinBox      },
                              'logp': {'label': 'LogP'                    , 'min': -1e6, 'max': 1e6, 'unit': ''        , 'step': 1., 'spinbox': QDoubleSpinBox},
                              'tpsa': {'label': 'Topological Polar Surface Area', 'min': 0   , 'max': 1e6, 'unit': 'Å²'      , 'step': 5., 'spinbox': QDoubleSpinBox},
                              'rb'  : {'label': 'Rotatable Bonds'         , 'min': 0   , 'max': 1e6, 'unit': ''        , 'step': 1 , 'spinbox': QSpinBox      },
                              'nor' : {'label': 'Number of Rings'         , 'min': 0   , 'max': 1e6, 'unit': ''        , 'step': 1 , 'spinbox': QSpinBox      },
                              'fc'  : {'label': 'Formal Charge'           , 'min': -1e6, 'max': 1e6, 'unit': ''        , 'step': 1 , 'spinbox': QSpinBox      },
                              'nha' : {'label': 'Number of Heavy Atoms'   , 'min': 0   , 'max': 1e6, 'unit': ''        , 'step': 1 , 'spinbox': QSpinBox      },
                              'mr'  : {'label': 'Molar Refractivity'      , 'min': 0   , 'max': 1e6, 'unit': ''        , 'step': 1 , 'spinbox': QDoubleSpinBox},
                              'na'  : {'label': 'Number of Atoms'         , 'min': 0   , 'max': 1e6, 'unit': ''        , 'step': 1 , 'spinbox': QSpinBox      },
                              'QED' : {'label': 'QED'                     , 'min': 0   , 'max': 1  , 'unit': ''        , 'step': .1, 'spinbox': QDoubleSpinBox},}
        
        ### Chemical rule layout ###
        self.widget_mapping = {}
        
        chemprop_label = QLabel('<b>Filters :</b>')
        chemprop_rule_layout.addWidget(chemprop_label, 0, 0, 1, 4)
        
        ### Add Name row ###
        checkbox = QCheckBox()
        checkbox.setText('Name :')
        checkbox.setChecked(False)
        checkbox.stateChanged.connect(self.change_downstream_options)
        
        condition_layout = QVBoxLayout()
        
        add_button = QToolButton()
        add_button.setText("+")
        add_button.setStyleSheet('font-family: "Courier New", Courier, monospace; font-weight: 1000; font-size: 12px')
        add_button.clicked.connect(lambda _, k='Name': self.add_condition_row(k))
        add_button.setDisabled(True)
        
        chemprop_rule_layout.addWidget(checkbox, 1, 0)
        chemprop_rule_layout.addLayout(condition_layout, 1, 1)
        chemprop_rule_layout.addWidget(add_button, 1, 3)
        h_line = QFrame()
        h_line.setFrameShape(QFrame.Shape.HLine)
        chemprop_rule_layout.addWidget(h_line, 2, 0, 1, 4)
        
        self.widget_mapping['Name'] = {'checkbox'        : checkbox,
                                       'add_button'      : add_button,
                                       'condition_layout': condition_layout,
                                       'conditions'      : [],}
        
        for i, (config_key, config_dict) in enumerate(self.config_values.items(), 2):
            checkbox = QCheckBox()
            checkbox.setText(f'{config_dict['label']} :')
            checkbox.setChecked(False)
            checkbox.stateChanged.connect(self.change_downstream_options)
            
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
            # self.add_condition_row(config_key)
        
        self.update_chemprops_with_dict(self.chem_filter_dict)
        
        ### Chemprop Column List ###
        column_filter_label = QLabel('<b>Columns :</b>')
        self.column_filter_list = QListWidget()
        self.column_filter_list.setSelectionMode(QListWidget.NoSelection)
        for column, checked in self.chem_column_dict.items():
            self.add_column_filter(column, checked)
        
        ### Between-filters
        filter_ops_between_label = QLabel('<b>Logic for Different Filters :</b>')
        between_logic_layout.addWidget(filter_ops_between_label)
        and_or_button_group = QButtonGroup()
        self.ops_between_radios = {}
        for name in ['And', 'Or']:
            radio = QRadioButton(name)
            radio.setChecked(self.between_chem_ops_dict[name])
            and_or_button_group.addButton(radio)
            self.ops_between_radios[name] = radio
            between_logic_layout.addWidget(radio)
        
        chemprop_name_layout.addWidget(column_filter_label)
        chemprop_name_layout.addWidget(self.column_filter_list)
        chemprop_name_layout.addWidget(between_logic_frame)
        
        chemprop_rule_name_layout.addWidget(chemprop_rule_scroll_area)
        chemprop_rule_name_layout.addWidget(chemprop_name_frame)
        
        overall_layout.addLayout(chemprop_rule_name_layout)
        overall_layout.addLayout(button_layout)
        self.setLayout(overall_layout)
        
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # QApplication.instance().processEvents()
        # self.adjustSize()
    
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
                                    if len(widgets[k]) == 2 and k != 'Name':
                                        if v != 'layout':
                                            v_dict[v].setEnabled(state)
                                        widgets['add_button'].setDisabled(True)
                                    else:
                                        if v not in ['layout', 'rm_button']:
                                            v_dict[v].setEnabled(state)
                                            
    def add_condition_row(self, key):
        if key != 'Name':
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
            value_spinbox.setFixedWidth(100)
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
        else:
            enabled = self.widget_mapping[key]['checkbox'].isChecked()
            condition_layout = self.widget_mapping[key]['condition_layout']
            
            regex_lineedit = QLineEdit()
            regex_lineedit.setMinimumWidth(180)
            regex_lineedit.setEnabled(enabled)
            regex_lineedit.setPlaceholderText('Matching Text...')
            
            rm_button = QToolButton()
            rm_button.setText("-")
            rm_button.setStyleSheet('font-family: "Courier New", Courier, monospace; font-weight: 1000; font-size: 20px')
            rm_button.clicked.connect(lambda _, k=key, rl=rm_button: self.rm_condition_row(k, rl))
            rm_button.setMaximumSize(QSize(15, 15))
            if self.widget_mapping[key]['conditions']:
                rm_button.setEnabled(enabled)
            else:
                rm_button.setEnabled(False)
            
            row_layout = QHBoxLayout()
            row_layout.addWidget(regex_lineedit, alignment=Qt.AlignmentFlag.AlignLeft)
            row_layout.addWidget(rm_button)
            row_layout.setSpacing(1)
            condition_layout.addLayout(row_layout)
            self.widget_mapping[key]['conditions'].append({'lineedit' : regex_lineedit,
                                                           'rm_button': rm_button,
                                                           'layout'   : row_layout})
            
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
    
    def remove_all_conditions(self, key):
        condition_list = self.widget_mapping[key]['conditions']
        for condition in condition_list:
            for i in reversed(range(condition['layout'].count())):
                widget = condition['layout'].itemAt(i).widget()
                if widget is not None:
                    widget.setParent(None)
                self.widget_mapping[key]['condition_layout'].removeItem(condition['layout'])
        self.widget_mapping[key]['conditions'] = []
        
    def update_chemprops_with_dict(self, chemprops_dict: dict):
        for chemprop_key, chemprop_value in chemprops_dict.items():
            self.remove_all_conditions(chemprop_key)
            self.add_condition_row(chemprop_key)
            if chemprop_key != 'Name':
                if not chemprop_value[0]:
                    self.widget_mapping[chemprop_key]['checkbox'].setCheckState(Qt.CheckState.Unchecked)
                elif len(chemprop_value) == 1:
                    comparison, value = chemprop_value[0]
                    self.widget_mapping[chemprop_key]['checkbox'].setCheckState(Qt.CheckState.Checked)
                    self.widget_mapping[chemprop_key]['conditions'][0]['combobox'].setCurrentText(comparison)
                    self.widget_mapping[chemprop_key]['conditions'][0]['spinbox'].setValue(value)
                elif len(chemprop_value) == 2:
                    comparison_1, value_1 = chemprop_value[0]
                    comparison_2, value_2 = chemprop_value[1]
                    self.add_condition_row(chemprop_key)
                    self.widget_mapping[chemprop_key]['checkbox'].setCheckState(Qt.CheckState.Checked)
                    self.widget_mapping[chemprop_key]['conditions'][0]['combobox'].setCurrentText(comparison_1)
                    self.widget_mapping[chemprop_key]['conditions'][0]['spinbox'].setValue(value_1)
                    self.widget_mapping[chemprop_key]['conditions'][1]['combobox'].setCurrentText(comparison_2)
                    self.widget_mapping[chemprop_key]['conditions'][1]['spinbox'].setValue(value_2)
                    self.widget_mapping[chemprop_key]['add_button'].setDisabled(True)
            else:
                if chemprop_value:
                    self.widget_mapping[chemprop_key]['checkbox'].setCheckState(Qt.CheckState.Checked)
                    for _ in range(len(chemprop_value)-1):
                        self.add_condition_row(chemprop_key)
                    for i, text in enumerate(chemprop_value):
                        self.widget_mapping[chemprop_key]['conditions'][i]['lineedit'].setText(text)
                
    def reset_filters(self):
        for chemprop_key in self.config_values:
            self.remove_all_conditions(chemprop_key)
            self.add_condition_row(chemprop_key)
            self.widget_mapping[chemprop_key]['checkbox'].setChecked(False)
            
        for index in range(self.column_filter_list.count()):
            item = self.column_filter_list.item(index)
            if index != 1:
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setCheckState(Qt.CheckState.Unchecked)
            
        # QApplication.instance().processEvents()
        # self.adjustSize()
    
    def add_column_filter(self, column_name, checked):
        item = QListWidgetItem(column_name)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        if checked:
            item.setCheckState(Qt.CheckState.Checked)
        else:
            item.setCheckState(Qt.CheckState.Unchecked)
        self.column_filter_list.addItem(item)
    
    def update_chem_filter_dict(self):
        for chemprop_key in self.widget_mapping:
            if chemprop_key != 'Name':
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
            else:
                all_regex = []
                if self.widget_mapping[chemprop_key]['checkbox'].isChecked():
                    for condition in self.widget_mapping[chemprop_key]['conditions']:
                        regex = condition['lineedit'].text()
                        try:
                            re.compile(regex)
                            all_regex.append(regex)
                        except:
                            QMessageBox.critical(self, 'Regex Error', f'{regex} is not a valid regex!')
                            return 'Failed'
                self.chem_filter_dict[chemprop_key] = all_regex
    
    def update_chem_column_dict(self):
        for index in range(self.column_filter_list.count()):
            item = self.column_filter_list.item(index)
            if item.checkState() == Qt.CheckState.Checked:
                self.chem_column_dict[item.text()] = True
            else:
                self.chem_column_dict[item.text()] = False
    
    def update_chem_ops_dict(self):
        for name, radio in self.ops_between_radios.items():
            self.between_chem_ops_dict[name] = radio.isChecked()
    
    def accept_changes(self):
        regex_check = self.update_chem_filter_dict()
        if regex_check == 'Failed':
            return
        self.update_chem_column_dict()
        self.update_chem_ops_dict()
        self.accept()

class PlotFilterDataframe:
    def __init__(self):
        self.df = None
        self.filtered_df = None
        self.chem_prop_to_full_name_map = {'eng' : 'Score'               , 'mw'  : 'Molecular Weight'        ,
                                           'hbd' : 'Hydrogen Bond Donors', 'hba' : 'Hydrogen Bond Acceptors' ,
                                           'logp': 'LogP'                , 'tpsa': 'Topological Polar Surface Area',
                                           'rb'  : 'Rotatable Bonds'     , 'nor' : 'Number of Rings'         ,
                                           'fc'  : 'Formal Charge'       , 'nha' : 'Number of Heavy Atoms'   ,
                                           'mr'  : 'Molar Refractivity'  , 'na'  : 'Number of Atoms'         ,
                                           'engr': 'Score Rank'         , 'QED' : 'QED'                     ,
                                           'olds': 'Old Score'           , 'cnns': 'CNN Score'               ,
                                           'cnna': 'CNN Affinity'}
        
    def set_df(self, result_dict: dict):
        if 'Old Score' in result_dict:
            appended = ['Old Score']
        if 'CNN Score' in result_dict:
            appended = ['CNN Score', 'CNN Affinity']
        else:
            appended = []
        if 'QED' in result_dict:
            self.df = pd.DataFrame(data=result_dict)[['Name', 'Score'] + appended + list(property_functions)]
        else:
            self.df = pd.DataFrame(data=result_dict)[['Name', 'Score'] + appended]
        
    def _compare_single_values(self, op, thres, target):
        if   op == '≥':
            return target >= thres
        elif op == '>':
            return target >  thres
        elif op == '<':
            return target <  thres
        elif op == '≤':
            return target <= thres
    
    def check_energy_ranking(self, list_operations: list[tuple], target: pd.Series):
        length = len(target)
        if len(list_operations) == 1:
            percentage = list_operations[0][1]
            partition_kth = max(0, int(length * percentage / 100) - 1)
            threshold = np.partition(target, partition_kth)[partition_kth]
            return self._compare_single_values(list_operations[0][0], threshold, target)
        else:
            op1, percentage1 = list_operations[0]
            op2, percentage2 = list_operations[1]
            partition_kth1 = max(0, int(length * percentage1 / 100) - 1)
            partition_kth2 = max(0, int(length * percentage2 / 100) - 1)
            thres1 = np.partition(target, partition_kth1)[partition_kth1]
            thres2 = np.partition(target, partition_kth2)[partition_kth2]
            or_option = False
            if (op1 in ('≥', '>') and op2 in ('≤', '<')):
                if thres1 >= thres2:
                    or_option = True
            elif (op1 in ('≤', '<') and op2 in ('≥', '>')):
                if thres1 >= thres2:
                    or_option = True
            op1_result = self._compare_single_values(op1, thres1, target)
            op2_result = self._compare_single_values(op2, thres2, target)
            if or_option:
                return op1_result | op2_result
            else:
                return op1_result & op2_result
    
    def check_chemprop_matching(self, list_operations: list[tuple], target: pd.Series):
        if len(list_operations) == 1:
            return self._compare_single_values(list_operations[0][0], list_operations[0][1], target)
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
            op1_result = self._compare_single_values(op1, thres1, target)
            op2_result = self._compare_single_values(op2, thres2, target)
            if or_option:
                return op1_result | op2_result
            else:
                return op1_result & op2_result
            
    def filter_dataframe(self, chem_filter_dict: dict):
        existing_columns = self.df.columns.to_list() + ['Score Rank']
        final_filter_dict = {}
        for prop, ops in chem_filter_dict.items():
            if prop != 'Name':
                full_name = self.chem_prop_to_full_name_map[prop]
                if ops[0] and full_name in existing_columns:
                    final_filter_dict[full_name] = ops
            else:
                if ops:
                    final_filter_dict['Name'] = ops
        bool_column = pd.Series([True] * len(self.df))
        for full_prop, ops in final_filter_dict.items():
            if full_prop == 'Score Rank':
                target = self.df['Score']
                bool_column &= self.check_energy_ranking(ops, target)
            elif full_prop == 'Name':
                text_match_bool_column = pd.Series([False] * len(self.df))
                name_series = self.df['Name']
                for match_txt in ops:
                    text_match_bool_column |= name_series.str.contains(match_txt, case=True, regex=True)
                bool_column &= text_match_bool_column
            else:
                target = self.df[full_prop]
                bool_column &= self.check_chemprop_matching(ops, target)    # currently set to "AND" between each chemical properties
        return bool_column
    
    def energy_outlier_detection(self, chem_filter_dict: dict):
        # set it to median ± 3.5 * IQR
        eng = self.df['Score'].to_numpy()
        median = np.median(eng)
        iqr = np.percentile(eng, 75) - np.percentile(eng, 25)
        chem_filter_dict['eng'] = [('≥', median - 3.5 * iqr), ('≤', median + 3.5 * iqr)]
        return chem_filter_dict
    
    def apply_filter(self, chemprop_filter_dict: dict):
        if self.df is not None:
            bool_filter = self.filter_dataframe(chemprop_filter_dict)
            self.filtered_df = self.df[bool_filter].reset_index(drop=True)
    
    def clear_everything(self):
        self.df = None
        self.filtered_df = None

class PlotFilterDialog(QDialog):
    def __init__(self, chem_filter_dict: dict):
        super().__init__()
        self.chem_filter_dict = copy.deepcopy(chem_filter_dict)
        self.initUI()
        screen = self.screen()
        screen_size = screen.availableGeometry()
        self.resize(screen_size.width() * 0.3, min(730, screen_size.height() * 0.85))
        
    def initUI(self):
        self.setWindowTitle('Property Filter')
        
        overall_layout = QVBoxLayout()
        chemprop_rule_name_layout = QHBoxLayout()
        chemprop_rule_scroll_area = QScrollArea()
        chemprop_rule_scroll_area.horizontalScrollBar().setDisabled(True)
        chemprop_rule_scroll_area.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        
        chemprop_rule_widget = QWidget()
        chemprop_rule_layout = QGridLayout(chemprop_rule_widget)
        chemprop_rule_scroll_area.setWidget(chemprop_rule_widget)
        chemprop_rule_scroll_area.setWidgetResizable(True)
        
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_filters)
        self.button_box = QDialogButtonBox(QBtn)
        self.button_box.accepted.connect(self.accept_changes)
        self.button_box.rejected.connect(self.reject)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.reset_button, alignment=Qt.AlignmentFlag.AlignLeft)
        button_layout.addWidget(self.button_box  , alignment=Qt.AlignmentFlag.AlignRight)
        
        self.config_values = {'eng' : {'label': 'Score'                   , 'min': -1e6, 'max': 1e6, 'unit': ''        , 'step': .5, 'spinbox': QDoubleSpinBox},
                              'engr': {'label': 'Score Rank'              , 'min': 0   , 'max': 100, 'unit': '%'       , 'step': 1 , 'spinbox': QDoubleSpinBox},
                              'cnna': {'label': 'CNN Affinity'            , 'min': -1e6, 'max': 1e6, 'unit': ''        , 'step': .5, 'spinbox': QDoubleSpinBox},
                              'cnns': {'label': 'CNN Score'               , 'min': 0   , 'max': 1  , 'unit': ''        , 'step': .1, 'spinbox': QDoubleSpinBox},
                              'olds': {'label': 'Old Score'               , 'min': -1e6, 'max': 1e6, 'unit': ''        , 'step': .5, 'spinbox': QDoubleSpinBox},
                              'mw'  : {'label': 'Molecular Weight'        , 'min': 0   , 'max': 1e6, 'unit': 'Da'      , 'step': 5., 'spinbox': QDoubleSpinBox},
                              'hbd' : {'label': 'Hydrogen Bond Donors'    , 'min': 0   , 'max': 1e6, 'unit': ''        , 'step': 1 , 'spinbox': QSpinBox      },
                              'hba' : {'label': 'Hydrogen Bond Acceptors' , 'min': 0   , 'max': 1e6, 'unit': ''        , 'step': 1 , 'spinbox': QSpinBox      },
                              'logp': {'label': 'LogP'                    , 'min': -1e6, 'max': 1e6, 'unit': ''        , 'step': 1., 'spinbox': QDoubleSpinBox},
                              'tpsa': {'label': 'Topological Polar Surface Area', 'min': 0   , 'max': 1e6, 'unit': 'Å²'      , 'step': 5., 'spinbox': QDoubleSpinBox},
                              'rb'  : {'label': 'Rotatable Bonds'         , 'min': 0   , 'max': 1e6, 'unit': ''        , 'step': 1 , 'spinbox': QSpinBox      },
                              'nor' : {'label': 'Number of Rings'         , 'min': 0   , 'max': 1e6, 'unit': ''        , 'step': 1 , 'spinbox': QSpinBox      },
                              'fc'  : {'label': 'Formal Charge'           , 'min': -1e6, 'max': 1e6, 'unit': ''        , 'step': 1 , 'spinbox': QSpinBox      },
                              'nha' : {'label': 'Number of Heavy Atoms'   , 'min': 0   , 'max': 1e6, 'unit': ''        , 'step': 1 , 'spinbox': QSpinBox      },
                              'mr'  : {'label': 'Molar Refractivity'      , 'min': 0   , 'max': 1e6, 'unit': ''        , 'step': 1 , 'spinbox': QDoubleSpinBox},
                              'na'  : {'label': 'Number of Atoms'         , 'min': 0   , 'max': 1e6, 'unit': ''        , 'step': 1 , 'spinbox': QSpinBox      },
                              'QED' : {'label': 'QED'                     , 'min': 0   , 'max': 1  , 'unit': ''        , 'step': .1, 'spinbox': QDoubleSpinBox},}
        
        ### Chemical rule layout ###
        self.widget_mapping = {}
        
        chemprop_label = QLabel('<b>Filters :</b>')
        chemprop_rule_layout.addWidget(chemprop_label, 0, 0, 1, 4)
        
        ### Add Name row ###
        checkbox = QCheckBox()
        checkbox.setText('Name :')
        checkbox.setChecked(False)
        checkbox.stateChanged.connect(self.change_downstream_options)
        
        condition_layout = QVBoxLayout()
        
        add_button = QToolButton()
        add_button.setText("+")
        add_button.setStyleSheet('font-family: "Courier New", Courier, monospace; font-weight: 1000; font-size: 12px')
        add_button.clicked.connect(lambda _, k='Name': self.add_condition_row(k))
        add_button.setDisabled(True)
        
        chemprop_rule_layout.addWidget(checkbox, 1, 0)
        chemprop_rule_layout.addLayout(condition_layout, 1, 1)
        chemprop_rule_layout.addWidget(add_button, 1, 3)
        h_line = QFrame()
        h_line.setFrameShape(QFrame.Shape.HLine)
        chemprop_rule_layout.addWidget(h_line, 2, 0, 1, 4)
        
        self.widget_mapping['Name'] = {'checkbox'        : checkbox,
                                       'add_button'      : add_button,
                                       'condition_layout': condition_layout,
                                       'conditions'      : [],}
        
        for i, (config_key, config_dict) in enumerate(self.config_values.items(), 2):
            checkbox = QCheckBox()
            checkbox.setText(f'{config_dict['label']} :')
            checkbox.setChecked(False)
            checkbox.stateChanged.connect(self.change_downstream_options)
            
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
        
        self.update_chemprops_with_dict(self.chem_filter_dict)
        chemprop_rule_name_layout.addWidget(chemprop_rule_scroll_area)
        
        overall_layout.addLayout(chemprop_rule_name_layout)
        overall_layout.addLayout(button_layout)
        self.setLayout(overall_layout)
        
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # QApplication.instance().processEvents()
        # self.adjustSize()
    
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
                                    if len(widgets[k]) == 2 and k != 'Name':
                                        if v != 'layout':
                                            v_dict[v].setEnabled(state)
                                        widgets['add_button'].setDisabled(True)
                                    else:
                                        if v not in ['layout', 'rm_button']:
                                            v_dict[v].setEnabled(state)
                                            
    def add_condition_row(self, key):
        if key != 'Name':
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
            value_spinbox.setFixedWidth(100)
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
        else:
            enabled = self.widget_mapping[key]['checkbox'].isChecked()
            condition_layout = self.widget_mapping[key]['condition_layout']
            
            regex_lineedit = QLineEdit()
            regex_lineedit.setMinimumWidth(180)
            regex_lineedit.setEnabled(enabled)
            regex_lineedit.setPlaceholderText('Matching Text...')
            
            rm_button = QToolButton()
            rm_button.setText("-")
            rm_button.setStyleSheet('font-family: "Courier New", Courier, monospace; font-weight: 1000; font-size: 20px')
            rm_button.clicked.connect(lambda _, k=key, rl=rm_button: self.rm_condition_row(k, rl))
            rm_button.setMaximumSize(QSize(15, 15))
            if self.widget_mapping[key]['conditions']:
                rm_button.setEnabled(enabled)
            else:
                rm_button.setEnabled(False)
            
            row_layout = QHBoxLayout()
            row_layout.addWidget(regex_lineedit, alignment=Qt.AlignmentFlag.AlignLeft)
            row_layout.addWidget(rm_button)
            row_layout.setSpacing(1)
            condition_layout.addLayout(row_layout)
            self.widget_mapping[key]['conditions'].append({'lineedit' : regex_lineedit,
                                                           'rm_button': rm_button,
                                                           'layout'   : row_layout})
            
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
    
    def remove_all_conditions(self, key):
        condition_list = self.widget_mapping[key]['conditions']
        for condition in condition_list:
            for i in reversed(range(condition['layout'].count())):
                widget = condition['layout'].itemAt(i).widget()
                if widget is not None:
                    widget.setParent(None)
                self.widget_mapping[key]['condition_layout'].removeItem(condition['layout'])
        self.widget_mapping[key]['conditions'] = []
        
    def update_chemprops_with_dict(self, chemprops_dict: dict):
        for chemprop_key, chemprop_value in chemprops_dict.items():
            self.remove_all_conditions(chemprop_key)
            self.add_condition_row(chemprop_key)
            if chemprop_key != 'Name':
                if not chemprop_value[0]:
                    self.widget_mapping[chemprop_key]['checkbox'].setCheckState(Qt.CheckState.Unchecked)
                elif len(chemprop_value) == 1:
                    comparison, value = chemprop_value[0]
                    self.widget_mapping[chemprop_key]['checkbox'].setCheckState(Qt.CheckState.Checked)
                    self.widget_mapping[chemprop_key]['conditions'][0]['combobox'].setCurrentText(comparison)
                    self.widget_mapping[chemprop_key]['conditions'][0]['spinbox'].setValue(value)
                elif len(chemprop_value) == 2:
                    comparison_1, value_1 = chemprop_value[0]
                    comparison_2, value_2 = chemprop_value[1]
                    self.add_condition_row(chemprop_key)
                    self.widget_mapping[chemprop_key]['checkbox'].setCheckState(Qt.CheckState.Checked)
                    self.widget_mapping[chemprop_key]['conditions'][0]['combobox'].setCurrentText(comparison_1)
                    self.widget_mapping[chemprop_key]['conditions'][0]['spinbox'].setValue(value_1)
                    self.widget_mapping[chemprop_key]['conditions'][1]['combobox'].setCurrentText(comparison_2)
                    self.widget_mapping[chemprop_key]['conditions'][1]['spinbox'].setValue(value_2)
                    self.widget_mapping[chemprop_key]['add_button'].setDisabled(True)
            else:
                if chemprop_value:
                    self.widget_mapping[chemprop_key]['checkbox'].setCheckState(Qt.CheckState.Checked)
                    for _ in range(len(chemprop_value)-1):
                        self.add_condition_row(chemprop_key)
                    for i, text in enumerate(chemprop_value):
                        self.widget_mapping[chemprop_key]['conditions'][i]['lineedit'].setText(text)
                
    def reset_filters(self):
        for chemprop_key in self.config_values:
            self.remove_all_conditions(chemprop_key)
            self.add_condition_row(chemprop_key)
            self.widget_mapping[chemprop_key]['checkbox'].setChecked(False)
            
        # QApplication.instance().processEvents()
        # self.adjustSize()
            
    def update_chem_filter_dict(self):
        for chemprop_key in self.widget_mapping:
            if chemprop_key != 'Name':
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
            else:
                all_regex = []
                if self.widget_mapping[chemprop_key]['checkbox'].isChecked():
                    for condition in self.widget_mapping[chemprop_key]['conditions']:
                        regex = condition['lineedit'].text()
                        try:
                            re.compile(regex)
                            all_regex.append(regex)
                        except:
                            QMessageBox.critical(self, 'Regex Error', f'{regex} is not a valid regex!')
                            return 'Failed'
                self.chem_filter_dict[chemprop_key] = all_regex
    
    def accept_changes(self):
        regex_check = self.update_chem_filter_dict()
        if regex_check == 'Failed':
            return
        self.accept()

class AutoFilterDialog(QDialog):
    apply_signal = Signal(dict, dict)
    
    def __init__(self, chem_filter_dict: dict, chem_filter_bool: dict):
        super().__init__()
        self.initUI(chem_filter_dict, chem_filter_bool)
        screen = self.screen()
        screen_size = screen.availableGeometry()
        self.resize(screen_size.width() * 0.3, min(730, screen_size.height() * 0.85))
        
    def initUI(self, chem_filter_dict, chem_filter_bool):
        self.setWindowTitle('Ligand Filters')
        
        self.overall_layout = QVBoxLayout()
        
        chemprop_rule_scroll_area = QScrollArea()
        chemprop_rule_scroll_area.horizontalScrollBar().setDisabled(True)
        chemprop_rule_scroll_area.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        
        chemprop_rule_widget = QWidget()
        chemprop_rule_layout = QGridLayout(chemprop_rule_widget)
        chemprop_rule_scroll_area.setWidget(chemprop_rule_widget)
        chemprop_rule_scroll_area.setWidgetResizable(True)
        
        example_layout = QHBoxLayout()
        example_frame = QFrame()
        example_frame.setFrameShape(QFrame.Shape.StyledPanel)
        example_frame.setLineWidth(2)
        
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
        
        QBtn = QDialogButtonBox.Apply | QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_filters)
        self.button_box = QDialogButtonBox(QBtn)
        self.button_box.accepted.connect(self.accept_changes)
        self.button_box.rejected.connect(self.reject)
        self.button_box.button(QDialogButtonBox.Apply).clicked.connect(self.apply_changes)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.reset_button, alignment=Qt.AlignmentFlag.AlignLeft)
        button_layout.addWidget(self.button_box, alignment=Qt.AlignmentFlag.AlignRight)
        self.chem_filter_dict = copy.deepcopy(chem_filter_dict)
        self.chem_filter_bool = copy.deepcopy(chem_filter_bool)
        
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
        
        ### Chemical rule layout ###
        self.widget_mapping = {}
        self.enabled_rdkit_filter = 0
        
        chemprop_label = QLabel('<b>Chemical Property Filters :</b>')
        chemprop_rule_layout.addWidget(chemprop_label, 0, 0, 1, 4)
        for i, (config_key, config_dict) in enumerate(self.config_values.items(), 1):
            checkbox = QCheckBox()
            checkbox.setText(f'{config_dict['label']} :')
            checkbox.setTristate(True)
            checkbox.setChecked(False)
            checkbox.stateChanged.connect(self.change_downstream_options)
            
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
            self.add_condition_row(config_key)
        
        chemprop_threshold_label = QLabel('Partial Filter threshold :')
        chemprop_threshold_label.setToolTip('Chemicals with hits "<" threshold will be removed. (0 = no filter)')
        self.chemprop_threshold_spinbox = QSpinBox()
        self.chemprop_threshold_spinbox.setSingleStep(1)
        self.chemprop_threshold_spinbox.setRange(0, self.chem_filter_bool['partial_filter_threshold'])
        chemprop_rule_layout.addWidget(chemprop_threshold_label, 2 * len(self.config_values) + 2, 0)
        chemprop_rule_layout.addWidget(self.chemprop_threshold_spinbox, 2 * len(self.config_values) + 2, 1)
        
        self.update_chemprops_with_dict(self.chem_filter_dict, self.chem_filter_bool)  # Update the chemprop filter to current settings
        self.chemprop_threshold_spinbox.setValue(self.chem_filter_bool['partial_filter_threshold'])
        
        self.overall_layout.addWidget(example_frame)
        self.overall_layout.addWidget(chemprop_rule_scroll_area)
        self.overall_layout.addLayout(button_layout)
        self.setLayout(self.overall_layout)
        
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.adjustSize()
    
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
    
    def update_chemprop_spinbox_value(self):
        cnt = 0
        curr_spin_value = self.chemprop_threshold_spinbox.value()
        for k in self.widget_mapping:
            if self.widget_mapping[k]['checkbox'].checkState() == Qt.CheckState.PartiallyChecked:
                cnt += 1
        self.chemprop_threshold_spinbox.setRange(0, cnt)
        if curr_spin_value + 1 == cnt:
            self.chemprop_threshold_spinbox.setValue(cnt)
                
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
            if chemprop_key in self.widget_mapping:
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
                    
    def reset_filters(self):
        for chemprop_key in self.config_values:
            self.remove_all_conditions(chemprop_key)
            self.add_condition_row(chemprop_key)
            self.widget_mapping[chemprop_key]['checkbox'].setChecked(False)
        self.chemprop_threshold_spinbox.setValue(0)
        self.chemprop_threshold_spinbox.setRange(0, 0)
        
    def update_chem_filter_dict(self, return_dict: bool = False):
        if return_dict:
            chem_filter_dict = copy.deepcopy(self.chem_filter_dict)
            chem_filter_bool = copy.deepcopy(self.chem_filter_bool)
            for chemprop_key in self.widget_mapping:
                if not self.widget_mapping[chemprop_key]['checkbox'].isChecked():
                    chem_filter_dict[chemprop_key] = [()]
                elif len(self.widget_mapping[chemprop_key]['conditions']) == 1:
                    comp  = self.widget_mapping[chemprop_key]['conditions'][0]['combobox'].currentText()
                    value = self.widget_mapping[chemprop_key]['conditions'][0]['spinbox'].value()
                    chem_filter_dict[chemprop_key] = [(comp, value)]
                else:
                    comp_1  = self.widget_mapping[chemprop_key]['conditions'][0]['combobox'].currentText()
                    value_1 = self.widget_mapping[chemprop_key]['conditions'][0]['spinbox'].value()
                    comp_2  = self.widget_mapping[chemprop_key]['conditions'][1]['combobox'].currentText()
                    value_2 = self.widget_mapping[chemprop_key]['conditions'][1]['spinbox'].value()
                    chem_filter_dict[chemprop_key] = [(comp_1, value_1), (comp_2, value_2)]
                state = self.widget_mapping[chemprop_key]['checkbox'].checkState()
                if state == Qt.CheckState.Checked:
                    chem_filter_bool[chemprop_key] = True
                elif state == Qt.CheckState.PartiallyChecked:
                    chem_filter_bool[chemprop_key] = 'partial'
                else:
                    chem_filter_bool[chemprop_key] = False
            chem_filter_bool['partial_filter_threshold'] = self.chemprop_threshold_spinbox.value()
            return chem_filter_dict, chem_filter_bool
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
        
    def accept_changes(self):
        self.update_chem_filter_dict()
        self.accept()
        
    def apply_changes(self):
        d_1, d_2 = self.update_chem_filter_dict(True)
        self.apply_signal.emit(d_1, d_2)
        
class ImageLabel(QLabel):
    def __init__(self, img: Image, resize: None | int = None):
        super().__init__()
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_save_menu)
        
        save_png_action = QAction('Save PNG', self)
        save_png_action.triggered.connect(self.create_png_and_save_dialog)
        
        self.popup_menu = QMenu()
        self.popup_menu.addAction(save_png_action)
        
        self.setStyleSheet("""QToolTip {font-family: "Courier New", Courier, monospace;}""")
        self.setToolTipDuration(3e4)
        
        self.img = img
        self.set_pixmap(None, resize)
        
    def set_pixmap(self, img=None, size=None):
        if img is not None:
            self.img = img
        if size is None:
            size = 200
        pixmap = QPixmap(ImageQt(self.img.resize((size, size))))
        self.setPixmap(pixmap)
    
    def show_save_menu(self, pos):
        self.popup_menu.exec(self.mapToGlobal(pos))
    
    def create_png_and_save_dialog(self):
        save_file, _ = QFileDialog.getSaveFileName(self, 'Select Saved PNG Path', '', 'Portable Network Graphics (*.png)')
        if save_file:
            self.img.save(save_file)
            
    def set_tool_tips(self, property_dict):
        tool_tip = ''
        max_prop_name_len = max(len(n) for n in property_dict)
        for k, v in property_dict.items():
            if k in ['Molecular Weight', 'LogP', 'Topological Polar Surface Area', 'Molar Refractivity']:
                tool_tip += f'{k:<{max_prop_name_len}}: {v:.4f}\n'
            else:
                tool_tip += f'{k:<{max_prop_name_len}}: {int(v)}\n'
        tool_tip = tool_tip.strip()
        self.setToolTip(tool_tip)

class ImageWithSearchLabel(ImageLabel):
    def __init__(self, img: Image):
        super().__init__(img)
        
        self.browser_js_dict = {'Ambinter'     : {'url' : 'https://www.ambinter.com/',
                                                  'js'  : """
                                                  document.querySelector('#textarea_search-list').value = '{smiles}';
                                                  document.querySelector('p.m-5:nth-child(2) > button:nth-child(1)').click();
                                                  """},
                                'molport'      : {'url' : 'https://www.molport.com/shop/find-chemicals-by-smiles',
                                                  'js'  : """
                                                  document.querySelector('#DrawSearchStructureSearchType4').click();
                                                  document.querySelector('#Structure').value = '{smiles}';
                                                  document.querySelector('#button-search-by-structure').click();
                                                  """},
                                'eMolecules'   : {'url' : 'https://search.emolecules.com/#?click=screening-compounds',
                                                  'js'  : """
                                                  document.querySelector('#list-search-tab').click();
                                                  document.querySelector('#list').value = '{smiles}';
                                                  document.querySelector('#datatype').value = 'smiles';
                                                  document.querySelector('div.col-sm-12:nth-child(6) > div:nth-child(4) > button:nth-child(1)').click();
                                                  """},
                                'ChemExper'    : {'url' : 'http://www.chemexper.com/index.shtml',
                                                  'js'  : """
                                                  document.querySelector('#quickSearchField').value = '{smiles}';
                                                  document.querySelector('#search_table > tbody:nth-child(1) > tr:nth-child(1) > td:nth-child(2) > article:nth-child(1) > section:nth-child(2) > form:nth-child(2) > fieldset:nth-child(1) > input:nth-child(3)').click();
                                                  """},
                                'Sigma Aldrich': {'url' : 'https://www.aldrichmarketselect.com/list_search.php?ID=list',
                                                  'js'  : """
                                                  document.querySelector('.acceptbuttons > div:nth-child(2) > input:nth-child(1)').click();
                                                  document.querySelector('table.copy:nth-child(1) > tbody:nth-child(1) > tr:nth-child(1) > td:nth-child(1) > div:nth-child(1) > textarea:nth-child(1)').value = '{smiles}';
                                                  document.querySelector('div.copy > select:nth-child(6)').value = 'SMILES';
                                                  document.querySelector('input.myButton:nth-child(32)').click();
                                                  """},
                                'ChemSpider'   : {'url' : 'https://www.chemspider.com/Search.aspx',
                                                  'js'  : """
                                                  function waitForElement(selector, callback) {{
                                                      var element = document.querySelector(selector);
                                                      if (element) {{
                                                          callback(element);
                                                      }} else {{
                                                          setTimeout(function() {{
                                                              waitForElement(selector, callback);
                                                          }}, 100);
                                                      }}
                                                  }}
                                                  waitForElement('#onetrust-accept-btn-handler', function(cookiesBtn) {{
                                                      cookiesBtn.click();
                                                      document.querySelector('#input-left-icon').value = '{smiles}';
                                                  }});
                                                  """},
                                'Chemspace'    : {'url' : 'https://chem-space.com/search',
                                                  'js'  : """
                                                  document.querySelector('#SearchText').value = '{smiles}';
                                                  document.querySelector('#text-search-btn').click();
                                                  document.querySelector('div.selector:nth-child(1) > select:nth-child(2)').value = '64';
                                                  document.querySelector('div.selector:nth-child(1) > select:nth-child(2)').click();
                                                  """},
                                'ZINC22'       : {'url' : 'https://cartblanche.docking.org/search/smiles',
                                                  'js'  : """
                                                  document.querySelector('#smilesTextarea').value = '{smiles}';
                                                  document.querySelector('#searchZincBtn2').click();
                                                  """},
                                'ZINC20'       : {'url' : 'https://zinc20.docking.org/substances/home/',
                                                  'js'  : """
                                                  document.querySelector('#jsme-smiles').value = '{smiles}';
                                                  document.querySelector('div.subsets-filter > div:nth-child(2) > ul:nth-child(1) > li:nth-child(1) > a:nth-child(1)').click();
                                                  document.querySelector('#query-exact').click();
                                                  """},
                                'Clipboard'    : None}
        
        browser_menu = QMenu()
        self.main_browser_action = QAction('Database', self)
        browser_actions = []
        self.browser_search_signal = Signal(str)
        
        for name in self.browser_js_dict:
            action = QAction(name, self)
            action.triggered.connect(lambda _, x=name: self.signal_search_browser(x))
            browser_actions.append(action)
        
        browser_menu.addActions(browser_actions)
        self.main_browser_action.setMenu(browser_menu)
        self.popup_menu.addAction(self.main_browser_action)
        self.main_browser_action.setDisabled(True)
        self.smiles = None
        
    def set_pixmap_smiles(self, img=None, size=None, smiles=None):
        self.smiles = smiles
        self.main_browser_action.setDisabled(self.smiles is None)
        self.set_pixmap(img, size)
    
    def signal_search_browser(self, name: str):
        if name == 'Clipboard':
            clipboard = QApplication.clipboard()
            clipboard.setText(self.smiles)
        else:
            url, js = self.browser_js_dict[name]['url'], self.browser_js_dict[name]['js']
            SearchDBBrowserWindow(self, url, js, self.smiles)

class SVGImageWidget(QSvgWidget):
    def __init__(self, svg_bytes: str):
        super().__init__()
        
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_save_menu)
        
        save_png_action = QAction('Save PNG', self)
        save_png_action.triggered.connect(self.create_png_and_save_dialog)
        
        self.popup_menu = QMenu()
        self.popup_menu.addAction(save_png_action)
        
        self.img = svg_bytes
        self.renderer().load(svg_bytes)
    
    def create_png_and_save_dialog(self):
        save_file, _ = QFileDialog.getSaveFileName(self, 'Select Saved PNG Path', '', 'Portable Network Graphics (*.png)')
        if save_file:
            svgtopng(self.img, save_file)
    
    def show_save_menu(self, pos):
        self.popup_menu.exec(self.mapToGlobal(pos))
        
    def set_image(self, svg_bytes: str):
        self.img = svg_bytes
        self.renderer().load(self.img)

class SVGImageWithSearchWidget(SVGImageWidget):
    def __init__(self, svg_bytes: str):
        super().__init__(svg_bytes)
        
        self.browser_js_dict = {'Ambinter'     : {'url' : 'https://www.ambinter.com/',
                                                  'js'  : """
                                                  document.querySelector('#textarea_search-list').value = '{smiles}';
                                                  document.querySelector('p.m-5:nth-child(2) > button:nth-child(1)').click();
                                                  """},
                                'molport'      : {'url' : 'https://www.molport.com/shop/find-chemicals-by-smiles',
                                                  'js'  : """
                                                  document.querySelector('#DrawSearchStructureSearchType4').click();
                                                  document.querySelector('#Structure').value = '{smiles}';
                                                  document.querySelector('#button-search-by-structure').click();
                                                  """},
                                'eMolecules'   : {'url' : 'https://search.emolecules.com/#?click=screening-compounds',
                                                  'js'  : """
                                                  document.querySelector('#list-search-tab').click();
                                                  document.querySelector('#list').value = '{smiles}';
                                                  document.querySelector('#datatype').value = 'smiles';
                                                  document.querySelector('div.col-sm-12:nth-child(6) > div:nth-child(4) > button:nth-child(1)').click();
                                                  """},
                                'ChemExper'    : {'url' : 'http://www.chemexper.com/index.shtml',
                                                  'js'  : """
                                                  document.querySelector('#quickSearchField').value = '{smiles}';
                                                  document.querySelector('#search_table > tbody:nth-child(1) > tr:nth-child(1) > td:nth-child(2) > article:nth-child(1) > section:nth-child(2) > form:nth-child(2) > fieldset:nth-child(1) > input:nth-child(3)').click();
                                                  """},
                                'Sigma Aldrich': {'url' : 'https://www.aldrichmarketselect.com/list_search.php?ID=list',
                                                  'js'  : """
                                                  document.querySelector('.acceptbuttons > div:nth-child(2) > input:nth-child(1)').click();
                                                  document.querySelector('table.copy:nth-child(1) > tbody:nth-child(1) > tr:nth-child(1) > td:nth-child(1) > div:nth-child(1) > textarea:nth-child(1)').value = '{smiles}';
                                                  document.querySelector('div.copy > select:nth-child(6)').value = 'SMILES';
                                                  document.querySelector('input.myButton:nth-child(32)').click();
                                                  """},
                                'ChemSpider'   : {'url' : 'https://www.chemspider.com/Search.aspx',
                                                  'js'  : """
                                                  function waitForElement(selector, callback) {{
                                                      var element = document.querySelector(selector);
                                                      if (element) {{
                                                          callback(element);
                                                      }} else {{
                                                          setTimeout(function() {{
                                                              waitForElement(selector, callback);
                                                          }}, 100);
                                                      }}
                                                  }}
                                                  waitForElement('#onetrust-accept-btn-handler', function(cookiesBtn) {{
                                                      cookiesBtn.click();
                                                      document.querySelector('#input-left-icon').value = '{smiles}';
                                                  }});
                                                  """},
                                'Chemspace'    : {'url' : 'https://chem-space.com/search',
                                                  'js'  : """
                                                  document.querySelector('#SearchText').value = '{smiles}';
                                                  document.querySelector('#text-search-btn').click();
                                                  document.querySelector('div.selector:nth-child(1) > select:nth-child(2)').value = '64';
                                                  document.querySelector('div.selector:nth-child(1) > select:nth-child(2)').click();
                                                  """},
                                'ZINC22'       : {'url' : 'https://cartblanche.docking.org/search/smiles',
                                                  'js'  : """
                                                  document.querySelector('#smilesTextarea').value = '{smiles}';
                                                  document.querySelector('#searchZincBtn2').click();
                                                  """},
                                'ZINC20'       : {'url' : 'https://zinc20.docking.org/substances/home/',
                                                  'js'  : """
                                                  document.querySelector('#jsme-smiles').value = '{smiles}';
                                                  document.querySelector('div.subsets-filter > div:nth-child(2) > ul:nth-child(1) > li:nth-child(1) > a:nth-child(1)').click();
                                                  document.querySelector('#query-exact').click();
                                                  """},
                                'Clipboard'    : None}
        
        browser_menu = QMenu()
        self.main_browser_action = QAction('Database', self)
        browser_actions = []
        self.browser_search_signal = Signal(str)
        
        for name in self.browser_js_dict:
            action = QAction(name, self)
            action.triggered.connect(lambda _, x=name: self.signal_search_browser(x))
            browser_actions.append(action)
        
        browser_menu.addActions(browser_actions)
        self.main_browser_action.setMenu(browser_menu)
        self.popup_menu.addAction(self.main_browser_action)
        self.main_browser_action.setDisabled(True)
        self.smiles = None
        
    def set_image_smiles(self, svg_bytes=None, smiles=None):
        self.smiles = smiles
        self.main_browser_action.setDisabled(self.smiles is None)
        self.set_image(svg_bytes)
    
    def signal_search_browser(self, name: str):
        if name == 'Clipboard':
            clipboard = QApplication.clipboard()
            clipboard.setText(self.smiles)
        else:
            url, js = self.browser_js_dict[name]['url'], self.browser_js_dict[name]['js']
            SearchDBBrowserWindow(self, url, js, self.smiles)

class SupplierSignals(QObject):
    change_dark_light_mode = Signal()
    closed_window = Signal()

class ZINCSupplierFinderWidget(QWidget):
    def __init__(self, parent, display_mode: str):
        super().__init__(parent)
        supplier_dict = {'Name': [], 'SMILES': []}
        supplier_dict.update({prop_name: [] for prop_name in property_functions})
        self.supplier_df = pd.DataFrame(supplier_dict)
        self.name_smiles_df = pd.DataFrame({'Name': [], 'SMILES': []})
        self.browser = None
        self.display_mode = display_mode
        self.setupUI()
    
    def setupUI(self):
        overall_layout = QHBoxLayout()
        list_name_layout = QVBoxLayout()
        search_option_layout = QGridLayout()
        search_option_progress_layout = QVBoxLayout()
        table_information_layout = QGridLayout()
        table_layout = QHBoxLayout()
        information_table_layout = QVBoxLayout()
        final_btn_layout = QHBoxLayout()
        table_information_and_image_layout = QHBoxLayout()
        result_table_layout = QVBoxLayout()
        dist_label = QLabel('<b>Dist:</b>')
        self.first_dist_spinbox = QSpinBox()
        self.first_dist_spinbox.setRange(0, 99)
        self.first_dist_spinbox.setValue(0)
        self.first_dist_spinbox.valueChanged.connect(self.change_second_dist_range)
        self.first_dist_spinbox.setMinimumWidth(75)
        dist_spinbox_between = QLabel('<b>~</b>')
        self.second_dist_spinbox = QSpinBox()
        self.second_dist_spinbox.setRange(0, 99)
        self.second_dist_spinbox.setValue(4)
        self.second_dist_spinbox.setMinimumWidth(75)
        anon_dist_label = QLabel('<b>Anon Dist:</b>')
        self.first_anon_dist_spinbox = QSpinBox()
        self.first_anon_dist_spinbox.setRange(0, 99)
        self.first_anon_dist_spinbox.setValue(0)
        self.first_anon_dist_spinbox.valueChanged.connect(self.change_second_anon_dist_range)
        self.first_anon_dist_spinbox.setMinimumWidth(80)
        anon_dist_spinbox_between = QLabel('<b>~</b>')
        self.second_anon_dist_spinbox = QSpinBox()
        self.second_anon_dist_spinbox.setRange(0, 99)
        self.second_anon_dist_spinbox.setValue(2)
        self.second_anon_dist_spinbox.setMinimumWidth(75)
        smallworld_length_label = QLabel('<b>Length:</b>')
        self.smallworld_length_spinbox = QSpinBox()
        self.smallworld_length_spinbox.setRange(1, 200) # Set max to 200 or else ZINC database might explode.
        self.smallworld_length_spinbox.setValue(10)
        self.smallworld_length_spinbox.setMinimumWidth(75)
        self.search_button = QPushButton('Search')
        self.search_button.setMinimumWidth(75)
        self.search_button.clicked.connect(self.search_through_zinc22)
        if not len(self.name_smiles_df):
            self.search_button.setDisabled(True)
        
        search_option_layout.addWidget(dist_label, 0, 0)
        search_option_layout.addWidget(self.first_dist_spinbox, 0, 1)
        search_option_layout.addWidget(dist_spinbox_between, 0, 2, Qt.AlignmentFlag.AlignCenter)
        search_option_layout.addWidget(self.second_dist_spinbox, 0, 3)
        
        search_option_layout.addWidget(anon_dist_label, 1, 0)
        search_option_layout.addWidget(self.first_anon_dist_spinbox, 1, 1)
        search_option_layout.addWidget(anon_dist_spinbox_between, 1, 2, Qt.AlignmentFlag.AlignCenter)
        search_option_layout.addWidget(self.second_anon_dist_spinbox, 1, 3)
        
        search_option_layout.addWidget(smallworld_length_label, 2, 0)
        search_option_layout.addWidget(self.smallworld_length_spinbox, 2, 1)
        search_option_layout.addWidget(self.search_button, 2, 2, 1, 2)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        if len(self.name_smiles_df):
            self.progress_bar.setMaximum(len(self.name_smiles_df))
        else:
            self.progress_bar.setMaximum(1)
            
        search_option_progress_layout.addLayout(search_option_layout)
        search_option_progress_layout.addWidget(self.progress_bar)
        
        self.list_of_names = DropFileDirListWidget()
        self.list_of_names.itemSelectionChanged.connect(self.update_combo_with_zincid)
        self.fill_list_with_names()
        self.list_of_names.setEditTriggers(QListWidget.EditTrigger.DoubleClicked)
        self.list_of_names.supplierdictSignal.connect(self.add_new_mols_to_supplier_df)
        self.list_of_names.itemClicked.connect(self.keep_name_before_changing)
        self.list_of_names.itemDoubleClicked.connect(self.start_editing_text)
        self.list_of_names.itemChanged.connect(self.change_name_of_dfs_to_new_name)
        self.list_of_names.currCountChanged.connect(self.check_search_button)
        self.list_of_names.itemRemovedSignal.connect(self.remove_text_from_dfs)
        
        curr_name_frame = QFrame()
        curr_name_frame.setFrameShape(QFrame.Shape.StyledPanel)
        curr_name_frame.setLineWidth(10)
        curr_name_layout = QHBoxLayout(curr_name_frame)
        curr_name_layout.setContentsMargins(0, 0, 0, 0)
        self.empty_img = self.generate_placeholder_svg()
        self.curr_name_img_label = SVGImageWithSearchWidget(self.empty_img)
        self.curr_name_img_label.setFixedSize(220, 220)
        curr_name_layout.addWidget(self.curr_name_img_label)
        
        list_name_layout.addWidget(curr_name_frame, alignment=Qt.AlignmentFlag.AlignCenter)
        list_name_layout.addLayout(search_option_progress_layout)
        list_name_layout.addWidget(self.list_of_names)
        
        zinc_id_combo_label = QLabel('<b>Matching ZINC ID :</b>')
        self.zinc_id_combobox = QComboBox()
        self.zinc_id_combobox.currentTextChanged.connect(self.update_table_with_zincid)
        self.zinc_id_label = QLabel('<b>ZINC ID :</b>')
        self.zinc_id_lineedit = CopyLineEdit()
        self.smiles_label = QLabel('<b>SMILES :</b>')
        self.smiles_lineedit = CopyLineEdit()
        similarity_label = QLabel('<b>Similarity :</b>')
        self.similarity_lineedit = QLineEdit()
        self.similarity_lineedit.setReadOnly(True)
        self.similarity_lineedit.setMaximumWidth(70)
        table_information_layout.addWidget(zinc_id_combo_label, 0, 0)
        table_information_layout.addWidget(self.zinc_id_combobox, 0, 1, 1, 3)
        table_information_layout.addWidget(self.zinc_id_label, 1, 0)
        table_information_layout.addWidget(self.zinc_id_lineedit, 1, 1)
        table_information_layout.addWidget(similarity_label, 1, 2)
        table_information_layout.addWidget(self.similarity_lineedit, 1, 3)
        table_information_layout.addWidget(self.smiles_label, 2, 0)
        table_information_layout.addWidget(self.smiles_lineedit, 2, 1, 1, 3)
        
        chem_image_frame = QFrame()
        chem_image_frame.setFrameShape(QFrame.Shape.StyledPanel)
        chem_image_frame.setLineWidth(10)
        chem_image_layout = QHBoxLayout(chem_image_frame)
        chem_image_layout.setContentsMargins(0, 0, 0, 0)
        self.chem_image_label = SVGImageWithSearchWidget(self.empty_img)
        self.chem_image_label.setFixedSize(220, 220)
        chem_image_layout.addWidget(self.chem_image_label)
        table_information_and_image_layout.addLayout(table_information_layout)
        table_information_and_image_layout.addWidget(chem_image_frame)
        
        self.purchasability_table = QTableWidget()
        self.purchasability_table.setMinimumWidth(550)
        self.purchasability_table.setColumnCount(2)
        self.purchasability_table.setHorizontalHeaderLabels(['Catalog Name', 'Supplier Code'])
        header = self.purchasability_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        self.chemprop_table = QTableWidget()
        self.chemprop_table.setMinimumWidth(300)
        self.chemprop_table.setColumnCount(3)
        self.chemprop_table.setHorizontalHeaderLabels(['Properties', 'Query', 'Target'])
        self.chemprop_table_set_prop_rows()
        self.chemprop_table.verticalHeader().setVisible(False)
        self.chemprop_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header = self.chemprop_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        
        table_layout.addWidget(self.purchasability_table)
        table_layout.addWidget(self.chemprop_table, alignment=Qt.AlignmentFlag.AlignRight)
        
        information_table_layout.addLayout(table_information_and_image_layout)
        information_table_layout.addLayout(table_layout)
        
        btn_right_layout = QHBoxLayout()
        btn_right_widget = QWidget()
        
        # self.custom_search_button = QPushButton('Custom Search')
        # self.custom_search_button.clicked.connect(self.custom_search_dialog)
        self.save_files_button = QPushButton('Save', self)
        self.save_menu_setup()
        # btn_right_layout.addWidget(self.custom_search_button)
        btn_right_layout.addWidget(self.save_files_button)
        btn_right_layout.setContentsMargins(0, 0, 0, 0)
        btn_right_widget.setLayout(btn_right_layout)
        final_btn_layout.addWidget(btn_right_widget, alignment=Qt.AlignmentFlag.AlignRight)
        
        result_table_layout.addLayout(information_table_layout)
        result_table_layout.addLayout(final_btn_layout)
        
        overall_layout.addLayout(list_name_layout, 1)
        overall_layout.addLayout(result_table_layout, 9)
        
        self.setLayout(overall_layout)
        
        self.start_editing_listitem_text = False
        self.currently_searching = False
        self.failed_or_non_smiles = []
        self.name_zinc_map = {}
    
    def generate_placeholder_svg(self):
        placeholder_svg = '''
        <svg width="150" height="150" xmlns="http://www.w3.org/2000/svg">
            <rect width="150%" height="150%" fill="none" stroke="gray" stroke-width="1"/>
            <text x="75%" y="50%" font-size="13" dominant-baseline="middle" text-anchor="middle" fill="gray">
                No
            </text>
            <text x="75%" y="75%" font-size="13" dominant-baseline="middle" text-anchor="middle" fill="gray">
                Molecule
            </text>
            <text x="75%" y="100%" font-size="13" dominant-baseline="middle" text-anchor="middle" fill="gray">
                Selected
            </text>
        </svg>
        '''
        return placeholder_svg.encode('utf-8')
    
    def draw_svg_bytes(self, mol):
        drawer = Draw.MolDraw2DSVG(600, 600)
        options = drawer.drawOptions()
        if self.display_mode == 'dark':
            SetDarkMode(options)
            bgcolor = '#000000'
        else:
            bgcolor = '#FFFFFF'
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        strings = drawer.GetDrawingText().replace(f"<rect style='opacity:1.0;fill:{bgcolor}", "<rect style='opacity:1.0;fill:none")
        return strings.encode('utf-8')
    
    def change_second_dist_range(self):
        self.second_dist_spinbox.setRange(self.first_dist_spinbox.value(), 99)
    
    def change_second_anon_dist_range(self):
        self.second_anon_dist_spinbox.setRange(self.first_anon_dist_spinbox.value(), 99)
    
    def add_new_mols_to_supplier_df(self, supplier_dict: dict):
        if supplier_dict:
            existed_names = self.name_smiles_df['Name'].values
            for idx, name in enumerate(supplier_dict['Name']):
                if name in existed_names:
                    i = 1
                    while f'{name}_{i}' in existed_names:
                        i += 1
                    supplier_dict['Name'][idx] = f'{name}_{i}'
            df = pd.DataFrame(supplier_dict)
            self.supplier_df = pd.concat([self.supplier_df, df]).reset_index(drop=True)
            self.name_smiles_df = self.supplier_df[['Name', 'SMILES']]
            for idx, name in enumerate(supplier_dict['Name']):
                item = QListWidgetItem()
                item.setText(name)
                if self.currently_searching:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable & ~Qt.ItemFlag.ItemIsEnabled & ~Qt.ItemFlag.ItemIsEditable)
                else:
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                self.list_of_names.addItem(item)
    
    def start_editing_text(self):
        self.start_editing_listitem_text = True
    
    def keep_name_before_changing(self, item: QListWidgetItem):
        self.curr_item_text = item.text()
        
    def change_name_of_dfs_to_new_name(self, item: QListWidgetItem):
        if self.start_editing_listitem_text and item.isSelected():
            new_text = item.text()
            if not new_text:
                item.setText(self.curr_item_text)
            elif new_text != self.curr_item_text:
                if new_text in self.supplier_df['Name'].to_list():
                    QMessageBox.critical(self, 'NameError', f'{new_text}, already exists!')
                    item.setText(self.curr_item_text)
                    return
                self.supplier_df[['Name']] = self.supplier_df[['Name']].replace(self.curr_item_text, new_text)
                self.name_smiles_df[['Name']] = self.name_smiles_df[['Name']].replace(self.curr_item_text, new_text)
                self.name_zinc_map[new_text] = self.name_zinc_map.pop(self.curr_item_text)
            self.start_editing_listitem_text = False
    
    def remove_text_from_dfs(self, name):
        kept_rows = self.name_smiles_df['Name'] != name
        self.supplier_df = self.supplier_df[kept_rows].reset_index(drop=True)
        self.name_smiles_df = self.name_smiles_df[kept_rows].reset_index(drop=True)
    
    def check_search_button(self, num):
        if num:
            self.search_button.setEnabled(True)
        else:
            self.search_button.setEnabled(False)
    
    def chemprop_table_set_prop_rows(self):
        self.chemprop_table.setRowCount(len(property_functions))
        for r, k in enumerate(property_functions):
            item = QTableWidgetItem()
            item.setText(k)
            self.chemprop_table.setItem(r, 0, item)
    
    def save_menu_setup(self):
        options = ['Current Similar Molecule', 'All Similar Molecules', 'Full Result']
        all_actions = []
        for name in options:
            curr_action = QAction(name, self)
            curr_png_action = QAction('Image', self)
            curr_catalog_table_action = QAction('Catalog Table', self)
            curr_smiles_action = QAction('SMILES', self)
            
            submenu = QMenu(self)
            submenu.addActions([curr_png_action,
                                curr_catalog_table_action,
                                curr_smiles_action])
            curr_action.setMenu(submenu)
            
            curr_png_action.triggered.connect(lambda _, n=name: self.save_to_png(n))
            curr_catalog_table_action.triggered.connect(lambda _, n=name: self.save_to_catalog_table(n))
            curr_smiles_action.triggered.connect(lambda _, n=name: self.save_to_smiles(n))
            
            all_actions.append(curr_action)
        
        save_menu = QMenu(self)
        save_menu.addActions(all_actions)
        self.save_files_button.setMenu(save_menu)
    
    def fill_list_with_names(self):
        names = self.name_smiles_df['Name'].to_list()
        self.list_of_names.addItems(names)
        default_brush = QListWidgetItem().foreground()
        for idx in range(len(names)):
            item = self.list_of_names.item(idx)
            f = item.font()
            f.setStrikeOut(False)
            item.setFont(f)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            item.setForeground(default_brush)
    
    def search_through_zinc22(self):
        if not self.currently_searching:
            all_unique_smiles_list = list(set(self.name_smiles_df['SMILES']))
            params = {'dist'  : f'{self.first_dist_spinbox.value()}-{self.second_dist_spinbox.value()}',
                      'adist' : f'{self.first_anon_dist_spinbox.value()}-{self.second_anon_dist_spinbox.value()}',
                      'length': f'{self.smallworld_length_spinbox.value()}',
                      'db'    : 'ZINC-All-22Q2-1.6B',
                      'fmt'   : 'csv',}
            
            default_brush = QListWidgetItem().foreground()
            for idx in range(len(self.name_smiles_df)):
                item = self.list_of_names.item(idx)
                f = item.font()
                f.setStrikeOut(False)
                item.setFont(f)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable & ~Qt.ItemFlag.ItemIsEnabled & ~Qt.ItemFlag.ItemIsEditable)
                item.setForeground(default_brush)
            self.list_of_names.clearSelection()
            self.purchasability_table.clearContents()
            self.chemprop_table.clearContents()
            self.chemprop_table_set_prop_rows()
            self.zinc_id_combobox.clear()
            self.curr_name_img_label.set_image_smiles(self.empty_img)
            with open(os.path.join(os.path.dirname(__file__), 'fp_param', 'fp_param.json'), 'r') as f:
                fp_settings = json.load(f)
            self.fpgen = retrieve_fp_generator(fp_settings)
            self.sim_method_func = retrieve_similarity_method(fp_settings['sim'])
            
            self.currently_searching = True
            self.name_zinc_map = {}
            self.zinc_id_catalog_map = {}
            self.failed_or_non_smiles = []
            self.progress_bar.setValue(0)
            self.progress_bar.setMaximum(len(self.supplier_df.index))
            self.smiles_lineedit.clear()
            self.zinc_id_lineedit.clear()
            self.search_button.setText('Stop')
            # self.search_button.setDisabled(True)
            self.thread = QThread()
            self.worker = MultiThreadZINCSearch(params, all_unique_smiles_list)
            self.worker.moveToThread(self.thread)
            self.worker.curr_step_text.connect(self.update_progress_bar_and_df)
            
            self.worker.canceled.connect(self.cancel_search)
            
            self.worker.finished.connect(self.worker.deleteLater)
            self.worker.finished.connect(self.thread.quit)
            self.thread.finished.connect(self.thread.deleteLater)
            self.thread.finished.connect(self.searching_finished)
            self.thread.started.connect(self.worker.run)
            self.thread.start()
        else:
            self.worker.stop()
            self.search_button.setEnabled(False)
            self.search_button.setText('Stopping')
            
    def cancel_search(self):
        self.thread.quit()
        self.thread.wait()
        self.worker.deleteLater()
        self.thread.deleteLater()
        self.search_button.setText('Search')
        self.search_button.setEnabled(True)
        for i in range(self.list_of_names.count()):
            item = self.list_of_names.item(i)
            if item.flags() & ~Qt.ItemFlag.ItemIsEnabled:
                color = QColor()
                color.setRgb(255, 80, 80)
                item.setForeground(color)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable)
                item.setToolTip('Search Stopped')
        self.progress_bar.setValue(self.progress_bar.maximum())
    
    def update_progress_bar_and_df(self, csv_text: str, smiles_list: list):
        smiles_series = self.name_smiles_df['SMILES']
        for smiles_str in smiles_list:
            idxs = smiles_series[smiles_series == smiles_str].index
            if csv_text.strip():
                csv_data = io.StringIO(csv_text)
                df = pd.read_csv(csv_data)  # the first is the padded zinc_id bc ZINC22 search sucks
                for idx in idxs:
                    item = self.list_of_names.item(idx)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable)
                self.parse_single_downloaded_dataframe(df, smiles_str)
            elif csv_text == ' ':  # smallworld has no matching result
                self.failed_or_non_smiles.extend(smiles_str)
                for idx in idxs:
                    item = self.list_of_names.item(idx)
                    f = item.font()
                    f.setStrikeOut(True)
                    item.setFont(f)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable)
                    item.setToolTip('No match found')
            elif csv_text == '':    # smallworld search failed.
                self.failed_or_non_smiles.extend(smiles_str)
                for idx in idxs:
                    item = self.list_of_names.item(idx)
                    color = QColor()
                    color.setRgb(255, 165, 0)
                    item.setForeground(color)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable)
                    item.setToolTip('SmallWorld API Search failed')
            elif csv_text == '  ':  # ZINC22 search failed
                self.failed_or_non_smiles.extend(smiles_str)
                for idx in idxs:
                    item = self.list_of_names.item(idx)
                    color = QColor()
                    color.setRgb(255, 165, 0)
                    item.setForeground(color)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable)
                    item.setToolTip('ZINC22 Search failed')
        self.progress_bar.setValue(self.progress_bar.value() + len(idxs))
        
    def parse_single_downloaded_dataframe(self, single_downloaded_df: pd.DataFrame, smiles_str: str):
        bool_row = self.name_smiles_df['SMILES'] == smiles_str
        query_mol = Chem.MolFromSmiles(smiles_str)
        query_smiles = Chem.MolToSmiles(query_mol)
        query_fps = self.fpgen(query_mol)
        
        names = self.name_smiles_df[bool_row]['Name'].to_list()
        for name_idx, name in enumerate(names):
            if name_idx == 0:
                similarity_list = []    # used for sorting
                first_name = name
                self.name_zinc_map[name] = {'zinc_ids': []}
                for _, rrow in single_downloaded_df.iterrows():
                    zinc_id, zinc_smiles = rrow['zinc_id'], rrow['smiles']
                    zinc_mol = Chem.MolFromSmiles(zinc_smiles)
                    zinc_smiles = Chem.MolToSmiles(zinc_mol)
                    zinc_fps = self.fpgen(zinc_mol)
                    similarity = self.sim_method_func(query_fps, zinc_fps)
                    similarity_list.append(similarity)
                    if zinc_smiles == query_smiles:
                        zinc_id += f' (Exact)'
                    self.name_zinc_map[name]['zinc_ids'].append(zinc_id)
                    i = 0
                    catalog_name = rrow[f'catalogs_{i}_catalog_name']
                    catalog_dict = {'catalog_names'         : [],
                                    'catalog_supplier_codes': [],
                                    'catalog_url'           : [],}
                    while catalog_name == catalog_name: # exit if NaN is found
                        catalog_dict['catalog_names'].append(catalog_name)
                        supplier_code = rrow.get(f'catalogs_{i}_supplier_code', float('nan'))
                        catalog_dict['catalog_supplier_codes'].append(supplier_code)
                        url_base = rrow.get(f'catalogs_{i}_url', float('nan'))
                        if url_base == url_base and supplier_code == supplier_code:
                            full_url = url_base.replace('%%s', str(supplier_code), 1).strip()
                            catalog_dict['catalog_url'].append(full_url)
                        elif 'zinc20' in catalog_name or catalog_name in ['informer', 'informer2']:
                            full_url = f'https://cartblanche.docking.org/substance/{rrow[f'catalogs_{i}_supplier_code']}'
                            catalog_dict['catalog_url'].append(full_url)
                        else:
                            catalog_dict['catalog_url'].append(url_base)
                        i += 1
                        catalog_name = rrow.get(f'catalogs_{i}_catalog_name', float('nan'))
                    mol = Chem.MolFromSmiles(zinc_smiles)
                    mol = Chem.AddHs(mol)
                    properties = {f'{chem_prop}': func(mol) for chem_prop, func in property_functions.items()}
                    self.zinc_id_catalog_map[zinc_id] = {'smiles'    : zinc_smiles,
                                                         'catalogs'  : pd.DataFrame(catalog_dict),
                                                         'properties': properties,
                                                         'similarity': similarity}
                try:
                    exact_id_idx = next(idx for idx, id in enumerate(self.name_zinc_map[name]['zinc_ids']) if id.endswith(' (Exact)'))
                    exact_id = self.name_zinc_map[name]['zinc_ids'].pop(exact_id_idx)
                    similarity_list.pop(exact_id_idx)
                    sorted_similarity_zinc = [x for _, x in sorted(zip(similarity_list, self.name_zinc_map[name]['zinc_ids']), reverse=True)]
                    self.name_zinc_map[name]['zinc_ids'] = [exact_id] + sorted_similarity_zinc
                except:
                    sorted_similarity_zinc = [x for _, x in sorted(zip(similarity_list, self.name_zinc_map[name]['zinc_ids']), reverse=True)]
                    self.name_zinc_map[name]['zinc_ids'] = sorted_similarity_zinc
            else:
                self.name_zinc_map[name] = copy.deepcopy(self.name_zinc_map[first_name])    # just copy everything, no need to do it again
                
    def searching_finished(self):
        self.currently_searching = False
        self.search_button.setEnabled(True)
        self.search_button.setText('Search')
            
    def update_combo_with_zincid(self):
        items = self.list_of_names.selectedItems()
        if items:
            name = items[0].text()
            curr_smiles = self.name_smiles_df.loc[self.name_smiles_df['Name'] == name]['SMILES'].iloc[0]
            self.curr_name_img_label.set_image_smiles(self.draw_svg_bytes(Chem.MolFromSmiles(curr_smiles)), curr_smiles)
            if name not in self.name_zinc_map:
                self.zinc_id_combobox.clear()
                self.purchasability_table.clearContents()
                self.chemprop_table.clearContents()
                self.chemprop_table_set_prop_rows()
                self.zinc_id_lineedit.clear()
                self.smiles_lineedit.clear()
                return
            zinc_ids = self.name_zinc_map[name]['zinc_ids']
            self.zinc_id_combobox.clear()
            self.zinc_id_combobox.addItems(zinc_ids)
        else:
            self.curr_name_img_label.set_image_smiles(self.empty_img, None)
            self.curr_name_img_label.setToolTip('')
            self.purchasability_table.clearContents()
    
    def update_table_with_zincid(self, zinc_id):
        if zinc_id:
            self.zinc_id_lineedit.setText(zinc_id.split(' (')[0])
            smiles = self.zinc_id_catalog_map[zinc_id]['smiles']
            self.smiles_lineedit.setText(smiles)
            self.similarity_lineedit.setText(f'{self.zinc_id_catalog_map[zinc_id]['similarity']:.4f}')
            self.zinc_id_lineedit.setCursorPosition(0)
            self.smiles_lineedit.setCursorPosition(0)
            self.chem_image_label.set_image_smiles(self.draw_svg_bytes(Chem.MolFromSmiles(smiles)), smiles)
            self.populate_catalog_table(self.zinc_id_catalog_map[zinc_id]['catalogs'])
            self.compare_query_and_target_properties()
        else:
            self.chem_image_label.set_image_smiles(self.empty_img, None)
            self.chem_image_label.setToolTip('')
            self.chemprop_table.clearContents()
            self.purchasability_table.clearContents()
            self.purchasability_table.setRowCount(0)
    
    def populate_catalog_table(self, catalog_df: pd.DataFrame):
        self.purchasability_table.setSortingEnabled(False)
        self.purchasability_table.clearContents()
        table_rows = len(catalog_df)
        self.purchasability_table.setRowCount(table_rows)
        for idx, row in catalog_df.iterrows():
            cat_name, cat_supplier_code, cat_url = row
            
            cat_name_item = QTableWidgetItem()
            cat_name_item.setText(cat_name)
            cat_name_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            cat_name_item.setFlags(cat_name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if cat_url == cat_url:
                cat_supplier_label = QLabel(f'<a href="{cat_url}">{cat_supplier_code}</a>')
                cat_supplier_label.setOpenExternalLinks(False)
                cat_supplier_label.linkActivated.connect(self.show_custom_webbrowser)
            else:
                cat_supplier_label = QLabel(str(cat_supplier_code))
            cat_supplier_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            self.purchasability_table.setItem(idx, 0, cat_name_item)
            self.purchasability_table.setCellWidget(idx, 1, cat_supplier_label)
        self.purchasability_table.resizeColumnsToContents()
        header = self.purchasability_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.purchasability_table.setSortingEnabled(True)
        
    def compare_query_and_target_properties(self):
        items = self.list_of_names.selectedItems()
        if not items:
            return
        query_name = items[0].text()
        query_series = self.supplier_df[self.supplier_df['Name'] == query_name]
        target_chemprop_dict = self.zinc_id_catalog_map[self.zinc_id_combobox.currentText()]['properties']
        self.chemprop_table.clearContents()
        for r, k in enumerate(property_functions):
            query_value = query_series.get(k).to_list()[0]
            target_value = target_chemprop_dict[k]
            
            color = QTableWidgetItem().foreground()
            if target_value - query_value >= 1e-3:
                color = QColor()
                color.setRgb(229, 115, 115)
            elif query_value - target_value >= 1e-3:
                color = QColor()
                color.setRgb(76, 175, 80)
                
            prop_item = QTableWidgetItem()
            query_item = QTableWidgetItem()
            target_item = QTableWidgetItem()
            
            prop_item.setText(k)
            if k in ['Molecular Weight', 'LogP', 'Topological Polar Surface Area', 'Molar Refractivity', 'QED']:
                query_item.setText(f'{query_value:.3f}')
                target_item.setText(f'{target_value:.3f}')
            else:
                query_item.setText(f'{int(query_value)}')
                target_item.setText(f'{int(target_value)}')
            target_item.setForeground(color)
            
            self.chemprop_table.setItem(r, 0, prop_item)
            self.chemprop_table.setItem(r, 1, query_item)
            self.chemprop_table.setItem(r, 2, target_item)
    
    def show_custom_webbrowser(self, url):
        if self.browser is None:
            self.browser = BrowserWithTabs(self, url)
            self.browser.closed.connect(lambda: setattr(self, 'browser', None))
        else:
            self.browser.tab_browser.add_new_tab(url)
            self.browser.raise_()
            self.browser.activateWindow()
    
    def save_to_png(self, name):
        if name == 'Current Similar Molecule':
            current_id = self.zinc_id_combobox.currentText()
            if current_id:
                current_molecule_name = self.list_of_names.selectedItems()[0].text()
                dir = QFileDialog.getExistingDirectory(self, 'Select Save Directory', '')
                if dir:
                    new_dir = os.path.join(dir, current_molecule_name)
                    os.makedirs(new_dir, exist_ok=True)
                    similarity = f'_{self.zinc_id_catalog_map[current_id]['similarity']:.4f}'
                    curr_dir = os.path.join(new_dir, current_id.replace(' ', '_') + similarity)
                    os.makedirs(curr_dir, exist_ok=True)
                    curr_molecule_pth = os.path.join(curr_dir, current_id.replace(' ', '_') + '.png')  # replace the space in " (Exact)"
                    mol = Chem.MolFromSmiles(self.zinc_id_catalog_map[current_id]['smiles'])
                    svgtopng(self.draw_svg_bytes(mol), curr_molecule_pth)
        elif name == 'All Similar Molecules':
            current_items = self.list_of_names.selectedItems()
            if current_items:
                current_molecule_name = current_items[0].text()
                if current_molecule_name in self.name_zinc_map:
                    dir = QFileDialog.getExistingDirectory(self, 'Select Save Directory', '')
                    if dir:
                        zinc_id_list = self.name_zinc_map[current_molecule_name]['zinc_ids']
                        new_dir = os.path.join(dir, current_molecule_name)
                        os.makedirs(new_dir, exist_ok=True)
                        for current_id in zinc_id_list:
                            similarity = f'_{self.zinc_id_catalog_map[current_id]['similarity']:.4f}'
                            curr_dir = os.path.join(new_dir, current_id.replace(' ', '_') + similarity)
                            os.makedirs(curr_dir, exist_ok=True)
                            curr_molecule_pth = os.path.join(curr_dir, current_id.replace(' ', '_') + '.png')
                            mol = Chem.MolFromSmiles(self.zinc_id_catalog_map[current_id]['smiles'])
                            svgtopng(self.draw_svg_bytes(mol), curr_molecule_pth)
        else:
            num_of_items = self.list_of_names.count()
            dir = QFileDialog.getExistingDirectory(self, 'Select Save Directory', '')
            if dir:
                for r in range(num_of_items):
                    item = self.list_of_names.item(r)
                    current_molecule_name = item.text()
                    new_dir = os.path.join(dir, current_molecule_name)
                    os.makedirs(new_dir, exist_ok=True)
                    if current_molecule_name in self.name_zinc_map:
                        zinc_id_list = self.name_zinc_map[current_molecule_name]['zinc_ids']
                        for current_id in zinc_id_list:
                            similarity = f'_{self.zinc_id_catalog_map[current_id]['similarity']:.4f}'
                            curr_dir = os.path.join(new_dir, current_id.replace(' ', '_') + similarity)
                            os.makedirs(curr_dir, exist_ok=True)
                            curr_molecule_pth = os.path.join(curr_dir, current_id.replace(' ', '_') +'.png')
                            mol = Chem.MolFromSmiles(self.zinc_id_catalog_map[current_id]['smiles'])
                            svgtopng(self.draw_svg_bytes(mol), curr_molecule_pth)
        
    def save_to_catalog_table(self, name):
        if name == 'Current Similar Molecule':
            current_id = self.zinc_id_combobox.currentText()
            if current_id:
                current_molecule_name = self.list_of_names.selectedItems()[0].text()
                dir = QFileDialog.getExistingDirectory(self, 'Select Save Directory', '')
                if dir:
                    new_dir = os.path.join(dir, current_molecule_name)
                    os.makedirs(new_dir, exist_ok=True)
                    similarity = f'_{self.zinc_id_catalog_map[current_id]['similarity']:.4f}'
                    curr_dir = os.path.join(new_dir, current_id.replace(' ', '_') + similarity)
                    os.makedirs(curr_dir, exist_ok=True)
                    curr_molecule_pth = os.path.join(curr_dir, current_id.replace(' ', '_') + '.csv')
                    df: pd.DataFrame = self.zinc_id_catalog_map[current_id]['catalogs']
                    df.to_csv(curr_molecule_pth, index=None)
        elif name == 'All Similar Molecules':
            current_items = self.list_of_names.selectedItems()
            if current_items:
                current_molecule_name = current_items[0].text()
                if current_molecule_name in self.name_zinc_map:
                    dir = QFileDialog.getExistingDirectory(self, 'Select Save Directory', '')
                    if dir:
                        zinc_id_list = self.name_zinc_map[current_molecule_name]['zinc_ids']
                        new_dir = os.path.join(dir, current_molecule_name)
                        os.makedirs(new_dir, exist_ok=True)
                        for current_id in zinc_id_list:
                            similarity = f'_{self.zinc_id_catalog_map[current_id]['similarity']:.4f}'
                            curr_dir = os.path.join(new_dir, current_id.replace(' ', '_') + similarity)
                            os.makedirs(curr_dir, exist_ok=True)
                            curr_molecule_pth = os.path.join(curr_dir, current_id.replace(' ', '_') + '.csv')
                            df: pd.DataFrame = self.zinc_id_catalog_map[current_id]['catalogs']
                            if not df.empty:
                                df.to_csv(curr_molecule_pth, index=None)
        else:
            num_of_items = self.list_of_names.count()
            dir = QFileDialog.getExistingDirectory(self, 'Select Save Directory', '')
            if dir:
                for r in range(num_of_items):
                    item = self.list_of_names.item(r)
                    current_molecule_name = item.text()
                    new_dir = os.path.join(dir, current_molecule_name)
                    os.makedirs(new_dir, exist_ok=True)
                    if current_molecule_name in self.name_zinc_map:
                        zinc_id_list = self.name_zinc_map[current_molecule_name]['zinc_ids']
                        new_dir = os.path.join(dir, current_molecule_name)
                        os.makedirs(new_dir, exist_ok=True)
                        for current_id in zinc_id_list:
                            similarity = f'{self.zinc_id_catalog_map[current_id]['similarity']:.4f}'
                            curr_dir = os.path.join(new_dir, current_id.replace(' ', '_') + similarity)
                            os.makedirs(curr_dir, exist_ok=True)
                            curr_molecule_pth = os.path.join(curr_dir, current_id.replace(' ', '_') + similarity +'.csv')
                            df: pd.DataFrame = self.zinc_id_catalog_map[current_id]['catalogs']
                            if not df.empty:
                                df.to_csv(curr_molecule_pth, index=None)
                                
    def save_to_smiles(self, name):
        if name == 'Current Similar Molecule':
            current_id = self.zinc_id_combobox.currentText()
            if current_id:
                current_molecule_name = self.list_of_names.selectedItems()[0].text()
                dir = QFileDialog.getExistingDirectory(self, 'Select Save Directory', '')
                if dir:
                    new_dir = os.path.join(dir, current_molecule_name)
                    os.makedirs(new_dir, exist_ok=True)
                    similarity = f'{self.zinc_id_catalog_map[current_id]['similarity']:.4f}'
                    curr_dir = os.path.join(new_dir, current_id.replace(' ', '_') + similarity)
                    os.makedirs(curr_dir, exist_ok=True)
                    curr_molecule_pth = os.path.join(curr_dir, current_id.replace(' ', '_') + similarity +'.smi')
                    smiles = self.zinc_id_catalog_map[current_id]['smiles']
                    with open(curr_molecule_pth, 'w') as f:
                        f.write(f'{smiles} {current_id}')
        elif name == 'All Similar Molecules':
            current_items = self.list_of_names.selectedItems()
            if current_items:
                current_molecule_name = current_items[0].text()
                dir = QFileDialog.getExistingDirectory(self, 'Select Save Directory', '')
                if dir:
                    zinc_id_list = self.name_zinc_map[current_molecule_name]['zinc_ids']
                    new_dir = os.path.join(dir, current_molecule_name)
                    os.makedirs(new_dir, exist_ok=True)
                    for current_id in zinc_id_list:
                        similarity = f'{self.zinc_id_catalog_map[current_id]['similarity']:.4f}'
                        curr_dir = os.path.join(new_dir, current_id.replace(' ', '_') + similarity)
                        os.makedirs(curr_dir, exist_ok=True)
                        curr_molecule_pth = os.path.join(curr_dir, current_id.replace(' ', '_') + similarity +'.smi')
                        smiles = self.zinc_id_catalog_map[current_id]['smiles']
                        with open(curr_molecule_pth, 'w') as f:
                            f.write(f'{smiles} {current_id}')
        else:
            num_of_items = self.list_of_names.count()
            dir = QFileDialog.getExistingDirectory(self, 'Select Save Directory', '')
            if dir:
                for r in range(num_of_items):
                    item = self.list_of_names.item(r)
                    current_molecule_name = item.text()
                    if current_molecule_name in self.name_zinc_map:
                        zinc_id_list = self.name_zinc_map[current_molecule_name]['zinc_ids']
                        new_dir = os.path.join(dir, current_molecule_name)
                        os.makedirs(new_dir, exist_ok=True)
                        for current_id in zinc_id_list:
                            similarity = f'{self.zinc_id_catalog_map[current_id]['similarity']:.4f}'
                            curr_dir = os.path.join(new_dir, current_id.replace(' ', '_') + similarity)
                            os.makedirs(curr_dir, exist_ok=True)
                            curr_molecule_pth = os.path.join(curr_dir, current_id.replace(' ', '_') + similarity +'.smi')
                            smiles = self.zinc_id_catalog_map[current_id]['smiles']
                            with open(curr_molecule_pth, 'w') as f:
                                f.write(f'{smiles} {current_id.split(' ')[0]}')
    
    def change_dark_light_mode(self, display_mode):
        self.display_mode = display_mode
        items = self.list_of_names.selectedItems()
        if items:
            name = items[0].text()
            curr_smiles = self.name_smiles_df.loc[self.name_smiles_df['Name'] == name]['SMILES'].iloc[0]
            self.curr_name_img_label.set_image_smiles(self.draw_svg_bytes(Chem.MolFromSmiles(curr_smiles)), curr_smiles)
        zinc_id = self.zinc_id_combobox.currentText()
        if zinc_id:
            smiles = self.zinc_id_catalog_map[zinc_id]['smiles']
            mol = Chem.MolFromSmiles(smiles)
            self.chem_image_label.set_image_smiles(self.draw_svg_bytes(mol), smiles)
            
    def clear_all(self):
        self.list_of_names.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(1)
        self.purchasability_table.clearContents()
        self.chemprop_table.clearContents()
        self.chemprop_table_set_prop_rows()
        self.zinc_id_combobox.clear()
        self.curr_name_img_label.set_image_smiles(self.empty_img, None)
        self.chem_image_label.set_image_smiles(self.empty_img, None)
        supplier_dict = {'Name': [], 'SMILES': []}
        supplier_dict.update({prop_name: [] for prop_name in property_functions})
        self.supplier_df = pd.DataFrame(supplier_dict)
        self.name_smiles_df = pd.DataFrame({'Name': [], 'SMILES': []})
        self.zinc_id_catalog_map = {}
        self.failed_or_non_smiles = []
        self.name_zinc_map = {}
        self.smiles_lineedit.clear()
        self.zinc_id_lineedit.clear()

class PubChemSupplierFinderWidget(QWidget):
    def __init__(self, parent, display_mode: str):
        super().__init__(parent)
        supplier_dict = {'Name': [], 'SMILES': []}
        supplier_dict.update({prop_name: [] for prop_name in property_functions})
        self.supplier_df = pd.DataFrame(supplier_dict)
        self.name_smiles_df = pd.DataFrame({'Name': [], 'SMILES': []})
        self.browser = None
        self.display_mode = display_mode
        self.setupUI()
    
    def setupUI(self):
        overall_layout = QHBoxLayout()
        list_name_layout = QVBoxLayout()
        search_option_layout = QGridLayout()
        search_option_progress_layout = QVBoxLayout()
        table_information_layout = QGridLayout()
        table_layout = QHBoxLayout()
        information_table_layout = QVBoxLayout()
        final_btn_layout = QHBoxLayout()
        table_information_and_image_layout = QHBoxLayout()
        result_table_layout = QVBoxLayout()
        similarity_label = QLabel('<b>Sim. Thres.:</b>')
        self.similarity_spinbox = QSpinBox()
        self.similarity_spinbox.setRange(1, 100)
        self.similarity_spinbox.setValue(100)
        self.similarity_spinbox.setMinimumWidth(75)
        # pubchem_max_label = QLabel('<b>Max Records:</b>')
        # self.pubchem_max_record_spinbox = QSpinBox()
        # self.pubchem_max_record_spinbox.setRange(1, 150) # PubChem set default to 2_000_000, 150 seems fine
        # self.pubchem_max_record_spinbox.setValue(5)
        self.search_button = QPushButton('Search')
        self.search_button.clicked.connect(self.search_through_pubchem)
        if not len(self.name_smiles_df):
            self.search_button.setDisabled(True)
        
        search_option_layout.addWidget(similarity_label, 0, 0)
        search_option_layout.addWidget(self.similarity_spinbox, 0, 1)
        search_option_layout.addWidget(self.search_button, 0, 2)
        
        # search_option_layout.addWidget(pubchem_max_label, 2, 0)
        # search_option_layout.addWidget(self.pubchem_max_record_spinbox, 2, 1)
        # search_option_layout.addWidget(self.search_button, 3, 0, 1, 2)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        if len(self.name_smiles_df):
            self.progress_bar.setMaximum(len(self.name_smiles_df))
        else:
            self.progress_bar.setMaximum(1)
            
        search_option_progress_layout.addLayout(search_option_layout)
        search_option_progress_layout.addWidget(self.progress_bar)
        
        self.list_of_names = DropFileDirListWidget()
        self.list_of_names.itemSelectionChanged.connect(self.update_combo_with_cid)
        self.fill_list_with_names()
        self.list_of_names.setEditTriggers(QListWidget.EditTrigger.DoubleClicked)
        self.list_of_names.supplierdictSignal.connect(self.add_new_mols_to_supplier_df)
        self.list_of_names.itemClicked.connect(self.keep_name_before_changing)
        self.list_of_names.itemDoubleClicked.connect(self.start_editing_text)
        self.list_of_names.itemChanged.connect(self.change_name_of_dfs_to_new_name)
        self.list_of_names.currCountChanged.connect(self.check_search_button)
        self.list_of_names.itemRemovedSignal.connect(self.remove_text_from_dfs)
        
        # self.empty_img = Image.new('RGB', (200, 200), (255, 255, 255))
        # self.curr_name_img_label = ImageWithSearchLabel(self.empty_img)
        curr_name_frame = QFrame()
        curr_name_frame.setFrameShape(QFrame.Shape.StyledPanel)
        curr_name_frame.setLineWidth(10)
        curr_name_layout = QHBoxLayout(curr_name_frame)
        curr_name_layout.setContentsMargins(0, 0, 0, 0)
        self.empty_img = self.generate_placeholder_svg()
        self.curr_name_img_label = SVGImageWithSearchWidget(self.empty_img)
        self.curr_name_img_label.setFixedSize(220, 220)
        curr_name_layout.addWidget(self.curr_name_img_label)
        
        list_name_layout.addWidget(curr_name_frame, alignment=Qt.AlignmentFlag.AlignCenter)
        list_name_layout.addLayout(search_option_progress_layout)
        list_name_layout.addWidget(self.list_of_names)
        
        pubchem_id_combo_label = QLabel('<b>Matching PubChem ID :</b>')
        self.pubchem_id_combobox = QComboBox()
        self.pubchem_id_combobox.currentTextChanged.connect(self.update_table_with_cid)
        pubchem_id_label = QLabel('<b>PubChem ID :</b>')
        self.pubchem_id_lineedit = CopyLineEdit()
        self.smiles_label = QLabel('<b>SMILES :</b>')
        self.smiles_lineedit = CopyLineEdit()
        similarity_label = QLabel('<b>Similarity :</b>')
        self.similarity_lineedit = QLineEdit()
        self.similarity_lineedit.setReadOnly(True)
        self.similarity_lineedit.setMaximumWidth(70)
        table_information_layout.addWidget(pubchem_id_combo_label, 0, 0)
        table_information_layout.addWidget(self.pubchem_id_combobox, 0, 1, 1, 3)
        table_information_layout.addWidget(pubchem_id_label, 1, 0)
        table_information_layout.addWidget(self.pubchem_id_lineedit, 1, 1)
        table_information_layout.addWidget(similarity_label, 1, 2)
        table_information_layout.addWidget(self.similarity_lineedit, 1, 3)
        table_information_layout.addWidget(self.smiles_label, 2, 0)
        table_information_layout.addWidget(self.smiles_lineedit, 2, 1, 1, 3)
        
        # self.chem_image_label = ImageWithSearchLabel(self.empty_img)
        chem_image_frame = QFrame()
        chem_image_frame.setFrameShape(QFrame.Shape.StyledPanel)
        chem_image_frame.setLineWidth(10)
        chem_image_layout = QHBoxLayout(chem_image_frame)
        chem_image_layout.setContentsMargins(0, 0, 0, 0)
        self.chem_image_label = SVGImageWithSearchWidget(self.empty_img)
        self.chem_image_label.setFixedSize(220, 220)
        chem_image_layout.addWidget(self.chem_image_label)
        table_information_and_image_layout.addLayout(table_information_layout)
        table_information_and_image_layout.addWidget(chem_image_frame)
        
        self.purchasability_table = QTableWidget()
        self.purchasability_table.setMinimumWidth(550)
        self.purchasability_table.setColumnCount(2)
        self.purchasability_table.setHorizontalHeaderLabels(['Catalog Name', 'Supplier Code'])
        header = self.purchasability_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        self.chemprop_table = QTableWidget()
        self.chemprop_table.setMinimumWidth(300)
        self.chemprop_table.setColumnCount(3)
        self.chemprop_table.setHorizontalHeaderLabels(['Properties', 'Query', 'Target'])
        self.chemprop_table_set_prop_rows()
        self.chemprop_table.verticalHeader().setVisible(False)
        self.chemprop_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header = self.chemprop_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        
        table_layout.addWidget(self.purchasability_table)
        table_layout.addWidget(self.chemprop_table, alignment=Qt.AlignmentFlag.AlignRight)
        
        information_table_layout.addLayout(table_information_and_image_layout)
        information_table_layout.addLayout(table_layout)
        
        btn_right_layout = QHBoxLayout()
        btn_right_widget = QWidget()
        
        # self.custom_search_button = QPushButton('Custom Search')
        # self.custom_search_button.clicked.connect(self.custom_search_dialog)
        self.save_files_button = QPushButton('Save', self)
        self.save_menu_setup()
        # btn_right_layout.addWidget(self.custom_search_button)
        btn_right_layout.addWidget(self.save_files_button)
        btn_right_layout.setContentsMargins(0, 0, 0, 0)
        btn_right_widget.setLayout(btn_right_layout)
        final_btn_layout.addWidget(btn_right_widget, alignment=Qt.AlignmentFlag.AlignRight)
        
        result_table_layout.addLayout(information_table_layout)
        result_table_layout.addLayout(final_btn_layout)
        
        overall_layout.addLayout(list_name_layout, 1)
        overall_layout.addLayout(result_table_layout, 9)
        
        self.setLayout(overall_layout)
        
        self.start_editing_listitem_text = False
        self.currently_searching = False
        self.failed_or_non_smiles = []
        self.name_pubchem_map = {}
    
    def generate_placeholder_svg(self):
        placeholder_svg = '''
        <svg width="150" height="150" xmlns="http://www.w3.org/2000/svg">
            <rect width="150%" height="150%" fill="none" stroke="gray" stroke-width="1"/>
            <text x="75%" y="50%" font-size="13" dominant-baseline="middle" text-anchor="middle" fill="gray">
                No
            </text>
            <text x="75%" y="75%" font-size="13" dominant-baseline="middle" text-anchor="middle" fill="gray">
                Molecule
            </text>
            <text x="75%" y="100%" font-size="13" dominant-baseline="middle" text-anchor="middle" fill="gray">
                Selected
            </text>
        </svg>
        '''
        return placeholder_svg.encode('utf-8')
    
    def generate_failed_svg(self):
        failed_svg = '''
        <svg width="150" height="150" xmlns="http://www.w3.org/2000/svg">
            <rect width="150%" height="150%" fill="none" stroke="gray" stroke-width="1"/>
            <text x="75%" y="50%" font-size="13" dominant-baseline="middle" text-anchor="middle" fill="gray">
                Image
            </text>
            <text x="75%" y="75%" font-size="13" dominant-baseline="middle" text-anchor="middle" fill="gray">
                Generation
            </text>
            <text x="75%" y="100%" font-size="13" dominant-baseline="middle" text-anchor="middle" fill="gray">
                Failed
            </text>
        </svg>
        '''
        return failed_svg.encode('utf-8')
    
    def draw_svg_bytes(self, mol):
        drawer = Draw.MolDraw2DSVG(600, 600)
        options = drawer.drawOptions()
        if self.display_mode == 'dark':
            SetDarkMode(options)
            bgcolor = '#000000'
        else:
            bgcolor = '#FFFFFF'
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        strings = drawer.GetDrawingText().replace(f"<rect style='opacity:1.0;fill:{bgcolor}", "<rect style='opacity:1.0;fill:none")
        return strings.encode('utf-8')
    
    def add_new_mols_to_supplier_df(self, supplier_dict: dict):
        if supplier_dict:
            existed_names = self.name_smiles_df['Name'].values
            for idx, name in enumerate(supplier_dict['Name']):
                if name in existed_names:
                    i = 1
                    while f'{name}_{i}' in existed_names:
                        i += 1
                    supplier_dict['Name'][idx] = f'{name}_{i}'
            df = pd.DataFrame(supplier_dict)
            self.supplier_df = pd.concat([self.supplier_df, df]).reset_index(drop=True)
            self.name_smiles_df = self.supplier_df[['Name', 'SMILES']]
            for idx, name in enumerate(supplier_dict['Name']):
                item = QListWidgetItem()
                item.setText(name)
                if self.currently_searching:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable & ~Qt.ItemFlag.ItemIsEnabled & ~Qt.ItemFlag.ItemIsEditable)
                else:
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                self.list_of_names.addItem(item)
    
    def start_editing_text(self):
        self.start_editing_listitem_text = True
    
    def keep_name_before_changing(self, item: QListWidgetItem):
        self.curr_item_text = item.text()
        
    def change_name_of_dfs_to_new_name(self, item: QListWidgetItem):
        if self.start_editing_listitem_text and item.isSelected():
            new_text = item.text()
            if not new_text:
                item.setText(self.curr_item_text)
            elif new_text != self.curr_item_text:
                if new_text in self.supplier_df['Name'].to_list():
                    QMessageBox.critical(self, 'NameError', f'{new_text}, already exists!')
                    item.setText(self.curr_item_text)
                    return
                self.supplier_df[['Name']] = self.supplier_df[['Name']].replace(self.curr_item_text, new_text)
                self.name_smiles_df[['Name']] = self.name_smiles_df[['Name']].replace(self.curr_item_text, new_text)
                self.name_pubchem_map[new_text] = self.name_pubchem_map.pop(self.curr_item_text)
            self.start_editing_listitem_text = False
    
    def remove_text_from_dfs(self, name):
        kept_rows = self.name_smiles_df['Name'] != name
        self.supplier_df = self.supplier_df[kept_rows].reset_index(drop=True)
        self.name_smiles_df = self.name_smiles_df[kept_rows].reset_index(drop=True)
    
    def check_search_button(self, num):
        if num:
            self.search_button.setEnabled(True)
        else:
            self.search_button.setEnabled(False)
    
    def chemprop_table_set_prop_rows(self):
        self.chemprop_table.setRowCount(len(property_functions))
        for r, k in enumerate(property_functions):
            item = QTableWidgetItem()
            item.setText(k)
            self.chemprop_table.setItem(r, 0, item)
    
    def save_menu_setup(self):
        options = ['Current Similar Molecule', 'All Similar Molecules', 'Full Result']
        all_actions = []
        for name in options:
            curr_action = QAction(name, self)
            curr_png_action = QAction('Image', self)
            curr_catalog_table_action = QAction('Catalog Table', self)
            curr_smiles_action = QAction('SMILES', self)
            
            submenu = QMenu(self)
            submenu.addActions([curr_png_action,
                                curr_catalog_table_action,
                                curr_smiles_action])
            curr_action.setMenu(submenu)
            
            curr_png_action.triggered.connect(lambda _, n=name: self.save_to_png(n))
            curr_catalog_table_action.triggered.connect(lambda _, n=name: self.save_to_catalog_table(n))
            curr_smiles_action.triggered.connect(lambda _, n=name: self.save_to_smiles(n))
            
            all_actions.append(curr_action)
        
        save_menu = QMenu(self)
        save_menu.addActions(all_actions)
        self.save_files_button.setMenu(save_menu)
    
    def fill_list_with_names(self):
        names = self.name_smiles_df['Name'].to_list()
        self.list_of_names.addItems(names)
        default_brush = QListWidgetItem().foreground()
        for idx in range(len(names)):
            item = self.list_of_names.item(idx)
            f = item.font()
            f.setStrikeOut(False)
            item.setFont(f)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            item.setForeground(default_brush)
    
    def search_through_pubchem(self):
        if not self.currently_searching:
            all_unique_smiles_list = list(set(self.name_smiles_df['SMILES']))
            params = {'Threshold' : self.similarity_spinbox.value(),}
            
            default_brush = QListWidgetItem().foreground()
            for idx in range(len(self.name_smiles_df)):
                item = self.list_of_names.item(idx)
                f = item.font()
                f.setStrikeOut(False)
                item.setFont(f)
                item.setFlags(item.flags() & 
                              ~Qt.ItemFlag.ItemIsSelectable & 
                              ~Qt.ItemFlag.ItemIsEnabled & 
                              ~Qt.ItemFlag.ItemIsEditable)
                item.setForeground(default_brush)
            self.list_of_names.clearSelection()
            self.purchasability_table.clearContents()
            self.chemprop_table.clearContents()
            self.chemprop_table_set_prop_rows()
            self.pubchem_id_combobox.clear()
            self.curr_name_img_label.set_image_smiles(self.empty_img, None)
            self.chem_image_label.set_image_smiles(self.empty_img, None)
            with open(os.path.join(os.path.dirname(__file__), 'fp_param', 'fp_param.json'), 'r') as f:
                fp_settings = json.load(f)
            self.fpgen = retrieve_fp_generator(fp_settings)
            self.sim_method_func = retrieve_similarity_method(fp_settings['sim'])
            
            self.currently_searching = True
            self.name_pubchem_map = {}
            self.pubchem_id_catalog_map = {}
            self.failed_or_non_smiles = []
            self.progress_bar.setValue(0)
            self.progress_bar.setMaximum(len(self.name_smiles_df.index))
            self.smiles_lineedit.clear()
            self.pubchem_id_lineedit.clear()
            self.search_button.setText('Stop')
            # self.search_button.setDisabled(True)
            self.thread = QThread()
            self.worker = MultiThreadPubChemSearch(params, all_unique_smiles_list)
            self.worker.moveToThread(self.thread)
            self.worker.curr_step_text.connect(self.update_progress_bar_and_df)
            self.worker.canceled.connect(self.cancel_search)
            self.worker.finished.connect(self.thread.quit)
            self.thread.finished.connect(self.searching_finished)
            self.thread.started.connect(self.worker.run)
            self.thread.start()
        else:
            self.worker.stop()
            self.search_button.setEnabled(False)
            self.search_button.setText('Stopping')
            
    def cancel_search(self):
        self.thread.quit()
        self.thread.wait()
        self.worker.deleteLater()
        self.thread.deleteLater()
        self.search_button.setText('Search')
        self.search_button.setEnabled(True)
        for i in range(self.list_of_names.count()):
            item = self.list_of_names.item(i)
            if item.flags() & ~Qt.ItemFlag.ItemIsEnabled:
                color = QColor()
                color.setRgb(255, 80, 80)
                item.setForeground(color)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable)
                item.setToolTip('Search Stopped')
        self.progress_bar.setValue(self.progress_bar.maximum())
    
    def update_progress_bar_and_df(self, cids_dict: dict, smiles_str: str):
        smiles_series = self.name_smiles_df['SMILES']
        idxs = smiles_series[smiles_series == smiles_str].index
        first_key = next(iter(cids_dict))
        if first_key == '':    # search failed
            self.failed_or_non_smiles.extend(smiles_str)
            for idx in idxs:
                item = self.list_of_names.item(idx)
                color = QColor()
                color.setRgb(255, 165, 0)
                item.setForeground(color)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable)
                item.setToolTip('Searched failed.')
        elif first_key == ' ':
            self.failed_or_non_smiles.extend(smiles_str)
            for idx in idxs:
                item = self.list_of_names.item(idx)
                f = item.font()
                f.setStrikeOut(True)
                item.setFont(f)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable)
                item.setToolTip('No match found')
        else:
            for idx in idxs:
                item = self.list_of_names.item(idx)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable)
            self.parse_single_downloaded_dict(cids_dict, smiles_str)
        self.progress_bar.setValue(self.progress_bar.value() + len(idxs))
        
    def parse_single_downloaded_dict(self, cids_dict: dict, smiles_str: str):
        bool_row = self.name_smiles_df['SMILES'] == smiles_str
        query_mol = Chem.MolFromSmiles(smiles_str)
        query_smiles = Chem.MolToSmiles(query_mol)
        query_fps = self.fpgen(query_mol)
        names = self.name_smiles_df[bool_row]['Name'].to_list()
        
        for name_idx, name in enumerate(names): # in-case there are multiple names with same smiles, then just reference it
            if name_idx == 0:
                similarity_list = []    # used for sorting
                first_name = name
                self.name_pubchem_map[name] = {'pubchem_ids': []}
                for cid, vendor_dict in cids_dict.items():
                    cid_smiles = vendor_dict['smiles']
                    cid_mol = Chem.MolFromSmiles(cid_smiles)
                    if cid_mol is not None:
                        cid_smiles = Chem.MolToSmiles(cid_mol)
                        cid_fps = self.fpgen(cid_mol)
                        similarity = self.sim_method_func(query_fps, cid_fps)
                        similarity_list.append(similarity)
                        if cid_smiles == query_smiles:
                            cid += f' (Exact)'
                        self.name_pubchem_map[name]['pubchem_ids'].append(cid)
                        del vendor_dict['smiles']
                        catalog_dict = vendor_dict
                        mol = Chem.MolFromSmiles(cid_smiles)
                        mol = Chem.AddHs(mol)
                        properties = {f'{chem_prop}': func(mol) for chem_prop, func in property_functions.items()}
                        self.pubchem_id_catalog_map[cid] = {'smiles'    : cid_smiles,
                                                            'catalogs'  : pd.DataFrame(catalog_dict),
                                                            'properties': properties,
                                                            'similarity': similarity}
                    else:
                        ori_smiles = vendor_dict['smiles']
                        del vendor_dict['smiles']
                        catalog_dict = vendor_dict
                        self.pubchem_id_catalog_map[cid] = {'smiles'    : ori_smiles,
                                                            'catalogs'  : pd.DataFrame(catalog_dict),
                                                            'properties': {f'{chem_prop}': float('nan') for chem_prop in property_functions},
                                                            'similarity': float('nan')}
                try:
                    exact_id_idx = next(idx for idx, id in enumerate(self.name_pubchem_map[name]['pubchem_ids']) if id.endswith(' (Exact)'))
                    exact_id = self.name_pubchem_map[name]['pubchem_ids'].pop(exact_id_idx)
                    similarity_list.pop(exact_id_idx)
                    sorted_similarity_zinc = [x for _, x in sorted(zip(similarity_list, self.name_pubchem_map[name]['pubchem_ids']), reverse=True)]
                    self.name_pubchem_map[name]['pubchem_ids'] = [exact_id] + sorted_similarity_zinc
                except:
                    sorted_similarity_zinc = [x for _, x in sorted(zip(similarity_list, self.name_pubchem_map[name]['pubchem_ids']), reverse=True)]
                    self.name_pubchem_map[name]['pubchem_ids'] = sorted_similarity_zinc
            else:
                self.name_pubchem_map[name] = self.name_pubchem_map[first_name]
                
    def searching_finished(self):
        self.currently_searching = False
        self.search_button.setEnabled(True)
        self.search_button.setText('Search')
        self.thread.wait()
        self.worker.deleteLater()
        self.thread.deleteLater()
    
    def update_combo_with_cid(self):
        items = self.list_of_names.selectedItems()
        if items:
            name = items[0].text()
            curr_smiles = self.name_smiles_df.loc[self.name_smiles_df['Name'] == name]['SMILES'].iloc[0]
            self.curr_name_img_label.set_image_smiles(self.draw_svg_bytes(Chem.MolFromSmiles(curr_smiles)), curr_smiles)
            if name not in self.name_pubchem_map:
                self.pubchem_id_combobox.clear()
                self.purchasability_table.clearContents()
                self.chemprop_table.clearContents()
                self.chemprop_table_set_prop_rows()
                self.pubchem_id_lineedit.clear()
                self.smiles_lineedit.clear()
                return
            pubchem_ids = self.name_pubchem_map[name]['pubchem_ids']
            self.pubchem_id_combobox.clear()
            self.pubchem_id_combobox.addItems(pubchem_ids)
        else:
            self.curr_name_img_label.set_image_smiles(self.empty_img, None)
            self.purchasability_table.clearContents()
    
    def update_table_with_cid(self, pubchem_id):
        if pubchem_id:
            self.pubchem_id_lineedit.setText(pubchem_id.split(' (')[0])
            smiles = self.pubchem_id_catalog_map[pubchem_id]['smiles']
            self.smiles_lineedit.setText(smiles)
            self.similarity_lineedit.setText(f'{self.pubchem_id_catalog_map[pubchem_id]['similarity']:.4f}')
            self.pubchem_id_lineedit.setCursorPosition(0)
            self.smiles_lineedit.setCursorPosition(0)
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                self.chem_image_label.set_image_smiles(self.generate_failed_svg(), smiles)
            else:
                self.chem_image_label.set_image_smiles(self.draw_svg_bytes(mol), smiles)
            self.populate_catalog_table(self.pubchem_id_catalog_map[pubchem_id]['catalogs'])
            self.compare_query_and_target_properties()
        else:
            self.chem_image_label.set_image_smiles(self.empty_img, None)
            self.chemprop_table.clearContents()
            self.purchasability_table.clearContents()
            self.purchasability_table.setRowCount(0)
    
    def populate_catalog_table(self, catalog_df: pd.DataFrame):
        self.purchasability_table.setSortingEnabled(False)
        self.purchasability_table.clearContents()
        table_rows = len(catalog_df)
        self.purchasability_table.setRowCount(table_rows)
        for idx, row in catalog_df.iterrows():
            cat_name, cat_supplier_code, cat_url = row
            
            cat_name_item = QTableWidgetItem()
            cat_name_item.setText(cat_name)
            cat_name_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            cat_name_item.setFlags(cat_name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if cat_url is not None:
                cat_supplier_label = QLabel(f'<a href="{cat_url}">{cat_supplier_code}</a>')
                cat_supplier_label.setOpenExternalLinks(False)
                cat_supplier_label.linkActivated.connect(self.show_custom_webbrowser)
            else:
                cat_supplier_label = QLabel(str(cat_supplier_code))
            cat_supplier_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            self.purchasability_table.setItem(idx, 0, cat_name_item)
            self.purchasability_table.setCellWidget(idx, 1, cat_supplier_label)
        self.purchasability_table.resizeColumnsToContents()
        header = self.purchasability_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.purchasability_table.setSortingEnabled(True)
        
    def compare_query_and_target_properties(self):
        items = self.list_of_names.selectedItems()
        if not items:
            return
        query_name = items[0].text()
        query_series = self.supplier_df[self.supplier_df['Name'] == query_name]
        target_chemprop_dict = self.pubchem_id_catalog_map[self.pubchem_id_combobox.currentText()]['properties']
        self.chemprop_table.clearContents()
        for r, k in enumerate(property_functions):
            query_value = query_series.get(k).to_list()[0]
            target_value = target_chemprop_dict[k]
            
            color = QTableWidgetItem().foreground()
            if target_value - query_value >= 1e-3:
                color = QColor()
                color.setRgb(229, 115, 115)
            elif query_value - target_value >= 1e-3:
                color = QColor()
                color.setRgb(76, 175, 80)
                
            prop_item = QTableWidgetItem()
            query_item = QTableWidgetItem()
            target_item = QTableWidgetItem()
            flags = prop_item.flags()
            
            prop_item.setText(k)
            if k in ['Molecular Weight', 'LogP', 'Topological Polar Surface Area', 'Molar Refractivity', 'QED']:
                query_item.setText(f'{query_value:.3f}')
                target_item.setText(f'{target_value:.3f}')
            else:
                query_item.setText(f'{int(query_value)}')
                target_item.setText(f'{int(target_value)}')
            target_item.setForeground(color)
            prop_item.setFlags(flags & ~Qt.ItemFlag.ItemIsEditable)
            query_item.setFlags(flags & ~Qt.ItemFlag.ItemIsEditable)
            target_item.setFlags(flags & ~Qt.ItemFlag.ItemIsEditable)
            
            self.chemprop_table.setItem(r, 0, prop_item)
            self.chemprop_table.setItem(r, 1, query_item)
            self.chemprop_table.setItem(r, 2, target_item)
    
    def show_custom_webbrowser(self, url):
        if self.browser is None:
            self.browser = BrowserWithTabs(self, url)
            self.browser.closed.connect(lambda: setattr(self, 'browser', None))
        else:
            self.browser.tab_browser.add_new_tab(url)
            self.browser.raise_()
            self.browser.activateWindow()
    
    def save_to_png(self, name):
        if name == 'Current Similar Molecule':
            current_id = self.pubchem_id_combobox.currentText()
            if current_id:
                current_molecule_name = self.list_of_names.selectedItems()[0].text()
                dir = QFileDialog.getExistingDirectory(self, 'Select Save Directory', '')
                if dir:
                    new_dir = os.path.join(dir, current_molecule_name)
                    os.makedirs(new_dir, exist_ok=True)
                    similarity = f'_{self.pubchem_id_catalog_map[current_id]['similarity']:.4f}'
                    curr_dir = os.path.join(new_dir, current_id.replace(' ', '_') + similarity)
                    os.makedirs(curr_dir, exist_ok=True)
                    curr_molecule_pth = os.path.join(curr_dir, current_id.replace(' ', '_') + '.png')  # replace the space in " (Exact)"
                    mol = Chem.MolFromSmiles(self.pubchem_id_catalog_map[current_id]['smiles'])
                    svgtopng(self.draw_svg_bytes(mol), curr_molecule_pth)
        elif name == 'All Similar Molecules':
            current_items = self.list_of_names.selectedItems()
            if current_items:
                current_molecule_name = current_items[0].text()
                if current_molecule_name in self.name_pubchem_map:
                    dir = QFileDialog.getExistingDirectory(self, 'Select Save Directory', '')
                    if dir:
                        pubchem_id_list = self.name_pubchem_map[current_molecule_name]['pubchem_ids']
                        new_dir = os.path.join(dir, current_molecule_name)
                        os.makedirs(new_dir, exist_ok=True)
                        for current_id in pubchem_id_list:
                            similarity = f'_{self.pubchem_id_catalog_map[current_id]['similarity']:.4f}'
                            curr_dir = os.path.join(new_dir, current_id.replace(' ', '_') + similarity)
                            os.makedirs(curr_dir, exist_ok=True)
                            curr_molecule_pth = os.path.join(curr_dir, current_id.replace(' ', '_') + '.png')
                            mol = Chem.MolFromSmiles(self.pubchem_id_catalog_map[current_id]['smiles'])
                            svgtopng(self.draw_svg_bytes(mol), curr_molecule_pth)
        else:
            num_of_items = self.list_of_names.count()
            dir = QFileDialog.getExistingDirectory(self, 'Select Save Directory', '')
            if dir:
                for r in range(num_of_items):
                    item = self.list_of_names.item(r)
                    current_molecule_name = item.text()
                    new_dir = os.path.join(dir, current_molecule_name)
                    os.makedirs(new_dir, exist_ok=True)
                    if current_molecule_name in self.name_pubchem_map:
                        pubchem_id_list = self.name_pubchem_map[current_molecule_name]['pubchem_ids']
                        for current_id in pubchem_id_list:
                            similarity = f'_{self.pubchem_id_catalog_map[current_id]['similarity']:.4f}'
                            curr_dir = os.path.join(new_dir, current_id.replace(' ', '_') + similarity)
                            os.makedirs(curr_dir, exist_ok=True)
                            curr_molecule_pth = os.path.join(curr_dir, current_id.replace(' ', '_') +'.png')
                            mol = Chem.MolFromSmiles(self.pubchem_id_catalog_map[current_id]['smiles'])
                            svgtopng(self.draw_svg_bytes(mol), curr_molecule_pth)
        
    def save_to_catalog_table(self, name):
        if name == 'Current Similar Molecule':
            current_id = self.pubchem_id_combobox.currentText()
            if current_id:
                current_molecule_name = self.list_of_names.selectedItems()[0].text()
                dir = QFileDialog.getExistingDirectory(self, 'Select Save Directory', '')
                if dir:
                    new_dir = os.path.join(dir, current_molecule_name)
                    os.makedirs(new_dir, exist_ok=True)
                    similarity = f'_{self.pubchem_id_catalog_map[current_id]['similarity']:.4f}'
                    curr_dir = os.path.join(new_dir, current_id.replace(' ', '_') + similarity)
                    os.makedirs(curr_dir, exist_ok=True)
                    curr_molecule_pth = os.path.join(curr_dir, current_id.replace(' ', '_') + '.csv')
                    df: pd.DataFrame = self.pubchem_id_catalog_map[current_id]['catalogs']
                    df.to_csv(curr_molecule_pth, index=None)
        elif name == 'All Similar Molecules':
            current_items = self.list_of_names.selectedItems()
            if current_items:
                current_molecule_name = current_items[0].text()
                if current_molecule_name in self.name_pubchem_map:
                    dir = QFileDialog.getExistingDirectory(self, 'Select Save Directory', '')
                    if dir:
                        pubchem_id_list = self.name_pubchem_map[current_molecule_name]['pubchem_ids']
                        new_dir = os.path.join(dir, current_molecule_name)
                        os.makedirs(new_dir, exist_ok=True)
                        for current_id in pubchem_id_list:
                            similarity = f'_{self.pubchem_id_catalog_map[current_id]['similarity']:.4f}'
                            curr_dir = os.path.join(new_dir, current_id.replace(' ', '_') + similarity)
                            os.makedirs(curr_dir, exist_ok=True)
                            curr_molecule_pth = os.path.join(curr_dir, current_id.replace(' ', '_') + '.csv')
                            df: pd.DataFrame = self.pubchem_id_catalog_map[current_id]['catalogs']
                            df.to_csv(curr_molecule_pth, index=None)
        else:
            num_of_items = self.list_of_names.count()
            dir = QFileDialog.getExistingDirectory(self, 'Select Save Directory', '')
            if dir:
                for r in range(num_of_items):
                    item = self.list_of_names.item(r)
                    current_molecule_name = item.text()
                    new_dir = os.path.join(dir, current_molecule_name)
                    os.makedirs(new_dir, exist_ok=True)
                    if current_molecule_name in self.name_pubchem_map:
                        pubchem_id_list = self.name_pubchem_map[current_molecule_name]['pubchem_ids']
                        for current_id in pubchem_id_list:
                            similarity = f'_{self.pubchem_id_catalog_map[current_id]['similarity']:.4f}'
                            curr_dir = os.path.join(new_dir, current_id.replace(' ', '_') + similarity)
                            os.makedirs(curr_dir, exist_ok=True)
                            curr_molecule_pth = os.path.join(curr_dir, current_id.replace(' ', '_') + '.csv')
                            df: pd.DataFrame = self.pubchem_id_catalog_map[current_id]['catalogs']
                            if not df.empty:
                                df.to_csv(curr_molecule_pth, index=None)
                            
    def save_to_smiles(self, name):
        if name == 'Current Similar Molecule':
            current_id = self.pubchem_id_combobox.currentText()
            if current_id:
                current_molecule_name = self.list_of_names.selectedItems()[0].text()
                dir = QFileDialog.getExistingDirectory(self, 'Select Save Directory', '')
                if dir:
                    new_dir = os.path.join(dir, current_molecule_name)
                    os.makedirs(new_dir, exist_ok=True)
                    similarity = f'_{self.pubchem_id_catalog_map[current_id]['similarity']:.4f}'
                    curr_dir = os.path.join(new_dir, current_id.replace(' ', '_') + similarity)
                    os.makedirs(curr_dir, exist_ok=True)
                    curr_molecule_pth = os.path.join(curr_dir, current_id.replace(' ', '_') + '.smi')
                    smiles = self.pubchem_id_catalog_map[current_id]['smiles']
                    with open(curr_molecule_pth, 'w') as f:
                        f.write(f'{smiles} {current_id}')
        elif name == 'All Similar Molecules':
            current_items = self.list_of_names.selectedItems()
            if current_items:
                current_molecule_name = current_items[0].text()
                dir = QFileDialog.getExistingDirectory(self, 'Select Save Directory', '')
                if dir:
                    zinc_id_list = self.name_pubchem_map[current_molecule_name]['pubchem_ids']
                    new_dir = os.path.join(dir, current_molecule_name)
                    os.makedirs(new_dir, exist_ok=True)
                    for current_id in zinc_id_list:
                        similarity = f'_{self.pubchem_id_catalog_map[current_id]['similarity']:.4f}'
                        curr_dir = os.path.join(new_dir, current_id.replace(' ', '_') + similarity)
                        os.makedirs(curr_dir, exist_ok=True)
                        curr_molecule_pth = os.path.join(curr_dir, current_id.replace(' ', '_') + '.smi')
                        smiles = self.pubchem_id_catalog_map[current_id]['smiles']
                        with open(curr_molecule_pth, 'w') as f:
                            f.write(f'{smiles} {current_id}')
        else:
            num_of_items = self.list_of_names.count()
            dir = QFileDialog.getExistingDirectory(self, 'Select Save Directory', '')
            if dir:
                for r in range(num_of_items):
                    item = self.list_of_names.item(r)
                    current_molecule_name = item.text()
                    new_dir = os.path.join(dir, current_molecule_name)
                    os.makedirs(new_dir, exist_ok=True)
                    if current_molecule_name in self.name_pubchem_map:
                        zinc_id_list = self.name_pubchem_map[current_molecule_name]['pubchem_ids']
                        for current_id in zinc_id_list:
                            similarity = f'_{self.pubchem_id_catalog_map[current_id]['similarity']:.4f}'
                            curr_dir = os.path.join(new_dir, current_id.replace(' ', '_') + similarity)
                            os.makedirs(curr_dir, exist_ok=True)
                            curr_molecule_pth = os.path.join(curr_dir, current_id.replace(' ', '_') + '.smi')
                            smiles = self.pubchem_id_catalog_map[current_id]['smiles']
                            with open(curr_molecule_pth, 'w') as f:
                                f.write(f'{smiles} {current_id.split(' ')[0]}')
    
    def change_dark_light_mode(self, display_mode: str):
        self.display_mode = display_mode
        items = self.list_of_names.selectedItems()
        if items:
            name = items[0].text()
            curr_smiles = self.name_smiles_df.loc[self.name_smiles_df['Name'] == name]['SMILES'].iloc[0]
            self.curr_name_img_label.set_image_smiles(self.draw_svg_bytes(Chem.MolFromSmiles(curr_smiles)), curr_smiles)
        pubchem_id = self.pubchem_id_combobox.currentText()
        if pubchem_id:
            smiles = self.pubchem_id_catalog_map[pubchem_id]['smiles']
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                self.chem_image_label.set_image_smiles(self.generate_failed_svg(), smiles)
            else:
                self.chem_image_label.set_image_smiles(self.draw_svg_bytes(mol), smiles)
                
    def clear_all(self):
        self.list_of_names.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(1)
        self.purchasability_table.clearContents()
        self.chemprop_table.clearContents()
        self.chemprop_table_set_prop_rows()
        self.pubchem_id_combobox.clear()
        self.curr_name_img_label.set_image_smiles(self.empty_img, None)
        self.chem_image_label.set_image_smiles(self.empty_img, None)
        supplier_dict = {'Name': [], 'SMILES': []}
        supplier_dict.update({prop_name: [] for prop_name in property_functions})
        self.supplier_df = pd.DataFrame(supplier_dict)
        self.name_smiles_df = pd.DataFrame({'Name': [], 'SMILES': []})
        self.name_pubchem_map = {}
        self.pubchem_id_catalog_map = {}
        self.failed_or_non_smiles = []
        self.smiles_lineedit.clear()
        self.pubchem_id_lineedit.clear()

class LocalDatabaseFinderWidget(QWidget):
    def __init__(self, parent, display_mode: str):
        super().__init__(parent)
        supplier_dict = {'Name': [], 'SMILES': []}
        supplier_dict.update({prop_name: [] for prop_name in property_functions})
        self.supplier_df = pd.DataFrame(supplier_dict)
        self.name_smiles_df = pd.DataFrame({'Name': [], 'SMILES': []})
        self.browser = None
        self.display_mode = display_mode
        self.setupUI()
    
    def setupUI(self):
        db_pth = os.path.join(os.path.dirname(__file__), 'database')
        overall_layout = QHBoxLayout()
        list_name_layout = QVBoxLayout()
        search_option_layout = QGridLayout()
        search_option_progress_layout = QVBoxLayout()
        table_information_layout = QGridLayout()
        table_layout = QHBoxLayout()
        information_table_layout = QVBoxLayout()
        final_btn_layout = QHBoxLayout()
        table_information_and_image_layout = QHBoxLayout()
        result_table_layout = QVBoxLayout()
        similarity_label = QLabel('<b>Sim. Thres.:</b>')
        self.similarity_spinbox = QSpinBox()
        self.similarity_spinbox.setRange(1, 100)
        self.similarity_spinbox.setValue(40)
        self.search_button = QPushButton('Search')
        self.search_button.clicked.connect(self.search_through_db)
        self.search_button.setDisabled(True)
        
        self.db_table = DropDBTableWidget()
        self.db_table.setColumnCount(4)
        self.db_table.setHorizontalHeaderLabels(['Search', 'Database', 'Num', 'URL Template'])
        self.db_table.verticalHeader().setVisible(False)
        self.db_table.chemDBSignal.connect(self.check_search_status)
        self.db_table.setMaximumHeight(120)
        header = self.db_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        with open(os.path.join(db_pth, 'url_template.txt')) as f:
            url_map = {}
            for l in f:
                name, template = l.strip().split()
                url_map[name] = template
        for f in os.listdir(db_pth):
            if f.endswith('.mddb'):
                db_pth = os.path.join(db_pth, f)
                self.db_table.add_file_to_dict(db_pth,
                                               url_map.get(f.rsplit('.', 1)[0], ''),
                                               self.compare_db_and_curr_settings(db_pth))
        
        search_option_layout.addWidget(similarity_label, 0, 0)
        search_option_layout.addWidget(self.similarity_spinbox, 0, 1)
        search_option_layout.addWidget(self.search_button, 0, 2)
        search_option_layout.addWidget(self.db_table, 1, 0, 1, 3)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        if len(self.name_smiles_df):
            self.progress_bar.setMaximum(len(self.name_smiles_df))
        else:
            self.progress_bar.setMaximum(1)
        
        search_option_progress_layout.addLayout(search_option_layout)
        search_option_progress_layout.addWidget(self.progress_bar)
        
        self.list_of_names = DropFileDirListWidget()
        self.list_of_names.itemSelectionChanged.connect(self.update_combo_with_cid)
        self.fill_list_with_names()
        self.list_of_names.setEditTriggers(QListWidget.EditTrigger.DoubleClicked)
        self.list_of_names.supplierdictSignal.connect(self.add_new_mols_to_supplier_df)
        self.list_of_names.itemClicked.connect(self.keep_name_before_changing)
        self.list_of_names.itemDoubleClicked.connect(self.start_editing_text)
        self.list_of_names.itemChanged.connect(self.change_name_of_dfs_to_new_name)
        self.list_of_names.currCountChanged.connect(self.check_search_button)
        self.list_of_names.itemRemovedSignal.connect(self.remove_text_from_dfs)
        
        curr_name_frame = QFrame()
        curr_name_frame.setFrameShape(QFrame.Shape.StyledPanel)
        curr_name_frame.setLineWidth(10)
        curr_name_layout = QHBoxLayout(curr_name_frame)
        curr_name_layout.setContentsMargins(0, 0, 0, 0)
        self.empty_img = self.generate_placeholder_svg()
        self.curr_name_img_label = SVGImageWithSearchWidget(self.empty_img)
        self.curr_name_img_label.setFixedSize(220, 220)
        curr_name_layout.addWidget(self.curr_name_img_label)
        list_name_layout.addWidget(curr_name_frame, alignment=Qt.AlignmentFlag.AlignCenter)
        list_name_layout.addLayout(search_option_progress_layout)
        list_name_layout.addWidget(self.list_of_names)
        
        database_id_combo_label = QLabel('<b>Matching Database ID :</b>')
        self.database_id_combobox = QComboBox()
        self.database_id_combobox.currentTextChanged.connect(self.update_table_with_cid)
        database_id_label = QLabel('<b>Database ID :</b>')
        self.database_id_lineedit = CopyLineEdit()
        self.smiles_label = QLabel('<b>SMILES :</b>')
        self.smiles_lineedit = CopyLineEdit()
        similarity_label = QLabel('<b>Similarity :</b>')
        self.similarity_lineedit = QLineEdit()
        self.similarity_lineedit.setReadOnly(True)
        self.similarity_lineedit.setMaximumWidth(70)
        table_information_layout.addWidget(database_id_combo_label, 0, 0)
        table_information_layout.addWidget(self.database_id_combobox, 0, 1, 1, 3)
        table_information_layout.addWidget(database_id_label, 1, 0)
        table_information_layout.addWidget(self.database_id_lineedit, 1, 1)
        table_information_layout.addWidget(similarity_label, 1, 2)
        table_information_layout.addWidget(self.similarity_lineedit, 1, 3)
        table_information_layout.addWidget(self.smiles_label, 2, 0)
        table_information_layout.addWidget(self.smiles_lineedit, 2, 1, 1, 3)
        
        chem_image_frame = QFrame()
        chem_image_frame.setFrameShape(QFrame.Shape.StyledPanel)
        chem_image_frame.setLineWidth(10)
        chem_image_layout = QHBoxLayout(chem_image_frame)
        chem_image_layout.setContentsMargins(0, 0, 0, 0)
        self.chem_image_label = SVGImageWithSearchWidget(self.empty_img)
        self.chem_image_label.setFixedSize(220, 220)
        chem_image_layout.addWidget(self.chem_image_label)
        table_information_and_image_layout.addLayout(table_information_layout)
        table_information_and_image_layout.addWidget(chem_image_frame)
        
        self.purchasability_table = QTableWidget()
        self.purchasability_table.setMinimumWidth(550)
        self.purchasability_table.setColumnCount(2)
        self.purchasability_table.setHorizontalHeaderLabels(['Catalog Name', 'Supplier Code'])
        header = self.purchasability_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        self.chemprop_table = QTableWidget()
        self.chemprop_table.setMinimumWidth(300)
        self.chemprop_table.setColumnCount(3)
        self.chemprop_table.setHorizontalHeaderLabels(['Properties', 'Query', 'Target'])
        self.chemprop_table_set_prop_rows()
        self.chemprop_table.verticalHeader().setVisible(False)
        self.chemprop_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header = self.chemprop_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        
        table_layout.addWidget(self.purchasability_table)
        table_layout.addWidget(self.chemprop_table, alignment=Qt.AlignmentFlag.AlignRight)
        
        information_table_layout.addLayout(table_information_and_image_layout)
        information_table_layout.addLayout(table_layout)
        
        btn_right_layout = QHBoxLayout()
        btn_right_widget = QWidget()
        
        self.save_files_button = QPushButton('Save', self)
        self.save_menu_setup()
        btn_right_layout.addWidget(self.save_files_button)
        btn_right_layout.setContentsMargins(0, 0, 0, 0)
        btn_right_widget.setLayout(btn_right_layout)
        final_btn_layout.addWidget(btn_right_widget, alignment=Qt.AlignmentFlag.AlignRight)
        
        result_table_layout.addLayout(information_table_layout)
        result_table_layout.addLayout(final_btn_layout)
        
        overall_layout.addLayout(list_name_layout, 1)
        overall_layout.addLayout(result_table_layout, 9)
        
        self.setLayout(overall_layout)
        
        self.start_editing_listitem_text = False
        self.currently_searching = False
        self.failed_or_non_smiles = []
        self.name_dbid_map = {}
    
    def compare_db_and_curr_settings(self, file_path: str, curr_settings: dict=None):
        conn = sqlite3.connect(file_path)
        cur = conn.cursor()
        cur.execute('SELECT format FROM DBInfo WHERE id = ?', (1,))
        row = cur.fetchone()
        db_fp_settings = json.loads(row[0])
        conn.close()
        if not curr_settings:
            with open(os.path.join(os.path.dirname(__file__), 'fp_param', 'fp_param.json'), 'r') as f:
                curr_fp_dict = json.load(f)
        else:
            curr_fp_dict = dict(curr_settings)
        del curr_fp_dict['sim']
        return db_fp_settings == curr_fp_dict
    
    def check_search_status(self, num: int):
        if num:
            self.search_button.setEnabled(bool(len(self.name_smiles_df)))
        else:
            self.search_button.setEnabled(False)
    
    def generate_placeholder_svg(self):
        placeholder_svg = '''
        <svg width="150" height="150" xmlns="http://www.w3.org/2000/svg">
            <rect width="150%" height="150%" fill="none" stroke="gray" stroke-width="1"/>
            <text x="75%" y="50%" font-size="13" dominant-baseline="middle" text-anchor="middle" fill="gray">
                No
            </text>
            <text x="75%" y="75%" font-size="13" dominant-baseline="middle" text-anchor="middle" fill="gray">
                Molecule
            </text>
            <text x="75%" y="100%" font-size="13" dominant-baseline="middle" text-anchor="middle" fill="gray">
                Selected
            </text>
        </svg>
        '''
        return placeholder_svg.encode('utf-8')
    
    def generate_failed_svg(self):
        failed_svg = '''
        <svg width="150" height="150" xmlns="http://www.w3.org/2000/svg">
            <rect width="150%" height="150%" fill="none" stroke="gray" stroke-width="1"/>
            <text x="75%" y="50%" font-size="13" dominant-baseline="middle" text-anchor="middle" fill="gray">
                Image
            </text>
            <text x="75%" y="75%" font-size="13" dominant-baseline="middle" text-anchor="middle" fill="gray">
                Generation
            </text>
            <text x="75%" y="100%" font-size="13" dominant-baseline="middle" text-anchor="middle" fill="gray">
                Failed
            </text>
        </svg>
        '''
        return failed_svg.encode('utf-8')
    
    def draw_svg_bytes(self, mol):
        drawer = Draw.MolDraw2DSVG(600, 600)
        options = drawer.drawOptions()
        if self.display_mode == 'dark':
            SetDarkMode(options)
            bgcolor = '#000000'
        else:
            bgcolor = '#FFFFFF'
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        strings = drawer.GetDrawingText().replace(f"<rect style='opacity:1.0;fill:{bgcolor}", "<rect style='opacity:1.0;fill:none")
        return strings.encode('utf-8')
    
    def add_new_mols_to_supplier_df(self, supplier_dict: dict):
        if supplier_dict:
            existed_names = self.name_smiles_df['Name'].values
            for idx, name in enumerate(supplier_dict['Name']):
                if name in existed_names:
                    i = 1
                    while f'{name}_{i}' in existed_names:
                        i += 1
                    supplier_dict['Name'][idx] = f'{name}_{i}'
            df = pd.DataFrame(supplier_dict)
            self.supplier_df = pd.concat([self.supplier_df, df]).reset_index(drop=True)
            self.name_smiles_df = self.supplier_df[['Name', 'SMILES']]
            for idx, name in enumerate(supplier_dict['Name']):
                item = QListWidgetItem()
                item.setText(name)
                if self.currently_searching:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable & ~Qt.ItemFlag.ItemIsEnabled & ~Qt.ItemFlag.ItemIsEditable)
                else:
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                self.list_of_names.addItem(item)
    
    def start_editing_text(self):
        self.start_editing_listitem_text = True
    
    def keep_name_before_changing(self, item: QListWidgetItem):
        self.curr_item_text = item.text()
        
    def change_name_of_dfs_to_new_name(self, item: QListWidgetItem):
        if self.start_editing_listitem_text and item.isSelected():
            new_text = item.text()
            if not new_text:
                item.setText(self.curr_item_text)
            elif new_text != self.curr_item_text:
                if new_text in self.supplier_df['Name'].to_list():
                    QMessageBox.critical(self, 'NameError', f'{new_text}, already exists!')
                    item.setText(self.curr_item_text)
                    return
                self.supplier_df[['Name']] = self.supplier_df[['Name']].replace(self.curr_item_text, new_text)
                self.name_smiles_df[['Name']] = self.name_smiles_df[['Name']].replace(self.curr_item_text, new_text)
                self.name_dbid_map[new_text] = self.name_dbid_map.pop(self.curr_item_text)
            self.start_editing_listitem_text = False
    
    def remove_text_from_dfs(self, name):
        kept_rows = self.name_smiles_df['Name'] != name
        self.supplier_df = self.supplier_df[kept_rows].reset_index(drop=True)
        self.name_smiles_df = self.name_smiles_df[kept_rows].reset_index(drop=True)
    
    def check_search_button(self, num):
        if num:
            self.search_button.setEnabled(True)
        else:
            self.search_button.setEnabled(False)
    
    def chemprop_table_set_prop_rows(self):
        self.chemprop_table.setRowCount(len(property_functions))
        for r, k in enumerate(property_functions):
            item = QTableWidgetItem()
            item.setText(k)
            self.chemprop_table.setItem(r, 0, item)
    
    def save_menu_setup(self):
        options = ['Current Similar Molecule', 'All Similar Molecules', 'Full Result']
        all_actions = []
        for name in options:
            curr_action = QAction(name, self)
            curr_png_action = QAction('Image', self)
            curr_catalog_table_action = QAction('Catalog Table', self)
            curr_smiles_action = QAction('SMILES', self)
            
            submenu = QMenu(self)
            submenu.addActions([curr_png_action,
                                curr_catalog_table_action,
                                curr_smiles_action])
            curr_action.setMenu(submenu)
            
            curr_png_action.triggered.connect(lambda _, n=name: self.save_to_png(n))
            curr_catalog_table_action.triggered.connect(lambda _, n=name: self.save_to_catalog_table(n))
            curr_smiles_action.triggered.connect(lambda _, n=name: self.save_to_smiles(n))
            
            all_actions.append(curr_action)
        
        save_menu = QMenu(self)
        save_menu.addActions(all_actions)
        self.save_files_button.setMenu(save_menu)
    
    def fill_list_with_names(self):
        names = self.name_smiles_df['Name'].to_list()
        self.list_of_names.addItems(names)
        default_brush = QListWidgetItem().foreground()
        for idx in range(len(names)):
            item = self.list_of_names.item(idx)
            f = item.font()
            f.setStrikeOut(False)
            item.setFont(f)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            item.setForeground(default_brush)
    
    def search_through_db(self):
        if not self.currently_searching:
            all_unique_smiles_list = list(set(self.name_smiles_df['SMILES']))
            default_brush = QListWidgetItem().foreground()
            for idx in range(len(self.name_smiles_df)):
                item = self.list_of_names.item(idx)
                f = item.font()
                f.setStrikeOut(False)
                item.setFont(f)
                item.setFlags(item.flags() & 
                              ~Qt.ItemFlag.ItemIsSelectable & 
                              ~Qt.ItemFlag.ItemIsEnabled & 
                              ~Qt.ItemFlag.ItemIsEditable)
                item.setForeground(default_brush)
            self.list_of_names.clearSelection()
            self.purchasability_table.clearContents()
            self.chemprop_table.clearContents()
            self.chemprop_table_set_prop_rows()
            self.database_id_combobox.clear()
            self.curr_name_img_label.set_image_smiles(self.empty_img, None)
            self.chem_image_label.set_image_smiles(self.empty_img, None)
            with open(os.path.join(os.path.dirname(__file__), 'fp_param', 'fp_param.json'), 'r') as f:
                fp_settings = json.load(f)
            self.fpgen = retrieve_fp_generator(fp_settings)
            self.sim_method_func = retrieve_similarity_method(fp_settings['sim'])
            
            db_pths_map = {}
            for row, (db_pth, ckbox) in enumerate(self.db_table.pth_record_dict.items()):
                if ckbox.isChecked():
                    if self.compare_db_and_curr_settings(db_pth, fp_settings):
                        template = self.db_table.item(row, 3).text()
                        db_name = self.db_table.item(row, 1).text()
                        if '{id}' not in template:
                            template = ''
                        db_pths_map[db_pth] = {'name'    : db_name,
                                               'template': template}
                    else:
                        ckbox.setChecked(False)
            
            self.currently_searching = True
            self.name_dbid_map = {}
            self.db_id_catalog_map = {}
            self.failed_or_non_smiles = []
            self.progress_bar.setValue(0)
            self.progress_bar.setMaximum(len(self.name_smiles_df.index))
            self.smiles_lineedit.clear()
            self.database_id_lineedit.clear()
            self.search_button.setText('Stop')
            self.thread = QThread()
            self.worker = MultiprocessDBSearch(db_pths_map,
                                               all_unique_smiles_list,
                                               round(self.similarity_spinbox.value() / 100, 2),
                                               fp_settings)
            self.worker.moveToThread(self.thread)
            self.worker.currSmilesResult.connect(self.update_progress_bar_and_df)
            self.worker.canceled.connect(self.cancel_search)
            self.worker.finished.connect(self.thread.quit)
            self.thread.finished.connect(self.searching_finished)
            self.thread.started.connect(self.worker.run)
            self.thread.start()
        else:
            self.worker.stop()
            self.search_button.setEnabled(False)
            self.search_button.setText('Stopping')
    
    def cancel_search(self):
        self.thread.quit()
        self.thread.wait()
        self.worker.deleteLater()
        self.thread.deleteLater()
        self.search_button.setText('Search')
        self.search_button.setEnabled(True)
        for i in range(self.list_of_names.count()):
            item = self.list_of_names.item(i)
            if item.flags() & ~Qt.ItemFlag.ItemIsEnabled:
                color = QColor()
                color.setRgb(255, 80, 80)
                item.setForeground(color)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable)
                item.setToolTip('Search Stopped')
        self.progress_bar.setValue(self.progress_bar.maximum())
    
    def update_progress_bar_and_df(self, result_dict: dict, smiles_str: str):
        smiles_series = self.name_smiles_df['SMILES']
        idxs = smiles_series[smiles_series == smiles_str].index
        if not result_dict:
            self.failed_or_non_smiles.extend(smiles_str)
            for idx in idxs:
                item = self.list_of_names.item(idx)
                f = item.font()
                f.setStrikeOut(True)
                item.setFont(f)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable)
                item.setToolTip('No match found')
        elif next(iter(result_dict)) == '':    # Failed to process input SMILES (shouldn't happen, just in case)
            self.failed_or_non_smiles.extend(smiles_str)
            for idx in idxs:
                item = self.list_of_names.item(idx)
                color = QColor()
                color.setRgb(255, 165, 0)
                item.setForeground(color)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable)
                item.setToolTip('Failed to read SMILES')
        else:
            for idx in idxs:
                item = self.list_of_names.item(idx)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable)
            self.parse_single_downloaded_dict(result_dict, smiles_str)
        self.progress_bar.setValue(self.progress_bar.value() + len(idxs))
        
    def parse_single_downloaded_dict(self, result_dict: dict, smiles_str: str):
        bool_row = self.name_smiles_df['SMILES'] == smiles_str
        names = self.name_smiles_df[bool_row]['Name'].to_list()
        
        for name_idx, name in enumerate(names):
            if name_idx == 0:
                similarity_list = []    # used for sorting
                first_name = name
                self.name_dbid_map[name] = {'db_ids': []}
                for id, vendor_dict in result_dict.items():
                    self.name_dbid_map[name]['db_ids'].append(id)
                    id_smiles = vendor_dict.pop('smiles')
                    similarity = vendor_dict.pop('similarity')
                    properties = vendor_dict.pop('properties')
                    similarity_list.append(similarity)
                    catalog_dict = vendor_dict
                    self.db_id_catalog_map[id] = {'smiles'    : id_smiles,
                                                  'catalogs'  : pd.DataFrame(catalog_dict),
                                                  'properties': properties,
                                                  'similarity': similarity}
                try:
                    exact_id_idx = next(idx for idx, id in enumerate(self.name_dbid_map[name]['db_ids']) if id.endswith(' (Exact)'))
                    exact_id = self.name_dbid_map[name]['db_ids'].pop(exact_id_idx)
                    similarity_list.pop(exact_id_idx)
                    sorted_similarity = [x for _, x in sorted(zip(similarity_list, self.name_dbid_map[name]['db_ids']), reverse=True)]
                    self.name_dbid_map[name]['db_ids'] = [exact_id] + sorted_similarity
                except:
                    sorted_similarity = [x for _, x in sorted(zip(similarity_list, self.name_dbid_map[name]['db_ids']), reverse=True)]
                    self.name_dbid_map[name]['db_ids'] = sorted_similarity
            else:
                self.name_dbid_map[name] = self.name_dbid_map[first_name]
                
    def searching_finished(self):
        self.currently_searching = False
        self.search_button.setEnabled(True)
        self.search_button.setText('Search')
        self.thread.wait()
        self.worker.deleteLater()
        self.thread.deleteLater()
    
    def update_combo_with_cid(self):
        items = self.list_of_names.selectedItems()
        if items:
            name = items[0].text()
            curr_smiles = self.name_smiles_df.loc[self.name_smiles_df['Name'] == name]['SMILES'].iloc[0]
            self.curr_name_img_label.set_image_smiles(self.draw_svg_bytes(Chem.MolFromSmiles(curr_smiles)), curr_smiles)
            if name not in self.name_dbid_map:
                self.database_id_combobox.clear()
                self.purchasability_table.clearContents()
                self.chemprop_table.clearContents()
                self.chemprop_table_set_prop_rows()
                self.database_id_lineedit.clear()
                self.smiles_lineedit.clear()
                return
            db_ids = self.name_dbid_map[name]['db_ids']
            self.database_id_combobox.clear()
            self.database_id_combobox.addItems(db_ids)
        else:
            self.curr_name_img_label.set_image_smiles(self.empty_img, None)
            self.purchasability_table.clearContents()
    
    def update_table_with_cid(self, db_id):
        if db_id:
            self.database_id_lineedit.setText(db_id.split(' (')[0].split('_', 1)[-1])
            smiles = self.db_id_catalog_map[db_id]['smiles']
            self.smiles_lineedit.setText(smiles)
            self.similarity_lineedit.setText(f'{self.db_id_catalog_map[db_id]['similarity']:.4f}')
            self.database_id_lineedit.setCursorPosition(0)
            self.smiles_lineedit.setCursorPosition(0)
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                self.chem_image_label.set_image_smiles(self.generate_failed_svg(), smiles)
            else:
                self.chem_image_label.set_image_smiles(self.draw_svg_bytes(mol), smiles)
            self.populate_catalog_table(self.db_id_catalog_map[db_id]['catalogs'])
            self.compare_query_and_target_properties()
        else:
            self.chem_image_label.set_image_smiles(self.empty_img, None)
            self.chemprop_table.clearContents()
            self.purchasability_table.clearContents()
            self.purchasability_table.setRowCount(0)
    
    def populate_catalog_table(self, catalog_df: pd.DataFrame):
        self.purchasability_table.setSortingEnabled(False)
        self.purchasability_table.clearContents()
        table_rows = len(catalog_df)
        self.purchasability_table.setRowCount(table_rows)
        for idx, row in catalog_df.iterrows():
            cat_name, cat_supplier_code, cat_url = row
            
            cat_name_item = QTableWidgetItem()
            cat_name_item.setText(cat_name)
            cat_name_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            cat_name_item.setFlags(cat_name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if cat_url is not None:
                cat_supplier_label = QLabel(f'<a href="{cat_url}">{cat_supplier_code}</a>')
                cat_supplier_label.setOpenExternalLinks(False)
                cat_supplier_label.linkActivated.connect(self.show_custom_webbrowser)
            else:
                cat_supplier_label = QLabel(str(cat_supplier_code))
            cat_supplier_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            self.purchasability_table.setItem(idx, 0, cat_name_item)
            self.purchasability_table.setCellWidget(idx, 1, cat_supplier_label)
        self.purchasability_table.resizeColumnsToContents()
        header = self.purchasability_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.purchasability_table.setSortingEnabled(True)
        
    def compare_query_and_target_properties(self):
        items = self.list_of_names.selectedItems()
        if not items:
            return
        query_name = items[0].text()
        query_series = self.supplier_df[self.supplier_df['Name'] == query_name]
        target_chemprop_dict = self.db_id_catalog_map[self.database_id_combobox.currentText()]['properties']
        self.chemprop_table.clearContents()
        for r, k in enumerate(property_functions):
            query_value = query_series.get(k).to_list()[0]
            target_value = target_chemprop_dict[k]
            
            color = QTableWidgetItem().foreground()
            if target_value - query_value >= 1e-3:
                color = QColor()
                color.setRgb(229, 115, 115)
            elif query_value - target_value >= 1e-3:
                color = QColor()
                color.setRgb(76, 175, 80)
                
            prop_item = QTableWidgetItem()
            query_item = QTableWidgetItem()
            target_item = QTableWidgetItem()
            flags = prop_item.flags()
            
            prop_item.setText(k)
            if k in ['Molecular Weight', 'LogP', 'Topological Polar Surface Area', 'Molar Refractivity', 'QED']:
                query_item.setText(f'{query_value:.3f}')
                target_item.setText(f'{target_value:.3f}')
            else:
                query_item.setText(f'{int(query_value)}')
                target_item.setText(f'{int(target_value)}')
            target_item.setForeground(color)
            prop_item.setFlags(flags & ~Qt.ItemFlag.ItemIsEditable)
            query_item.setFlags(flags & ~Qt.ItemFlag.ItemIsEditable)
            target_item.setFlags(flags & ~Qt.ItemFlag.ItemIsEditable)
            
            self.chemprop_table.setItem(r, 0, prop_item)
            self.chemprop_table.setItem(r, 1, query_item)
            self.chemprop_table.setItem(r, 2, target_item)
    
    def show_custom_webbrowser(self, url):
        if self.browser is None:
            self.browser = BrowserWithTabs(self, url)
            self.browser.closed.connect(lambda: setattr(self, 'browser', None))
        else:
            self.browser.tab_browser.add_new_tab(url)
            self.browser.raise_()
            self.browser.activateWindow()
    
    def save_to_png(self, name):
        if name == 'Current Similar Molecule':
            current_id = self.database_id_combobox.currentText()
            if current_id:
                current_molecule_name = self.list_of_names.selectedItems()[0].text()
                dir = QFileDialog.getExistingDirectory(self, 'Select Save Directory', '')
                if dir:
                    new_dir = os.path.join(dir, current_molecule_name)
                    os.makedirs(new_dir, exist_ok=True)
                    similarity = f'_{self.db_id_catalog_map[current_id]['similarity']:.4f}'
                    curr_dir = os.path.join(new_dir, current_id.replace(' ', '_') + similarity)
                    os.makedirs(curr_dir, exist_ok=True)
                    curr_molecule_pth = os.path.join(curr_dir, current_id.replace(' ', '_') + '.png')  # replace the space in " (Exact)"
                    mol = Chem.MolFromSmiles(self.db_id_catalog_map[current_id]['smiles'])
                    svgtopng(self.draw_svg_bytes(mol), curr_molecule_pth)
        elif name == 'All Similar Molecules':
            current_items = self.list_of_names.selectedItems()
            if current_items:
                current_molecule_name = current_items[0].text()
                if current_molecule_name in self.name_dbid_map:
                    dir = QFileDialog.getExistingDirectory(self, 'Select Save Directory', '')
                    if dir:
                        pubchem_id_list = self.name_dbid_map[current_molecule_name]['db_ids']
                        new_dir = os.path.join(dir, current_molecule_name)
                        os.makedirs(new_dir, exist_ok=True)
                        for current_id in pubchem_id_list:
                            similarity = f'_{self.db_id_catalog_map[current_id]['similarity']:.4f}'
                            curr_dir = os.path.join(new_dir, current_id.replace(' ', '_') + similarity)
                            os.makedirs(curr_dir, exist_ok=True)
                            curr_molecule_pth = os.path.join(curr_dir, current_id.replace(' ', '_') + '.png')
                            mol = Chem.MolFromSmiles(self.db_id_catalog_map[current_id]['smiles'])
                            svgtopng(self.draw_svg_bytes(mol), curr_molecule_pth)
        else:
            num_of_items = self.list_of_names.count()
            dir = QFileDialog.getExistingDirectory(self, 'Select Save Directory', '')
            if dir:
                for r in range(num_of_items):
                    item = self.list_of_names.item(r)
                    current_molecule_name = item.text()
                    new_dir = os.path.join(dir, current_molecule_name)
                    os.makedirs(new_dir, exist_ok=True)
                    if current_molecule_name in self.name_dbid_map:
                        pubchem_id_list = self.name_dbid_map[current_molecule_name]['db_ids']
                        for current_id in pubchem_id_list:
                            similarity = f'_{self.db_id_catalog_map[current_id]['similarity']:.4f}'
                            curr_dir = os.path.join(new_dir, current_id.replace(' ', '_') + similarity)
                            os.makedirs(curr_dir, exist_ok=True)
                            curr_molecule_pth = os.path.join(curr_dir, current_id.replace(' ', '_') +'.png')
                            mol = Chem.MolFromSmiles(self.db_id_catalog_map[current_id]['smiles'])
                            svgtopng(self.draw_svg_bytes(mol), curr_molecule_pth)
        
    def save_to_catalog_table(self, name):
        if name == 'Current Similar Molecule':
            current_id = self.database_id_combobox.currentText()
            if current_id:
                current_molecule_name = self.list_of_names.selectedItems()[0].text()
                dir = QFileDialog.getExistingDirectory(self, 'Select Save Directory', '')
                if dir:
                    new_dir = os.path.join(dir, current_molecule_name)
                    os.makedirs(new_dir, exist_ok=True)
                    similarity = f'_{self.db_id_catalog_map[current_id]['similarity']:.4f}'
                    curr_dir = os.path.join(new_dir, current_id.replace(' ', '_') + similarity)
                    os.makedirs(curr_dir, exist_ok=True)
                    curr_molecule_pth = os.path.join(curr_dir, current_id.replace(' ', '_') + '.csv')
                    df: pd.DataFrame = self.db_id_catalog_map[current_id]['catalogs']
                    df.to_csv(curr_molecule_pth, index=None)
        elif name == 'All Similar Molecules':
            current_items = self.list_of_names.selectedItems()
            if current_items:
                current_molecule_name = current_items[0].text()
                if current_molecule_name in self.name_dbid_map:
                    dir = QFileDialog.getExistingDirectory(self, 'Select Save Directory', '')
                    if dir:
                        pubchem_id_list = self.name_dbid_map[current_molecule_name]['db_ids']
                        new_dir = os.path.join(dir, current_molecule_name)
                        os.makedirs(new_dir, exist_ok=True)
                        for current_id in pubchem_id_list:
                            similarity = f'_{self.db_id_catalog_map[current_id]['similarity']:.4f}'
                            curr_dir = os.path.join(new_dir, current_id.replace(' ', '_') + similarity)
                            os.makedirs(curr_dir, exist_ok=True)
                            curr_molecule_pth = os.path.join(curr_dir, current_id.replace(' ', '_') + '.csv')
                            df: pd.DataFrame = self.db_id_catalog_map[current_id]['catalogs']
                            df.to_csv(curr_molecule_pth, index=None)
        else:
            num_of_items = self.list_of_names.count()
            dir = QFileDialog.getExistingDirectory(self, 'Select Save Directory', '')
            if dir:
                for r in range(num_of_items):
                    item = self.list_of_names.item(r)
                    current_molecule_name = item.text()
                    new_dir = os.path.join(dir, current_molecule_name)
                    os.makedirs(new_dir, exist_ok=True)
                    if current_molecule_name in self.name_dbid_map:
                        pubchem_id_list = self.name_dbid_map[current_molecule_name]['db_ids']
                        for current_id in pubchem_id_list:
                            similarity = f'_{self.db_id_catalog_map[current_id]['similarity']:.4f}'
                            curr_dir = os.path.join(new_dir, current_id.replace(' ', '_') + similarity)
                            os.makedirs(curr_dir, exist_ok=True)
                            curr_molecule_pth = os.path.join(curr_dir, current_id.replace(' ', '_') + '.csv')
                            df: pd.DataFrame = self.db_id_catalog_map[current_id]['catalogs']
                            if not df.empty:
                                df.to_csv(curr_molecule_pth, index=None)
                            
    def save_to_smiles(self, name):
        if name == 'Current Similar Molecule':
            current_id = self.database_id_combobox.currentText()
            if current_id:
                current_molecule_name = self.list_of_names.selectedItems()[0].text()
                dir = QFileDialog.getExistingDirectory(self, 'Select Save Directory', '')
                if dir:
                    new_dir = os.path.join(dir, current_molecule_name)
                    os.makedirs(new_dir, exist_ok=True)
                    similarity = f'_{self.db_id_catalog_map[current_id]['similarity']:.4f}'
                    curr_dir = os.path.join(new_dir, current_id.replace(' ', '_') + similarity)
                    os.makedirs(curr_dir, exist_ok=True)
                    curr_molecule_pth = os.path.join(curr_dir, current_id.replace(' ', '_') + '.smi')
                    smiles = self.db_id_catalog_map[current_id]['smiles']
                    with open(curr_molecule_pth, 'w') as f:
                        f.write(f'{smiles} {current_id}')
        elif name == 'All Similar Molecules':
            current_items = self.list_of_names.selectedItems()
            if current_items:
                current_molecule_name = current_items[0].text()
                dir = QFileDialog.getExistingDirectory(self, 'Select Save Directory', '')
                if dir:
                    zinc_id_list = self.name_dbid_map[current_molecule_name]['db_ids']
                    new_dir = os.path.join(dir, current_molecule_name)
                    os.makedirs(new_dir, exist_ok=True)
                    for current_id in zinc_id_list:
                        similarity = f'_{self.db_id_catalog_map[current_id]['similarity']:.4f}'
                        curr_dir = os.path.join(new_dir, current_id.replace(' ', '_') + similarity)
                        os.makedirs(curr_dir, exist_ok=True)
                        curr_molecule_pth = os.path.join(curr_dir, current_id.replace(' ', '_') + '.smi')
                        smiles = self.db_id_catalog_map[current_id]['smiles']
                        with open(curr_molecule_pth, 'w') as f:
                            f.write(f'{smiles} {current_id}')
        else:
            num_of_items = self.list_of_names.count()
            dir = QFileDialog.getExistingDirectory(self, 'Select Save Directory', '')
            if dir:
                for r in range(num_of_items):
                    item = self.list_of_names.item(r)
                    current_molecule_name = item.text()
                    new_dir = os.path.join(dir, current_molecule_name)
                    os.makedirs(new_dir, exist_ok=True)
                    if current_molecule_name in self.name_dbid_map:
                        zinc_id_list = self.name_dbid_map[current_molecule_name]['db_ids']
                        for current_id in zinc_id_list:
                            similarity = f'_{self.db_id_catalog_map[current_id]['similarity']:.4f}'
                            curr_dir = os.path.join(new_dir, current_id.replace(' ', '_') + similarity)
                            os.makedirs(curr_dir, exist_ok=True)
                            curr_molecule_pth = os.path.join(curr_dir, current_id.replace(' ', '_') + '.smi')
                            smiles = self.db_id_catalog_map[current_id]['smiles']
                            with open(curr_molecule_pth, 'w') as f:
                                f.write(f'{smiles} {current_id.split(' ')[0]}')
    
    def change_dark_light_mode(self, display_mode: str):
        self.display_mode = display_mode
        items = self.list_of_names.selectedItems()
        if items:
            name = items[0].text()
            curr_smiles = self.name_smiles_df.loc[self.name_smiles_df['Name'] == name]['SMILES'].iloc[0]
            self.curr_name_img_label.set_image_smiles(self.draw_svg_bytes(Chem.MolFromSmiles(curr_smiles)), curr_smiles)
        pubchem_id = self.database_id_combobox.currentText()
        if pubchem_id:
            smiles = self.db_id_catalog_map[pubchem_id]['smiles']
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                self.chem_image_label.set_image_smiles(self.generate_failed_svg(), smiles)
            else:
                self.chem_image_label.set_image_smiles(self.draw_svg_bytes(mol), smiles)
                
    def clear_all(self):
        self.list_of_names.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(1)
        self.purchasability_table.clearContents()
        self.chemprop_table.clearContents()
        self.chemprop_table_set_prop_rows()
        self.database_id_combobox.clear()
        self.curr_name_img_label.set_image_smiles(self.empty_img, None)
        self.chem_image_label.set_image_smiles(self.empty_img, None)
        supplier_dict = {'Name': [], 'SMILES': []}
        supplier_dict.update({prop_name: [] for prop_name in property_functions})
        self.supplier_df = pd.DataFrame(supplier_dict)
        self.name_smiles_df = pd.DataFrame({'Name': [], 'SMILES': []})
        self.name_dbid_map = {}
        self.db_id_catalog_map = {}
        self.failed_or_non_smiles = []
        self.smiles_lineedit.clear()
        self.database_id_lineedit.clear()

class _FileNameTree(QTreeView): # Depracated, slower
    checkedSignal = Signal(str, str)
    uncheckedSignal = Signal(str, str)
    contactSignal = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):
        self.model = QStandardItemModel()
        self.model.setColumnCount(4)
        self.model.setHorizontalHeaderLabels(['Name', 'Score', 'RMSD L.B.', 'RMSD U.B.'])
        self.setModel(self.model)
        
        self.setEditTriggers(QTreeView.NoEditTriggers)
        self.setHeaderHidden(False)
        self.setSelectionMode(QTreeView.SingleSelection)
        self.setUniformRowHeights(True)
        
        self.model.itemChanged.connect(self.check_status)
        self.clicked.connect(self.update_contact_information)
        self.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.header().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        
    def populate_tree(self, name_energy_dict: dict):
        self.model.itemChanged.disconnect(self.check_status)
        self.model.beginResetModel()
        
        self.model.clear()
        self.model.setHorizontalHeaderLabels(['Name', 'Score', 'RMSD L.B.', 'RMSD U.B.'])
        for name, subs in name_energy_dict.items():
            parent_item = QStandardItem(name)
            parent_item.setFlags(parent_item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            parent_item.setData(Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
            
            parent_energy = QStandardItem('')
            parent_rmsd_lb = QStandardItem('')
            parent_rmsd_ub = QStandardItem('')
            self.model.appendRow([parent_item, parent_energy, parent_rmsd_lb, parent_rmsd_ub])
            
            for i, eng_lb_ub in enumerate(subs, 1):
                child_name = QStandardItem(f'#{i}')
                child_name.setFlags(child_name.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                child_name.setData(Qt.Unchecked, Qt.CheckStateRole)
                
                child_energy = QStandardItem(str(eng_lb_ub[0]))
                child_rmsd_lb = QStandardItem(str(eng_lb_ub[1]))
                child_rmsd_ub = QStandardItem(str(eng_lb_ub[2]))
                
                parent_item.appendRow([child_name, child_energy, child_rmsd_lb, child_rmsd_ub])
        
        self.model.endResetModel()
        self.model.itemChanged.connect(self.check_status)
        self.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.header().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        
    def check_status(self, item: QStandardItem):
        if item.parent() is None:
            state = item.checkState()
            for row in range(item.rowCount()):
                child = item.child(row, 0)
                if child.checkState() != state:
                    child.setCheckState(state)
            self.contactSignal.emit('')
        else:
            parent = item.parent()
            if item.checkState() == Qt.Checked:
                self.checkedSignal.emit(parent.text(), item.text())
            else:
                self.uncheckedSignal.emit(parent.text(), item.text())
                self.contactSignal.emit('')
            parent.setCheckState(self.count_parent_checked(parent))
            
    def update_contact_information(self, index: QModelIndex):
        item = self.model.itemFromIndex(index)
        if item and item.parent():
            contact_info = f'{item.parent().text()} {item.text()}'
            self.contactSignal.emit(contact_info)
        else:
            self.contactSignal.emit('')
            
    def count_parent_checked(self, parent_item: QStandardItem):
        checked = 0
        total = parent_item.rowCount()
        
        for row in range(total):
            child = parent_item.child(row, 0)
            if child.checkState() == Qt.CheckState.Checked:
                checked += 1
            elif child.checkState() == Qt.CheckState.PartiallyChecked:
                return Qt.CheckState.PartiallyChecked
            
        if checked == 0:
            return Qt.CheckState.Unchecked
        elif checked == total:
            return Qt.CheckState.Checked
        else:
            return Qt.CheckState.PartiallyChecked
        
    def clear_everything(self):
        self.model.clear()
        self.model.setHorizontalHeaderLabels(['Name', 'Score', 'RMSD L.B.', 'RMSD U.B.'])

class FileNameTree(QTreeWidget):
    checkedSignal = Signal(str, str, bool)
    uncheckedSignal = Signal(str, str)
    contactSignal = Signal(str)
    
    def __init__(self, parent):
        super().__init__(parent)
        self.initUI()
        self.is_parent_checked = False
        self.selected_list = []
    
    def initUI(self):
        self.setColumnCount(4)
        self.setHeaderLabels(['Name', 'Score', 'RMSD L.B.', 'RMSD U.B.'])
        for i in range(self.columnCount()):
            self.resizeColumnToContents(i)
        # self.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.itemChanged.connect(self.check_status)
        self.itemClicked.connect(self.update_contact_information)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    
    def populate_tree(self, name_energy_dict: dict):
        self.itemChanged.disconnect(self.check_status)
        self.itemClicked.disconnect(self.update_contact_information)
        self.clear()
        self.selected_list = []
        for name, subs in name_energy_dict.items():
            item = QTreeWidgetItem(self)
            item.setText(0, name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Unchecked)
            for i, eng_lb_ub in enumerate(subs, 1):
                sub_item = QTreeWidgetItem(item)
                sub_item.setText(0, f'#{i}')
                sub_item.setText(1, str(eng_lb_ub[0]))
                sub_item.setText(2, str(eng_lb_ub[1]))
                sub_item.setText(3, str(eng_lb_ub[2]))
                sub_item.setFlags(sub_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                sub_item.setCheckState(0, Qt.CheckState.Unchecked)
        self.itemChanged.connect(self.check_status)
        self.itemClicked.connect(self.update_contact_information)
        for i in range(self.columnCount()):
            self.resizeColumnToContents(i)
        self.setColumnWidth(1, self.columnWidth(1) + 5)
    
    def check_status(self, item: QTreeWidgetItem):
        if item.parent() is None:
            self.is_parent_checked = True
            ck_state = item.checkState(0)
            for i in range(item.childCount()-1, 0, -1):
                item.child(i).setCheckState(0, ck_state)
            self.is_parent_checked = False
            item.child(0).setCheckState(0, ck_state)
            self.clearSelection()
            item.child(0).setSelected(True)
            if ck_state == Qt.CheckState.Checked:
                self.expandItem(item)
            else:
                self.collapseItem(item)
        else:
            if item.checkState(0) == Qt.CheckState.Checked:
                self.checkedSignal.emit(f'{item.parent().text(0)}', f'{item.text(0)}', self.is_parent_checked)
                self.selected_list.append(item)
            else:
                self.uncheckedSignal.emit(f'{item.parent().text(0)}', f'{item.text(0)}')
                self.selected_list.remove(item)
                if self.selected_list:
                    prev_item = self.selected_list[-1]
                    self.contactSignal.emit(f'{prev_item.parent().text(0)} {prev_item.text(0)}')
                else:
                    self.contactSignal.emit('')
            self.blockSignals(True)
            item.parent().setCheckState(0, self.count_parent_checked(item.parent()))
            self.blockSignals(False)
            
    def update_contact_information(self, item: QTreeWidgetItem):
        if item.parent() is not None and item.checkState(0):
            self.contactSignal.emit(f'{item.parent().text(0)} {item.text(0)}')
    
    def count_parent_checked(self, parent_item: QTreeWidgetItem):
        child_cnt = parent_item.childCount()
        summed = sum(1 for i in range(child_cnt) if parent_item.child(i).checkState(0) == Qt.CheckState.Checked)
        if not summed:
            return Qt.CheckState.Unchecked
        elif summed == child_cnt:
            return Qt.CheckState.Checked
        return Qt.CheckState.PartiallyChecked
    
    def clear_everything(self):
        self.selected_list = []
        self.clear()

class TabBar(QTabBar):
    # Copied from https://stackoverflow.com/a/51230694
    def tabSizeHint(self, index):
        s = QTabBar.tabSizeHint(self, index)
        s.transpose()
        return s

    def paintEvent(self, event):
        painter = QStylePainter(self)
        opt = QStyleOptionTab()
        
        for i in range(self.count()):
            self.initStyleOption(opt, i)
            painter.drawControl(QStyle.CE_TabBarTabShape, opt)
            painter.save()
            
            s = opt.rect.size()
            s.transpose()
            r = QRect(QPoint(), s)
            r.moveCenter(opt.rect.center())
            opt.rect = r
            
            c = self.tabRect(i).center()
            painter.translate(c)
            painter.rotate(90)
            painter.translate(-c)
            painter.drawControl(QStyle.CE_TabBarTabLabel, opt);
            painter.restore()

class ContactTabTables(QToolBox):
    def __init__(self, parent):
        super().__init__(parent)
        self.all_tables_dict = {}
        
        for name in ['Hydrogen bond', 'Hydrophobic contact', 'Halogen bond',
                     'Ionic interaction', 'Cation-pi interaction', 'Pi-pi stacking']:
            table = ContactTable(self)
            self.all_tables_dict[name] = table
            self.addItem(table, name)
        
        btns = self.findChildren(QAbstractButton)
        self.btns = [btn for btn in btns if btn.metaObject().className() == "QToolBoxButton"]
        for btn in self.btns:
            btn.setStyleSheet('color: #E57373')
    
    def clear_tables(self):
        for tables in self.all_tables_dict.values():
            tables.setRowCount(0)
        for btn in self.btns:
            btn.setStyleSheet('color: #E57373')
            
    def change_button_color(self):
        for idx, tables in enumerate(self.all_tables_dict.values()):
            if tables.rowCount():
                self.btns[idx].setStyleSheet('color: #4CAF50')

class ContactTable(QTableWidget):
    def __init__(self, parent):
        super().__init__(parent)
        small_font = QFont()
        small_font.setPointSize(12)
        self.setFont(small_font)
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(['Protein', 'Ligand', 'Dist.'])
        h = self.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            
    def add_contact_to_table(self, p_contact: str, l_contact: str, dist: str):
        row_position = self.rowCount()
        self.insertRow(row_position)
        p_item = QTableWidgetItem(p_contact)
        l_item = QTableWidgetItem(l_contact)
        d_item = QTableWidgetItem(dist)
        flags = p_item.flags()
        p_item.setFlags(flags & ~Qt.ItemFlag.ItemIsEditable)
        l_item.setFlags(flags & ~Qt.ItemFlag.ItemIsEditable)
        d_item.setFlags(flags & ~Qt.ItemFlag.ItemIsEditable)
        p_item.setToolTip(p_contact)
        l_item.setToolTip(l_contact)
        d_item.setToolTip(dist)
        self.setItem(row_position, 0, p_item)
        self.setItem(row_position, 1, l_item)
        self.setItem(row_position, 2, d_item)
    
    def resizeEvent(self, event):
        h = self.horizontalHeader()
        for i in range(h.count() - 1):
            curr_size = h.sectionSize(i)
            h.resizeSection(i, curr_size)
        h.setSectionResizeMode(i + 1, QHeaderView.ResizeMode.Stretch)
        super().resizeEvent(event)

class SlidingStackedWidget(QStackedWidget):
    # Modified from https://stackoverflow.com/a/52597972, thanks!
    def __init__(self, parent=None, direction=Qt.Orientation.Vertical, speed: int=400):
        super().__init__(parent)
        
        self.m_direction = direction
        self.m_speed = speed
        self.m_animationtype = QEasingCurve.Type.OutCubic
        self.m_now = 0
        self.m_next = 0
        self.m_pnow = QPoint(0, 0)
        self.m_active = False
        
    @Slot()
    def moveToIndex(self, target_index: int):
        self.slideInWgt(self.widget(target_index))
        
    def slideInWgt(self, newwidget):
        if self.m_active:
            return
        self.m_active = True
        
        _now = self.currentIndex()
        _next = self.indexOf(newwidget)
        
        if _now == _next:
            self.m_active = False
            return
        
        offsetx, offsety = self.frameRect().width(), self.frameRect().height()
        self.widget(_next).setGeometry(self.frameRect())
        
        if not self.m_direction == Qt.Orientation.Horizontal:
            if _now < _next:
                offsetx, offsety = 0, -offsety
            else:
                offsetx = 0
        else:
            if _now < _next:
                offsetx, offsety = -offsetx, 0
            else:
                offsety = 0
                
        pnext = self.widget(_next).pos()
        pnow = self.widget(_now).pos()
        self.m_pnow = pnow
        
        offset = QPoint(offsetx, offsety)
        self.widget(_next).move(pnext - offset)
        self.widget(_next).show()
        self.widget(_next).raise_()
        
        anim_group = QParallelAnimationGroup(
            self, finished=self.animationDoneSlot
        )
        
        for index, start, end in zip(
            (_now, _next), (pnow, pnext - offset), (pnow + offset, pnext)
        ):
            animation = QPropertyAnimation(
                self.widget(index),
                b"pos",
                duration=self.m_speed,
                easingCurve=self.m_animationtype,
                startValue=start,
                endValue=end,
            )
            anim_group.addAnimation(animation)
            
        self.m_next = _next
        self.m_now = _now
        self.m_active = True
        anim_group.start(QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)
        
    @Slot()
    def animationDoneSlot(self):
        self.setCurrentIndex(self.m_next)
        self.widget(self.m_now).hide()
        self.widget(self.m_now).move(self.m_pnow)
        self.m_active = False

property_functions = {'Molecular Weight'        : Descriptors.MolWt,
                      'Hydrogen Bond Donors'    : Descriptors.NumHDonors,
                      'Hydrogen Bond Acceptors' : Descriptors.NumHAcceptors,
                      'LogP'                    : Descriptors.MolLogP,
                      'Topological Polar Surface Area': Descriptors.TPSA,
                      'Rotatable Bonds'         : Descriptors.NumRotatableBonds,
                      'Number of Rings'         : Descriptors.RingCount,
                      'Formal Charge'           : lambda mol: sum([atom.GetFormalCharge() for atom in mol.GetAtoms()]),
                      'Number of Heavy Atoms'   : Descriptors.HeavyAtomCount,
                      'Molar Refractivity'      : Descriptors.MolMR,
                      'Number of Atoms'         : lambda mol: mol.GetNumAtoms(),
                      'QED'                     : Descriptors.qed}

class ThreadedProteinDownloader(QObject):
    proteinDownloadStatus = Signal(str, str, str, bool)
    conversionString = Signal(str, bool)
    proteinString = Signal(str, str, dict)
    finished = Signal()
    
    def __init__(self, id: str, database: str, is_pdbqt: bool, fill_gap: bool, ph: float):
        super().__init__()
        self.db_map = {'PDB'        : 'https://files.rcsb.org/download/{id}.pdb',
                       'AF Database': 'https://alphafold.ebi.ac.uk/files/AF-{id}-F1-model_v4.pdb'}
        self.regex_title = {'PDB'        : r'TITLE\s+(.*)',
                            'AF Database': r'TITLE.{5}\s?(.*)',
                            'PDB_CIF'    : r'_citation.title\s+(.*)'}
        self.id = id
        self.url = self.db_map[database].format(id=id)
        self.db_type = database
        self.is_pdbqt = is_pdbqt
        self.fill_gap = fill_gap
        self.ph = ph
        
    def run(self):
        def protein_downloader():
            try:
                cif = False
                r = requests.get(self.url)
                if r.status_code == 404:
                    cif = True
                    r = requests.get(self.url.rsplit('.', 1)[0] + '.cif')  # try CIF
                if r.status_code == 200:
                    pdb_text = r.text
                    if not cif:
                        title = ''.join([s.strip() for s in re.findall(self.regex_title[self.db_type], pdb_text)])
                    else:
                        title = ''.join([s.strip() for s in re.findall(self.regex_title['PDB_CIF'], pdb_text)])
                    self.proteinDownloadStatus.emit(f'Status : {self.id} retrieved.', self.db_type, title, True)
                    pdb_text, hetatm_dict = clean_pdb(pdb_text, True, self.fill_gap, 'cif' if cif else 'pdb')
                    if not pdb_text:
                        text = f'{self.id} contains no protein!'
                        self.conversionString.emit(text, False)
                        self.proteinString.emit('Failed', self.id, {})
                        self.finished.emit()
                        return
                    with tempfile.NamedTemporaryFile('w+', delete=False) as tmp_f:
                        tmp_f.write(pdb_text)
                        pdb_file = tmp_f.name
                    if self.is_pdbqt:
                        protein_string = fix_and_convert(pdb_file, fill_gap=self.fill_gap, ph=self.ph)
                        if isinstance(protein_string, tuple):
                            text = f'Failed to convert .pdb to .pdbqt :\n{protein_string[0]}'
                            self.conversionString.emit(text, False)
                            self.proteinString.emit('Failed', self.id, {})
                        else:
                            text = 'Protein structure loaded. (PDBQT Format)'
                            self.conversionString.emit(text, True)
                            self.proteinString.emit(protein_string, self.id, hetatm_dict)
                    else:
                        protein_string = fix_pdb_missing_atoms(pdb_file, fill_gap=self.fill_gap, ph=self.ph)
                        text = 'Protein structure loaded. (PDB Format)'
                        self.conversionString.emit(text, True)
                        self.proteinString.emit(protein_string, self.id, hetatm_dict)
                    os.remove(pdb_file)
                else:
                    self.proteinDownloadStatus.emit(f'Status : Failed to retrieve {self.id}. Status Code: {r.status_code}.', self.db_type, '', False)
                    self.conversionString.emit(f'Failed to retrieve protein from {self.db_type}.', False)
                    self.proteinString.emit('Failed', self.id, {})
            except requests.ConnectionError as _:
                self.proteinDownloadStatus.emit(f'Status : Failed to retrieve {self.id}. Connection Error.', self.db_type, '', False)
                self.conversionString.emit(f'Failed to retrieve protein from {self.db_type}.', False) 
                self.proteinString.emit('Failed', self.id, {})
            self.finished.emit()
            return
        protein_downloader()

class ThreadedLocalPDBLoader(QObject):
    conversionString = Signal(str, bool)
    proteinString = Signal(str, str, dict)
    finished = Signal()
    
    def __init__(self, local_pth: str, is_pdbqt: bool, fill_gap: bool, ph: float):
        super().__init__()
        self.id = os.path.basename(local_pth).rsplit('.', 1)[0]
        self.is_pdbqt = is_pdbqt
        self.local_pth = local_pth
        self.fill_gap = fill_gap
        self.ph = ph
        
    def run(self):
        def protein_loader():
            format = 'cif' if self.local_pth.endswith('.cif') else 'pdb'
            with open(self.local_pth) as f:
                pdb_text = f.read()
            pdb_text, hetatm_dict = clean_pdb(pdb_text, True, self.fill_gap, format)
            if not pdb_text:
                text = f'"{self.local_pth}" contains no protein!'
                self.conversionString.emit(text, False)
                self.proteinString.emit('Failed', self.id, {})
                self.finished.emit()
                return
            with tempfile.NamedTemporaryFile('w+', delete=False, suffix='.'+format) as tmp_f:
                tmp_f.write(pdb_text)
                pdb_file = tmp_f.name
            if self.is_pdbqt:
                protein_string = fix_and_convert(pdb_file, fill_gap=self.fill_gap, ph=self.ph)
                if isinstance(protein_string, tuple):
                    text = f'Failed to convert .pdb to .pdbqt :\n{protein_string[0]}'
                    self.conversionString.emit(text, False)
                    self.proteinString.emit('Failed', self.id, {})
                else:
                    text = 'Protein structure loaded. (PDBQT Format)'
                    self.conversionString.emit(text, True)
                    self.proteinString.emit(protein_string, self.id, hetatm_dict)
            else:
                protein_string = fix_pdb_missing_atoms(pdb_file, fill_gap=self.fill_gap, ph=self.ph)
                text = 'Protein structure loaded. (PDB Format)'
                self.conversionString.emit(text, True)
                self.proteinString.emit(protein_string, self.id, hetatm_dict)
            os.remove(pdb_file)
            self.finished.emit()
            return
        protein_loader()

class ThreadedPDBConverter(QObject):
    proteinString = Signal(str, dict)
    finished = Signal()
    
    def __init__(self, pdb_str: str, display_flex_dict: dict, ph: float):
        super().__init__()
        with tempfile.NamedTemporaryFile('w+', delete=False) as f:
            f.write(pdb_str)
            file = f.name
        self.local_pth = file
        self.d = display_flex_dict
        self.ph = ph
        
    def run(self):
        def protein_loader():
            protein_string = fix_and_convert(self.local_pth, ph=self.ph)
            os.remove(self.local_pth)
            self.proteinString.emit(protein_string, self.d)
            self.finished.emit()
        protein_loader()

class BoundedThreadPoolExecutor(ThreadPoolExecutor):
    ### copied from https://stackoverflow.com/a/78071937
    def __init__(self, max_size=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._semaphore = Semaphore(max_size)
        
    def submit(self, *args, **kwargs):
        self._semaphore.acquire()
        future = super().submit(*args, **kwargs)
        future.add_done_callback(lambda _: self._semaphore.release())
        return future

def retrieve_energy_and_string_from_unidock_sdf(name: str, ligand_sdf_str: str, protein_data: list, complex_cache: str):
    mdlname_pdbqtcombiner_map = {}
    for compiled_regex in sdf_regex_list:
        if re.search(compiled_regex, ligand_sdf_str) is not None:
            energy_dict = {name: [re.findall(sdf_match_eng_rmsd_compiled, s) 
                                  for s in re.findall(compiled_regex, ligand_sdf_str)]}
            break
    supp = Chem.SDMolSupplier()
    supp.SetData(ligand_sdf_str, sanitize=False)
    for i, mol in enumerate(supp, start=1):
        Chem.RemoveHs(mol)
        mdl_str = Chem.MolToPDBBlock(mol)
        mdl_name = f'{name} #{i}'
        pdbqt_combiner = PDBQTCombiner(mdl_name)
        pdbqt_combiner.cache_pth = complex_cache
        conn = sqlite3.connect(complex_cache)
        cur = conn.cursor()
        cur.execute(f"""SELECT EXISTS(SELECT 1 FROM ProLigDB WHERE name = '{mdl_name}')""")
        if not cur.fetchone()[0]:
            protein_delta_data, ligand_str = pdbqt_combiner.process_strings(protein_data, mdl_str)
            protein_compressed = sqlite3.Binary(lzma.compress(pickle.dumps(protein_delta_data)))
            ligand_compressed = sqlite3.Binary(lzma.compress(ligand_str.encode('utf-8')))
            cur.execute("""
                        INSERT INTO ProLigDB (name, protein, ligand) 
                        VALUES (?, ?, ?)
                        """, (mdl_name, protein_compressed, ligand_compressed))
            conn.commit()
        mdlname_pdbqtcombiner_map[mdl_name] = pdbqt_combiner
    return energy_dict, mdlname_pdbqtcombiner_map

def retrieve_energy_and_string_from_smina_sdf(name: str, ligand_sdf_str: str, protein_data: list, complex_cache: str):
    mdlname_pdbqtcombiner_map = {}
    energy_list = []
    for g in re.finditer(smina_eng_compiled, ligand_sdf_str):
        eng = g.group(1)
        energy_list.append([eng, 0., 0.,])
    energy_dict = {name: energy_list}
    supp = Chem.SDMolSupplier()
    supp.SetData(ligand_sdf_str)
    for i, mol in enumerate(supp, start=1):
        Chem.RemoveHs(mol)
        flex_pdb = None
        if mol.HasProp('Flex Sidechains PDB'):
            flex_pdb = mol.GetProp('Flex Sidechains PDB')
        mdl_str = Chem.MolToPDBBlock(mol)
        mdl_name = f'{name} #{i}'
        pdbqt_combiner = PDBQTCombiner(mdl_name)
        pdbqt_combiner.cache_pth = complex_cache
        conn = sqlite3.connect(complex_cache)
        cur = conn.cursor()
        cur.execute(f"""SELECT EXISTS(SELECT 1 FROM ProLigDB WHERE name = '{mdl_name}')""")
        if not cur.fetchone()[0]:
            protein_delta_data, ligand_str = pdbqt_combiner.process_strings(protein_data, mdl_str, flex_pdb)
            protein_compressed = sqlite3.Binary(lzma.compress(pickle.dumps(protein_delta_data)))
            ligand_compressed = sqlite3.Binary(lzma.compress(ligand_str.encode('utf-8')))
            cur.execute("""
                        INSERT INTO ProLigDB (name, protein, ligand) 
                        VALUES (?, ?, ?)
                        """, (mdl_name, protein_compressed, ligand_compressed))
            conn.commit()
        mdlname_pdbqtcombiner_map[mdl_name] = pdbqt_combiner
    return energy_dict, mdlname_pdbqtcombiner_map

def retrieve_energy_and_string_from_autodock_pdbqt(name: str, ligand_pdbqt_str: str, protein_data: list, db_cache_pth: str):
    mdlname_pdbqtcombiner_map = {}
    energy_dict = {name: [s.strip().split()[3:] 
                          for s in re.findall(vina_eng_compiled, ligand_pdbqt_str)]}  # energy, rmsd L.B., rmsd U.B.
    models = re.findall(vina_mdl_compiled, ligand_pdbqt_str)
    for i, m in enumerate(models, start=1):
        mdl_str = m[0]
        mdl_name = f'{name} #{i}'
        pdbqt_combiner = PDBQTCombiner(mdl_name)
        pdbqt_combiner.cache_pth = db_cache_pth
        conn = sqlite3.connect(db_cache_pth)
        cur = conn.cursor()
        cur.execute(f"""SELECT EXISTS(SELECT 1 FROM ProLigDB WHERE name = '{mdl_name}')""")
        if not cur.fetchone()[0]:
            protein_delta_data, ligand_str = pdbqt_combiner.process_strings(protein_data, mdl_str)
            protein_compressed = sqlite3.Binary(lzma.compress(pickle.dumps(protein_delta_data)))
            ligand_compressed = sqlite3.Binary(lzma.compress(ligand_str.encode('utf-8')))
            cur.execute("""
                        INSERT INTO ProLigDB (name, protein, ligand) 
                        VALUES (?, ?, ?)
                        """, (mdl_name, protein_compressed, ligand_compressed))
            conn.commit()
            conn.close()
        mdlname_pdbqtcombiner_map[mdl_name] = pdbqt_combiner
    return energy_dict, mdlname_pdbqtcombiner_map

def retrieve_energy_and_string_from_gnina_pdbqt(name: str, ligand_pdbqt_str: str, protein_data: list, db_cache_pth: str):
    mdlname_pdbqtcombiner_map = {}
    energy_dict = {name: [], 'CNN Score': [], 'CNN Affinity': []}
    for matched in re.finditer(gnina_eng_compiled, ligand_pdbqt_str):
        vina_aff, cnn_score, cnn_aff = matched.group(1, 2, 3)
        energy_dict[name].append([vina_aff.strip().split()[0], float(cnn_score), float(cnn_aff)])
        energy_dict['CNN Score'].append(float(cnn_score))
        energy_dict['CNN Affinity'].append(float(cnn_aff))
    models = re.findall(vina_mdl_compiled, ligand_pdbqt_str)
    for i, m in enumerate(models, start=1):
        mdl_str = m[0]
        mdl_name = f'{name} #{i}'
        pdbqt_combiner = PDBQTCombiner(mdl_name)
        pdbqt_combiner.cache_pth = db_cache_pth
        conn = sqlite3.connect(db_cache_pth)
        cur = conn.cursor()
        cur.execute(f"""SELECT EXISTS(SELECT 1 FROM ProLigDB WHERE name = '{mdl_name}')""")
        if not cur.fetchone()[0]:
            protein_delta_data, ligand_str = pdbqt_combiner.process_strings(protein_data, mdl_str)
            protein_compressed = sqlite3.Binary(lzma.compress(pickle.dumps(protein_delta_data)))
            ligand_compressed = sqlite3.Binary(lzma.compress(ligand_str.encode('utf-8')))
            cur.execute("""
                        INSERT INTO ProLigDB (name, protein, ligand) 
                        VALUES (?, ?, ?)
                        """, (mdl_name, protein_compressed, ligand_compressed))
            conn.commit()
            conn.close()
        mdlname_pdbqtcombiner_map[mdl_name] = pdbqt_combiner
    return energy_dict, mdlname_pdbqtcombiner_map

def retrieve_energy_and_string_from_mdm(mdm_file_pth: str, calc_chemprop: bool, analyze_frags: bool):
    name = os.path.basename(mdm_file_pth).rsplit('_output.mdm')[0]
    mdlname_mdmprocessor_map = {}
    with lzma.open(mdm_file_pth) as f:
        mdm_content_dict = pickle.load(f)
    energy_dict = {name: [(round(mdm_content_dict['binding_energy'], 3), '0.00', '0.00')]}
    mol = None
    chem_prop_dict = {}
    if calc_chemprop:
        if 'rdmol' in mdm_content_dict:
            mol = mdm_content_dict['rdmol']
        else:
            mol = Chem.MolFromMolBlock(mdm_content_dict['lig_sdf']) # old format
        smiles = Chem.MolToSmiles(mol)
        chem_prop_dict = {k: round(float(mdm_content_dict[k]), 3) for k in property_functions}
        chem_prop_dict['SMILES'] = smiles
    if analyze_frags:
        if mol is None:
            if 'rdmol' in mdm_content_dict:
                mol = mdm_content_dict['rdmol']
            else:
                mol = Chem.MolFromMolBlock(mdm_content_dict['lig_sdf'])
        fragments = retrieve_fragments(mol)
        chem_prop_dict['Fragments'] = fragments
    mdlname = f'{name} #1'
    processor = MDMFileProcessor(mdlname)
    processor.cache_pth = mdm_file_pth
    mdlname_mdmprocessor_map[mdlname] = processor
    
    prop_dict = {'Name': name, 'Score': energy_dict[name][0][0], 'File Path': mdm_file_pth,
                 'Energies': energy_dict, 'Structure': mdlname_mdmprocessor_map,
                 'Old Score': mdm_content_dict['old_score']}
    if chem_prop_dict:
        prop_dict.update(chem_prop_dict)
    return prop_dict

def retrieve_fragments(mol):
    mols = [m for m in MacFrag(mol, asMols=True)]
    smiles_list = []
    for m in mols:
        for atom in m.GetAtoms():
            atom.SetIsotope(0)
        params = Chem.AdjustQueryParameters()
        params.makeDummiesQueries = True
        m = Chem.AdjustQueryProperties(m, params)
        smiles_list.append(Chem.MolToSmiles(m))
    return list(dict.fromkeys(smiles_list))

def retrieve_chemprops(name: str, db_cache_pth: str,
                       ligand_str: str, format: str, 
                       props: dict, analyze_fragments: bool):
    conn = sqlite3.connect(db_cache_pth)
    cur = conn.cursor()
    cur.execute(f"""SELECT EXISTS(SELECT 1 FROM ChemProp WHERE name = '{name}')""")
    if not cur.fetchone()[0]:
        chemprops = {}
        if format == 'pdbqt':
            pdbqt_mol = PDBQTMolecule(ligand_str, poses_to_read=1, skip_typing=True)
            mol = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol, add_Hs=False)[0] # fragment cannot have hydrogen
            smiles = Chem.MolToSmiles(mol)
        elif format == 'sdf':
            mol = Chem.MolFromMolBlock(ligand_str)  # H is automatically removed
            if mol is None:
                smiles = ''
            else:
                smiles = Chem.MolToSmiles(mol)
        chemprops['SMILES'] = smiles
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if analyze_fragments:
                frags = retrieve_fragments(mol)
                frags_byte = sqlite3.Binary(lzma.compress(pickle.dumps(frags)))
            else:
                frags_byte = None
            mol = Chem.AddHs(mol)
            try:
                chemprops.update({k: f(mol) for k, f in property_functions.items()})
            except:
                chemprops.update({k: float('nan') for k in property_functions})
        else:
            chemprops.update({k: float('nan') for k in property_functions})
            frags_byte = None
        props_byte = sqlite3.Binary(lzma.compress(pickle.dumps(chemprops)))
        cur.execute("""
                    INSERT INTO ChemProp (name, prop, fragment) 
                    VALUES (?, ?, ?)
                    """, (name, props_byte, frags_byte))
        conn.commit()
        conn.close()
        props.update(chemprops)
        if frags_byte is not None:
            props['Fragments'] = frags
        return props
    else:
        cur.execute(f"""SELECT * FROM ChemProp WHERE name = '{name}'""")
        _, props_byte, frags_byte = cur.fetchone()
        chemprops = pickle.loads(lzma.decompress(props_byte))
        if analyze_fragments:
            if frags_byte is None:
                smiles = chemprops['SMILES']
                mol = Chem.MolFromSmiles(smiles)
                frags = retrieve_fragments(mol)
                frags_byte = sqlite3.Binary(lzma.compress(pickle.dumps(frags)))
                cur.execute("""
                            UPDATE ChemProp
                            SET fragment = ?
                            WHERE name = ?;
                        """, (frags_byte, name))
                conn.commit()
            chemprops['Fragments'] = pickle.loads(lzma.decompress(frags_byte))
        conn.close()
        props.update(chemprops)
        return props

def extract_docked_properties(docked_file_pth: str, protein_data: list,
                              analyze_fragment: bool, db_cache_pth: str,
                              calc_chemprop: bool, only_rank1: bool):
    if docked_file_pth.endswith(('.pdbqt', '.sdf')):
        with open(docked_file_pth, 'r') as f:
            if only_rank1:
                r = []
                if docked_file_pth.endswith('.pdbqt'):
                    for l in f:
                        r.append(l)
                        if l.startswith('ENDMDL'):
                            break
                elif docked_file_pth.endswith('.sdf'):
                    for l in f:
                        r.append(l)
                        if l.startswith('$$$$'):
                            break
                ligand_str = ''.join(r)
            else:
                ligand_str = f.read()
        file_name = os.path.basename(docked_file_pth)
        if file_name.endswith('.pdbqt'):    # AutoDock VINA / QuickVINA / smina / gnina
            format = 'pdbqt'
            name = file_name.rsplit('.pdbqt', 1)[0]
            if name.endswith('_docked'):
                name = name.rsplit('_docked', 1)[0]
            elif name.endswith('_out'):
                name = name.rsplit('_out', 1)[0]
            if 'REMARK CNNscore ' in ligand_str:
                energies, structure = retrieve_energy_and_string_from_gnina_pdbqt(name, ligand_str, protein_data, db_cache_pth)
            else:
                energies, structure = retrieve_energy_and_string_from_autodock_pdbqt(name, ligand_str, protein_data, db_cache_pth)
        elif file_name.endswith('.sdf'):    # UniDock SDF / LeDock SDF / smina SDF
            format = 'sdf'
            name = file_name.rsplit('.sdf', 1)[0]
            if name.endswith('_out'):
                name = name.rsplit('_out', 1)[0]
                energies, structure = retrieve_energy_and_string_from_unidock_sdf(name, ligand_str, protein_data, db_cache_pth)
            elif name.endswith('_docked'):
                name = name.rsplit('_docked', 1)[0]
                energies, structure = retrieve_energy_and_string_from_smina_sdf(name, ligand_str, protein_data, db_cache_pth)
        min_energy = float(energies[name][0][0])
        if 'CNN Score' in energies:
            cnn_aff_list = energies.pop('CNN Affinity')
            cnn_sco_list = energies.pop('CNN Score')
            property = {'Name': name, 'Score': min_energy, 'File Path': docked_file_pth,
                        'Energies': energies, 'Structure': structure, 'CNN Affinity': cnn_aff_list,
                        'CNN Score': cnn_sco_list}
        else:
            property = {'Name': name, 'Score': min_energy, 'File Path': docked_file_pth,
                        'Energies': energies, 'Structure': structure}
        if calc_chemprop:
            property = retrieve_chemprops(name, db_cache_pth, ligand_str, format, property, analyze_fragment)
    elif docked_file_pth.endswith('.mdm'):
        property = retrieve_energy_and_string_from_mdm(docked_file_pth, calc_chemprop, analyze_fragment)
    return property

class MultiprocessReader(QObject):
    progress = Signal(int, dict, float, float)
    finished = Signal()
    
    def __init__(self, docked_file_paths: list[str],
                 analyze_fragments: bool,
                 protein_data: list,
                 db_cache: str,
                 calc_chemprop: bool,
                 only_rank1: bool):
        super().__init__()
        self.docked_pths = docked_file_paths
        self.analyze_fragments_bool = analyze_fragments
        self.protein_data = protein_data
        self.db_cache = db_cache
        self.calc_chemprop = calc_chemprop
        self.only_rank1 = only_rank1
        
    def run(self):
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(extract_docked_properties, f_pth, self.protein_data, 
                                       self.analyze_fragments_bool, self.db_cache,
                                       self.calc_chemprop, self.only_rank1) 
                       for f_pth in self.docked_pths]
            run_avg_step = max(1, len(futures) // 10)    # at least 1 file in case less that 10 files are loaded
            step = 0
            tik = time.perf_counter()
            mid_t = tik
            for f in as_completed(futures):
                result = f.result()
                step += 1
                tok = time.perf_counter()
                if step % run_avg_step == 0:
                    mid_t = time.perf_counter()
                    step = 1
                self.progress.emit(step, result, tok-tik, tok-mid_t)
        self.finished.emit()

class MultiThreadZINCSearch(QObject):
    curr_step_text = Signal(str, list)
    finished = Signal()
    canceled = Signal()
    
    def __init__(self, params: dict, all_smiles_list: list):
        super().__init__()
        self.params = params
        self.all_smiles_list = all_smiles_list
        self.is_running = True
    
    def run(self):
        def smallworld_zinc_search(params: dict, smiles: str, session: requests.Session, retries: int=3):
            local_params = copy.deepcopy(params)
            local_params.update({'smi': smiles})
            for _ in range(retries):
                if not self.is_running:
                    return '', smiles
                try:
                    response = session.get('https://sw.docking.org/search/view', params=local_params)
                    if response.status_code == 200:
                        df = pd.read_csv(io.StringIO(response.text))
                        max_dist = int(params['dist'].split('-')[-1])
                        max_anon_dist = int(params['adist'].split('-')[-1])
                        df = df[(df['Dist'] <= max_dist) & (df['AnonDist'] <= max_anon_dist)]
                        if len(df) == 0:
                            zinc_ids = ' '
                        else:
                            zinc_ids = [s.split(' ')[-1] for s in df['Smiles'].to_list()]
                        return zinc22_id_catalog_search(zinc_ids, smiles, retries)
                    else:
                        time.sleep(1)
                except requests.RequestException as e:
                    time.sleep(1)
            return '', smiles
        
        def zinc22_id_catalog_search(zinc_ids, smiles, retries=3):
            if zinc_ids == ' ':
                return ' ', smiles
            else:
                all_csv_strs = asyncio.run(fetch_vendor_info(zinc_ids, retries))
                dfs = [pd.read_csv(io.StringIO(csv_text))[1:] for csv_text in all_csv_strs if csv_text is not None]
                csv_strs = pd.concat(dfs).to_csv(index=None)
                return csv_strs, smiles
        
        async def fetch_vendor_info(zinc_ids: list, retries: int):
            conn = aiohttp.TCPConnector(limit_per_host=25)
            async with aiohttp.ClientSession(connector=conn) as session:
                tasks = [retrieve_zinc_with_url(session, 'ZINCms000002NiP3\n'+id, retries) for id in zinc_ids]
                return await asyncio.gather(*tasks)
            
        async def retrieve_zinc_with_url(session: aiohttp.ClientSession, ids: str, retries: int):
            for _ in range(retries):
                try:
                    file = io.BytesIO(ids.encode('utf8'))
                    form_data = aiohttp.FormData()
                    form_data.add_field('zinc_ids', file, filename='zinc_ids.txt', content_type='text/plain')
                    form_data.add_field('output_fields', 'zinc_id,smiles,catalogs')
                    async with session.get('https://cartblanche22.docking.org/substances.csv', data=form_data) as result:
                        if result.status == 200:
                            return await result.text()
                        else:
                            await asyncio.sleep(2)
                except aiohttp.ClientError as e:
                    await asyncio.sleep(1)
            return None
        
        with BoundedThreadPoolExecutor(5) as executor:
            with requests.Session() as session:
                futures = [executor.submit(smallworld_zinc_search,
                                           self.params, smiles, session) for smiles in self.all_smiles_list]
                for f in as_completed(futures):
                    if not self.is_running:
                        self.canceled.emit()
                        break
                    else:
                        respond_txt, smiles_str = f.result()
                        self.curr_step_text.emit(respond_txt, [smiles_str])
        if self.is_running:
            self.finished.emit()
    
    def stop(self):
        self.is_running = False
        tasks = asyncio.all_tasks()
        for task in tasks:
            task.cancel()
        self.executor.shutdown(wait=True, cancel_futures=False)

class MultiThreadPubChemSearch(QObject):
    curr_step_text = Signal(dict, str)
    finished = Signal()
    canceled = Signal()
    
    def __init__(self, params: dict, all_smiles_list: list):
        super().__init__()
        self.params = params
        self.all_smiles_list = all_smiles_list
        self.is_running = True
        self.smiles_url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cids}/property/CanonicalSMILES/TXT'
        self.vendor_url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/categories/compound/{cid}/JSON'
        # I don't understand the XREF order returned by PUG REST, so PUG View it is!
        # Since I have to search it one-by-one, it is a lot slower than PUG REST.
        # TODO: Review the PUG REST documentation again to see if it is possible to use PUG REST.
    
    def run(self):
        def pubchem_id_search(params: dict, smiles: str, session: requests.Session, retries: int=3):
            parsed_smi = parse.quote(smiles)
            for _ in range(retries):
                if not self.is_running:
                    return {'': None}, smiles
                try:
                    response = session.get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound'
                                           f'/fastsimilarity_2d/smiles/{parsed_smi}/cids/TXT', params=params)
                    if response.status_code == 200:
                        all_cids = response.text.strip().split('\n')
                        if all_cids[0]:
                            response = session.get(self.smiles_url.format(cids=','.join(all_cids)), params=params)
                            all_smiles = response.text.strip().split('\n')
                            cid_smiles_dict = {id: smi for id, smi in zip(all_cids, all_smiles)}
                            return pubchem_vendor_search(cid_smiles_dict, smiles)
                        else:
                            return {' ': None}, smiles
                    else:
                        time.sleep(2)
                except requests.RequestException as e:
                    time.sleep(3)
            return {'': None}, smiles
        
        async def retrieve_json_with_url(session: aiohttp.ClientSession, cid: str, retries: int):
            url = self.vendor_url.format(cid=cid)
            for _ in range(retries):
                try:
                    async with session.get(url) as result:
                        if result.status == 200:
                            return cid, await result.json()
                        else:
                            await asyncio.sleep(2)
                except aiohttp.ClientError as e:
                    await asyncio.sleep(1)
            return cid, None
        
        async def fetch_vendor_info(cids_list: list, retries: int):
            conn = aiohttp.TCPConnector(limit_per_host=5)
            async with aiohttp.ClientSession(connector=conn) as session:
                tasks = [retrieve_json_with_url(session, cid, retries) for cid in cids_list]
                return await asyncio.gather(*tasks)
        
        def pubchem_vendor_search(cid_smiles_dict: dict, smiles: str, retries: int=3):
            cid_vendors_dict = {}   # {cid: vendor_dict}, where vendor_dict = {'vendor': [...], 'ID': [...], 'url': [...], 'smiles': str}
            
            cids_list = list(cid_smiles_dict)
            all_respond_tups = asyncio.run(fetch_vendor_info(cids_list, retries))
            for cid_json_tup in all_respond_tups:
                cid, json_result = cid_json_tup
                vendor_dict = {'catalog_names'         : [] ,
                               'catalog_supplier_codes': [] ,
                               'catalog_url'           : [] ,
                               'smiles'                : cid_smiles_dict[cid],}
                if json_result is None:
                    cid_vendors_dict[cid] = vendor_dict
                else:
                    first = json_result['SourceCategories']['Categories'][0]
                    if first['Category'] == "Chemical Vendors":
                        vendors = json_result['SourceCategories']['Categories'][0]['Sources']
                        for informations in vendors:
                            vendor_dict['catalog_names'].append(informations['SourceName'])
                            vendor_dict['catalog_supplier_codes'].append(informations['RegistryID'])
                            vendor_dict['catalog_url'].append(informations.get('SourceRecordURL', None))
                    cid_vendors_dict[cid] = vendor_dict
            return cid_vendors_dict, smiles
        
        with BoundedThreadPoolExecutor(max_size=5) as self.executor:
            with requests.Session() as session:
                futures = [self.executor.submit(pubchem_id_search, self.params, smiles, session) for smiles in self.all_smiles_list]
                for f in as_completed(futures):
                    if not self.is_running:
                        self.canceled.emit()
                        break
                    else:
                        respond_dict, smiles_str = f.result()
                        self.curr_step_text.emit(respond_dict, smiles_str)
        self.finished.emit()
    
    def stop(self):
        self.is_running = False
        tasks = asyncio.all_tasks()
        for task in tasks:
            task.cancel()
        self.executor.shutdown(wait=True, cancel_futures=False)
        # kill_child_processes(os.getpid())

def search_through_db(smiles: str, db_pth_url_map: dict, threshold: float, fp_settings: dict, is_searching):
    if not is_searching.value:
        return None, smiles
    fp_gen = retrieve_fp_generator(fp_settings)
    dctx = zstd.ZstdDecompressor()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: # Shouldn't happen, since the molecule were checked during input
        return {'': None}, smiles
    query_fp = fp_gen(mol)
    final_vendor_dict = {}
    for pth in db_pth_url_map:
        if not is_searching.value:
            return None, smiles
        template = db_pth_url_map[pth]['template']
        if not template:
            template = None
        db_name = db_pth_url_map[pth]['name']
        conn = sqlite3.connect(pth)
        db = pd.read_sql('SELECT name, fp, smi From MolDB', conn, chunksize=3500)
        for row in db:
            if not is_searching.value:
                conn.close()
                return None, smiles
            fps = [CreateFromBinaryText(dctx.decompress(fp)) for fp in row.fp]
            sims = np.array(retrieve_similarity_method(fp_settings['sim'], True)(query_fp, fps))
            bool_mask = sims >= threshold
            all_names = np.array(row.name)[bool_mask]
            all_smi   = np.array(row.smi)[bool_mask]
            all_sims  = sims[bool_mask]
            for n, s, si in zip(all_names, all_smi, all_sims):
                matched_smi = dctx.decompress(s).decode()
                matched_mol = Chem.AddHs(Chem.MolFromSmiles(matched_smi))
                properties = {f'{chem_prop}': func(matched_mol) for chem_prop, func in property_functions.items()}
                vendor_dict = {'catalog_names'         : [db_name],
                               'catalog_supplier_codes': [n],
                               'catalog_url'           : [template.format(id=n) if template else None],
                               'smiles'                : matched_smi,
                               'properties'            : properties,
                               'similarity'            : si}
                final_vendor_dict[f'{db_name}_{n}'] = vendor_dict
        conn.close()
        if not is_searching.value:
            return None, smiles
    if not is_searching.value:
        return None, smiles
    return final_vendor_dict, smiles

class MultiprocessDBSearch(QObject):
    currSmilesResult = Signal(dict, str)
    finished = Signal()
    canceled = Signal()
    
    def __init__(self, db_pth_url_map: dict, all_smiles_list: list, threshold: float, fp_settings: dict):
        super().__init__()
        self.db_pth_url_map = db_pth_url_map
        self.all_smiles_list = all_smiles_list
        self.threshold = threshold
        self.futures = []
        self.fp_settings = fp_settings
    
    def run(self):
        self.manager = Manager()
        self.is_searching = self.manager.Value('b', True)
        self.curr_searching = True
        self.executor = ProcessPoolExecutor()
        for smiles in self.all_smiles_list:
            future = self.executor.submit(search_through_db,
                                          smiles,
                                          self.db_pth_url_map,
                                          self.threshold,
                                          self.fp_settings,
                                          self.is_searching)
            future.add_done_callback(self.process_future)
            self.futures.append(future)
    
    def process_future(self, future):
        try:
            result, smiles = future.result()
            if result is not None:
                self.currSmilesResult.emit(result, smiles)
        except:
            pass
        
        if all(f.done() for f in self.futures):
            if self.curr_searching:
                self.finished.emit()
    
    @Slot()
    def stop(self):
        self.is_searching.value = False
        self.curr_searching = False
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.manager.shutdown()
        self.canceled.emit()

def get_fragment_img_score_name(input_dict: dict, frag: str, total_docked: int, uniqueness: float):
    count = len(input_dict['Score'])
    mol = Chem.MolFromSmiles(frag, sanitize=False)  # Some SMILES cannot be converted... But since they are prepared with RDKit, just force convert it.
    weighted_tf = sum(input_dict['Score'])
    idf = np.log10(total_docked / count)
    frag_score = weighted_tf * idf * np.log10(mol.GetNumAtoms())
    
    img = Draw.MolToImage(mol, size=(300, 300))
    
    return {'Score': frag_score,
            'Names': input_dict['Names'],
            'Image': img,}, frag

class MultiprocessFragmentScore(QObject):
    progress = Signal(int, dict, str)
    finished = Signal()
    
    def __init__(self, fragment_score_name_count_dict: dict, total_docked: int, uniqueness: float):
        super().__init__()
        self.fragment_dict = fragment_score_name_count_dict
        self.num = total_docked
        self.unique = uniqueness
        
    def run(self):
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(get_fragment_img_score_name, frag_dict, frag, self.num, self.unique) for frag, frag_dict in self.fragment_dict.items()]
            step = 0
            for f in as_completed(futures):
                result_dict, frag = f.result()
                step += 1
                self.progress.emit(step, result_dict, frag)
        self.finished.emit()

def plot_energy_fragment(name_energy_df: pd.DataFrame, names: list, html_pth: str):
    plotly_html_config = {'displaylogo': False,
                          'toImageButtonOptions': {'format': 'png',
                                                   'scale' : 3},
                          'edits': {'legendPosition'  : True,
                                    'legendText'      : True,
                                    'titleText'       : True,
                                    'colorbarPosition': True},
                          'showTips': False}
    extracted_df = name_energy_df[name_energy_df['Name'].isin(names)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=name_energy_df.index,
                             y=name_energy_df['Score'],
                             name='All',
                             mode='markers',
                             marker={'opacity': 0.4, 'size': 5},
                             customdata=name_energy_df['Name'],
                             hovertemplate="Name: %{customdata}<br>Score: %{y:.4f}<extra></extra>",))
    fig.add_trace(go.Scatter(x=extracted_df.index,
                             y=extracted_df['Score'],
                             name='With Fragment',
                             mode='markers',
                             marker={'opacity': 0.8, 'size': 5},
                             customdata=extracted_df['Name'],
                             hovertemplate="Name: %{customdata}<br>Score: %{y:.4f}<extra></extra>",))
    fig['layout']['xaxis']['title']['text'] = 'Rank'
    fig['layout']['yaxis']['title']['text'] = 'Score'
    fig['layout'].update({'title': 'Score v.s. Score Rank'})
    fig.write_html(html_pth, plotly_html_config)

def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in children:
        process.send_signal(sig)
    # os.killpg(os.getpgid(parent_pid), sig)

def ebox_size(pdbqt_pth: str):
    xyz = []
    with open(pdbqt_pth) as f:
        xyz = [[l[30:38], l[38:46], l[46:54]] for l in f if l.startswith(('ATOM', 'HETATM')) and l[13] != 'H']
    xyz = np.array(xyz, float)
    return ((((xyz - xyz.mean(0)) ** 2).sum() / xyz.shape[0]) ** 0.5) / 0.23   # ebox algorithm, no clue why.

class CustomSortFilterProxyModel(QSortFilterProxyModel):
    def lessThan(self, left, right):
        sort_column = self.sortColumn()
        
        if left.column() != sort_column or right.column() != sort_column:
            return super().lessThan(left, right)
        
        left_data = self.sourceModel().data(left, Qt.ItemDataRole.DisplayRole)
        right_data = self.sourceModel().data(right, Qt.ItemDataRole.DisplayRole)
        
        try:
            left_num = float(left_data)
            left_is_num = True
        except (ValueError, TypeError):
            left_is_num = False
            
        try:
            right_num = float(right_data)
            right_is_num = True
        except (ValueError, TypeError):
            right_is_num = False
            
        if left_is_num and right_is_num:
            return left_num < right_num
        elif left_is_num:
            return True
        elif right_is_num:
            return False
        else:
            return str(left_data) < str(right_data)

class ManualProgressBarDelegate(QStyledItemDelegate):
    # This part of the code heavily relies on ChatGPT.
    # The QStyledProgressBar isn't working for MacOS as described here https://bugreports.qt.io/browse/PYSIDE-1464,
    # so other methods have to be used. Here, an entire new progrss bar is created with the help of ChatGPT.
    def __init__(self, parent=None):
        super().__init__(parent)
        self.bg_color = QColor('transparent')
        self.fg_color = QColor(66 , 165, 245)
        
    def paint(self, painter: QPainter, option, index):
        if index.column() == 2:
            progress = index.data(Qt.ItemDataRole.DisplayRole)
            try:
                progress = int(progress)
                progress = max(0, min(progress, 51))
            except (ValueError, TypeError):
                progress = 0
            
            progress_ratio = progress / 51
            rect = option.rect.adjusted(4, 4, -4, -4)   # padding of the rectangle
            
            # Draw the rectangle of progress bar
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setBrush(self.bg_color)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(rect, 5, 5)
            
            # Draw the progress
            progress_width = int(rect.width() * progress_ratio)
            progress_rect = rect.adjusted(0, 0, -rect.width() + progress_width, 0)
            painter.setBrush(self.fg_color)
            painter.drawRoundedRect(progress_rect, 5, 5)
            
            # Draw the text (centered)
            painter.setPen(QApplication.palette().text().color())
            text = f'{progress_ratio:.0%}'
            font_metrics = QFontMetrics(painter.font())
            text_width = font_metrics.horizontalAdvance(text)
            text_height = font_metrics.height()
            text_x = rect.x() + (rect.width() - text_width) / 2
            text_y = rect.y() + (rect.height() + text_height) / 2 - 2  # Adjust vertically
            painter.drawText(text_x, text_y, text)
            painter.restore()
        else:
            # For other columns, use the default painting
            super().paint(painter, option, index)
            
    def sizeHint(self, option, index):
        return super().sizeHint(option, index)

class DockProgressModel(QAbstractTableModel):
    def __init__(self, data, header_labels):
        super().__init__()
        self._data = data
        self._header_labels = header_labels
        
    def rowCount(self, parent=QModelIndex()):
        return len(self._data)
    
    def columnCount(self, parent=QModelIndex()):
        return len(self._header_labels)
    
    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row, col = index.row(), index.column()
        
        if role == Qt.ItemDataRole.DisplayRole:
            return self._data[index.row()][index.column()]
        
        if role == Qt.ItemDataRole.ForegroundRole and col == 1:
            status_text = self._data[row][col]
            if status_text == "Failed":
                return QColor(229, 115, 115)
            elif status_text == "Docking..." or status_text == "Refining...":
                return QColor(66 , 165, 245)
            elif status_text == 'Pending...':
                return QApplication.palette().text().color()
            else:
                return QColor(76 , 175, 80 )    # Done
            
        return None
    
    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return self._header_labels[section]
            elif orientation == Qt.Orientation.Vertical:
                return str(section + 1)
        return None
    
    def update_status(self, name: str, new_stat: str):
        for row, row_data in enumerate(self._data):
            if row_data[0] == name:
                row_data[1] = new_stat
                index = self.index(row, 1)
                self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ForegroundRole])
                break
    
    def update_progress(self, name: str, new_value: int):
        for row, row_data in enumerate(self._data):
            if row_data[0] == name:
                row_data[2] = new_value
                index = self.index(row, 2)
                self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole])
                break
    
    def add_to_progress(self, name: str, added: int):
        for row, row_data in enumerate(self._data):
            if row_data[0] == name:
                row_data[2] += added
                index = self.index(row, 2)
                self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole])
                break
    
    def __change_everything_to_false(self):
        for row, row_data in enumerate(self._data):
            if row_data[1] < 51:
                row_data[2] = 'Failed'
                row_data[1] = 51
        index_topleft = self.index(0, 1)
        index_bottomright = self.index(row, 2)
        self.dataChanged

class DockProgressTableView(QTableView):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.column_map = {'Name': 0, 'Status': 1, 'Progress': 2}
        self.model = DockProgressModel([], list(self.column_map))
        self.proxy = CustomSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.setModel(self.proxy)
        
        self.setItemDelegateForColumn(2, ManualProgressBarDelegate(self))
        self.setSortingEnabled(True)
        self.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        
        self.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self.set_column_width()
    
    def set_column_width(self):
        header = self.horizontalHeader()
        # for i, column in enumerate(self.column_map):
        #     if column == 'Progress':
        #         header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
        #     else:
        #         header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        
    def resizeEvent(self, event):
        h = self.horizontalHeader()
        for i in range(2):
            curr_size = h.sectionSize(i)
            h.resizeSection(i, curr_size)
        h.setSectionResizeMode(i + 1, QHeaderView.ResizeMode.Stretch)
        super().resizeEvent(event)
    
    def init_progresses(self, name_progress_status_dict: dict):
        data = [[name, d['status'], d['progress']] for name, d in name_progress_status_dict.items()]
        self.model = DockProgressModel(data, list(self.column_map))
        self.proxy.setSourceModel(self.model)
        self.setModel(self.proxy)
        
    def update_progress_bar_by_add(self, name: str, value: int):
        self.model.add_to_progress(name, value)
        
    def set_progress_bar_value(self, name: str, value: int):
        self.model.update_progress(name, value)
        
    def get_current_progress(self, name: str):
        for row_data in self.model._data:
            if row_data[0] == name:
                try:
                    return int(row_data[2])
                except (ValueError, TypeError):
                    return 0
        return 0
        
    def update_progress_status(self, name: str, status: str):
        self.model.update_status(name, status)
    
    def set_filter(self, regex_str: str, column_name: str):
        column_idx = self.column_map[column_name]
        regex = QRegularExpression(regex_str)
        if regex.isValid():
            self.proxy.setFilterRegularExpression(regex)
            self.proxy.setFilterKeyColumn(column_idx)
    
    def clear_table(self):
        self.model = DockProgressModel([], list(self.column_map))
        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.setModel(self.proxy)

class ThreadedVINAPreprocess(QObject):
    finalResult = Signal(dict, dict, str, int)
    
    def __init__(self, input_output_dict, out_dir, cache_dir, ligand_ext, program_type):
        super().__init__()
        self.input_output_dict = input_output_dict
        self.out_dir = out_dir
        self.cache_dir = cache_dir
        self.ligand_ext = ligand_ext
        self.program_type = program_type
        
    def run(self):
        def preprocess_vina_stuff():
            docked_files = []
            for f in os.listdir(self.out_dir):
                dir_f = os.path.join(self.out_dir, f)
                if os.path.isfile(dir_f) and os.path.getsize(dir_f) == 0:
                    os.remove(dir_f)
                elif f.endswith(self.ligand_ext):
                    docked_files.append(f)
                    
            progress_csv = os.path.join(self.cache_dir, 'dock_progress.csv')
            if os.path.isfile(progress_csv):
                docked_file_name = [f.rsplit(f'_docked.')[0] for f in docked_files]
                if docked_file_name:
                    # in case someone modified the csv file to create new entries that doesn't exist.
                    # Is this really necessary though?
                    df = pd.read_csv(progress_csv)
                    df = df[(df['name'].isin(docked_file_name)) | (~df['score'].notna())]
                    df.to_csv(progress_csv, index=None)
                    name_score_map = {name: float(score) for name, score in zip(df['name'].to_list(), df['score'].to_list())}
                else:
                    # the docked files have higher priority, so if no docked files are detected, reset everything
                    with open(progress_csv, 'w') as f:
                        f.write('name,score\n')
                    name_score_map = {}
            else:
                name_score_map = {}
                program_regex_map = {'AutoDock VINA': vina_score_only_compiled,
                                     'smina'        : smina_eng_compiled,
                                     'qvina2'       : vina_score_only_compiled,
                                     'qvinaw'       : vina_score_only_compiled,}
                with open(progress_csv, 'w') as f:
                    f.write('name,score\n')
                if docked_files:
                    # the csv file is a new addition, so this part is mainly for backward compatability
                    final_line = []
                    for f in docked_files:
                        docked_f = os.path.join(self.out_dir, f)
                        name = f.rsplit('_docked.')[0]
                        with open(docked_f) as d_f:
                            docked_str = d_f.read()
                        for g in re.finditer(program_regex_map[self.program_type], docked_str):
                            eng = float(g.group(1))    # min score
                            break
                        final_line.append(f'{name},{eng:.2f}')
                        name_score_map[name] = round(eng, 2)
                    with open(progress_csv, 'a') as f:
                        f.write('\n'.join(final_line)+'\n')
            
            curr_step = 0
            final_lig_out_dict = {}
            name_progress_status_dict = {}
            
            for lig_pth, docked_name_dict in self.input_output_dict.items():
                name = docked_name_dict['name']
                if name in name_score_map:
                    curr_step += 1
                    score = name_score_map[name]
                    if score != score:
                        score = 'Failed'
                    name_progress_status_dict[name] = {'status'   : score,
                                                        'progress': 51}
                else:
                    final_lig_out_dict[lig_pth] = os.path.join(self.out_dir, docked_name_dict['docked_name'])
                    name_progress_status_dict[name] = {'status'   : 'Pending...',
                                                        'progress': 0 }
            
            self.finalResult.emit(final_lig_out_dict, name_progress_status_dict, self.program_type, curr_step)
        
        preprocess_vina_stuff()

class ModifyNameURLTemplateDialog(QDialog):
    def __init__(self, url_template_pth: str):
        super().__init__()
        self.url_template_pth = url_template_pth
        self.name_template_map = {}
        with open(self.url_template_pth) as f:
            for l in f:
                db_name, url = l.split()[:2]    # should only be 2, but just in case
                self.name_template_map[db_name] = url
        self.setup_ui()
        
    def setup_ui(self):
        overall_layout = QVBoxLayout()
        input_layout = QGridLayout()
        
        self.name_template_table = RemovableTableWidget(len(self.name_template_map), 2)
        self.name_template_table.setHorizontalHeaderLabels(['Name', 'Template'])
        self.name_template_table.verticalHeader().hide()
        for row, (name, template) in enumerate(self.name_template_map.items()):
            name_item = QTableWidgetItem(name)
            temp_item = QTableWidgetItem(template)
            self.name_template_table.setItem(row, 0, name_item)
            self.name_template_table.setItem(row, 1, temp_item)
        self.name_template_table.itemChanged.connect(self.check_name_and_template)
        self.name_template_table.cellClicked.connect(self.keep_old_names)
        
        name_label = QLabel('Name')
        template_label = QLabel('Template')
        self.name_lineedit = QLineEdit()
        self.template_lineedit = QLineEdit()
        self.template_lineedit.setPlaceholderText('https://www.example.com/catalog/{id}')
        add_button = QPushButton('+')
        add_button.clicked.connect(self.add_name_template)
        
        input_layout.addWidget(name_label, 0, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        input_layout.addWidget(template_label, 1, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        input_layout.addWidget(self.name_lineedit, 0, 1)
        input_layout.addWidget(self.template_lineedit, 1, 1)
        input_layout.addWidget(add_button, 1, 2, alignment=Qt.AlignmentFlag.AlignRight)
        
        QBtn = QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        button_box = QDialogButtonBox(QBtn)
        button_box.accepted.connect(self.accept_changes)
        button_box.rejected.connect(self.reject)
        
        overall_layout.addWidget(self.name_template_table, 1)
        overall_layout.addLayout(input_layout)
        overall_layout.addWidget(button_box)
        
        self.setLayout(overall_layout)
        self.setWindowTitle('Modify Local Database Name & URL Template')
        screen = self.screen()
        screen_size = screen.availableGeometry()
        self.resize(screen_size.width() * 0.4, screen_size.height() * 0.55)
        
    def add_name_template(self):
        name = self.name_lineedit.text()
        if not name:
            QMessageBox.critical(self, 'NameError', 'Name is required!')
            return
        if name in self.name_template_table.get_all_names():
            QMessageBox.critical(self, 'NameError', 'Name alread existed!')
            return
        
        template_url = self.template_lineedit.text()
        if not template_url:
            QMessageBox.critical(self, 'TemplateError', 'URL template is required!')
            return
        if '{id}' not in template_url:
            QMessageBox.critical(self, 'TemplateError', 'URL template requires "{id}"!')
            return
        
        self.name_template_table.blockSignals(True)
        
        curr_row = self.name_template_table.rowCount()
        self.name_template_table.insertRow(curr_row)
        name_item = QTableWidgetItem(name)
        temp_item = QTableWidgetItem(template_url)
        self.name_template_table.setItem(curr_row, 0, name_item)
        self.name_template_table.setItem(curr_row, 1, temp_item)
        
        self.name_template_table.blockSignals(False)
    
    def keep_old_names(self, row, column):
        self.last_string = self.name_template_table.item(row, column).text()
    
    def check_name_and_template(self, item: QTableWidgetItem):
        col = self.name_template_table.column(item)
        if col == 0:
            new_text = item.text()
            for r in range(self.name_template_table.rowCount()):
                if r != col:
                    if self.name_template_table.item(r, 0).text() == new_text:
                        QMessageBox.critical(self, 'NameError', f'{new_text} already existed!')
                        self.name_template_table.blockSignals(True)
                        item.setText(self.last_string)
                        self.name_template_table.blockSignals(False)
                        return
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Return:
            if self.name_lineedit.hasFocus():
                self.template_lineedit.setFocus()
            elif self.template_lineedit.hasFocus():
                self.add_name_template()
        else:
            super().keyPressEvent(event)
    
    def accept_changes(self):
        self.final_map = {}
        for r in range(self.name_template_table.rowCount()):
            self.final_map[self.name_template_table.item(r, 0).text()] = self.name_template_table.item(r, 1).text()
        with open(self.url_template_pth, 'w') as f:
            for name, template in self.final_map.items():
                f.write(f'{name} {template}\n')
        self.accept()
