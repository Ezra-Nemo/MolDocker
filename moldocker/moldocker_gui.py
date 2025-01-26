import os, io, re, gc, stat, copy, math, time, json, shutil
import pickle, sqlite3, zipfile, platform, warnings
import qdarktheme, subprocess

import numpy as np
import pandas as pd
import plotly.io as pio

from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import pybel
pybel.ob.obErrorLog.SetOutputLevel(0)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout,
                               QPushButton, QFileDialog, QLabel,
                               QTextEdit, QHBoxLayout, QProgressBar,
                               QSpinBox, QSpacerItem, QGridLayout,
                               QTabWidget, QMainWindow, QMessageBox,
                               QComboBox, QFrame, QDoubleSpinBox,
                               QMenu, QScrollArea, QProgressDialog,
                               QCheckBox, QLineEdit, QRadioButton, QSplitter,
                               QMenuBar, QSizePolicy, QButtonGroup, QListWidgetItem,
                               QWidgetAction)
from PySide6.QtGui import (QIcon, QTextCursor, QAction, QFont, QRegularExpressionValidator,
                           QColor, QPixmap)
from PySide6.QtCore import Qt, QThread, QSize, QPropertyAnimation, QEasingCurve, Signal, Slot
from PySide6.QtWebEngineCore import QWebEngineDownloadRequest
from multiprocessing import cpu_count

from .utilities.utilis import RDKitMolCreate, PDBQTMolecule, pdbqt_to_pdb

from .utilities.protein_utilis import PDBEditor
from .utilities.browser_utilis import (ProteinLigandEmbedBrowserWidget, PlotViewer, FPocketBrowser, BrowserWithTabs,
                                       DocumentationWindow, LigPlotWidget)
from .utilities.convert_utilis import (DirSplitterDialog, DirCombinerDialog, DBSearchDialog, InputFileDirListWidget,
                                       OutputDirLineEdit, LogTextEdit, ConvertFilterDialog, CSVTSVColumnDialog,
                                       ThreadedMaxMinPicker, MultiprocessConversion, MultiprocessConvertReader,
                                       SDFIDDialog, MultiProcessDBSimilarityPicker)
from .utilities.general_utilis import (DropFileLineEdit, DropDirLineEdit, SettingDialog,
                                       DockingTextExtractor, DockResultTable, MultiprocessReader,
                                       TableFilterDialog, AutoFilterDialog, CopyLineEdit, ImageLabel,
                                       ShowPlotLabel, MultiprocessFragmentScore, HiddenExpandLabel,
                                       FileNameTree, PlotFilterDialog, PlotFilterDataframe,
                                       ThreadedProteinDownloader, ContactTabTables, ThreadedLocalPDBLoader,
                                       SlidingStackedWidget, ThreadedPDBConverter, ZINCSupplierFinderWidget,
                                       PubChemSupplierFinderWidget, DockProgressTableView, ThreadedVINAPreprocess,
                                       chem_prop_to_full_name_map, LocalDatabaseFinderWidget, ModifyNameURLTemplateDialog)
from .utilities.fingerprint_utilis import (FingerprintSettingDialog)
from .utilities.general_utilis import (sdf_match_eng_rmsd_compiled, sdf_regex_list, vina_eng_compiled, vina_mdl_compiled,
                                       smina_eng_compiled)
from .utilities.docking_utilis import MultiprocessLeDock, MultiprocessVINADock, MultiprocessRefine, DockingThread
from .utilities.rdeditor_utilis import ChemEditorDialog

class MolDocker(QMainWindow):
    dockStopSignal = Signal()
    
    def __init__(self, curr_theme: str):
        super().__init__()
        global theme
        theme = curr_theme
        self.initUI()
        self.create_menu_bar()
    
    def initUI(self):
        self.curr_dir = os.path.dirname(__file__)
        curr_os = platform.system()
        if curr_os == 'Darwin':
            self.vina_exec = os.path.join(self.curr_dir, 'docking_exec', 'vina_file', 'vina_mac')
            self.ledock_exec = os.path.join(self.curr_dir, 'docking_exec', 'ledock_file', 'ledock_mac')
            self.smina_exec = os.path.join(self.curr_dir, 'docking_exec', 'smina_file', 'smina_mac')
            processor = platform.processor()
            self.qvina2_exec = os.path.join(self.curr_dir, 'docking_exec', 'qvina2_file', f'qvina2_mac_{processor}')
            self.qvinaw_exec = os.path.join(self.curr_dir, 'docking_exec', 'qvinaw_file', f'qvinaw_mac_{processor}')
            mode = os.stat(self.vina_exec).st_mode
            for exec in [self.vina_exec, self.smina_exec, self.qvina2_exec, self.qvinaw_exec]:
                os.chmod(exec, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                subprocess.run(['xattr', '-d' 'com.apple.quarantine', f'{exec}'], stdout=subprocess.DEVNULL)
        elif curr_os == 'Windows':
            self.vina_exec = os.path.join(self.curr_dir, 'docking_exec', 'vina_file', 'vina_win.exe')
            # self.ledock_exec = os.path.join(self.curr_dir, 'docking_exec', 'Ledock.win32', 'LeDock.exe')
            self.ledock_exec = None
            self.smina_exec = os.path.join(self.curr_dir, 'docking_exec', 'smina_file', 'smina_win', 'smina.exe')
            self.qvina2_exec = os.path.join(self.curr_dir, 'docking_exec', 'qvina2_file', 'qvina2_win', 'qvina2.exe')
            self.qvinaw_exec = os.path.join(self.curr_dir, 'docking_exec', 'qvinaw_file', 'qvinaw_win', 'qvinaw.exe')
        elif curr_os == 'Linux':
            self.vina_exec = os.path.join(self.curr_dir, 'docking_exec', 'vina_file', 'vina_linux')
            self.ledock_exec = os.path.join(self.curr_dir, 'docking_exec', 'ledock_file', 'ledock_linux')
            self.smina_exec = os.path.join(self.curr_dir, 'docking_exec', 'smina_file', 'smina_linux')
            self.qvina2_exec = os.path.join(self.curr_dir, 'docking_exec', 'qvina2_file', 'qvina2_linux')
            self.qvinaw_exec = os.path.join(self.curr_dir, 'docking_exec', 'qvinaw_file', 'qvinaw_linux')
            mode = os.stat(self.vina_exec).st_mode
            os.chmod(self.vina_exec, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            os.chmod(self.smina_exec, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            os.chmod(self.qvina2_exec, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            os.chmod(self.qvinaw_exec, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.empty_light_plot_html = os.path.join(self.curr_dir, 'plot_empty', 'empty_light.html')
        self.empty_dark_plot_html = os.path.join(self.curr_dir, 'plot_empty', 'empty_dark.html')
        self.tmp_plotly_file = os.path.join(self.curr_dir, 'plot_empty', '_tmp.html')
        self.stacked_widget = SlidingStackedWidget(self)
        self.tab_widget = QTabWidget(self)
        self.available_formats = ('.smi', '.sdf',  '.mol2', '.mol',
                                  '.mrv', '.pdb',  '.xyz',  '.pdbqt',
                                  '.zip', '.mddb',)
        
        central_widget = QWidget()
        self.central_layout = QHBoxLayout()
        self.central_layout.setContentsMargins(0, 0, 0, 0)
        central_widget.setLayout(self.central_layout)
        self.setCentralWidget(central_widget)
        
        self.swap_widget_button_frame = QFrame()
        swap_widget_button_layout = QVBoxLayout()
        self.swap_widget_button_frame.setLayout(swap_widget_button_layout)
        self.swap_widget_button_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        self.swap_widget_button_frame.setLineWidth(4)
        
        toggle_btn_container = QWidget()
        toggle_btn_layout = QVBoxLayout()
        toggle_btn_layout.setContentsMargins(0, 0, 0, 0)
        toggle_btn_layout.setSpacing(0)
        toggle_btn_container.setLayout(toggle_btn_layout)
        
        left_layout = QVBoxLayout()
        left_layout.setSpacing(0)
        left_layout.addWidget(toggle_btn_container, alignment=Qt.AlignmentFlag.AlignTop)
        left_layout.addSpacerItem(QSpacerItem(0, 0))
        left_layout.addWidget(self.swap_widget_button_frame)
        
        self.allow_decision_plot = False
        self.pdbqt_editor = None
        convert_widget = self.setup_convert_widget()
        docking_widget = self.setup_dock_widget()
        table_widget = self.setup_table_widget()
        structure_widget = self.setup_structure_widget()
        shopper_widget = self.setup_shopper_widget()
        plot_widget = self.setup_plot_widget()
        decision_widget = self.setup_decision_widget()
        fragment_widget = self.setup_fragment_widget()
        
        ### Add widgets to stack
        self.stacked_widget.addWidget(convert_widget)
        self.stacked_widget.addWidget(docking_widget)
        self.stacked_widget.addWidget(table_widget)
        self.stacked_widget.addWidget(structure_widget)
        self.stacked_widget.addWidget(plot_widget)
        self.stacked_widget.addWidget(shopper_widget)
        self.stacked_widget.addWidget(decision_widget)
        self.stacked_widget.addWidget(fragment_widget)
        self.stacked_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        
        ### Create buttons to swap
        self.all_swap_buttons = {}
        self.tab_names = ['converter', 'docker', 'table', 'structure', 'figure', 'shopper', 'threshold', 'fragment']
        for i, name in enumerate(self.tab_names):
            btn = QPushButton()
            btn.setToolTip(name.capitalize())
            btn.setIconSize(QSize(35, 35))
            btn.clicked.connect(lambda _, i=i: self.stacked_widget.moveToIndex(i))
            btn.setStyleSheet("QPushButton {border-style: outset; border-width: 0px;}")
            self.all_swap_buttons[name] = btn
            swap_widget_button_layout.addWidget(btn)
            swap_widget_button_layout.addSpacerItem(QSpacerItem(0, 10))
        swap_widget_button_layout.addStretch()
        swap_widget_button_layout.addWidget(self.dark_light_swap_button)
        swap_widget_button_layout.setContentsMargins(3, 20, 3, 20)
        self.change_icon_light_dark()
        
        self.central_layout.setSpacing(0)
        self.central_layout.addLayout(left_layout)
        self.central_layout.addWidget(self.stacked_widget, 1)
        
        self.sidebar_minimum_width = self.swap_widget_button_frame.sizeHint().width()
        self.swap_widget_button_frame.setMaximumWidth(self.sidebar_minimum_width)
        self.toggle_sidebar_button = QPushButton("≡")
        self.sidebar_btn_expand_font = QFont('Arial', 20)
        self.sidebar_btn_expand_font.setBold(True)
        self.sidebar_btn_hide_font = QFont('Arial', 20)
        self.sidebar_btn_hide_font.setBold(True)
        self.toggle_sidebar_button.setFont(self.sidebar_btn_expand_font)
        self.toggle_sidebar_button.setToolTip('Hide tab')
        self.toggle_sidebar_button.setStyleSheet("""QPushButton { 
                                                 padding: 0px;
                                                 text-align: top;
                                                 }""")
        self.toggle_sidebar_button.clicked.connect(self.toggle_sidebar)
        toggle_btn_layout.addWidget(self.toggle_sidebar_button)
        
        self.sidebar_animation = QPropertyAnimation(self.swap_widget_button_frame, b"maximumWidth")
        self.sidebar_animation.setDuration(250)
        self.sidebar_animation.setEasingCurve(QEasingCurve.Type.OutCurve)
        self.sidebar_animation.valueChanged.connect(self.update_other_positions)
        
        self.setWindowTitle('MolDocker')
        
        self.now_docking = False
        self.sidebar_visible = True
        
        self.param_dict = {'dock_center'   : {'x': None, 'y': None, 'z': None},
                           'dock_width'    : {'x': None, 'y': None, 'z': None},
                           'exhaustiveness': 12                               ,
                           'eval_poses'    : 5                                ,
                           'center_color'  : QColor(0  , 255,   0),
                           'width_color'   : QColor(255, 255, 255, 128),
                           'fpocket'       : None,
                           'hetatm'        : None,}
        self.param_to_full_name_map = {'exhaustiveness': 'Exhaustiveness',
                                       'eval_poses'    : 'Num. Poses',
                                       'dock_center'   : 'Dock Center',
                                       'dock_width'    : 'Dock Size'}
        self.webserver_map = {'Uni-Dock' : {'colab': 'https://colab.research.google.com/github/Ezra-Nemo/MolDocker_Colab_Notebooks/blob/main/UniDock.ipynb',
                                            'hf'   : 'https://huggingface.co/spaces/Cpt-Nemo/MolDocker_UniDock_Interface'},
                              'DiffDock' : {'colab': 'https://colab.research.google.com/github/Ezra-Nemo/MolDocker_Colab_Notebooks/blob/main/DiffDock.ipynb',
                                            'hf'   : 'https://huggingface.co/spaces/Cpt-Nemo/MolDocker_DiffDock_Interface'},
                              'gnina'    : {'colab': 'https://colab.research.google.com/github/Ezra-Nemo/MolDocker_Colab_Notebooks/blob/main/gnina.ipynb',
                                            'hf'   : 'https://huggingface.co/spaces/Cpt-Nemo/MolDocker_VINA_and_Variants_Interface'},
                              'Refine'   : {'colab': 'https://colab.research.google.com/github/Ezra-Nemo/MolDocker_Colab_Notebooks/blob/main/Refinement.ipynb',
                                            'hf'   : 'https://huggingface.co/spaces/Cpt-Nemo/MolDocker_Refinement_Interface'},
                              'RunPod'   : {'url'  : 'https://www.runpod.io/console/pods'},}
        self.protein_loader_thread = None
        self.allow_plot = False
        # self.supplier_search_window_map = {'ZINC': None, 'PubChem': None}
        self.aminoAcids = [
        '[ALA]', '[ARG]', '[ASN]', '[ASP]', 
        '[ASH]', '[ASX]', '[CYS]', '[GLN]',
        '[GLU]', '[GLH]', '[GLY]', '[GLX]',
        '[HIS]', '[ILE]', '[LEU]', '[LYS]',
        '[MET]', '[PHE]', '[PRO]', '[SER]',
        '[THR]', '[TRP]', '[TYR]', '[VAL]',
        '[HID]', '[HIP]', '[CYX]', '[CYM]',
        '[HIE]', '[LYN]']
        # self.showMaximized()
    
    def create_menu_bar(self):
        self.menu_bar = QMenuBar()
        self.setMenuBar(self.menu_bar)
        
        convert_utilities_menu = QMenu('Convert Utilities', self)
        dock_utilities_menu = QMenu('Dock Utilities', self)
        shopper_utilities_menu = QMenu('Shop Utilities', self)
        general_utilities_menu = QMenu('General Utilities', self)
        help_utilities_menu = QMenu('Help', self)
        self.menu_bar.addMenu(convert_utilities_menu)
        self.menu_bar.addMenu(dock_utilities_menu)
        self.menu_bar.addMenu(shopper_utilities_menu)
        self.menu_bar.addMenu(general_utilities_menu)
        self.menu_bar.addMenu(help_utilities_menu)
        
        fill_gap_widget = QWidget()
        fill_gap_layout = QHBoxLayout()
        fill_gap_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        fill_gap_layout.setContentsMargins(10, 0, 0, 0)
        fill_gap_widget.setLayout(fill_gap_layout)
        self.fill_protein_gap_ckbox = QCheckBox('Fill Gap', self)
        fill_gap_layout.addWidget(self.fill_protein_gap_ckbox)
        fill_protein_gap_ckbox_action = QWidgetAction(self)
        fill_protein_gap_ckbox_action.setDefaultWidget(fill_gap_widget)
        dock_utilities_menu.addAction(fill_protein_gap_ckbox_action)
        
        spinbox_widget = QWidget()
        spinbox_layout = QHBoxLayout()
        spinbox_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        spinbox_layout.setContentsMargins(10, 0, 0, 0)
        spinbox_widget.setLayout(spinbox_layout)
        spinbox_label = QLabel('Protein pH:')
        self.protein_ph_spinbox = QDoubleSpinBox()
        self.protein_ph_spinbox.setRange(0, 14)
        self.protein_ph_spinbox.setValue(7.)
        self.protein_ph_spinbox.setSingleStep(0.1)
        spinbox_layout.addWidget(spinbox_label)
        spinbox_layout.addWidget(self.protein_ph_spinbox)
        protein_ph_spinbox_action = QWidgetAction(self)
        protein_ph_spinbox_action.setDefaultWidget(spinbox_widget)
        dock_utilities_menu.addAction(protein_ph_spinbox_action)
        
        fpocket_web_action = QAction('FPocketWeb', self)
        dock_utilities_menu.addAction(fpocket_web_action)
        fpocket_web_action.triggered.connect(self.open_fpocket_web)
        
        open_browser_action = QAction('Web Browser', self)
        general_utilities_menu.addAction(open_browser_action)
        open_browser_action.triggered.connect(self.open_browser)
        
        dark_light_action = QAction('Dark/Light Mode', self)
        general_utilities_menu.addAction(dark_light_action)
        dark_light_action.triggered.connect(self.change_dark_light_mode)
        
        hide_tabbar_action = QAction('Hide/Show Tabs', self)
        general_utilities_menu.addAction(hide_tabbar_action)
        hide_tabbar_action.triggered.connect(lambda: self.toggle_sidebar_button.click())
        
        file_split_action = QAction('Split Directories', self)
        general_utilities_menu.addAction(file_split_action)
        file_split_action.triggered.connect(self.open_directory_splitter)
        
        file_combine_action = QAction('Combine Directories', self)
        general_utilities_menu.addAction(file_combine_action)
        file_combine_action.triggered.connect(self.open_directory_combiner)
        
        fp_setting_action = QAction('Fingerprint / Similarity Settings', self)
        general_utilities_menu.addAction(fp_setting_action)
        fp_setting_action.triggered.connect(self.change_fp_setting_dialog)
        
        db_search_action = QAction('Search Databases', self)
        convert_utilities_menu.addAction(db_search_action)
        db_search_action.triggered.connect(self.open_db_dialog)
        
        chemdraw_action = QAction('Draw Chemicals', self)
        convert_utilities_menu.addAction(chemdraw_action )
        chemdraw_action.triggered.connect(self.open_chemdraw_dialog)
        
        sdf_id_action = QAction('SDF Props as Name', self)
        convert_utilities_menu.addAction(sdf_id_action)
        sdf_id_action.triggered.connect(self.modify_sdf_id_name)
        
        fix_sdf_action = QAction('Fix UTF-8 Formatting', self)
        convert_utilities_menu.addAction(fix_sdf_action)
        fix_sdf_action.triggered.connect(self.fix_chemstr_format)
        
        dpi_spinbox_widget = QWidget()
        dpi_spinbox_layout = QHBoxLayout()
        dpi_spinbox_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        dpi_spinbox_layout.setContentsMargins(10, 0, 0, 0)
        dpi_spinbox_widget.setLayout(dpi_spinbox_layout)
        dpi_spinbox_label = QLabel('PNG DPI:')
        self.png_dpi_spinbox = QSpinBox()
        self.png_dpi_spinbox.setRange(1, 100_000)
        self.png_dpi_spinbox.setValue(1000)
        self.png_dpi_spinbox.setSingleStep(10)
        dpi_spinbox_layout.addWidget(dpi_spinbox_label)
        dpi_spinbox_layout.addWidget(self.png_dpi_spinbox)
        png_dpi_spinbox_action = QWidgetAction(self)
        png_dpi_spinbox_action.setDefaultWidget(dpi_spinbox_widget)
        convert_utilities_menu.addAction(png_dpi_spinbox_action)
        self.png_dpi_spinbox.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        
        size_spinbox_widget = QWidget()
        size_spinbox_layout = QHBoxLayout()
        size_spinbox_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        size_spinbox_layout.setContentsMargins(10, 0, 0, 0)
        size_spinbox_widget.setLayout(size_spinbox_layout)
        size_spinbox_label = QLabel('PNG size:')
        self.png_size_spinbox = QSpinBox()
        self.png_size_spinbox.setRange(1, 100_000)
        self.png_size_spinbox.setValue(300)
        self.png_size_spinbox.setSingleStep(10)
        size_spinbox_layout.addWidget(size_spinbox_label)
        size_spinbox_layout.addWidget(self.png_size_spinbox)
        png_size_spinbox_action = QWidgetAction(self)
        png_size_spinbox_action.setDefaultWidget(size_spinbox_widget)
        convert_utilities_menu.addAction(png_size_spinbox_action)
        self.png_size_spinbox.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        
        db_name_url_action = QAction('Database Name and URL', self)
        shopper_utilities_menu.addAction(db_name_url_action)
        db_name_url_action.triggered.connect(self.modify_dbname_to_url_map)
        
        open_manual_action = QAction('Manual', self)
        help_utilities_menu.addAction(open_manual_action)
        open_manual_action.triggered.connect(self.open_manual_browser)
        self.doc_dialog = None
        
        for pro in self.webserver_map:
            if pro == 'RunPod':
                continue
            program_menu = QMenu(pro)
            colab_action = QAction('Google Colab', self)
            hf_action = QAction('Hugging Face Spaces', self)
            colab_action.triggered.connect(lambda _, x=pro: self.open_web_docking_programs(x, 'colab'))
            hf_action.triggered.connect(lambda _, x=pro: self.open_web_docking_programs(x, 'hf'))
            program_menu.addActions([colab_action, hf_action])
            dock_utilities_menu.addMenu(program_menu)
        
        runpod_action = QAction('RunPod', self)
        runpod_action.triggered.connect(lambda _: self.open_web_docking_programs('RunPod', 'url'))
        dock_utilities_menu.addAction(runpod_action)
    
    def setup_convert_widget(self):
        overall_widget = QWidget()
        overall_layout = QVBoxLayout()
        inp_layout = QHBoxLayout()
        inp_btn_layout = QVBoxLayout()
        out_dir_layout = QHBoxLayout()
        btn_layout = QHBoxLayout()
        progressbar_layout = QHBoxLayout()
        basic_setting_layout = QHBoxLayout()
        basic_setting_widget = QWidget()
        
        self.input_file_label = QLabel('<b>Select Input Files:</b>', self)
        overall_layout.addWidget(self.input_file_label)
        
        self.input_files_list = InputFileDirListWidget(self, self.available_formats)
        self.input_files_list.setMaximumHeight(110)
        self.input_files_list.currCountChanged.connect(self.check_conversion)
        self.input_files_list.currNameChanged.connect(self.check_if_db_id_changed)
        self.input_files_list.currNameRemoved.connect(self.check_if_db_id_removed)
        self.input_files_list.signalLineName.connect(self.check_if_file_pth_or_dbname_existed)
        inp_layout.addWidget(self.input_files_list)
        
        self.convert_input_file_button = QPushButton('Browse Files', self)
        self.convert_input_file_button.clicked.connect(self.select_convert_input_file)
        self.convert_input_file_button.setMinimumWidth(100)
        inp_btn_layout.addWidget(self.convert_input_file_button)
        self.convert_input_dir_button = QPushButton('Browse Dir.', self)
        self.convert_input_dir_button.setMinimumWidth(100)
        self.convert_input_dir_button.clicked.connect(self.select_convert_input_dir)
        inp_btn_layout.addWidget(self.convert_input_dir_button)
        self.read_table_file_button = QPushButton('Read Table', self)
        self.read_table_file_button.setMinimumWidth(100)
        self.read_table_file_button.clicked.connect(self.read_csv_tsv_file)
        inp_btn_layout.addWidget(self.read_table_file_button, alignment=Qt.AlignmentFlag.AlignHCenter)
        
        inp_layout.addLayout(inp_btn_layout)
        overall_layout.addLayout(inp_layout)
        
        self.convert_output_dir_label = QLabel('<b>Select Output Directory / File:</b>', self)
        self.convert_default_fontsize = 15
        overall_layout.addWidget(self.convert_output_dir_label)
        
        self.convert_output_dir_file_line = OutputDirLineEdit(self)
        self.convert_output_dir_file_line.textChanged.connect(self.check_conversion)
        self.convert_output_dir_file_line.textChanged.connect(self.update_dock_input_lineedit)
        out_dir_layout.addWidget(self.convert_output_dir_file_line)
        
        self.convert_output_extension_combo = QComboBox(self)
        self.convert_output_extension_combo.addItems(['.pdbqt', '.sdf', '.mol', '.mol2', '.xyz', '.smi', '.csv (DiffDock)', '.png', '.mddb'])
        self.convert_output_extension_combo.setCurrentText('.pdbqt')
        self.convert_output_extension_combo.setMinimumWidth(150)
        self.convert_output_extension_combo.currentTextChanged.connect(self.update_single_file_checkbox_and_add_h)
        out_dir_layout.addWidget(self.convert_output_extension_combo)
        
        self.single_file_checkbox = QCheckBox(self)
        self.single_file_checkbox.setText('Single File')
        self.single_file_checkbox.setToolTip('Save .sdf/.smi to a single file.')
        self.single_file_checkbox.setEnabled(False)
        self.single_file_checkbox.setChecked(False)
        self.single_file_checkbox.checkStateChanged.connect(self.check_conversion)
        out_dir_layout.addWidget(self.single_file_checkbox)
        
        self.convert_output_dir_button = QPushButton('Browse', self)
        self.convert_output_dir_button.clicked.connect(self.select_convert_output_directory)
        out_dir_layout.addWidget(self.convert_output_dir_button, alignment=Qt.AlignmentFlag.AlignRight)
        
        overall_layout.addLayout(out_dir_layout)
        
        self.filter_dialog_button = QPushButton('Filters', self)
        self.filter_dialog_button.clicked.connect(self.show_filter_dialog)
        self.add_h_checkbox = QCheckBox(self)
        self.add_h_checkbox.setText('Add H')
        self.add_h_checkbox.setStyleSheet('QCheckBox { font-weight: bold; }')
        self.add_h_checkbox.setChecked(True)
        self.add_h_checkbox.setDisabled(True)   # default option is .pdbqt, so set to disabled
        self.add_h_checkbox.setToolTip('.pdbqt file must add H')
        self.desalt_checkbox = QCheckBox(self)
        self.desalt_checkbox.setText('Desalt')
        self.desalt_checkbox.setStyleSheet('QCheckBox { font-weight: bold; }')
        self.desalt_checkbox.setChecked(True)
        self.desalt_checkbox.setToolTip('Recommend enabling it if molecules are to be used for docking')
        self.retain_3d = QCheckBox(self)
        self.retain_3d.setText('Retain 3D')
        self.retain_3d.setStyleSheet('QCheckBox { font-weight: bold; }')
        self.retain_3d.setChecked(True)
        self.retain_3d.setToolTip('Keep the original 3D coordinates for 3D input')
        self.flexible_macrocycle_ckbox = QCheckBox('Flexible Macrocycle', self)
        self.flexible_macrocycle_ckbox.setStyleSheet('QCheckBox { font-weight: bold; }')
        self.flexible_macrocycle_ckbox.setChecked(True)
        self.flexible_macrocycle_ckbox.setToolTip('Flexible macrocycle docking (ONLY for AutoDock VINA!)')
        
        timeout_layout = QHBoxLayout()
        timeout_layout.setContentsMargins(0, 0, 0, 0)
        timeout_widget = QWidget()
        timeout_widget.setLayout(timeout_layout)
        timeout_label = QLabel('<b>Timeout :</b>')
        self.timeout_spinbox = QSpinBox(self)
        self.timeout_spinbox.setRange(0, 999_999)
        self.timeout_spinbox.setValue(0)
        timeout_layout.addWidget(timeout_label)
        timeout_layout.addWidget(self.timeout_spinbox)
        
        self.convert_fontsize_label = QLabel('<b>Font Size :</b>', self)
        self.convert_fontsize_spinbox = QSpinBox(self)
        self.convert_fontsize_spinbox.setRange(1, 99)
        self.convert_fontsize_spinbox.setValue(self.convert_default_fontsize)
        self.convert_fontsize_spinbox.setSingleStep(1)
        self.convert_fontsize_spinbox.valueChanged.connect(self.change_convert_log_font_size)
        self.convert_fontsize_spinbox.setMinimumWidth(60)
        
        separator_line = QFrame()
        separator_line.setFrameShape(QFrame.Shape.VLine)
        
        basic_setting_layout.setSpacing(10)
        basic_setting_layout.setContentsMargins(0, 0, 0, 0)
        basic_setting_layout.addWidget(self.filter_dialog_button)
        basic_setting_layout.addItem(QSpacerItem(20, 0))
        basic_setting_layout.addWidget(self.add_h_checkbox)
        basic_setting_layout.addItem(QSpacerItem(20, 0))
        basic_setting_layout.addWidget(self.desalt_checkbox)
        basic_setting_layout.addItem(QSpacerItem(20, 0))
        basic_setting_layout.addWidget(self.retain_3d)
        basic_setting_layout.addItem(QSpacerItem(20, 0))
        basic_setting_layout.addWidget(self.flexible_macrocycle_ckbox)
        basic_setting_layout.addItem(QSpacerItem(20, 0))
        basic_setting_layout.addWidget(timeout_widget)
        basic_setting_layout.addItem(QSpacerItem(25, 0))
        basic_setting_layout.addWidget(separator_line)
        basic_setting_layout.addItem(QSpacerItem(25, 0))
        basic_setting_layout.addWidget(self.convert_fontsize_label)
        basic_setting_layout.addWidget(self.convert_fontsize_spinbox)
        basic_setting_widget.setLayout(basic_setting_layout)
        btn_layout.addWidget(basic_setting_widget, alignment=Qt.AlignmentFlag.AlignLeft)
        
        right_button_layout = QHBoxLayout()
        right_button_widget = QWidget()
        right_button_widget.setLayout(right_button_layout)
        right_button_layout.setContentsMargins(0, 0, 0, 0)
        self.save_convert_log_text_button = QPushButton('Save Log', self)
        self.save_convert_log_text_button.clicked.connect(self.save_convert_log_file_dialog)
        right_button_layout.addWidget(self.save_convert_log_text_button)
        self.convert_files_button = QPushButton('Convert', self)
        self.convert_files_button.clicked.connect(self.convert_start_reading)
        self.convert_files_button.setDisabled(True)
        right_button_layout.addWidget(self.convert_files_button)
        btn_layout.addWidget(right_button_widget, alignment=Qt.AlignmentFlag.AlignRight)
        overall_layout.addLayout(btn_layout)
        
        ### progress bar layout ###
        self.convert_progress = QProgressBar(self)
        progressbar_layout.addWidget(self.convert_progress)
        
        self.convert_progress_label = QLabel(self)
        self.convert_progress_label.setStyleSheet('font-family: "Courier New", Courier, monospace;')
        self.convert_progress_label.setText('?/?')
        progressbar_layout.addWidget(self.convert_progress_label)
        
        self.convert_eta_label = QLabel(self)
        self.convert_eta_label.setStyleSheet('font-family: "Courier New", Courier, monospace;')
        self.convert_eta_label.setText('[00:00<00:00, ??.??it/s]')
        progressbar_layout.addWidget(self.convert_eta_label)
        
        overall_layout.addLayout(progressbar_layout)
        
        ### dark/light mode setup
        self.curr_display_mode = theme
        self.dark_light_swap_button = QPushButton(self)
        self.light_icon = QIcon(os.path.join(self.curr_dir, 'icon', 'to_light.png'))
        self.dark_icon = QIcon(os.path.join(self.curr_dir, 'icon', 'to_dark.png'))
        self.dark_light_swap_button.setIcon(self.light_icon) if self.curr_display_mode == 'dark' else self.dark_light_swap_button.setIcon(self.dark_icon)
        self.dark_light_swap_button.clicked.connect(self.change_dark_light_mode)
        self.dark_light_swap_button.setToolTip('To light mode') if self.curr_display_mode == 'dark' else self.dark_light_swap_button.setToolTip('To dark mode')
        self.dark_light_swap_button.setStyleSheet("QPushButton {border-style: outset; border-width: 0px;}")
        
        ### conversion log layout ###
        self.convert_textedit = LogTextEdit(self, self.curr_display_mode, self.convert_default_fontsize)
        self.convert_textedit.setReadOnly(True)
        self.convert_textedit.append(f'<p align=\"center\"><span style="color:Gray;">Conversion Log...</span></p>')
        overall_layout.addWidget(self.convert_textedit)
        
        overall_widget.setLayout(overall_layout)
        
        self.start_conversion = False
        
        self.rdkit_filters_names = ['PAINS', 'BRENK', 'NIH', 'ZINC', 'CHEMBL_BMS',
                                    'CHEMBL_Dundee', 'CHEMBL_Glaxo', 'CHEMBL_Inpharmatica',
                                    'CHEMBL_LINT', 'CHEMBL_MLSMR', 'CHEMBL_SureChEMBL']
        self.chem_filter_dict = {'mw'  : [()], 'hbd' : [()], 'hba' : [()], 'logp': [()], 'tpsa': [()],
                                 'rb'  : [()], 'nor' : [()], 'fc'  : [()], 'nha' : [()], 'mr'  : [()],
                                 'na'  : [()], 'QED' : [()]}
        self.chem_filter_bool = {'mw'  : False, 'hbd' : False, 'hba' : False, 'logp': False, 'tpsa': False,
                                 'rb'  : False, 'nor' : False, 'fc'  : False, 'nha' : False, 'mr'  : False,
                                 'na'  : False, 'QED' : False, 'partial_filter_threshold': 0, 'match_type': 'Include'}
        self.rdkit_filter_dict = {f: False for f in self.rdkit_filters_names}
        self.rdkit_filter_dict['partial_filter_threshold'] = 0
        self.rdkit_filter_dict['match_type'] = 'Exclude'
        self.chem_prop_to_full_name_map = {'mw'  : 'Molecular Weight        ', 'hbd' : 'Hydrogen Bond Donors    ',
                                           'hba' : 'Hydrogen Bond Acceptors ', 'logp': 'LogP                    ',
                                           'tpsa': 'Topological Polar Surface Area', 'rb'  : 'Rotatable Bonds         ',
                                           'nor' : 'Number of Rings         ', 'fc'  : 'Formal Charge           ',
                                           'nha' : 'Number of Heavy Atoms   ', 'mr'  : 'Molar Refractivity      ',
                                           'na'  : 'Number of Atoms         ', 'QED' : 'Quant. Est. of Drug-likeness'}
        self.sampling_setting_dict = {'ckbox': False, 'random': False, 'maxmin': False, 'seed': '0', 'count': '100'}
        self.similarity_db_dict = {'db_pth' : [], 'sim_type': 'Include', 'db_sim' : (0.5, 1.0)}
        
        self.table_file_params = {}
        self.db_small_molecule = {}
        self.now_converting = False
        self.now_picking = False
        return overall_widget
    
    def setup_dock_widget(self):
        overall_layout = QVBoxLayout()
        input_structure_layout = QGridLayout()
        upper_layout = QVBoxLayout()
        self.online_id_layout = QGridLayout()
        self.input_ligand_layout = QHBoxLayout()
        self.out_dir_layout = QHBoxLayout()
        self.btn_layout = QHBoxLayout()
        self.progressbar_layout = QHBoxLayout()
        self.basic_setting_layout = QHBoxLayout()
        self.basic_setting_widget = QWidget()
        
        self.input_file_label = QLabel('<b>Load Protein Structure :</b>', self)
        upper_layout.addWidget(self.input_file_label)
        
        local_radio_button = QRadioButton('Local', self)
        local_radio_button.setChecked(True)
        local_radio_button.clicked.connect(lambda _, x='Local': self.change_protein_input_type(x))
        self.protein_radio_button_group = QButtonGroup()
        self.protein_radio_button_group.addButton(local_radio_button)
        
        self.local_file_line = DropFileLineEdit(self, ('.pdb', '.pdbqt', '.cif', '.mds'))
        self.local_file_line.textChanged.connect(self.check_protein_pth)
        self.local_file_button = QPushButton('Browse', self)
        self.local_file_button.clicked.connect(self.select_dock_input_file)
        self.local_file_line.setFocus()
        self.protein_convert_button = QPushButton('Load', self)
        self.protein_convert_button.clicked.connect(self.load_file_from_local)
        self.protein_convert_button.setDisabled(True)
        
        input_structure_layout.setContentsMargins(0, 0, 0, 0)
        input_structure_layout.addWidget(local_radio_button, 0, 0)
        input_structure_layout.addWidget(self.local_file_line, 0, 1)
        input_structure_layout.addWidget(self.local_file_button, 0, 2)
        input_structure_layout.addWidget(self.protein_convert_button, 0, 3)
        
        self.input_id_dict = {'Local': {'LineEdit'      : self.local_file_line,
                                        'Browse_Button' : self.local_file_button,
                                        'Load_Button'   : self.protein_convert_button,}}
        name_regex_dict = {'PDB'        : {'regex'      : '[1-9][a-zA-Z0-9]{3}',
                                           'placeholder': 'PDB ID...'},
                           'AF Database': {'regex'      : '[opqOPQ][0-9][a-zA-Z0-9]{3}[0-9]|[a-nA-Nr-zR-Z][0-9]([a-zA-Z][a-zA-Z0-9]{2}[0-9]){1,2}',
                                           'placeholder': 'UniProt ID...'}}
        for row, name in enumerate(name_regex_dict, 1):
            label = QRadioButton(name)
            label.clicked.connect(lambda _, x=name: self.change_protein_input_type(x))
            self.protein_radio_button_group.addButton(label)
            tmp_layout = QHBoxLayout()
            lineedit = QLineEdit()
            lineedit.setMaximumWidth(150)
            lineedit.setPlaceholderText(name_regex_dict[name]['placeholder'])
            regex_validator = QRegularExpressionValidator(name_regex_dict[name]['regex'])
            lineedit.setValidator(regex_validator)
            lineedit.setDisabled(True)
            lineedit.textChanged.connect(lambda _, x=name: self.force_upper_case(x))
            status_label = QLabel('<b>Status :</b>')
            status_label.setDisabled(True)
            btn = QPushButton('Load')
            btn.setDisabled(True)
            btn.clicked.connect(lambda _, x=name: self.load_from_online_db(x))
            tmp_layout.addWidget(lineedit)
            tmp_layout.addWidget(status_label)
            tmp_layout.setContentsMargins(0, 0, 0, 0)
            input_structure_layout.addWidget(label, row, 0)
            input_structure_layout.addLayout(tmp_layout, row, 1, 1, 2, Qt.AlignmentFlag.AlignLeft)
            input_structure_layout.addWidget(btn, row, 3)
            self.input_id_dict[name] = {'LineEdit'    : lineedit,
                                        'Status_Label': status_label,
                                        'Load_Button' : btn,}
        
        upper_layout.addLayout(input_structure_layout)
        
        self.input_ligand_label = QLabel('<b>Select Ligand Directory :</b>', self)
        upper_layout.addWidget(self.input_ligand_label)
        
        self.input_ligand_line = DropDirLineEdit(self)
        self.input_ligand_line.textChanged.connect(self.check_if_enable_docking)
        self.input_ligand_button = QPushButton('Browse', self)
        self.input_ligand_button.clicked.connect(self.select_dock_ligand_directory)
        self.input_ligand_layout.addWidget(self.input_ligand_line)
        self.input_ligand_layout.addWidget(self.input_ligand_button, alignment=Qt.AlignmentFlag.AlignRight)
        upper_layout.addLayout(self.input_ligand_layout)
        
        self.output_folder_label = QLabel('<b>Select Output Directory :</b>', self)
        self.default_fontsize = 15
        upper_layout.addWidget(self.output_folder_label)
        
        self.output_directory_line = DropDirLineEdit(self)
        self.output_directory_line.textChanged.connect(self.check_if_enable_docking)
        self.output_directory_button = QPushButton('Browse', self)
        self.output_directory_button.clicked.connect(self.select_dock_output_directory)
        self.out_dir_layout.addWidget(self.output_directory_line)
        self.out_dir_layout.addWidget(self.output_directory_button, alignment=Qt.AlignmentFlag.AlignRight)
        
        upper_layout.addLayout(self.out_dir_layout)
        
        self.setting_dialog_button = QPushButton('Setting', self)
        self.setting_dialog_button.clicked.connect(self.show_docking_setting_dialog)
        self.fontsize_label = QLabel('<b>Font Size :</b>', self)
        self.fontsize_spinbox = QSpinBox(self)
        self.fontsize_spinbox.setRange(1, 99)
        self.fontsize_spinbox.setValue(self.default_fontsize)
        self.fontsize_spinbox.setSingleStep(1)
        self.fontsize_spinbox.valueChanged.connect(self.change_log_font_size)
        self.fontsize_spinbox.setMinimumWidth(70)
        self.docking_concurrency_label = QLabel('<b>Concurrent :</b>', self)
        self.docking_concurrency_spinbox = QSpinBox(self)
        cpu_num = cpu_count()
        self.docking_concurrency_spinbox.setRange(1, cpu_num)
        self.docking_concurrency_spinbox.setToolTip(f'Max: {cpu_num}')
        self.docking_concurrency_spinbox.setValue(max(1, cpu_num - 2))
        self.docking_concurrency_spinbox.setMinimumWidth(70)
        self.basic_setting_layout.setSpacing(10)
        self.basic_setting_layout.setContentsMargins(0, 0, 0, 0)
        self.basic_setting_layout.addWidget(self.setting_dialog_button)
        self.basic_setting_layout.addItem(QSpacerItem(30, 0))
        self.basic_setting_layout.addWidget(self.fontsize_label)
        self.basic_setting_layout.addWidget(self.fontsize_spinbox)
        self.basic_setting_layout.addItem(QSpacerItem(30, 0))
        self.basic_setting_layout.addWidget(self.docking_concurrency_label)
        self.basic_setting_layout.addWidget(self.docking_concurrency_spinbox)
        
        radio_layout = QHBoxLayout()
        radio_layout.setContentsMargins(0, 0, 0, 0)
        radio_widget = QWidget()
        radio_widget.setLayout(radio_layout)
        format_label = QLabel('<b>Format :</b>')
        radio_layout.addWidget(format_label)
        self.protein_format_radios = {}
        self.protein_format_button_group = QButtonGroup()
        for format in ['pdbqt', 'pdb']:
            radio = QRadioButton(format)
            radio.clicked.connect(lambda _, x=format: self.change_protein_format(x))
            self.protein_format_button_group.addButton(radio)
            radio.setStyleSheet('QRadioButton { font-weight: bold; }')
            self.protein_format_radios[format] = radio
            radio_layout.addWidget(radio)
            radio_layout.addSpacerItem(QSpacerItem(15, 0))
        self.protein_format_radios['pdbqt'].setChecked(True)
        
        self.basic_setting_layout.addSpacerItem(QSpacerItem(30, 0))
        self.basic_setting_layout.addWidget(radio_widget)
        self.basic_setting_layout.addSpacerItem(QSpacerItem(30, 0))
        # self.basic_setting_layout.addWidget(self.fill_protein_gap_ckbox)
        self.basic_setting_widget.setLayout(self.basic_setting_layout)
        self.btn_layout.addWidget(self.basic_setting_widget, alignment=Qt.AlignmentFlag.AlignLeft)
        
        small_widget = QWidget()
        small_layout = QHBoxLayout()
        small_widget.setLayout(small_layout)
        small_layout.setContentsMargins(0, 0, 0, 0)
        
        self.clear_docking_settings_button = QPushButton('Clear', self, clicked=self.clear_dock_data)
        small_layout.addWidget(self.clear_docking_settings_button)
        
        self.save_log_text_button = QPushButton('Save Log', self)
        self.save_log_text_button.clicked.connect(self.save_log_file_dialog)
        self.save_log_text_button.setDisabled(True)
        small_layout.addWidget(self.save_log_text_button)
        
        docking_submenu = QMenu(self)
        self.docking_action_dict = {}
        self.supported_docking_programs = ['AutoDock VINA', 'smina', 'qvina2', 'qvinaw', 'LeDock']
        for docking_program in self.supported_docking_programs + ['Refinement']:
            action = QAction(docking_program, self)
            action.triggered.connect(lambda _, x=docking_program: self.start_process_docking(x))
            docking_submenu.addAction(action)
            if docking_program == 'LeDock' and self.ledock_exec is None:
                action.setToolTip('LeDock is not supported on Windows')
            action.setDisabled(True)
            self.docking_action_dict[docking_program] = action
        self.start_docking_button = QPushButton('Dock', self)
        self.start_docking_button.setMenu(docking_submenu)
        small_layout.addWidget(self.start_docking_button)
        self.btn_layout.addWidget(small_widget, alignment=Qt.AlignmentFlag.AlignRight)
        
        upper_layout.addLayout(self.btn_layout)
        
        ### progress bar layout ###
        self.progress = QProgressBar(self)
        self.progressbar_layout.addWidget(self.progress)
        
        self.progress_label = QLabel(self)
        self.progress_label.setStyleSheet('font-family: "Courier New", Courier, monospace;')
        self.progress_label.setText('?/?')
        self.progressbar_layout.addWidget(self.progress_label)
        
        self.eta_label = QLabel(self)
        self.eta_label.setStyleSheet('font-family: "Courier New", Courier, monospace;')
        self.eta_label.setText('[00:00<00:00, ??.??it/s]')
        self.progressbar_layout.addWidget(self.eta_label)
        
        upper_layout.addLayout(self.progressbar_layout)
        
        ### docking log layout ###
        docking_splitter = QSplitter(Qt.Orientation.Horizontal)
        docking_splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        log_tableview_widget = QWidget()
        log_tableview_layout = QVBoxLayout()
        log_tableview_widget.setLayout(log_tableview_layout)
        log_search_layout = QHBoxLayout()
        log_search_layout.setContentsMargins(0, 0, 0, 0)
        log_search_label = QLabel('<b>Filter: <b/>')
        self.log_search_lineedit = QLineEdit()
        self.log_search_lineedit.setPlaceholderText('Searching term...')
        self.log_search_lineedit.textChanged.connect(self.search_log_tableview_text)
        self.log_search_combobox = QComboBox()
        log_search_col_label = QLabel('<b>Col: </b>')
        self.log_search_combobox.addItems(['Name', 'Status'])
        self.log_search_combobox.setMinimumWidth(150)
        self.log_search_combobox.currentTextChanged.connect(self.search_log_tableview_text)
        log_search_layout.addWidget(log_search_label)
        log_search_layout.addWidget(self.log_search_lineedit)
        log_search_layout.addWidget(log_search_col_label)
        log_search_layout.addWidget(self.log_search_combobox, alignment=Qt.AlignmentFlag.AlignRight)
        self.log_tableview = DockProgressTableView(self)
        log_tableview_layout.addLayout(log_search_layout)
        log_tableview_layout.addWidget(self.log_tableview)
        
        self.log_textedit = QTextEdit(self)
        self.log_textedit.setReadOnly(True)
        self.log_textedit.setStyleSheet(f'font-family: "Courier New", Courier, monospace; font-size:{int(self.default_fontsize)}px;')
        self.log_textedit.append(f'<p align=\"center\"><span style="color:Gray;">Protein / Docking Log...</span></p>')
        docking_splitter.addWidget(self.log_textedit)
        docking_splitter.addWidget(log_tableview_widget)
        docking_splitter.setStretchFactor(0, 15)
        docking_splitter.setStretchFactor(1, 5)
        
        upper_layout.setContentsMargins(0, 0, 0, 0)
        overall_layout.addLayout(upper_layout, 3)
        overall_layout.addWidget(docking_splitter, 7)
        
        self.text_extractor = DockingTextExtractor()
        self.text_extractor.update_text.connect(self.update_docking_progress_text)
        self.text_extractor.update_progress.connect(self.update_progress_bar)
        self.text_extractor.docking_complete.connect(self.docking_done_func)
        self.text_extractor.docking_cancel.connect(self.cancel_docking_func)
        
        docking_widget = QWidget()
        docking_widget.setLayout(overall_layout)
        self.is_docking = False
        return docking_widget
    
    def setup_table_widget(self):
        overall_layout = QVBoxLayout()
        docked_dir_layout = QHBoxLayout()
        table_progressbar_layout = QHBoxLayout()
        table_layout = QVBoxLayout()
        btn_layout = QHBoxLayout()
        right_btn_layout = QHBoxLayout()
        right_btn_widget = QWidget()
        
        ### Docked Directory Selection ###
        docked_dir_label = QLabel('<b>Select Docked Directory :</b>')
        self.docked_dir_line = DropDirLineEdit(self)
        self.docked_dir_line.textChanged.connect(self.check_if_enable_viewing)
        docked_dir_button = QPushButton('Browse')
        docked_dir_button.clicked.connect(self.select_docked_directory)
        self.show_dir_table_button = QPushButton('View')
        self.show_dir_table_button.clicked.connect(self.view_docking_result)
        self.show_dir_table_button.setDisabled(True)
        self.show_dir_table_button.setMinimumWidth(70)
        
        overall_layout.addWidget(docked_dir_label)
        docked_dir_layout.addWidget(self.docked_dir_line)
        docked_dir_layout.addWidget(docked_dir_button, alignment=Qt.AlignmentFlag.AlignRight)
        docked_dir_layout.addWidget(self.show_dir_table_button, alignment=Qt.AlignmentFlag.AlignRight)
        overall_layout.addLayout(docked_dir_layout)
        
        ### Progress bar For File Reader ###
        self.table_progress = QProgressBar(self)
        table_progressbar_layout.addWidget(self.table_progress)
        
        self.table_progress_label = QLabel(self)
        self.table_progress_label.setStyleSheet('font-family: "Courier New", Courier, monospace;')
        self.table_progress_label.setText('?/?')
        table_progressbar_layout.addWidget(self.table_progress_label)
        
        self.table_eta_label = QLabel(self)
        self.table_eta_label.setStyleSheet('font-family: "Courier New", Courier, monospace;')
        self.table_eta_label.setText('[00:00<00:00, ??.??it/s]')
        table_progressbar_layout.addWidget(self.table_eta_label)
        overall_layout.addLayout(table_progressbar_layout)
        
        ### Result Table Visualization ###
        result_table_setting_layout = QHBoxLayout()
        result_table_setting_widget = QWidget()
        
        result_label = QLabel('<b>Result Table</b>')
        self.filter_table_button = QPushButton('Filter')
        self.filter_table_button.clicked.connect(self.apply_table_filters)
        self.filter_table_button.setMinimumWidth(60)
        self.analyze_fragment_checkbox = QCheckBox(self)
        self.analyze_fragment_checkbox.setText('Analyze Fragments')
        self.analyze_fragment_checkbox.setChecked(False)
        self.calc_chemprop_checkbox = QCheckBox(self)
        self.calc_chemprop_checkbox.setText('Calculate Properties')
        self.calc_chemprop_checkbox.setChecked(True)
        self.show_only_rank_1_checkbox = QCheckBox(self)
        self.show_only_rank_1_checkbox.setText('Rank 1 Only')
        self.show_only_rank_1_checkbox.setChecked(True)
        
        ### Button Layout ###
        btn_widget = QWidget()
        
        self.combine_filtered_structure_action = QAction(text='Filtered Structure')
        # self.combine_filtered_pdbqt_action = QAction(text='pdbqt')    # depracated
        # self.combine_filtered_pdbqt_action.setToolTip('SINGLE .pdbqt file.')
        self.combine_filtered_sdf_action = QAction('sdf')
        self.combine_filtered_sdf_action.setToolTip('INDIVIDUAL .sdf files.')
        # self.mmgbsa_format_action = QAction('MM-GBSA Format')
        # self.mmgbsa_format_action.setToolTip('Save to format ready for performing MM-GB(PB)SA')
        self.save_filtered_table_action = QAction('Filtered Table')
        self.save_full_table_action = QAction('Full Table')
        
        openmm_submenu = QMenu(self)
        self.openmm_format_action = QAction('Refine Format')
        self.openmm_format_action.setToolTip('Save to format ready for refinement')
        self.openmm_format_dir = QAction('Directory')
        self.openmm_format_zip = QAction('ZIP')
        openmm_submenu.addActions([self.openmm_format_dir,
                                   self.openmm_format_zip])
        self.openmm_format_action.setMenu(openmm_submenu)
        
        # self.combine_filtered_pdbqt_action.triggered.connect(self.save_filtered_pdbqt)
        self.combine_filtered_sdf_action.triggered.connect(self.save_filtered_sdf)
        # self.mmgbsa_format_action.triggered.connect(self.save_filtered_mmgbsa)
        # self.openmm_format_action.triggered.connect(self.save_for_openmm_minimize)
        self.openmm_format_dir.triggered.connect(lambda x, type='dir': self.save_for_openmm_minimize(type))
        self.openmm_format_zip.triggered.connect(lambda x, type='zip': self.save_for_openmm_minimize(type))
        self.save_filtered_table_action.triggered.connect(self.save_filtered_df)
        self.save_full_table_action.triggered.connect(self.save_full_df)
        
        save_structure_submenu = QMenu(self)
        save_structure_submenu.addActions([self.combine_filtered_sdf_action,
                                        #    self.mmgbsa_format_action,
                                           self.openmm_format_action,])
        self.combine_filtered_structure_action.setMenu(save_structure_submenu)
        
        save_option_menu = QMenu(self)
        save_option_menu.addActions([self.combine_filtered_structure_action,
                                     self.save_filtered_table_action,
                                     self.save_full_table_action])
        self.save_options_button = QPushButton('Save', self)
        self.save_options_button.setMenu(save_option_menu)
        self.save_options_button.setDisabled(True)
        
        # supplier_type_menu = QMenu(self)
        # self.zinc_supplier_action = QAction('ZINC22')
        # self.pubchem_supplier_action = QAction('PubChem')
        # self.zinc_supplier_action.triggered.connect(lambda _, x='ZINC': self.show_search_catalog_dialog(x))
        # self.pubchem_supplier_action.triggered.connect(lambda _, x='PubChem': self.show_search_catalog_dialog(x))
        # supplier_type_menu.addActions([self.zinc_supplier_action,
        #                                self.pubchem_supplier_action])
        # self.search_purchasability_button = QPushButton('Search Supplier')
        # self.search_purchasability_button.setMenu(supplier_type_menu)
        # self.search_purchasability_button.clicked.connect(self.show_search_catalog_dialog)
        
        self.clear_table_results_button = QPushButton('Clear')
        self.clear_table_results_button.clicked.connect(self.clear_table_data_and_other)
        
        right_btn_layout.setContentsMargins(0, 0, 0, 0)
        right_btn_layout.addWidget(self.clear_table_results_button)
        # right_btn_layout.addWidget(self.search_purchasability_button)
        right_btn_layout.addWidget(self.save_options_button)
        right_btn_widget.setLayout(right_btn_layout)
        
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.addWidget(right_btn_widget, alignment=Qt.AlignmentFlag.AlignRight)
        btn_widget.setLayout(btn_layout)
        
        self.result_table = DockResultTable(self)
        self.result_table.setup_table()
        self.result_table.updateTreeSignal.connect(self.update_structure_tree_with_filterd_df)
        left_layout = QHBoxLayout()
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(result_label)
        left_layout.addWidget(self.filter_table_button)
        left_layout.addWidget(self.analyze_fragment_checkbox)
        left_layout.addWidget(self.calc_chemprop_checkbox)
        left_layout.addWidget(self.show_only_rank_1_checkbox)
        result_table_setting_layout.addWidget(left_widget, alignment=Qt.AlignmentFlag.AlignLeft)
        result_table_setting_layout.addWidget(btn_widget, alignment=Qt.AlignmentFlag.AlignRight)
        result_table_setting_layout.setContentsMargins(0, 0, 0, 0)
        result_table_setting_layout.setSpacing(30)
        result_table_setting_widget.setLayout(result_table_setting_layout)
        table_layout.addWidget(result_table_setting_widget)
        table_layout.addWidget(self.result_table)
        overall_layout.addLayout(table_layout)
        
        table_widget = QWidget()
        table_widget.setLayout(overall_layout)
        return table_widget
    
    def setup_structure_widget(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        contact_splitter = QSplitter(Qt.Orientation.Vertical)
        
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)
        self.name_tree = FileNameTree(self)
        self.name_tree.checkedSignal.connect(self.add_ligand_to_browser)
        self.name_tree.uncheckedSignal.connect(self.rm_ligand_from_browser)
        self.name_tree.contactSignal.connect(self.update_contacts_of_selected)
        
        self.contact_tabs = ContactTabTables(self)
        contact_splitter.addWidget(self.name_tree)
        contact_splitter.addWidget(self.contact_tabs)
        contact_splitter.setStretchFactor(0, 7)
        contact_splitter.setStretchFactor(1, 3)
        
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        self.update_structure_tree_btn = QPushButton('Update')
        self.update_structure_tree_btn.clicked.connect(self.update_structure_tree_with_filterd_df)
        self.update_structure_tree_btn.setDisabled(True)
        self.save_structure_btn = QPushButton('Save Selected')
        self.save_structure_btn.clicked.connect(self.save_current_selected_structure)
        self.save_structure_btn.setDisabled(True)
        self.reorient_structure_btn = QPushButton('Reorient')
        self.reorient_structure_btn.clicked.connect(self.reorient_embedded_structure)
        self.reorient_structure_btn.setDisabled(True)
        self.show_ligplot_btn = QPushButton('2D LigPlot')
        self.show_ligplot_btn.clicked.connect(self.display_2d_ligplot)
        self.show_ligplot_btn.setDisabled(True)
        self.show_ligplot_btn.setToolTip('(BETA) Contains a lot of bug & is not pretty!')
        btn_layout.addWidget(self.save_structure_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        btn_layout.addWidget(self.reorient_structure_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        btn_layout.addWidget(self.show_ligplot_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        btn_layout.addWidget(self.update_structure_tree_btn, alignment=Qt.AlignmentFlag.AlignRight)
        
        left_layout.addWidget(contact_splitter)
        left_layout.addLayout(btn_layout, 1)
        
        self.structure_browser_widget = ProteinLigandEmbedBrowserWidget(self.curr_display_mode)
        self.structure_browser_widget.signals.contactDone.connect(self.update_contacts_of_selected)
        self.curr_struct_contact_name = None
        
        splitter.addWidget(left_widget)
        splitter.addWidget(self.structure_browser_widget)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        
        return splitter
    
    def setup_shopper_widget(self):
        overall_widget = QWidget()
        overall_layout = QVBoxLayout()
        overall_widget.setLayout(overall_layout)
        
        btn_frame = QFrame()
        btn_frame.setFrameShape(QFrame.Shape.StyledPanel)
        btn_frame.setLineWidth(2)
        btn_layout = QHBoxLayout()
        btn_frame.setLayout(btn_layout)
        
        self.shopper_stacked_widget = SlidingStackedWidget(self, Qt.Orientation.Horizontal, 300)
        shopper_widgets_dict = {'ZINC22'        : ZINCSupplierFinderWidget,
                                'PubChem'       : PubChemSupplierFinderWidget,
                                'Local Database': LocalDatabaseFinderWidget}
        
        for i, (name, widget) in enumerate(shopper_widgets_dict.items()):
            btn = QPushButton(name)
            btn.setStyleSheet('QPushButton { font-weight: bold; }')
            shopper_widget = widget(self, self.curr_display_mode)
            btn.clicked.connect(lambda _, i=i: self.shopper_stacked_widget.moveToIndex(i))
            self.shopper_stacked_widget.addWidget(shopper_widget)
            btn_layout.addWidget(btn)
        btn_layout.addStretch(1)
        
        overall_layout.addWidget(btn_frame, alignment=Qt.AlignmentFlag.AlignTop)
        overall_layout.setSpacing(2)
        overall_layout.addWidget(self.shopper_stacked_widget)
        return overall_widget
    
    def setup_plot_widget(self):
        selection_layout = QHBoxLayout()
        overall_layout = QVBoxLayout()
        overall_widget = QWidget()
        
        ### Selection ###
        self.pio_templates = list(pio.templates.keys())
        
        selection_frame = QFrame()
        selection_frame.setFrameShape(QFrame.Shape.StyledPanel)
        selection_frame.setLineWidth(2)
        selection_frame.setLayout(selection_layout)
        x_layout = QHBoxLayout()
        x_widget = QWidget()
        y_layout = QHBoxLayout()
        y_widget = QWidget()
        color_widget = QWidget()
        color_layout = QHBoxLayout()
        template_layout = QHBoxLayout()
        template_widget = QWidget()
        x_layout.setContentsMargins(0, 0, 0, 0)
        y_layout.setContentsMargins(0, 0, 0, 0)
        color_layout.setContentsMargins(0, 0, 0, 0)
        template_layout.setContentsMargins(0, 0, 0, 0)
        
        x_label = QLabel('<b>X :</b>')
        self.x_combo = QComboBox(self)
        # self.x_combo.addItems(combo_texts)
        self.x_combo.setCurrentText('')
        self.x_combo.setMinimumWidth(220)
        self.x_combo.currentTextChanged.connect(self.plot_and_view_distribution)
        x_layout.addWidget(x_label)
        x_layout.addWidget(self.x_combo)
        x_widget.setLayout(x_layout)
        
        y_label = QLabel('<b>Y :</b>')
        self.y_combo = QComboBox(self)
        # self.y_combo.addItems(combo_texts)
        self.y_combo.setCurrentText('')
        self.y_combo.setMinimumWidth(220)
        self.y_combo.currentTextChanged.connect(self.plot_and_view_distribution)
        y_layout.addWidget(y_label)
        y_layout.addWidget(self.y_combo)
        y_widget.setLayout(y_layout)
        
        color_label = QLabel('<b>Color :</b>')
        self.color_combo = QComboBox(self)
        # self.color_combo.addItems(combo_texts)
        self.color_combo.setCurrentText('')
        self.color_combo.setMinimumWidth(220)
        self.color_combo.currentTextChanged.connect(self.plot_and_view_distribution)
        color_layout.addWidget(color_label)
        color_layout.addWidget(self.color_combo)
        color_widget.setLayout(color_layout)
        
        template_label = QLabel('<b>Template :</b>')
        self.template_combo = QComboBox(self)
        self.template_combo.addItems(self.pio_templates)
        if self.curr_display_mode == 'dark':
            self.template_combo.setCurrentText('plotly_dark')
        else:
            self.template_combo.setCurrentText('plotly')
        self.template_combo.setMinimumWidth(150)
        self.template_combo.currentTextChanged.connect(self.plot_and_view_distribution)
        template_layout.addWidget(template_label)
        template_layout.addWidget(self.template_combo)
        template_widget.setLayout(template_layout)
        
        filter_button = QPushButton('Filter')
        filter_button.clicked.connect(self.show_plot_filter)
        
        self.save_plot_button = QPushButton('Save')
        self.save_plot_button.clicked.connect(self.save_plot_to_html)
        self.save_plot_button.setDisabled(True)
        self.save_plot_button.setToolTip('Save html')
        
        selection_layout.setSpacing(10)
        selection_layout.addWidget(x_widget, alignment=Qt.AlignmentFlag.AlignCenter)
        selection_layout.addWidget(y_widget, alignment=Qt.AlignmentFlag.AlignCenter)
        selection_layout.addWidget(color_widget, alignment=Qt.AlignmentFlag.AlignCenter)
        selection_layout.addWidget(template_widget, alignment=Qt.AlignmentFlag.AlignCenter)
        selection_layout.addWidget(filter_button, alignment=Qt.AlignmentFlag.AlignLeft)
        selection_layout.addWidget(self.save_plot_button, alignment=Qt.AlignmentFlag.AlignRight)
        overall_layout.addWidget(selection_frame, alignment=Qt.AlignmentFlag.AlignTop)
        
        ### Browser ###
        self.browser_plot = PlotViewer()
        if self.curr_display_mode == 'light':
            self.browser_plot.setup_html(self.empty_light_plot_html)
            self.browser_plot.setup_background_color('')
        else:
            self.browser_plot.setup_html(self.empty_dark_plot_html)
            self.browser_plot.setup_background_color('_dark')
        self.browser_plot.showMaximized()
        self.browser_plot.page().profile().downloadRequested.connect(self.save_plot_to_image)
        overall_layout.addWidget(self.browser_plot, 1)
        
        overall_layout.setSpacing(10)
        overall_widget.setLayout(overall_layout)
        
        self.plot_filter_dict = {'Name': [],
                                 'eng' : [()],
                                 'engr': [()],
                                 'olds': [()], 
                                 'cnns': [()],
                                 'cnna': [()],
                                 'mw'  : [()], 'hbd' : [()], 'hba' : [()], 'logp': [()],
                                 'tpsa': [()], 'rb'  : [()], 'nor' : [()], 'fc'  : [()], 'nha' : [()],
                                 'mr'  : [()], 'na'  : [()], 'QED' : [()]}
        self.plot_filter_dataframe = PlotFilterDataframe()
        return overall_widget
    
    def setup_decision_widget(self):
        setting_layout = QVBoxLayout()
        first_layout = QHBoxLayout()
        second_layout = QHBoxLayout()
        overall_layout = QVBoxLayout()
        overall_widget = QWidget()
        
        setting_frame = QFrame()
        setting_frame.setFrameShape(QFrame.Shape.StyledPanel)
        setting_frame.setLineWidth(2)
        setting_frame.setLayout(setting_layout)
        
        energy_threshold_layout = QHBoxLayout()
        energy_threshold_widget = QWidget()
        energy_threshold_layout.setContentsMargins(0, 0, 0, 0)
        tpr_fpr_label_layout = QHBoxLayout()
        tpr_fpr_label_widget = QWidget()
        tpr_fpr_label_layout.setContentsMargins(0, 0, 0, 0)
        column_selection_layout = QHBoxLayout()
        column_selection_widget = QWidget()
        column_selection_layout.setContentsMargins(0, 0, 0, 0)
        filter_btn_layout = QHBoxLayout()
        filter_btn_widget = QWidget()
        filter_btn_layout.setContentsMargins(0, 0, 0, 0)
        
        energy_threshold_label = QLabel('<b>Score ≤</b>')
        self.energy_threshold_spinbox = QDoubleSpinBox(self)
        self.energy_threshold_spinbox.setRange(-1e2, 1e2)
        self.energy_threshold_spinbox.setValue(-6)
        self.energy_threshold_spinbox.setSingleStep(0.5)
        self.energy_threshold_spinbox.setMinimumWidth(100)
        tpr_threshold_label = QLabel('<b>TPR ≥</b>')
        self.tpr_threshold_spinbox = QDoubleSpinBox(self)
        self.tpr_threshold_spinbox.setRange(0, 1)
        self.tpr_threshold_spinbox.setValue(0.95)
        self.tpr_threshold_spinbox.setSingleStep(0.05)
        self.tpr_threshold_spinbox.setMinimumWidth(75)
        fpr_threshold_label = QLabel('<b>FPR ≤</b>')
        self.fpr_threshold_spinbox = QDoubleSpinBox(self)
        self.fpr_threshold_spinbox.setRange(0, 1)
        self.fpr_threshold_spinbox.setValue(0.60)
        self.fpr_threshold_spinbox.setSingleStep(0.05)
        self.fpr_threshold_spinbox.setMinimumWidth(75)
        self.auto_determine_button = QPushButton('Calculate')
        self.auto_determine_button.clicked.connect(self.calculate_threshold_metrics)
        self.auto_determine_button.setDisabled(True)
        energy_threshold_layout.addWidget(energy_threshold_label)
        energy_threshold_layout.addWidget(self.energy_threshold_spinbox)
        energy_threshold_layout.addWidget(tpr_threshold_label)
        energy_threshold_layout.addWidget(self.tpr_threshold_spinbox)
        energy_threshold_layout.addWidget(fpr_threshold_label)
        energy_threshold_layout.addWidget(self.fpr_threshold_spinbox)
        energy_threshold_layout.addSpacerItem(QSpacerItem(15, 0))
        energy_threshold_layout.addWidget(self.auto_determine_button)
        energy_threshold_widget.setLayout(energy_threshold_layout)
        
        combo_texts = ['Molecular Weight', 'Hydrogen Bond Donors',
                       'Hydrogen Bond Acceptors', 'LogP', 'Topological Polar Surface Area',
                       'Rotatable Bonds', 'Number of Rings', 'Formal Charge',
                       'Number of Heavy Atoms', 'Molar Refractivity', 'Number of Atoms', 'QED']
        
        self.tpr_label = QLabel('Overall TPR: ?.???')
        self.fpr_label = QLabel('Overall FPR: ?.???')
        tpr_color = 'LimeGreen' if self.curr_display_mode == 'dark' else 'ForestGreen'
        fpr_color = 'Tomato'    if self.curr_display_mode == 'dark' else 'Maroon'
        self.tpr_label.setStyleSheet(f'QLabel {{ font-size: 14px; color: {tpr_color}; font-weight: bold; }}')
        self.fpr_label.setStyleSheet(f'QLabel {{ font-size: 14px; color: {fpr_color}; font-weight: bold; }}')
        self.tpr_label.setToolTip('The closer to 1 the better.')
        self.fpr_label.setToolTip('The closer to 0 the better.')
        tpr_fpr_label_layout.addWidget(self.tpr_label)
        tpr_fpr_label_layout.addWidget(self.fpr_label)
        tpr_fpr_label_widget.setLayout(tpr_fpr_label_layout)
        
        column_selection_label = QLabel('<b>Y :</b>')
        self.column_selection_combo = QComboBox(self)
        self.column_selection_combo.addItems(combo_texts)
        self.column_selection_combo.setCurrentText('Molecular Weight')
        self.column_selection_combo.setMinimumWidth(210)
        self.column_selection_combo.currentTextChanged.connect(self.plot_and_view_scatter_or_roc)
        column_selection_layout.addWidget(column_selection_label)
        column_selection_layout.addWidget(self.column_selection_combo, alignment=Qt.AlignmentFlag.AlignLeft)
        column_selection_widget.setLayout(column_selection_layout)
        
        scatter_roc_template_label = QLabel('<b>Template :</b>')
        self.roc_template_combo = QComboBox(self)
        self.roc_template_combo.addItems(self.pio_templates)
        self.roc_template_combo.setMinimumWidth(150)
        self.roc_template_combo.currentTextChanged.connect(self.plot_and_view_scatter_or_roc)
        
        plot_type_label = QLabel('<b>Plot :</b>')
        self.plot_type_combo = QComboBox(self)
        self.plot_type_combo.addItems(['Scatter', 'ROC Curve'])
        self.plot_type_combo.setCurrentText('Scatter')
        self.plot_type_combo.setMinimumWidth(130)
        self.plot_type_combo.currentTextChanged.connect(self.plot_and_view_scatter_or_roc)
        
        # scatter_roc_template_layout.addWidget(scatter_roc_template_label)
        # scatter_roc_template_layout.addWidget(self.roc_template_combo)
        # scatter_roc_template_layout.addWidget(plot_type_label)
        # scatter_roc_template_layout.addWidget(self.plot_type_combo)
        column_selection_layout.addWidget(scatter_roc_template_label)
        column_selection_layout.addWidget(self.roc_template_combo)
        column_selection_layout.addWidget(plot_type_label)
        column_selection_layout.addWidget(self.plot_type_combo)
        
        self.setup_filter_button = QPushButton('View Filter')
        self.setup_filter_button.clicked.connect(self.update_threshold_filter_dialog)
        self.setup_filter_button.setDisabled(True)
        self.export_filter_button = QPushButton('Export Filter')
        self.export_filter_button.clicked.connect(self.export_filter_dict_to_json)
        self.export_filter_button.setDisabled(True)
        filter_btn_layout.addWidget(self.setup_filter_button)
        filter_btn_layout.addWidget(self.export_filter_button)
        filter_btn_widget.setLayout(filter_btn_layout)
        
        first_layout.addWidget(energy_threshold_widget, alignment=Qt.AlignmentFlag.AlignLeft)
        first_layout.addWidget(tpr_fpr_label_widget, alignment=Qt.AlignmentFlag.AlignRight)
        second_layout.addWidget(column_selection_widget, alignment=Qt.AlignmentFlag.AlignLeft)
        # second_layout.addWidget(roc_template_widget, alignment=Qt.AlignmentFlag.AlignLeft)
        second_layout.addWidget(filter_btn_widget, alignment=Qt.AlignmentFlag.AlignRight)
        setting_layout.addLayout(first_layout)
        setting_layout.setSpacing(10)
        setting_layout.addLayout(second_layout)
        overall_layout.addWidget(setting_frame, alignment=Qt.AlignmentFlag.AlignTop)
        
        ### Browser ###
        self.browser_roc_plot = PlotViewer()
        if self.curr_display_mode == 'light':
            self.browser_roc_plot.setup_html(self.empty_light_plot_html)
            self.roc_template_combo.setCurrentText('plotly')
            self.browser_roc_plot.setup_background_color('')
        else:
            self.browser_roc_plot.setup_html(self.empty_dark_plot_html)
            self.roc_template_combo.setCurrentText('plotly_dark')
            self.browser_roc_plot.setup_background_color('_dark')
        self.browser_roc_plot.showMaximized()
        overall_layout.addWidget(self.browser_roc_plot, 1)
        
        ### Setup Reverse mapping (from chemprop full name to abbreviation) ###
        self.full_name_to_chem_prop_map = dict()
        for syn, full in self.result_table.chem_prop_to_full_name_map.items():
            self.full_name_to_chem_prop_map[full] = syn
        
        overall_layout.setSpacing(10)
        overall_widget.setLayout(overall_layout)
        return overall_widget
    
    def setup_fragment_widget(self):
        overall_widget = QWidget()
        overall_layout = QVBoxLayout()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        grid_widget = QWidget()
        
        button_layout = QHBoxLayout()
        # threshold_label = QLabel('<b>Threshold :</b>')
        # self.fragment_threshold_spinbox = QDoubleSpinBox(self)
        # self.fragment_threshold_spinbox.setRange(-100, 100)
        # self.fragment_threshold_spinbox.setValue(-4.5)
        # self.fragment_threshold_spinbox.setSingleStep(0.5)
        self.fragment_button = QPushButton('Process Fragments')
        self.fragment_button.clicked.connect(self.process_fragment_img_and_score)
        self.fragment_button.setDisabled(True)
        # button_layout.addWidget(threshold_label, alignment=Qt.AlignmentFlag.AlignLeft)
        # button_layout.addWidget(self.fragment_threshold_spinbox, Qt.AlignmentFlag.AlignLeft)
        button_layout.addWidget(self.fragment_button, alignment=Qt.AlignmentFlag.AlignLeft)
        
        progress_layout = QHBoxLayout()
        progress_label = QLabel('Scoring Progress')
        self.fragment_progress_bar = QProgressBar(self)
        self.fragment_progress_text = QLabel('?/?')
        self.fragment_progress_text.setStyleSheet('font-family: "Courier New", Courier, monospace;')
        self.fragment_progress_text.setMaximumHeight(15)
        progress_layout.addWidget(progress_label, alignment=Qt.AlignmentFlag.AlignLeft)
        progress_layout.addWidget(self.fragment_progress_bar)
        progress_layout.addWidget(self.fragment_progress_text)
        
        font = QFont()
        font.setBold(True)
        header_grid = QGridLayout()
        self.fragment_grid = QGridLayout()
        
        for idx, header in enumerate(['Fragment', 'SMILES', 'Names', 'Scores']):
            head = QLabel(header)
            head.setFont(font)
            header_grid.addWidget(head, 0, idx, Qt.AlignmentFlag.AlignCenter)
        
        page_layout = QHBoxLayout()
        page_btn_layout = QHBoxLayout()
        page_btn_widget = QWidget()
        page_btn_layout.setContentsMargins(0, 0, 0, 0)
        self.prev_page_button = QPushButton('Prev')
        self.next_page_button = QPushButton('Next')
        self.prev_page_button.setDisabled(True)
        self.next_page_button.setDisabled(True)
        self.prev_page_button.clicked.connect(self.fragment_prev_page)
        self.next_page_button.clicked.connect(self.fragment_next_page)
        self.curr_page_lineedit = QLineEdit()
        page_regex_validator = QRegularExpressionValidator(r'^\d+$')
        self.curr_page_lineedit.setValidator(page_regex_validator)
        page_label = QLabel('Page :')
        self.page_total_label = QLabel()
        self.page_total_label.setText('/ ?')
        page_btn_layout.addWidget(self.prev_page_button)
        page_btn_layout.addWidget(self.next_page_button)
        page_btn_widget.setLayout(page_btn_layout)
        page_layout.setSpacing(1)
        page_layout.addWidget(page_btn_widget, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        page_layout.addWidget(page_label, alignment=Qt.AlignmentFlag.AlignRight)
        page_layout.addWidget(self.curr_page_lineedit, alignment=Qt.AlignmentFlag.AlignRight)
        page_layout.addWidget(self.page_total_label, alignment=Qt.AlignmentFlag.AlignRight)
        
        grid_widget.setLayout(self.fragment_grid)
        scroll_area.setWidget(grid_widget)
        
        overall_layout.addLayout(button_layout)
        overall_layout.addLayout(progress_layout)
        overall_layout.addLayout(header_grid)
        overall_layout.addWidget(scroll_area)
        overall_layout.addLayout(page_layout)
        
        overall_widget.setLayout(overall_layout)
        
        return overall_widget
    
    def change_icon_light_dark(self):
        icon_pth = os.path.join(curr_dir, 'icon')
        all_tab_icons = {name: QPixmap(os.path.join(icon_pth, self.curr_display_mode, f'{name}_{self.curr_display_mode}'))
                         for name in self.all_swap_buttons}
        for name, ico in all_tab_icons.items():
            self.all_swap_buttons[name].setIcon(ico)
    
    def modify_sdf_id_name(self):
        """
        Reads the sdf_id_names.txt file and show a dialog for user to modify it (basically a text editor)
        """
        id_file = os.path.join(curr_dir, 'utilities', 'sdf_id_names.txt')
        with open(id_file) as f:
            id_strs = f.read()
        sdf_dialog = SDFIDDialog(id_strs)
        if sdf_dialog.exec():
            new_ids = sdf_dialog.text
            with open(id_file, 'w') as f:
                f.write(new_ids)
    
    def fix_chemstr_format(self):
        # Sometimes, the chemical file contains non-utf-8 characters (some files downloaded from Sellek & ChemDiv), 
        # which is necessary for RDKit to read & process these files.
        # This function "fixes" the text file by encoding everything in utf-8.
        in_file, _ = QFileDialog.getOpenFileName(self,
                                                 'Select Input Text File',
                                                 '')
        if in_file:
            out_ext = in_file.rsplit('.', 1)[-1]
            out_file, _ = QFileDialog.getSaveFileName(self,
                                                      'Select Output Text File',
                                                      os.path.dirname(in_file),
                                                      f'Out Format (*.{out_ext})')
            if out_file:
                chunk_size = 100 * 1024 * 1024    # 100 MB
                in_file_size = os.path.getsize(in_file)
                progress_dialog = QProgressDialog('Writing...', 'Cancel', 0, 1, self)
                progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
                curr_progress = 0
                with open(in_file, 'r', encoding='ISO-8859-1') as i_f, open(out_file, 'w', encoding='utf-8') as o_f:
                    progress_dialog.show()
                    while chunk := i_f.read(chunk_size):
                        o_f.write(chunk)
                        curr_progress = min(curr_progress + chunk_size, in_file_size)
                        progress_dialog.setValue(curr_progress / in_file_size)
                        if progress_dialog.wasCanceled():
                            return
    
    def select_convert_input_file(self):
        filters = 'Molecule File ' + '(' + ' '.join(['*'+i for i in self.available_formats]) + ')'
        files, _ = QFileDialog.getOpenFileNames(self,
                                                'Select Input Files',
                                                '',
                                                filters)
        if files:
            self.input_files_list.addItems(files)
    
    def select_convert_input_dir(self):
        dir = QFileDialog.getExistingDirectory(self, 'Select Input Directory', '')
        if dir:
            self.input_files_list.addItem(dir)
    
    def read_csv_tsv_file(self):
        filters = 'Table File (*.csv *.tsv)'
        file, _ = QFileDialog.getOpenFileName(self,
                                              'Select Input Files',
                                              '',
                                              filters)
        if file:
            dialog = CSVTSVColumnDialog(file)
            if dialog.exec():
                self.table_file_params.update(dialog.table_params)
                self.input_files_list.addItems(list(dialog.table_params))
    
    def update_single_file_checkbox_and_add_h(self, text):
        if text in ['.sdf', '.smi']:
            self.single_file_checkbox.setEnabled(True)
            self.add_h_checkbox.setEnabled(True)
            if text == '.smi':
                self.retain_3d.setChecked(False)
                self.retain_3d.setEnabled(False)
        elif text == '.csv (DiffDock)':
            self.single_file_checkbox.setChecked(True)
            self.single_file_checkbox.setEnabled(False)
            self.add_h_checkbox.setChecked(False)
            self.add_h_checkbox.setEnabled(False)
            self.retain_3d.setChecked(False)
            self.retain_3d.setEnabled(False)
        elif text == '.mddb':
            self.single_file_checkbox.setChecked(True)
            self.single_file_checkbox.setEnabled(False)
            self.add_h_checkbox.setChecked(False)
            self.add_h_checkbox.setEnabled(False)
            self.retain_3d.setChecked(False)
            self.retain_3d.setEnabled(False)
        elif text == '.png':
            self.add_h_checkbox.setEnabled(True)
            self.single_file_checkbox.setEnabled(False)
            self.single_file_checkbox.setChecked(False)
            self.retain_3d.setChecked(False)
            self.retain_3d.setEnabled(False)
        else:
            self.add_h_checkbox.setEnabled(True)
            self.single_file_checkbox.setEnabled(False)
            self.single_file_checkbox.setChecked(False)
            self.retain_3d.setChecked(True)
            self.retain_3d.setEnabled(True)
        if text == '.pdbqt':
            self.add_h_checkbox.setChecked(True)
            self.add_h_checkbox.setEnabled(False)
            self.add_h_checkbox.setToolTip('.pdbqt file must add H')
            self.flexible_macrocycle_ckbox.setChecked(True)
            self.flexible_macrocycle_ckbox.setEnabled(True)
            self.flexible_macrocycle_ckbox.setToolTip('Flexible macrocycle docking (ONLY for AutoDock VINA!)')
        else:
            self.flexible_macrocycle_ckbox.setChecked(False)
            self.flexible_macrocycle_ckbox.setEnabled(False)
            self.flexible_macrocycle_ckbox.setToolTip('Only for .pdbqt')
    
    def select_convert_output_directory(self):
        if not self.single_file_checkbox.isChecked():
            folder = QFileDialog.getExistingDirectory(self, 'Select Output Directory', '')
            if folder:
                self.convert_output_dir_file_line.setText(folder)
        else:
            ext = self.convert_output_extension_combo.currentText()
            if ext == '.sdf':
                filter = 'Structure Data Format (*.sdf)'
            elif ext == '.smi':
                filter = 'SMILES (*.smi)'
            elif ext == '.csv (DiffDock)':
                filter = 'Comma Separated Value (*.csv)'
            elif ext == '.mddb':
                filter = 'MolDocker Database (*.mddb)'
            save_file, _ = QFileDialog.getSaveFileName(self, 'Select Output File Name', '', filter)
            if save_file:
                self.convert_output_dir_file_line.setText(save_file)
    
    def save_convert_log_file_dialog(self):
        save_file, _ = QFileDialog.getSaveFileName(self, 'Save Log File', '', 'Log File (*.log)')
        if save_file:
            with open(save_file, 'w') as f:
                f.write(self.convert_textedit.toPlainText())
    
    def check_conversion(self):
        if not self.now_converting:
            output_file_checked = bool(self.convert_output_dir_file_line.text())
            # if self.single_file_checkbox.isChecked():
            #     if output_file_checked:
            #         output_file_checked &= self.convert_output_dir_file_line.text().rsplit('.', 1)[1] == self.convert_output_extension_combo.currentText().split()[0][1:]
            self.convert_files_button.setEnabled(bool(self.input_files_list.count() and output_file_checked))
    
    def update_dock_input_lineedit(self, conv_out_pth: str):
        if not self.is_docking:
            self.input_ligand_line.setText(conv_out_pth)     
    
    def check_if_db_id_changed(self, previous_name: str, new_name: str):
        if previous_name in self.db_small_molecule:
            self.db_small_molecule[new_name] = self.db_small_molecule.pop(previous_name)
    
    def check_if_db_id_removed(self, previous_name: str):
        if previous_name in self.db_small_molecule:
            del self.db_small_molecule[previous_name]
    
    def check_if_file_pth_or_dbname_existed(self, pth_or_dbid: str):
        if os.path.isabs(pth_or_dbid):  # pth
            if not os.path.exists(pth_or_dbid):
                QMessageBox.critical(self, 'File Path Error', f'"{pth_or_dbid}" does not exist!')
                item = self.input_files_list.selectedItems()[0]
                item.setText(self.input_files_list.previous_path)
                return
        else:   # dbid
            if pth_or_dbid in self.db_small_molecule and pth_or_dbid != self.input_files_list.previous_path:
                QMessageBox.critical(self, 'Name Error', f'"{pth_or_dbid}" already existed!')
                item = self.input_files_list.selectedItems()[0]
                item.setText(self.input_files_list.previous_path)
                return
        if pth_or_dbid != self.input_files_list.previous_path:
            self.db_small_molecule[pth_or_dbid] = self.db_small_molecule.pop(self.input_files_list.previous_path)
    
    def remove_current_selected(self):
        self.input_files_list.remove_current_selected()
    
    @staticmethod
    def parse_operation_to_list(operation_tuple):
        operation_texts = []
        for operation_and_value in operation_tuple:
            if operation_and_value[0] == '>':
                operation_texts.append('&gt;' + f' {operation_and_value[1]}')
            elif operation_and_value[0] == '<':
                operation_texts.append('&lt;' + f' {operation_and_value[1]}')
            else:
                operation_texts.append(operation_and_value[0] + f' {operation_and_value[1]}')
        return operation_texts
    
    def convert_start_reading(self):
        if not self.now_converting and not self.now_picking:
            self.convert_progress.setValue(0)
            self.convert_textedit.clear()
            self.all_input_files = []
            self.molecules_dict = {}
            retain_3d = self.retain_3d.isChecked()
            for input_dir_or_file_or_dbid in self.input_files_list:
                if not os.path.exists(input_dir_or_file_or_dbid):
                    if input_dir_or_file_or_dbid not in self.db_small_molecule:
                        text = f'{input_dir_or_file_or_dbid} does not exist!'
                        self.convert_textedit.append_to_log(text, 'failed')
                    else:
                        chem = self.db_small_molecule[input_dir_or_file_or_dbid]
                        if '\n' in chem:
                            mol = Chem.MolFromMolBlock(chem)
                        else:
                            mol = Chem.MolFromSmiles(chem)
                        if mol is None:
                            text = f'Failed to load molecule "{input_dir_or_file_or_dbid}"!'
                            self.convert_textedit.append_to_log(text, 'failed')
                        else:
                            if retain_3d:
                                if mol.GetNumConformers() and mol.GetConformer().Is3D():
                                    mol_string = Chem.MolToMolBlock(mol)
                                else:
                                    mol_string = Chem.MolToSmiles(mol)
                            else:
                                mol_string = Chem.MolToSmiles(mol)
                        self.molecules_dict[input_dir_or_file_or_dbid] = mol_string
                else:
                    if os.path.isdir(input_dir_or_file_or_dbid):
                        # dir
                        self.all_input_files += [f for f in recursive_read_all_files(input_dir_or_file_or_dbid) if 
                                                 not f.startswith('.') and f.lower().endswith(self.available_formats)]
                    else:
                        # file
                        if input_dir_or_file_or_dbid.endswith(self.available_formats+('.csv', '.tsv')):
                            self.all_input_files.append(input_dir_or_file_or_dbid)
            self.all_input_files = list(dict.fromkeys(self.all_input_files))  # remove duplicates while preserving order
            
            with open(os.path.join(curr_dir, 'utilities', 'fp_param', 'fp_param.json'), 'r') as f:
                self.conv_fp_settings = json.load(f)
            
            # check the fp format of mddb file
            if self.convert_output_extension_combo.currentText() == '.mddb':
                out_pth = self.convert_output_dir_file_line.text()
                if not out_pth.endswith('.mddb'):
                    out_pth += '.mddb'
                if os.path.exists(out_pth):
                    conn = sqlite3.connect(out_pth)
                    cur = conn.cursor()
                    cur.execute('SELECT format FROM DBInfo WHERE id = ?', (1,))
                    row = cur.fetchone()
                    db_fp_settings = json.loads(row[0])
                    conn.close()
                    tmp_settings = dict(self.conv_fp_settings)
                    del tmp_settings['sim']
                    if db_fp_settings != tmp_settings:
                        db_param_txt = json.dumps(db_fp_settings, indent=4)
                        db_param_txt = db_param_txt.replace('    ', '&nbsp;&nbsp;&nbsp;&nbsp;').replace('\n', '<br>')
                        fp_param_txt = json.dumps(tmp_settings, indent=4)
                        fp_param_txt = fp_param_txt.replace('    ', '&nbsp;&nbsp;&nbsp;&nbsp;').replace('\n', '<br>')
                        final_txt = (f'Fingerprinting parameters of the database file '
                        f"is different from current parameters<br>Database:<br>{db_param_txt}<br><br>"
                        f"Current:<br>{fp_param_txt}")
                        self.convert_textedit.append_to_log(final_txt, 'failed', False)
                        return
            
            self.convert_files_button.setDisabled(True)
            self.convert_textedit.append('<b>Reading Files...</b>')
            self.convert_textedit.append(f'<hr/><br/>')
            
            self.read_failed = 0
            self.read_thread = QThread()
            self.read_worker = MultiprocessConvertReader(self.all_input_files, self.table_file_params,
                                                         retain_3d, self.available_formats)
            self.read_worker.moveToThread(self.read_thread)
            self.read_worker.processed.connect(self.successful_processed_log)
            self.read_worker.processedWithFaied.connect(self.successful_processed_and_failed_log)
            self.read_worker.failedLog.connect(self.failed_processed_log)
            self.read_worker.finished.connect(self.read_thread.quit)
            self.read_thread.finished.connect(self.start_dbsim_sampling)
            self.read_thread.started.connect(self.read_worker.run)
            self.read_thread.start()
        elif self.now_picking:
            self.db_sim_worker.stop()
            self.db_sim_thread.quit()
            self.db_sim_thread.wait()
        elif self.now_converting:
            self.worker.stop()
            self.convert_files_button.setDisabled(True)
            self.convert_files_button.setText('Stopping')
            self.conversion_done_output(True)
            self.thread.quit()
            self.thread.wait()
            self.worker.deleteLater()
            self.thread.deleteLater()
    
    def successful_processed_log(self, molecule_dict: dict, warning_str: str):
        tmp_dict = {}
        for name in molecule_dict:
            if name in self.molecules_dict:
                n = 1
                while name + f'_{n}' in self.molecules_dict:
                    n += 1
                warning_str += f'{name} already exists, renamed to {name}_{n}.'
                tmp_dict[f'{name}_{n}'] = molecule_dict[name]
            else:
                tmp_dict[name] = molecule_dict[name]
        self.molecules_dict.update(tmp_dict)
        self.convert_textedit.append_to_log(warning_str, 'warning', False)
    
    def successful_processed_and_failed_log(self, molecule_dict: dict, warning_str: str, failed_str: str):
        tmp_dict = {}
        for name in molecule_dict:
            if name in self.molecules_dict:
                n = 1
                while name + f'_{n}' in self.molecules_dict:
                    n += 1
                warning_str += f'{name} already exists, renamed to {name}_{n}.'
                tmp_dict[f'{name}_{n}'] = molecule_dict[name]
            else:
                tmp_dict[name] = molecule_dict[name]
        self.molecules_dict.update(tmp_dict)
        if failed_str:
            self.read_failed += failed_str.count('\n') + 1
            self.convert_textedit.append_to_log(failed_str, 'failed', False)
        if warning_str:
            self.convert_textedit.append_to_log(warning_str, 'warning', False)
    
    def failed_processed_log(self, failed_tuple: tuple):
        self.read_failed += 1
        failed_str, _type, exact_bool = failed_tuple
        self.convert_textedit.append_to_log(failed_str, _type, exact_bool)
    
    def start_dbsim_sampling(self):
        self.read_thread.wait()
        self.read_worker.deleteLater()
        self.read_thread.deleteLater()
        self.molecules_dict = {k: self.molecules_dict[k] for k in sorted(self.molecules_dict)}
        self.total_length = len(self.molecules_dict)
        total_files = len(self.all_input_files) - self.read_failed
        f_t = 'file' if total_files == 1 else 'files'
        c_t = 'molecule' if self.total_length == 1 else 'molecules'
        text = f'<b>{self.total_length} {c_t} from {total_files} {f_t} loaded.</b>'
        self.convert_textedit.append(text)
        if self.total_length == 0:
            self.convert_files_button.setText('Convert')
            self.convert_files_button.setEnabled(True)
            return
        self.convert_textedit.append(f'<hr/><br/>')
        self.max_name_length = max(len(i) for i in self.molecules_dict)
        self.ext = self.convert_output_extension_combo.currentText()
        
        # Setup output dir and output file name with molecule name
        self.output_dir_or_file = self.convert_output_dir_file_line.text()
        if not os.path.exists(self.output_dir_or_file):
            if not self.single_file_checkbox.isChecked():   # dir
                os.makedirs(self.output_dir_or_file, exist_ok=True)
                self.convert_textedit.append(f'Directory "{self.output_dir_or_file}" created.')
                self.convert_textedit.append('')
            else:   # file
                check_file = os.path.basename(self.output_dir_or_file)
                if check_file == check_file.rsplit('.', 1)[0]:
                    self.output_dir_or_file += self.ext
                    self.convert_textedit.append_to_log(f'"{self.ext}" appended after "{self.output_dir_or_file}"', 'warning')
                if self.ext == '.mddb':
                    conn = sqlite3.connect(self.output_dir_or_file)
                    cur = conn.cursor()
                    cur.execute("""CREATE TABLE MolDB
                                (name TEXT PRIMARY KEY,
                                fp BLOB NOT NULL,
                                smi BLOB NOT NULL
                                ) WITHOUT ROWID;""")
                    cur.execute("""CREATE TABLE DBInfo
                                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                                format TEXT NOT NULL);""")
                    conn.commit()
                    tmp_dict = dict(self.conv_fp_settings)
                    del tmp_dict['sim']
                    json_data_str = json.dumps(tmp_dict)
                    cur.execute("""INSERT INTO DBInfo (format) VALUES (?)""", (json_data_str,))
                    conn.commit()
                    conn.close()
                else:
                    with open(self.output_dir_or_file, 'w') as f:    # create empty text file if file does not exist
                        if self.ext == '.csv (DiffDock)':
                            f.write('complex_name,ligand_description\n')
                self.convert_textedit.append(f'File "{self.output_dir_or_file}" created.')
                self.convert_textedit.append('')
        else:
            if self.single_file_checkbox.isChecked():   # if file already exists, append to file
                check_file = os.path.basename(self.output_dir_or_file)
                # This will just append to this file, not matter the extension
                # If someone wants to save smiles into a file with txt extension, just allow them! They probably have their reasons.
                if check_file == check_file.rsplit('.', 1)[0]:
                    self.output_dir_or_file += self.ext
                    self.convert_textedit.append_to_log(f'"{self.ext}" appended after "{self.output_dir_or_file}"', 'warning')
                self.convert_textedit.append(f'File "{self.output_dir_or_file}" already exists, append to file.')
                self.convert_textedit.append('')
            else:
                os.makedirs(self.output_dir_or_file, exist_ok=True)
        
        if self.similarity_db_dict['db_pth']:
            self.convert_files_button.setEnabled(True)
            self.convert_files_button.setText('Stop')
            self.now_picking = True
            text = f'Database Similarity Macthing Enabled.'
            self.convert_textedit.append(text)
            self.convert_textedit.append('Target Databases: ')
            final_db_pths = []
            tmp_settings = dict(self.conv_fp_settings)
            del tmp_settings['sim']
            for pth in self.similarity_db_dict['db_pth']:
                conn = sqlite3.connect(pth)
                cur = conn.cursor()
                cur.execute('SELECT format FROM DBInfo WHERE id = ?', (1,))
                row = cur.fetchone()
                db_fp_settings = json.loads(row[0])
                if db_fp_settings != tmp_settings:
                    db_param_txt = json.dumps(db_fp_settings, indent=4)
                    db_param_txt = db_param_txt.replace('    ', '&nbsp;&nbsp;&nbsp;&nbsp;')
                    db_param_txt = db_param_txt.replace('\n', '<br>')
                    fp_param_txt = json.dumps(tmp_settings, indent=4)
                    fp_param_txt = fp_param_txt.replace('    ', '&nbsp;&nbsp;&nbsp;&nbsp;')
                    fp_param_txt = fp_param_txt.replace('\n', '<br>')
                    final_txt = f'"{os.path.basename(pth)}" ignored due to incompatible FP method.<br>'
                    self.convert_textedit.append_to_log(final_txt, 'warning', False)
                    conn.close()
                    continue
                cur.execute(f"SELECT COUNT(*) FROM MolDB;")
                row_cnt = cur.fetchone()[0]
                conn.close()
                self.convert_textedit.append(f'  {pth} (Num = {row_cnt})')
                final_db_pths.append(pth)
            if not final_db_pths:
                self.convert_textedit.append_to_log('All databases have mismatched fingerprinting parameters.', 'warning', False)
                self.start_maxmin_sampling(False)
            else:
                self.convert_textedit.append('Settings: ')
                self.convert_textedit.append(f"  Similarity Type: {self.conv_fp_settings['sim']}")
                one, two = [round(x, 2) for x in self.similarity_db_dict['db_sim']]
                if self.similarity_db_dict['sim_type'] == 'Include':
                    self.convert_textedit.append(f'  sim ≥ {one} & sim ≤ {two}')
                else:
                    self.convert_textedit.append(f'  sim < {one} | sim > {two}')
                self.db_sim_thread = QThread()
                self.db_sim_worker = MultiProcessDBSimilarityPicker(self.molecules_dict,
                                                                    final_db_pths,
                                                                    self.similarity_db_dict['sim_type'],
                                                                    self.similarity_db_dict['db_sim'],
                                                                    self.desalt_checkbox.isChecked(),
                                                                    self.conv_fp_settings)
                self.db_sim_worker.moveToThread(self.db_sim_thread)
                self.db_sim_worker.pickedResult.connect(lambda x: setattr(self, 'molecules_dict', x))
                self.db_sim_worker.pickedResult.connect(self.start_maxmin_sampling)
                self.db_sim_worker.pickedResult.connect(self.db_sim_thread.quit)
                self.db_sim_worker.pickingSuccess.connect(self.update_db_sim_success_text)
                self.db_sim_worker.pickingFail.connect(self.update_db_sim_failed_text)
                self.db_sim_worker.pickingStopped.connect(self.picking_force_stopped)
                self.db_sim_thread.finished.connect(self.db_sim_thread.deleteLater)
                self.db_sim_thread.finished.connect(self.db_sim_worker.deleteLater)
                self.db_sim_thread.started.connect(self.db_sim_worker.run)
                self.db_sim_thread.start()
        else:
            self.start_maxmin_sampling(False)
    
    def update_db_sim_success_text(self, success_str: str):
        self.convert_textedit.append_to_log(success_str, 'success', False)
    
    def update_db_sim_failed_text(self, fail_str: str):
        self.convert_textedit.append_to_log(fail_str, 'failed', False)
    
    def start_maxmin_sampling(self, sim_search=True):
        del self.conv_fp_settings['sim']
        if sim_search:
            self.total_length = len(self.molecules_dict)
            self.convert_textedit.append(f'<b>Similarity search completed. {self.total_length} molecules selected.<br></b>')
            self.now_picking = False
            if self.total_length == 0:
                return
        if self.sampling_setting_dict['maxmin']:
            sample_num = int(self.sampling_setting_dict['count'])
            seed = int(self.sampling_setting_dict['seed'])
            if sample_num <= self.total_length:
                text = f'<b>MaxMin Sampling Enabled. {sample_num} diverse molecules will be sampled.</b>'
                self.convert_textedit.append(text)
                self.maxmin_thread = QThread()
                self.maxminpicker_worker = ThreadedMaxMinPicker(self.molecules_dict, sample_num, seed, self.conv_fp_settings)
                self.maxminpicker_worker.moveToThread(self.maxmin_thread)
                self.maxminpicker_worker.pickedResult.connect(lambda x: setattr(self, 'molecules_dict', x))
                self.maxminpicker_worker.pickedResult.connect(lambda x: self.convert_textedit.append(
                    f'<b>Sampling done. {sample_num} out of {self.total_length} molecules sampled.<br></b>'
                    ))
                self.maxminpicker_worker.pickedResult.connect(lambda: setattr(self, 'total_length', sample_num))
                self.maxminpicker_worker.pickedResult.connect(self.continue_converting)
                self.maxminpicker_worker.pickedResult.connect(self.maxmin_thread.quit)
                self.maxmin_thread.finished.connect(self.maxmin_thread.deleteLater)
                self.maxmin_thread.finished.connect(self.maxminpicker_worker.deleteLater)
                self.maxmin_thread.started.connect(self.maxminpicker_worker.run)
                self.maxmin_thread.start()
            else:
                text = f'<b>Number of MaxMin samples greater than {self.total_length}, MaxMin sampling not applied.</b>'
                self.convert_textedit.append(text)
                self.convert_textedit.append('')
                self.continue_converting()
        else:
            # If MaxMin sampling is not enabled, proceed immediately
            self.continue_converting()
    
    def continue_converting(self):
        # Random Sampling
        if not self.molecules_dict:
            self.convert_textedit.append(f'<hr/><br/>')
            self.convert_textedit.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.convert_textedit.append_to_log(f'<b>No molecules selected.</b>', 'done')
            self.convert_textedit.append('')
            self.convert_files_button.setText('Convert')
            self.convert_files_button.setEnabled(True)
            self.now_picking = False
            return
        max_name_length = max(len(i) for i in self.molecules_dict)
        if self.sampling_setting_dict['random']:
            sample_num = int(self.sampling_setting_dict['count'])
            rng_generator = np.random.default_rng(int(self.sampling_setting_dict['seed']))
            if sample_num <= self.total_length:
                text = f'<b>Random Sampling Enabled. {sample_num} molecules will be sampled w/o replacement.</b>'
                self.convert_textedit.append(text)
                self.convert_textedit.append('')
                rng_idx = rng_generator.permutation(self.total_length)[:sample_num]
                self.molecules_dict = {os.path.join(self.output_dir_or_file, f+self.ext): m for i, (f, m) in enumerate(self.molecules_dict.items()) if i in rng_idx}
                self.total_length = sample_num
            else:
                text = f'<b>Number of random samples greater than {self.total_length}, random sampling not applied.</b>'
                self.convert_textedit.append_to_log(text, 'warning')
                self.convert_textedit.append('')
                self.molecules_dict = {os.path.join(self.output_dir_or_file, f+self.ext): m for f, m in self.molecules_dict.items()}
        else:
            self.molecules_dict = {os.path.join(self.output_dir_or_file, f+self.ext): m for f, m in self.molecules_dict.items()}
        # Setup filters
        exact_chemprop_filter_dict = {}
        partial_chemprop_filter_dict = {}
        for chemprop, check_state in self.chem_filter_bool.items():
            if chemprop not in ['partial_filter_threshold', 'match_type']:
                if check_state:
                    if check_state == True:
                        exact_chemprop_filter_dict[chemprop] = self.chem_filter_dict[chemprop]
                    else:
                        partial_chemprop_filter_dict[chemprop] = self.chem_filter_dict[chemprop]
            elif chemprop == 'partial_filter_threshold':
                partial_chemprop_filter_dict[chemprop] = self.chem_filter_bool[chemprop]
            elif chemprop == 'match_type':
                chemprop_match_type = self.chem_filter_bool[chemprop]
                
        if partial_chemprop_filter_dict['partial_filter_threshold'] == 0:
            partial_chemprop_filter_dict = {'partial_filter_threshold': 0}    # partial filter threshold set to 0 means no partial filter will be applied
            
        if (not exact_chemprop_filter_dict) and (partial_chemprop_filter_dict['partial_filter_threshold'] == 0):
            text = f'<b>Chemical Property Filter not applied.</b>'
            self.convert_textedit.append_to_log(text, 'property')
        else:
            exact_num_filter = len(exact_chemprop_filter_dict)
            partial_num_filter = len(partial_chemprop_filter_dict) - 1  # remove count of "partial_filter_threshold"
            f_t = 'Filter' if exact_num_filter + partial_num_filter == 1 else 'Filters'
            text = f'<b><u>{exact_num_filter} Exact</u> & {partial_num_filter} Partial Chemical Property {f_t} applied :</b>'
            self.convert_textedit.append_to_log(text, 'property')
            for chem_prop in exact_chemprop_filter_dict:
                full_name = self.chem_prop_to_full_name_map[chem_prop]
                operation_texts = self.parse_operation_to_list(exact_chemprop_filter_dict[chem_prop])
                filters_text = f'{full_name} : {", ".join(operation_texts)}'
                self.convert_textedit.append_to_log(filters_text, 'property_small', True)
            for chem_prop in partial_chemprop_filter_dict:
                if chem_prop != 'partial_filter_threshold':
                    full_name = self.chem_prop_to_full_name_map[chem_prop]
                    operation_texts = self.parse_operation_to_list(partial_chemprop_filter_dict[chem_prop])
                    filters_text = f'{full_name} : {", ".join(operation_texts)}'
                else:
                    if partial_num_filter >= 1:
                        filters_text = f'Partial Chemical Property Filter Threshold = {partial_num_filter}'
                    else:
                        filters_text = f'Partial Chemical Property Filter not applied.'
                self.convert_textedit.append_to_log(filters_text, 'property_small')
            match_type_text = 'Excluded' if chemprop_match_type == 'Exclude' else 'Included'
            self.convert_textedit.append_to_log(f'Chemicals matching above property filters will be {match_type_text}.', 'property_small')
        
        self.convert_textedit.append('')
        
        exact_rdkit_name = []
        partial_rdkit_name = []
        for name, state in self.rdkit_filter_dict.items():
            if name not in ['partial_filter_threshold', 'match_type']:
                if state is True:
                    exact_rdkit_name.append(name)
                elif state == 'partial':
                    partial_rdkit_name.append(name)
            elif name == 'match_type':
                rdkit_filter_match_type = state
        exact_rdkit_filter_dict = {'catalog': exact_rdkit_name}
        if self.rdkit_filter_dict['partial_filter_threshold'] == 0: # partial filter threshold set to 0 means no partial filter will be applied
            partial_rdkit_name = []
        partial_rdkit_filter_dict = {'catalog': partial_rdkit_name,
                                     'partial_filter_threshold': self.rdkit_filter_dict['partial_filter_threshold']}
        if not (len(exact_rdkit_name) + len(partial_rdkit_name)):
            text = f'<b>Structural Filter not applied.</b>'
            self.convert_textedit.append_to_log(text, 'structure')
        else:
            exact_num_filter = len(exact_rdkit_name)
            partial_num_filter = len(partial_rdkit_name)
            f_t = 'Filter' if exact_num_filter + partial_num_filter == 1 else 'Filters'
            text = f'<b><u>{exact_num_filter} Exact</u> & {partial_num_filter} Partial Structural {f_t} applied :</b>'
            self.convert_textedit.append_to_log(text, 'structure')
            for structure_filter_name in exact_rdkit_name:
                filters_text = f'{structure_filter_name}'
                self.convert_textedit.append_to_log(filters_text, 'structure_small', True)
            for structure_filter_name in partial_rdkit_name:
                filters_text = f'{structure_filter_name}'
                self.convert_textedit.append_to_log(filters_text, 'structure_small')
            if partial_num_filter >= 1:
                p_t = 'hit' if partial_num_filter == 1 else 'hits'
                match_type_text = '≥' if rdkit_filter_match_type == 'Exclude' else '<'
                filters_text = f'Partial Structural Filter Threshold = {partial_rdkit_filter_dict["partial_filter_threshold"]}'
            else:
                filters_text = f'Partial Structural Filter not applied.'
            self.convert_textedit.append_to_log(filters_text, 'structure_small')
            match_type_text = 'Excluded' if rdkit_filter_match_type == 'Exclude' else 'Included'
            self.convert_textedit.append_to_log(f'Chemicals matching above structural filters will be {match_type_text}.', 'structure_small')
        
        self.convert_textedit.append(f'<hr/><br/>') # need to set to new line! Or else every text behind it will also contain horizontal line
        
        ### setup progress bar ###
        self.convert_progress.setValue(0)
        self.convert_progress.setMaximum(self.total_length)
        self.step_str_length = len(str(self.total_length))
        self.convert_progress_label.setText(f'{0:{self.step_str_length}}/{self.total_length}')
        self.convert_files_button.setEnabled(True)
        self.convert_files_button.setText('Stop')
        self.now_converting = True
        self.stop_appending_text = False
        self.filtered_cnt = 0
        self.failed_cnt = 0
        self.tik = time.perf_counter()
        append_to_file_name = False
        if self.single_file_checkbox.isChecked():
            append_to_file_name = self.convert_output_extension_combo.currentText()[1:]
        
        ### setup and start multiprocessing to convert SMILES/SDF to other formats. ###
        ### Input used to be Chem.rdchem.Mol, but Nuitka cannot process it with Multiprocessing, so str it is! ###
        params = {'fp_settings': self.conv_fp_settings,
                  'macrocycle' : self.flexible_macrocycle_ckbox.isChecked(),
                  'dpi'        : self.png_dpi_spinbox.value(),
                  'size'       : self.png_size_spinbox.value(),}
        self.thread = QThread()
        self.worker = MultiprocessConversion(self.molecules_dict,
                                             exact_chemprop_filter_dict, partial_chemprop_filter_dict,
                                             exact_rdkit_filter_dict, partial_rdkit_filter_dict,
                                             max_name_length, append_to_file_name,
                                             self.add_h_checkbox.isChecked(), self.desalt_checkbox.isChecked(),
                                             self.timeout_spinbox.value(), chemprop_match_type,
                                             rdkit_filter_match_type, params)
        self.worker.moveToThread(self.thread)
        self.worker.progress.connect(self.convert_update_progress)
        self.worker.log.connect(self.convert_update_log)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.conversion_done_output)
        self.thread.started.connect(self.worker.run)
        self.thread.start()
    
    def convert_update_progress(self, value):
        self.convert_progress.setValue(value)
        self.convert_progress_label.setText(f'{value:{self.step_str_length}}/{self.total_length}')
        passed_time = time.perf_counter() - self.tik
        expected_sec = max(0, self.total_length * (passed_time / value) - passed_time)
        speed = f'{value / passed_time:.2f}'
        p_hr, p_min, p_sec = self.convert_secs(passed_time)
        e_hr, e_min, e_sec = self.convert_secs(expected_sec)
        if p_hr or e_hr:
            passed_string = f'{p_hr:02.0f}:{p_min:02.0f}:{p_sec:02.0f}'
            eta_string = f'{e_hr:02.0f}:{e_min:02.0f}:{e_sec:02.0f}'
        else:
            passed_string = f'{p_min:02.0f}:{p_sec:02.0f}'
            eta_string = f'{e_min:02.0f}:{e_sec:02.0f}'
        self.convert_eta_label.setText(f'[{passed_string}<{eta_string}, {speed}it/s]')
    
    def convert_update_log(self, log_tuple: tuple):
        msg, type, exact = log_tuple
        if type == 'failed':
            self.failed_cnt += 1
        else:
            self.filtered_cnt += 1
        if not self.stop_appending_text:
            self.convert_textedit.append_to_log(msg, type, exact)
        else:
            self.convert_textedit.temp_update(msg, type, exact)
    
    def conversion_done_output(self, forced=False):
        if not self.now_converting:
            return
        curr_pos = self.convert_textedit.verticalScrollBar().value()
        self.now_converting = False
        if curr_pos == self.convert_textedit.verticalScrollBar().maximum():
            to_max = True
        else:
            to_max = False
        self.convert_files_button.setText('Convert')
        self.convert_files_button.setEnabled(True)
        cnt_length = len(str(self.total_length))
        if forced:
            rest = self.total_length - self.convert_progress.value()
            self.failed_cnt += rest
            self.convert_progress.setValue(self.total_length)
            self.convert_update_progress(self.total_length)
            QApplication.instance().processEvents()
            time.sleep(0.1)
            QApplication.instance().processEvents()
            time.sleep(0.1)
        text_converted = f'{self.total_length - (self.filtered_cnt + self.failed_cnt):{cnt_length}}/{self.total_length} converted.'
        text_filtered  = f'{self.filtered_cnt:{cnt_length}}/{self.total_length}  filtered.'
        text_failed    = f'{self.failed_cnt:{cnt_length}}/{self.total_length}    failed.'
        self.convert_textedit.append('')
        self.convert_textedit.append(f'<hr/><br/>')
        self.convert_textedit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if forced:
            self.convert_textedit.append_to_log(f'<b>Conversion Forced Stop ({self.convert_eta_label.text().split("<")[0][1:]})</b>', 'done')
        else:
            self.convert_textedit.append_to_log(f'<b>Conversion Done ({self.convert_eta_label.text().split("<")[0][1:]})</b>', 'done')
        self.convert_textedit.append('')
        self.convert_textedit.append_to_log(text_converted, 'success')
        self.convert_textedit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.convert_textedit.append_to_log(text_filtered , 'property')
        self.convert_textedit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.convert_textedit.append_to_log(text_failed   , 'failed')
        self.convert_textedit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if not to_max:
            self.convert_textedit.verticalScrollBar().setValue(curr_pos)
        self.thread.wait()
        self.worker.deleteLater()
        self.thread.deleteLater()
        self.all_input_files = []
        self.molecules_dict = {}
        gc.collect()
    
    @Slot()
    def picking_force_stopped(self):
        self.convert_textedit.append(f'<hr/><br/>')
        self.convert_textedit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.convert_textedit.append_to_log(f'<b>Picking Forced Stop</b>', 'done')
        self.convert_textedit.append('')
        self.convert_files_button.setText('Convert')
        self.convert_files_button.setEnabled(True)
        self.now_picking = False
    
    def show_filter_dialog(self):
        self.filter_dialog = ConvertFilterDialog(self.chem_filter_dict, self.chem_filter_bool,
                                                 self.rdkit_filter_dict, self.sampling_setting_dict,
                                                 self.similarity_db_dict)
        if self.filter_dialog.exec():
            self.chem_filter_dict = copy.deepcopy(self.filter_dialog.chem_filter_dict)
            self.chem_filter_bool = copy.deepcopy(self.filter_dialog.chem_filter_bool)
            self.rdkit_filter_dict = copy.deepcopy(self.filter_dialog.rdkit_filter_dict)
            self.sampling_setting_dict = copy.deepcopy(self.filter_dialog.rng_setting_dict)
            self.similarity_db_dict = copy.deepcopy(self.filter_dialog.similarity_setting_dict)
    
    def change_convert_log_font_size(self, fontsize):
        self.convert_default_fontsize = fontsize
        self.stop_appending_text = True
        self.convert_textedit.update_fontsize_style(self.curr_display_mode, self.convert_default_fontsize)
        self.stop_appending_text = False
    
    def change_protein_input_type(self, name: str):
        for n, d in self.input_id_dict.items():
            if n == name:
                for widget in d.values():
                    widget.setEnabled(True)
                d['LineEdit'].setFocus()
                t = d['LineEdit'].text()
                if name == 'PDB':
                    d['Load_Button'].setEnabled(len(t) == 4)
                elif name == 'AF Database':
                    d['Load_Button'].setEnabled(any(((t) == 6, len(t) == 10)))
                else:
                    d['Load_Button'].setEnabled(os.path.isfile(t))
            else:
                for widget in d.values():
                    widget.setDisabled(True)
    
    def change_protein_format(self, format: str):
        if self.pdbqt_editor is not None:
            if format == self.curr_format:
                return
            else:
                self.protein_format_button_group.blockSignals(True)
                full_str = self.pdbqt_editor.convert_full_dict_to_text()
                if format == 'pdb':
                    _, string = pdbqt_to_pdb(full_str)
                    display_flex_dict = {}
                    for chain, df in self.pdbqt_editor.pdbqt_chain_dict.items():
                        display, flex = df['Display'], df['Flexible']
                        display_flex_dict[chain] = {'Display': display.to_list(), 'Flexible': flex.to_list()}
                    self.pdbqt_editor = PDBEditor(string)
                    self.pdbqt_editor.parse_pdbqt_text_to_dict(display_flex_dict)
                    self.log_textedit.clear()
                    self._print_settings_to_log()
                    self.curr_format = format
                    self.protein_format_button_group.blockSignals(False)
                else:
                    if self.protein_loader_thread is None:
                        display_flex_dict = {}
                        for chain, df in self.pdbqt_editor.pdbqt_chain_dict.items():
                            display, flex = df['Display'], df['Flexible']
                            display_flex_dict[chain] = {'Display': display.to_list(), 'Flexible': flex.to_list()}
                        self.protein_loader = ThreadedPDBConverter(full_str,
                                                                   display_flex_dict,
                                                                   self.protein_ph_spinbox.value())
                        self.protein_loader_thread = QThread()
                        self.protein_loader.moveToThread(self.protein_loader_thread)
                        self.protein_loader.proteinString.connect(self._set_protein_pdbqt_editor_new_format)
                        self.protein_loader.finished.connect(self.protein_loader_thread.quit)
                        self.protein_loader_thread.finished.connect(self._clean_up_protein_loader)
                        self.protein_loader_thread.started.connect(self.protein_loader.run)
                        self.protein_loader_thread.start()
            self.check_if_enable_docking()
    
    def force_upper_case(self, name: str):
        text = self.input_id_dict[name]['LineEdit'].text()
        text = text.upper()
        self.input_id_dict[name]['LineEdit'].setText(text)
        if name == 'PDB':
            self.input_id_dict[name]['Load_Button'].setEnabled(len(text) == 4)
        elif name == 'AF Database':
            self.input_id_dict[name]['Load_Button'].setEnabled(any((len(text) == 6, len(text) == 10)))
    
    def select_dock_input_file(self):
        files, _ = QFileDialog.getOpenFileName(self,
                                               'Select Input Files',
                                               '',
                                               'PDB Format (*.pdb);;AutoDock Format (*.pdbqt);;Molecule Docker Settings (*.mds)')
        if files:
            self.local_file_line.setText(files)
    
    def select_dock_ligand_directory(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Ligand Directory', '')
        if folder:
            self.input_ligand_line.setText(folder)
    
    def select_dock_output_directory(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Output Directory', '')
        if folder:
            self.output_directory_line.setText(folder)
    
    def select_docked_directory(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Directory Containing Docked Files', '')
        if folder:
            self.docked_dir_line.setText(folder)
    
    def check_if_enable_docking(self):
        def loop_over_param_dict():
            for k in ['dock_center', 'dock_width']:
                for vv in self.param_dict[k].values():
                    if vv is None:
                        return False
            return True
        in_out_condition = (os.path.isdir(self.input_ligand_line.text()),
                            bool(self.output_directory_line.text())     ,)  # output_dir only need to check if it is string
        self.docking_action_dict['Refinement'].setEnabled(all(in_out_condition))
        conditions = (self.pdbqt_editor is not None,
                      loop_over_param_dict()       ,
                      all(in_out_condition)        ,)
        for program in self.supported_docking_programs:
            if program == 'LeDock' and self.ledock_exec is None:
                self.docking_action_dict[program].setEnabled(False)
            else:
                if program == 'LeDock':
                    if all(conditions):
                        self.docking_action_dict[program].setEnabled(self.pdbqt_editor.check_format_type() == 'pdb')
                    else:
                        self.docking_action_dict[program].setEnabled(False)
                else:
                    if all(conditions):
                        self.docking_action_dict[program].setEnabled(self.pdbqt_editor.check_format_type() == 'pdbqt')
                    else:
                        self.docking_action_dict[program].setEnabled(False)
    
    def check_protein_pth(self):
        local_file_pth = self.local_file_line.text()
        if os.path.exists(local_file_pth) and os.path.isfile(local_file_pth) and local_file_pth.endswith(('.pdb', '.cif', '.pdbqt', '.mds')):
            self.protein_convert_button.setEnabled(True)
        else:
            self.protein_convert_button.setDisabled(True)
    
    def check_if_enable_viewing(self):
        t = self.docked_dir_line.text()
        if t and os.path.isdir(t):
            self.show_dir_table_button.setEnabled(True)
        else:
            self.show_dir_table_button.setEnabled(False)
    
    def save_log_file_dialog(self):
        save_file, _ = QFileDialog.getSaveFileName(self, 'Save Log File', '', 'Log File (*.log)')
        if save_file:
            with open(save_file, 'w') as f:
                f.write(self.log_textedit.toPlainText())
    
    def clear_dock_data(self):
        self.param_dict = {'dock_center'   : {'x': None, 'y': None, 'z': None},
                           'dock_width'    : {'x': None, 'y': None, 'z': None},
                           'exhaustiveness': 12                               ,
                           'eval_poses'    : 10                               ,
                           'center_color'  : QColor(0  , 255,   0),
                           'width_color'   : QColor(255, 255, 255, 128),
                           'fpocket'       : None,
                           'hetatm'        : None,}
        self.pdbqt_editor = None
        for d in self.input_id_dict.values():
            d['LineEdit'].clear()
            l = d.get('Status_Label', None)
            if l is not None:
                l.setText('<b>Status :            </b>')
                l.setStyleSheet('')
        self.input_ligand_line.clear()
        self.output_directory_line.clear()
        if not self.is_docking:
            self.log_textedit.clear()
            self.log_tableview.clear_table()
        gc.collect()
    
    def search_log_tableview_text(self):
        curr_text = self.log_search_lineedit.text()
        curr_colm = self.log_search_combobox.currentText()
        self.log_tableview.set_filter(curr_text, curr_colm)
    
    def start_process_docking(self, button_name: str):
        if button_name in ['AutoDock VINA', 'smina', 'qvina2', 'qvinaw']:
            self.start_preprocessing_vina(button_name)
        elif button_name == 'LeDock':
            self.start_processing_ledock()
        elif button_name == 'Refinement':
            self.start_processing_openmm()
    
    def start_preprocessing_vina(self, program_type: str):
        if not self.is_docking:
            out_dir = self.output_directory_line.text()
            lig_dir = self.input_ligand_line.text()
            cache_dir = os.path.join(out_dir, 'cache_files')
            processed_pdbqt_file_pth =  os.path.join(cache_dir, self.saved_pdbqt_file_name+'_processed.pdbqt')  # will be same as rigid for full rigid docking
            rigid_pdbqt_file_pth = os.path.join(cache_dir, self.saved_pdbqt_file_name+'_rigid.pdbqt')
            flex_pdbqt_file_pth = os.path.join(cache_dir, self.saved_pdbqt_file_name+'_flex.pdbqt')
            mds_path = os.path.join(cache_dir, self.saved_pdbqt_file_name+'.mds')
            if not os.path.isdir(cache_dir):
                os.makedirs(cache_dir)
            if self.pdbqt_editor.check_format_type() != 'pdbqt':
                self.log_textedit.append(f'<pre><span style="font-size:{int(self.default_fontsize)}px;color:FireBrick;">AutoDock VINA Require PDBQT Format!</span></pre>')
                return
            pdbqt_str = self.pdbqt_editor.convert_dict_to_pdbqt_text()
            if not pdbqt_str:
                self.log_textedit.append(f'<pre><span style="font-size:{int(self.default_fontsize)}px;color:FireBrick;">No protein chain/position selected.</span></pre>')
                return
            self.log_textedit.clear()
            self.log_textedit.append(f'<span style="font-size:{int(self.default_fontsize)}px;white-space:pre-line"> {str(self.pdbqt_editor)} </span>')
            self.log_textedit.append(f'<span style="font-size:{int(self.default_fontsize)}px;">Docking Program: {program_type}</span>')
            for param, param_v in self.param_dict.items():
                if param in ['exhaustiveness', 'eval_poses']:
                    full_name = self.param_to_full_name_map[param]
                    self.log_textedit.append(f'<span style="font-size:{int(self.default_fontsize)}px;">{full_name}: {param_v}</span>')
                elif param in ['dock_center', 'dock_width']:
                    full_name = self.param_to_full_name_map[param]
                    self.log_textedit.append(f'<span style="font-size:{int(self.default_fontsize)}px;">{full_name}:</span>')
                    self.log_textedit.append(f'<span style="font-size:{int(self.default_fontsize)}px;white-space:pre-line">&nbsp;&nbsp;X: {param_v['x']} </span>')
                    self.log_textedit.append(f'<span style="font-size:{int(self.default_fontsize)}px;white-space:pre-line">&nbsp;&nbsp;Y: {param_v['y']} </span>')
                    self.log_textedit.append(f'<span style="font-size:{int(self.default_fontsize)}px;white-space:pre-line">&nbsp;&nbsp;Z: {param_v['z']} </span>')
            self.log_textedit.append(f'<span style="font-size:{int(self.default_fontsize)}px;"><br/></span>')
            with open(processed_pdbqt_file_pth, 'w') as f:
                f.write(pdbqt_str)
            self.has_flex = self.pdbqt_editor.process_and_save_rigid_flex_pdbqt(rigid_pdbqt_file_pth, flex_pdbqt_file_pth)
            self.rigid_pdbqt_file_pth = rigid_pdbqt_file_pth
            self.flex_pdbqt_file_pth = flex_pdbqt_file_pth
            
            export_dict = {'dock_center'   : self.param_dict['dock_center'],
                           'dock_width'    : self.param_dict['dock_width'],
                           'exhaustiveness': self.param_dict['exhaustiveness'],
                           'eval_poses'    : self.param_dict['eval_poses'],
                           'pdbqt_editor'  : self.pdbqt_editor.pdbqt_chain_dict,
                           'ligand_dir'    : lig_dir,
                           'output_dir'    : out_dir,}
            with open(mds_path, 'wb') as f:
                pickle.dump(export_dict, f)
            
            if program_type in ['AutoDock VINA', 'qvina2', 'qvinaw']:
                ligand_ext = '.pdbqt'
            elif program_type == 'smina':
                ligand_ext = ('.sdf', '.pdbqt')
            
            input_output_dict = {}
            for f in os.listdir(lig_dir):
                if f.endswith(ligand_ext) and not f.startswith('.'):
                    if isinstance(ligand_ext, str):
                        name = f.rsplit(ligand_ext, 1)[0]
                        true_ext = ligand_ext
                    else:
                        for ext in ligand_ext:
                            if f.endswith(ext):
                                name = f.rsplit(ext, 1)[0]
                                true_ext = ext
                                break
                    input_output_dict[os.path.join(lig_dir, f)] = {'docked_name': f'{name}_docked{true_ext}',
                                                                   'name'       : name}
            
            text = f'{len(input_output_dict)} Ligands Found.'
            self.log_textedit.append(f'<p align=\"center\"><span style="font-size:{int(self.default_fontsize)}px;">{text}</span></p>')
            if len(input_output_dict) == 0:
                return
            self.total_length = len(input_output_dict)
            self.step_str_length = len(str(self.total_length))
            
            self.preprocess_thread = QThread()
            self.preprocess_worker = ThreadedVINAPreprocess(input_output_dict, out_dir, cache_dir,
                                                            ligand_ext, program_type)
            self.preprocess_worker.moveToThread(self.preprocess_thread)
            self.preprocess_worker.finalResult.connect(self.continue_processing_vina)
            self.preprocess_worker.finalResult.connect(self.preprocess_thread.quit)
            self.preprocess_thread.finished.connect(self.preprocess_thread.deleteLater)
            self.preprocess_thread.finished.connect(self.preprocess_worker.deleteLater)
            self.preprocess_thread.started.connect(self.preprocess_worker.run)
            self.preprocess_thread.start()
        else:
            self.is_docking = False
            self.dockStopSignal.emit()
            # self.multiprocess_dock_stop()
    
    def continue_processing_vina(self, final_lig_out_dict: dict, name_progress_status_dict: dict,
                                 program_type: str, curr_step: int):
        self.curr_step = curr_step
        param_dict = copy.deepcopy(self.param_dict)
        param_dict['receptor_path'] = self.rigid_pdbqt_file_pth
        if program_type == 'AutoDock VINA':
            param_dict['vina_exec'] = self.vina_exec
        elif program_type == 'smina':
            param_dict['vina_exec'] = self.smina_exec
        elif program_type == 'qvina2':
            param_dict['vina_exec'] = self.qvina2_exec
        elif program_type == 'qvinaw':
            param_dict['vina_exec'] = self.qvinaw_exec
        
        if self.has_flex:
            param_dict['flex_receptor_path'] = self.flex_pdbqt_file_pth
        
        self.progress.setMaximum(self.total_length * 51)    # each protein has 51 *
        if self.curr_step:
            self.log_textedit.append(f'<hr/><br/>')
            self.log_textedit.append(f'<pre><span style="font-size:{int(self.default_fontsize)}px;">{self.curr_step} ligands already docked.<br/></span></pre>')
        self.log_tableview.init_progresses(name_progress_status_dict)
        self.progress.setValue(self.curr_step * 51)
        self.within_dock_progress = -1
        self.remaining_step_for_time = self.total_length - self.curr_step
        self.curr_step_for_time = 0
        self.tik = time.perf_counter()
        
        if not final_lig_out_dict:
            return
        
        self.start_docking_button.setText('Stop')
        self.is_docking = True
        self.log_textedit.append('')
        self.dock_thread = DockingThread()
        self.dock_worker = MultiprocessVINADock(final_lig_out_dict, param_dict,
                                                program_type,
                                                self.docking_concurrency_spinbox.value())
        self.dock_worker.moveToThread(self.dock_thread)
        self.dockStopSignal.connect(self.dock_worker.stop, Qt.ConnectionType.QueuedConnection)
        self.dock_worker.doneSignal.connect(self.docking_single_done)
        self.dock_worker.startSignal.connect(self.name_start_docking)
        self.dock_worker.progressSignal.connect(self.update_name_docking_progress)
        self.dock_worker.finished.connect(self.dock_thread.quit)
        self.dock_worker.finished.connect(self.docking_done_func)
        self.dock_worker.canceled.connect(self.cancel_docking_func)
        self.dock_thread.started.connect(self.dock_worker.run)
        self.dock_thread.start()
        
    def docking_done_func(self, success=True):
        if success:
            self.dock_thread.wait()
            self.dockStopSignal.disconnect(self.dock_worker.stop)
            self.progress.setValue(self.progress.maximum())
            self.log_textedit.append(f'<hr/><br/>')
            self.log_textedit.append(f'<b><span style="font-size:{int(self.default_fontsize * 1.3)}px;">Docking Done ({self.eta_label.text().split('<')[0][1:]})</span></b>')
            self.log_textedit.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.start_docking_button.setEnabled(True)
            self.start_docking_button.setText('Dock')
            self.save_log_text_button.setEnabled(True)
            self.is_docking = False
            self.dock_thread.deleteLater()
            self.dock_worker.deleteLater()
    
    def cancel_docking_func(self):
        self.multiprocess_dock_stop()
        self.dockStopSignal.disconnect(self.dock_worker.stop)
        self.log_textedit.append(f'<hr/><br/>')
        self.log_textedit.append(f'<b><span style="font-size:{int(self.default_fontsize * 1.3)}px;">Docking canceled ({self.eta_label.text().split('<')[0][1:]})</span></b>')
        self.log_textedit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.start_docking_button.setEnabled(True)
        self.start_docking_button.setText('Dock')
        self.save_log_text_button.setEnabled(True)
        self.is_docking = False
        
    def update_docking_progress_text(self, text, raw=False):
        curr_pos = self.log_textedit.verticalScrollBar().value()
        at_max = curr_pos == self.log_textedit.verticalScrollBar().maximum()
        self.log_textedit.moveCursor(QTextCursor.MoveOperation.End)
        if raw:
            self.log_textedit.insertPlainText(text)
        else:
            self.log_textedit.append(text)
        self.log_textedit.moveCursor(QTextCursor.MoveOperation.End)
        if at_max:
            self.log_textedit.verticalScrollBar().setValue(self.log_textedit.verticalScrollBar().maximum())
        else:
            self.log_textedit.verticalScrollBar().setValue(curr_pos)
        
    def update_progress_bar(self):
        t = self.curr_step * 51 + self.within_dock_progress
        self.progress.setValue(t)
        self.progress_label.setText(f'{self.curr_step:{self.step_str_length}}/{self.total_length}')
        passed_time = round(time.perf_counter() - self.tik, 0)  # 59.5 ~ 60 sec will become 60 sec
        if self.curr_step_for_time * 51 + self.within_dock_progress:
            expected_sec = max(0, (self.remaining_step_for_time * 51) * 
                               (passed_time / (self.curr_step_for_time * 51 + self.within_dock_progress)) - passed_time)
            p_hr, p_min, p_sec = self.convert_secs(passed_time)
            e_hr, e_min, e_sec = self.convert_secs(expected_sec)
            # if p_hr or e_hr:
            passed_string = f'{p_hr:02.0f}:{p_min:02.0f}:{p_sec:02.0f}'
            eta_string = f'{e_hr:02.0f}:{e_min:02.0f}:{e_sec:02.0f}'
            # else:
            #     passed_string = f'{p_min:02.0f}:{p_sec:02.0f}'
            #     eta_string = f'{e_min:02.0f}:{e_sec:02.0f}'
        else:
            p_hr, p_min, p_sec = self.convert_secs(passed_time)
            # if p_hr:
            passed_string = f'{p_hr:02.0f}:{p_min:02.0f}:{p_sec:02.0f}'
            eta_string = '??:??:??'
            # else:
            #     passed_string = f'{p_min:02.0f}:{p_sec:02.0f}'
            #     eta_string = '??:??'
        speed = f'{(self.curr_step_for_time * 51 + self.within_dock_progress) / 51 / passed_time:.2f}'
        self.eta_label.setText(f'[{passed_string}<{eta_string}, {speed}it/s]')
        
    def convert_secs(self, sec):
        min, sec = divmod(sec, 60)
        if min >= 60:
            hr, min = divmod(min, 60)
        else:
            hr = 0
        return (hr, min, sec)
    
    def start_processing_ledock(self):
        if not self.is_docking:
            out_dir = self.output_directory_line.text()
            lig_dir = self.input_ligand_line.text()
            cache_dir = os.path.join(out_dir, 'cache_files')
            if self.pdbqt_editor.check_format_type() != 'pdb':
                self.log_textedit.append(f'<pre><span style="font-size:{int(self.default_fontsize)}px;'
                                         f'color:FireBrick;">LeDock Require PDB Format!</span></pre>')
                return
            processed_pdb_file_pth = os.path.join(cache_dir, self.saved_pdbqt_file_name+'_processed.pdb')
            pdb_str = self.pdbqt_editor.convert_dict_to_pdbqt_text()
            if not pdb_str:
                self.log_textedit.append(f'<pre><span style="font-size:{int(self.default_fontsize)}px;'
                                         f'color:FireBrick;">No protein chain/position selected.</span></pre>')
                return
            os.makedirs(cache_dir, exist_ok=True)
            with open(processed_pdb_file_pth, 'w') as f:
                f.write(pdb_str)
            self.log_textedit.clear()
            self.log_textedit.append(f'<span style="font-size:{int(self.default_fontsize)}px;'
                                     f'white-space:pre-line"> {str(self.pdbqt_editor)} </span>')
            
            for f in os.listdir(cache_dir):
                if f.endswith('.in'):
                    name = f.rsplit('.in')[0]
                    os.remove(os.path.join(cache_dir, f))
                    dok_file = os.path.join(out_dir, f'{name}.dok')
                    mol_file = os.path.join(out_dir, f'{name}.mol2')
                    lig_file = os.path.join(out_dir, f'{name}.list')
                    sdf_file = os.path.join(out_dir, f'{name}_out.sdf')
                    check_then_rm_list = [dok_file, mol_file, lig_file, sdf_file]
                    for file in check_then_rm_list:
                        if os.path.isfile(file):
                            os.remove(file)
            
            # {Original Path: Target Path}. LeDock will place the docked molecules in the same directory as input molecules, 
            # so copy the docked moelcules to output dir then remove it after it is done
            all_ligands = {os.path.join(lig_dir, f): os.path.join(out_dir, f) 
                           for f in os.listdir(lig_dir) if 
                           not f.startswith('.') and f.endswith('.mol2')}
            self.log_textedit.append(f'<span style="font-size:{int(self.default_fontsize)}px;'
                                     f'white-space:pre-line">{len(all_ligands)} Ligands Detected!</span>')
            
            ### Export mds settings ###
            export_dict = {'dock_center'   : self.param_dict['dock_center'],
                           'dock_width'    : self.param_dict['dock_width'],
                           'exhaustiveness': self.param_dict['exhaustiveness'],
                           'eval_poses'    : self.param_dict['eval_poses'],
                           'pdbqt_editor'  : self.pdbqt_editor.pdbqt_chain_dict,
                           'ligand_dir'    : lig_dir,
                           'output_dir'    : out_dir,}
            mds_path = os.path.join(cache_dir, self.saved_pdbqt_file_name+'.mds')
            with open(mds_path, 'wb') as f:
                pickle.dump(export_dict, f)
            
            ### Setup LeDock parameters ###
            ligand_list_file = os.path.join(out_dir, '{lig_name}.list')
            ledock_settings_file = os.path.join(cache_dir, '{lig_name}.in')
            center   = self.param_dict['dock_center']
            box_size = self.param_dict['dock_width']
            box_dist = {side: width / 2 for side, width in box_size.items()}
            param_dict = {'x_min'    : center['x'] - box_dist['x'], 'x_max': center['x'] + box_dist['x'],
                          'y_min'    : center['y'] - box_dist['y'], 'y_max': center['y'] + box_dist['y'],
                          'z_min'    : center['z'] - box_dist['z'], 'z_max': center['z'] + box_dist['z'],
                          'poses'    : self.param_dict['eval_poses'],
                          'receptor' : processed_pdb_file_pth,
                          'ligand_l' : ligand_list_file,
                          'setting_f': ledock_settings_file,
                          'dock_exe' : self.ledock_exec,}
            
            self.total_length = len(all_ligands)
            self.step_str_length = len(str(self.total_length))
            self.progress.setMaximum(self.total_length * 51)
            self.curr_step = 0
            docked_ligands = [f.rsplit('_out.sdf')[0] for f in os.listdir(out_dir) if f.endswith('_out.sdf')]
            
            progress_csv = os.path.join(cache_dir, 'dock_progress.csv')
            if os.path.isfile(progress_csv):
                df = pd.read_csv(progress_csv)
                if docked_ligands:
                    df = df[(df['name'].isin(docked_ligands)) | (~df['score'].notna())]
                    df.to_csv(progress_csv, index=None)
                name_score_map = {name: float(score) for name, score in zip(df['name'].to_list(), df['score'].to_list())}
            else:
                name_score_map = {}
                with open(progress_csv, 'w') as f:
                    f.write('name,score\n')
                if docked_ligands:
                    final_line = []
                    for name in docked_ligands:
                        docked_f = os.path.join(out_dir, name+'_out.sdf')
                        with open(docked_f) as d_f:
                            docked_str = d_f.read()
                        for g in re.finditer(ledock_eng_compiled, docked_str):
                            eng = float(g.group(1))    # min score
                            break
                        final_line.append(f'{name},{eng:.2f}')
                        name_score_map[name] = round(eng, 2)
                    with open(progress_csv, 'a') as f:
                        f.write('\n'.join(final_line)+'\n')
            
            name_progress_status_dict = {}
            for full_lig_pth in list(all_ligands):
                lig_name = os.path.basename(full_lig_pth).rsplit('.', 1)[0]
                if lig_name in name_score_map:
                    self.curr_step += 1
                    del all_ligands[full_lig_pth]
                    score = name_score_map[lig_name]
                    if score != score:
                        score = 'Failed'
                    name_progress_status_dict[lig_name] = {'status'  : score,
                                                           'progress': 51}
                else:
                    name_progress_status_dict[lig_name] = {'status'  : 'Pending...',
                                                           'progress': 0}
                    
            if self.curr_step:
                self.log_textedit.append(f'<hr/><br/>')
                self.log_textedit.append(f'<span style="font-size:{int(self.default_fontsize)}px;'
                                         f'white-space:pre-line">{self.curr_step} Ligands Already Docked!</span>')
            
            if not all_ligands:
                return
            
            self.start_docking_button.setText('Stop')
            self.log_tableview.init_progresses(name_progress_status_dict)
            self.progress.setValue(self.curr_step * 51)
            self.is_docking = True
            self.within_dock_progress = 0
            self.remaining_step_for_time = self.total_length - self.curr_step
            self.curr_step_for_time = 0
            self.tik = time.perf_counter()
            
            self.log_textedit.append('')
            self.dock_thread = QThread()
            self.dock_worker = MultiprocessLeDock(self.ledock_exec, all_ligands, param_dict,
                                                  self.docking_concurrency_spinbox.value())
            self.dock_worker.moveToThread(self.dock_thread)
            self.dock_worker.doneSignal.connect(self.docking_single_done)
            self.dockStopSignal.connect(self.dock_worker.stop, Qt.ConnectionType.QueuedConnection)
            self.dock_worker.startSignal.connect(self.name_start_docking)
            self.dock_worker.finished.connect(self.dock_thread.quit)
            self.dock_worker.finished.connect(self.docking_done_func)
            self.dock_worker.canceled.connect(self.cancel_docking_func)
            self.dock_thread.started.connect(self.dock_worker.run)
            self.dock_thread.start()
        else:
            self.is_docking = False
            self.dockStopSignal.emit()
    
    def docking_single_done(self, done_text: str, name: str, status: str):
        self.text_extractor.update_text.emit(f'<p><span style="font-size:'
                                             f'{int(self.default_fontsize)}px;">{done_text}</span></p>',
                                             False)
        self.log_tableview.update_progress_status(name, status)
        curr_progress = self.log_tableview.get_current_progress(name)
        self.log_tableview.set_progress_bar_value(name, 51)
        self.curr_step += 1
        self.curr_step_for_time += 1
        if self.within_dock_progress > 0:
            self.within_dock_progress -= curr_progress
        self.update_progress_bar()
    
    def name_start_docking(self, name: str, status_text: str='Docking...'):
        self.log_tableview.update_progress_status(name, status_text)
    
    def update_name_docking_progress(self, name: str, step: float=1):
        self.log_tableview.update_progress_bar_by_add(name, step)
        self.within_dock_progress += step
        self.update_progress_bar()
    
    def multiprocess_dock_stop(self):
        self.dock_thread.quit()
        self.dock_thread.wait()
        self.dock_worker.deleteLater()
        self.dock_thread.deleteLater()
    
    def start_processing_openmm(self):
        if not self.is_docking:
            self.log_textedit.clear()
            out_dir = self.output_directory_line.text()
            complex_dir = self.input_ligand_line.text()
            
            all_complexes_map = {}
            for d in os.listdir(complex_dir):
                dir = os.path.join(complex_dir, d)
                if os.path.isdir(dir):
                    all_complexes_map[d] = dir
            
            if not all_complexes_map:
                self.log_textedit.append(f'<pre><span style="font-size:{int(self.default_fontsize)}px;'
                                         f'color:FireBrick;">No complexes found.</span></pre>')
                return
            
            self.total_length = len(all_complexes_map)
            self.step_str_length = len(str(self.total_length))
            self.log_textedit.append(f'<pre><span style="font-size:{int(self.default_fontsize)}px;">'
                                     f'{self.total_length} Ligands Detected!</span></pre>')
            
            os.makedirs(out_dir, exist_ok=True)
            docked_name = [f.rsplit('_output.mdm')[0] for f in os.listdir(out_dir) if f.endswith('_output.mdm')]
            minimize_csv = os.path.join(out_dir, 'minimize.csv')
            if not os.path.isfile(minimize_csv):
                with open(minimize_csv, 'w') as f:
                    f.write('Name,Minimized Energy,Old Score,'+','.join([h for h in chem_prop_to_full_name_map.values()])+'\n')
            minimize_df = pd.read_csv(minimize_csv)
            if docked_name:
                minimize_df = minimize_df[(minimize_df['Name'].isin(docked_name)) | (~minimize_df['Minimized Energy'].notna())]
                minimize_df.to_csv(minimize_csv, index=None)
            docked_name_score_map =  {name: score for name, score in zip(minimize_df['Name'].to_list(),
                                                                         minimize_df['Minimized Energy'].to_list())}
            
            all_complexes_map = dict(sorted(all_complexes_map.items()))
            name_progress_dict = {name: {'progress': 0, 'status': 'Pending...'} for name in all_complexes_map}
            self.curr_step = 0
            for name, score in docked_name_score_map.items():
                self.curr_step += 1
                del all_complexes_map[name]
                name_progress_dict[name]['progress'] = 51
                name_progress_dict[name]['status'] = score if score == score else 'Failed'
            
            if not all_complexes_map:
                self.log_textedit.append(f'<pre><span style="font-size:{int(self.default_fontsize)}px;">'
                                         f'All ligands docked.</span></pre>')
                return
            
            self.log_tableview.init_progresses(name_progress_dict)
            self.start_docking_button.setText('Stop')
            self.progress.setMaximum(self.total_length * 51)
            if self.curr_step:
                self.log_textedit.append(f'<hr/><br/>')
                self.log_textedit.append(f'<pre><span style="font-size:{int(self.default_fontsize)}px;"'
                                         f'>{self.curr_step} ligands already docked.</span></pre>')
            self.progress.setValue(self.curr_step * 51)
            self.is_docking = True
            self.within_dock_progress = 0
            self.remaining_step_for_time = self.total_length - self.curr_step
            self.curr_step_for_time = 0
            self.tik = time.perf_counter()
            
            self.log_textedit.append('')
            self.dock_thread = DockingThread()
            self.dock_worker = MultiprocessRefine(all_complexes_map, out_dir, minimize_csv,
                                                  self.docking_concurrency_spinbox.value(),
                                                  self.protein_ph_spinbox.value())
            self.dock_worker.moveToThread(self.dock_thread)
            self.dockStopSignal.connect(self.dock_worker.stop, Qt.ConnectionType.QueuedConnection)
            self.dock_worker.doneSignal.connect(self.docking_single_done)
            self.dock_worker.startSignal.connect(self.name_start_docking)
            self.dock_worker.progressSignal.connect(self.update_name_docking_progress)
            self.dock_worker.finished.connect(self.dock_thread.quit)
            self.dock_worker.finished.connect(self.docking_done_func)
            self.dock_worker.canceled.connect(self.cancel_docking_func)
            self.dock_thread.started.connect(self.dock_worker.run)
            self.dock_thread.start()
        else:
            self.is_docking = False
            self.dockStopSignal.emit()
            
    def _openmm_minimize(self, all_complex_map: dict, out_dir: str, minimize_csv: str):
        # Depracated. Using multiprocessing instead since that is faster.
        from utilities.refine_utilis import minimize_complex
        for name, input_dir in all_complex_map.items():
            self.text_extractor.update_text.emit('<hr/><br/>', False)
            if self.is_docking:
                output_mdm = os.path.join(out_dir, f'{name}_output.mdm')
                protein_pth = os.path.join(input_dir, 'protein.pdb')
                ligand_pth = os.path.join(input_dir, f'{name}.sdf')
                self.text_extractor.update_text.emit(f'<p><span style="font-size:{int(self.default_fontsize)}px;"'
                                                     f'>Refining {name}...</span></p>',
                                                     False)
                self.log_tableview.update_progress_status(name, 'Refining...')
                minimize_complex(protein_pth, ligand_pth, output_mdm, minimize_csv, self.text_extractor, self.log_tableview)
                self.curr_step += 1
                self.curr_step_for_time += 1
                self.update_progress_bar()
        if self.is_docking:
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = None
            self.docking_done_func(True)
    
    def show_docking_setting_dialog(self):
        setting_dialog = SettingDialog(self.param_dict, self.curr_display_mode, copy.deepcopy(self.pdbqt_editor))
        if setting_dialog.exec():
            self.pdbqt_editor = copy.deepcopy(setting_dialog.pdbqt_editor)
            self.param_dict = copy.deepcopy(setting_dialog.vina_param)
            self.check_if_enable_docking()
            if not self.is_docking:
                self.log_textedit.clear()
                self._print_settings_to_log()
                
    def change_dark_light_mode(self):
        if self.curr_display_mode == 'dark':
            self.curr_display_mode = 'light'
            self.dark_light_swap_button.setIcon(self.dark_icon)
            self.dark_light_swap_button.setToolTip('To dark mode')
        else:
            self.curr_display_mode = 'dark'
            self.dark_light_swap_button.setIcon(self.light_icon)
            self.dark_light_swap_button.setToolTip('To light mode')
        self.structure_browser_widget.setup_theme(self.curr_display_mode)
        qdarktheme.setup_theme(self.curr_display_mode, custom_colors={'[light]': {'background': '#f0f0f0',}})
        self.stop_appending_text = True
        self.convert_textedit.update_color_style(self.curr_display_mode, self.convert_default_fontsize)
        self.stop_appending_text = False
        tpr_color = 'LimeGreen' if self.curr_display_mode == 'dark' else 'ForestGreen'
        fpr_color = 'Tomato'    if self.curr_display_mode == 'dark' else 'Maroon'
        self.tpr_label.setStyleSheet(f'QLabel {{ font-size: 14px; color: {tpr_color}; font-weight: bold; }}')
        self.fpr_label.setStyleSheet(f'QLabel {{ font-size: 14px; color: {fpr_color}; font-weight: bold; }}')
        if not self.allow_plot:
            if self.curr_display_mode == 'dark':
                self.template_combo.setCurrentText('plotly_dark')
                self.roc_template_combo.setCurrentText('plotly_dark')
                self.browser_plot.setup_html(self.empty_dark_plot_html)
                self.browser_plot.setup_background_color('_dark')
                self.browser_roc_plot.setup_html(self.empty_dark_plot_html)
                self.browser_roc_plot.setup_background_color('_dark')
            else:
                self.template_combo.setCurrentText('plotly')
                self.roc_template_combo.setCurrentText('plotly')
                self.browser_plot.setup_html(self.empty_light_plot_html)
                self.browser_plot.setup_background_color('')
                self.browser_roc_plot.setup_html(self.empty_light_plot_html)
                self.browser_roc_plot.setup_background_color('')
        self.change_icon_light_dark()
        for i in range(self.shopper_stacked_widget.count()):
            w = self.shopper_stacked_widget.widget(i)
            w.change_dark_light_mode(self.curr_display_mode)
    
    def change_log_font_size(self, fontsize):
        diff = fontsize - self.default_fontsize
        def increment_font_size(match):
            return f'font-size:{int(match.group(1)) + diff}px;'
        self.default_fontsize = fontsize
        self.log_textedit.setStyleSheet(f'font-family: "Courier New", Courier, monospace; font-size:{int(self.default_fontsize)}px;')
        t = self.log_textedit.toHtml()
        t = re.sub(r'font-size:(\d+)px;', increment_font_size, t)
        curr_pos = self.log_textedit.verticalScrollBar().value()
        self.log_textedit.setHtml(t)
        self.log_textedit.verticalScrollBar().setValue(curr_pos)
    
    def load_from_online_db(self, name):
        if self.protein_loader_thread is None:
            self.protein_format_button_group.blockSignals(True)
            is_pdbqt = self.protein_format_radios['pdbqt'].isChecked()
            self.protein_loader = ThreadedProteinDownloader(self.input_id_dict[name]['LineEdit'].text(),
                                                            name,
                                                            is_pdbqt,
                                                            self.fill_protein_gap_ckbox.isChecked(),
                                                            self.protein_ph_spinbox.value())
            self.param_dict['fpocket'] = None
            self.param_dict['hetatm'] = None
            self.input_id_dict[name]['Status_Label'].setText('<b>Retrieving...<b/>')
            self.log_textedit.clear()
            self.protein_loader_thread = QThread()
            self.protein_loader.moveToThread(self.protein_loader_thread)
            self.protein_loader.proteinDownloadStatus.connect(self._set_protein_download_status)
            self.protein_loader.conversionString.connect(self._protein_conversion_log_status)
            self.protein_loader.proteinString.connect(self._set_protein_pdbqt_editor)
            self.protein_loader.finished.connect(self.protein_loader_thread.quit)
            self.protein_loader_thread.finished.connect(self._clean_up_protein_loader)
            self.protein_loader_thread.started.connect(self.protein_loader.run)
            self.protein_loader_thread.start()
    
    def _set_protein_download_status(self, status_str: str, db_type: str, title: str, status_bool: bool):
        status_color = {True : 'SeaGreen',
                        False: 'PaleVioletRed'}
        self.input_id_dict[db_type]['Status_Label'].setText(f'<b><span style="color:{status_color[status_bool]};"'
                                                            f'>{status_str}</span><b/>')
        self.log_textedit.clear()
        if title:
            self.log_textedit.append(f'<pre><span style="font-size:{int(self.default_fontsize)}px;'
                                    f'">Title: {title}</span></pre>')
            self.log_textedit.append(f'<pre><span style="font-size:{int(self.default_fontsize)}px;'
                                    f'">Processing protein...</span></pre>')
    
    def _protein_conversion_log_status(self, conversion_str: str, status_bool: bool):
        status_color = {True : 'ForestGreen',
                        False: 'FireBrick'}
        # self.log_textedit.clear()
        self.log_textedit.append(f'<pre><span style="font-size:{int(self.default_fontsize)}px;'
                                 f'color:{status_color[status_bool]};">{conversion_str}</span></pre>')
    
    def _set_protein_pdbqt_editor(self, protein_str: str, protein_id: str, hetatm_dict: dict):
        if protein_str == 'Failed':
            self.pdbqt_editor = None
        else:
            self.saved_pdbqt_file_name = protein_id
            self.pdbqt_editor = PDBEditor(protein_str)
            try:
                self.pdbqt_editor.parse_pdbqt_text_to_dict()
                self.curr_format = self.pdbqt_editor.check_format_type()
                self.protein_format_radios[self.curr_format].setChecked(True)
                self.param_dict['hetatm'] = hetatm_dict if hetatm_dict else None
                self.check_if_enable_docking()
            except Exception as e:
                self.pdbqt_editor = None
                text = f'Failed to parse pdb file. Please make sure the input PDB format is correct:\n{e}'
                self.log_textedit.append(f'<pre><span style="font-size:{int(self.default_fontsize)}px;'
                                         f'color:FireBrick;">{text}</span></pre>')
    
    def _set_protein_pdbqt_editor_new_format(self, protein_str: str, display_flex_dict: dict):
        self.pdbqt_editor = PDBEditor(protein_str)
        self.pdbqt_editor.parse_pdbqt_text_to_dict(display_flex_dict)
        self.curr_format = 'pdbqt'
        self.protein_format_radios[self.curr_format].setChecked(True)
        self.log_textedit.clear()
        self._print_settings_to_log()
        self.protein_format_button_group.blockSignals(True)
    
    def _clean_up_protein_loader(self):
        self.protein_loader_thread.wait()
        self.protein_loader.deleteLater()
        self.protein_loader_thread.deleteLater()
        self.protein_loader_thread = None
        self.protein_format_button_group.blockSignals(False)
    
    def _print_settings_to_log(self):
        color_dict  = {True: 'SeaGreen', False: '#f03939' }
        marker_dict = {True: '&#10003;', False: '&#10799;'}
        check_dict = {'Dock Center': list(self.param_dict['dock_center'].values()),
                      'Dock Box'   : list(self.param_dict['dock_width'].values()),
                      'Protein'    : [self.pdbqt_editor]}
        for name, check_list in check_dict.items():
            check_bool = True
            for c in check_list:
                check_bool &= c is not None
            self.log_textedit.append(f'<span style="font-size:{int(self.default_fontsize)}px;color:{color_dict[check_bool]};"> {marker_dict[check_bool]} {name} </span>')
        if self.pdbqt_editor is not None:
            self.log_textedit.append(f'<span style="font-size:{int(self.default_fontsize)}px;white-space:pre-line"> {str(self.pdbqt_editor)} </span>')
    
    def load_file_from_local(self):
        local_pth = self.local_file_line.text()
        if not os.path.isfile(local_pth):
            return
        self.param_dict['fpocket'] = None
        self.param_dict['hetatm'] = None
        self.log_textedit.clear()
        if local_pth.endswith(('.pdb', '.cif')):
            is_pdbqt = self.protein_format_radios['pdbqt'].isChecked()
            self.curr_format = 'pdbqt' if is_pdbqt else 'pdb'
            if self.protein_loader_thread is None:
                self.protein_loader = ThreadedLocalPDBLoader(local_pth,
                                                             is_pdbqt,
                                                             self.fill_protein_gap_ckbox.isChecked(),
                                                             self.protein_ph_spinbox.value())
                self.protein_loader_thread = QThread()
                self.protein_loader.moveToThread(self.protein_loader_thread)
                self.protein_loader.conversionString.connect(self._protein_conversion_log_status)
                self.protein_loader.proteinString.connect(self._set_protein_pdbqt_editor)
                self.protein_loader.finished.connect(self.protein_loader_thread.quit)
                self.protein_loader_thread.finished.connect(self._clean_up_protein_loader)
                self.protein_loader_thread.started.connect(self.protein_loader.run)
                self.protein_loader_thread.start()
        elif local_pth.endswith('.pdbqt'):
            with open(local_pth, 'r') as f:
                pdbqt_string = f.readline()
            self.curr_format = 'pdbqt'
            self.protein_format_radios['pdbqt'].setChecked(True)
            self.pdbqt_editor = PDBEditor(pdbqt_string)
            self.pdbqt_editor.parse_pdbqt_text_to_dict()
            text = 'Protein structure loaded. (PDBQT Format)'
            self.log_textedit.append(f'<span style="font-size:{int(self.default_fontsize)}px;color:ForestGreen;">{text}</span>')
            self.check_if_enable_docking()
            self.saved_pdbqt_file_name = os.path.basename(local_pth).rsplit('.', 1)[0]
        else:   # mds file
            with open(local_pth, 'rb') as f:
                setting_dict = pickle.load(f)
                for k, v in setting_dict.items():
                    if k not in ['pdbqt_editor', 'ligand_dir', 'output_dir']:
                        self.param_dict[k] = v
                    elif k == 'pdbqt_editor':
                        self.pdbqt_editor = PDBEditor()
                        self.pdbqt_editor.pdbqt_chain_dict = setting_dict[k]
                        self.curr_format = self.pdbqt_editor.check_format_type()
                        self.protein_format_radios[self.curr_format].setChecked(True)
                    elif k == 'ligand_dir':
                        self.input_ligand_line.setText(setting_dict[k])
                    elif k == 'output_dir':
                        self.output_directory_line.setText(setting_dict[k])
            text = 'Settings loaded.'
            self.log_textedit.append(f'<span style="font-size:{int(self.default_fontsize)}px;color:DodgerBlue;">{text}</span>')
            self._print_settings_to_log()
            self.check_if_enable_docking()
            self.saved_pdbqt_file_name = os.path.basename(local_pth).rsplit('.', 1)[0]
    
    def view_docking_result(self):
        self.mdlname_pdbqtcombiner_map = {}
        self.name_eng_dict = {}
        docked_dir = self.docked_dir_line.text()
        is_min = False
        files = [os.path.join(docked_dir, f) for f in os.listdir(docked_dir) if f.endswith(('.pdbqt', '.sdf')) 
                 and not f.startswith('.') and os.path.getsize(os.path.join(docked_dir, f)) > 0]
        if not files:
            is_min = True
            files = [os.path.join(docked_dir, f) for f in os.listdir(docked_dir) if f.endswith('.mdm') and not f.startswith('.')]
            if not files:
                QMessageBox.critical(self,
                                     'Read Error',
                                     f'No docked Files Found in "{docked_dir}".')
                return
        if not is_min:
            cache_pth = os.path.join(docked_dir, 'cache_files')
            protein_ligand_cache = os.path.join(cache_pth, 'protein_ligand.db')
            if not os.path.isfile(protein_ligand_cache):
                if not os.path.isdir(cache_pth):
                    os.makedirs(cache_pth)
                conn = sqlite3.connect(protein_ligand_cache)
                cur = conn.cursor()
                cur.execute("""CREATE TABLE ProLigDB
                            (name TEXT NOT NULL UNIQUE,
                            protein BLOB NOT NULL,
                            ligand BLOB NOT NULL);""")
                conn.commit()
                cur.execute("""CREATE TABLE ChemProp
                            (name TEXT NOT NULL UNIQUE,
                            prop BLOB NOT NULL,
                            fragment BLOB);""")
                conn.commit()
                protein_file = None
                if os.path.isdir(cache_pth):
                    for f in os.listdir(cache_pth):
                        if f.endswith(('_processed.pdbqt', '_processed.pdb')):
                            protein_file = os.path.join(cache_pth, f)
                            break
                if protein_file is None:
                    protein_file, _ = QFileDialog.getOpenFileName(self,
                                                                'Protein PDB(QT) Path',
                                                                '', 'Protein File (*.pdb *.pdbqt)')
                if protein_file:
                    with open(protein_file, 'r') as f:
                        protein_pdbqt_str = f.read()
                    protein_data, protein_pdb = pdbqt_to_pdb(protein_pdbqt_str)
                    protein_data_byte = sqlite3.Binary(pickle.dumps(protein_data))
                    cur.execute("""CREATE TABLE ProteinReference
                                (Reference TEXT NOT NULL UNIQUE,
                                PDB_String TEXT NOT NULL UNIQUE,
                                PDB_Data BLOB NOT NULL);""")
                    cur.execute("""
                                INSERT INTO ProteinReference (Reference, PDB_String, PDB_Data) 
                                VALUES (?, ?, ?)
                                """, ('ref', protein_pdb, protein_data_byte))
                    conn.commit()
                else:
                    protein_data = None
                conn.close()
            else:
                conn = sqlite3.connect(protein_ligand_cache)
                cur = conn.cursor()
                cur.execute("""
                            SELECT name 
                            FROM sqlite_master 
                            WHERE type='table' 
                            AND name=?;
                            """, ('ProteinReference',))
                result = cur.fetchone()
                if not result:
                    protein_data = None
                else:
                    cur.execute(f"""SELECT * FROM ProteinReference WHERE Reference = 'ref'""")
                    _, _, protein_data = cur.fetchone()
                    protein_data = pickle.loads(protein_data)
                conn.close()
        else:
            protein_ligand_cache = None
            protein_data = None
        self.total_files = len(files)
        analyze_fragment_bool = self.analyze_fragment_checkbox.isChecked()
        calc_chemprop_bool = self.calc_chemprop_checkbox.isChecked()
        self.table_progress.setValue(0)
        self.table_progress.setRange(0, self.total_files)
        self.step_str_length = len(str(self.total_files))
        self.table_progress_label.setText(f'{0:{self.step_str_length}}/{self.total_files}')
        self.prop_dict = self.result_table.retrieve_empty_dict(analyze_fragment_bool, calc_chemprop_bool, is_min)
        self.show_dir_table_button.setText('Reading')
        self.show_dir_table_button.setDisabled(True)
        self.curr_view_step = 0
        
        self.thread = QThread()
        self.worker = MultiprocessReader(files, analyze_fragment_bool,
                                         protein_data, protein_ligand_cache,
                                         calc_chemprop_bool,
                                         self.show_only_rank_1_checkbox.isChecked())
        self.worker.moveToThread(self.thread)
        self.worker.progress.connect(self.update_table_progress)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.update_final_table_result)
        self.thread.started.connect(self.worker.run)
        self.thread.start()
    
    def update_table_progress(self, value, result_dict, passed_time, part_time):
        self.curr_view_step += 1
        self.table_progress.setValue(self.curr_view_step)
        self.table_progress_label.setText(f'{self.curr_view_step:{self.step_str_length}}/{self.total_files}')
        expected_sec = max(0, self.total_files / (self.curr_view_step / passed_time) - passed_time)
        if part_time > 0:
            speed = f'{value / part_time:.2f}'
        else:
            speed = f'{0:.2f}'
        p_hr, p_min, p_sec = self.convert_secs(passed_time)
        e_hr, e_min, e_sec = self.convert_secs(expected_sec)
        if p_hr or e_hr:
            passed_string = f'{p_hr:02.0f}:{p_min:02.0f}:{p_sec:02.0f}'
            eta_string = f'{e_hr:02.0f}:{e_min:02.0f}:{e_sec:02.0f}'
        else:
            passed_string = f'{p_min:02.0f}:{p_sec:02.0f}'
            eta_string = f'{e_min:02.0f}:{e_sec:02.0f}'
        if self.curr_view_step == self.total_files:
            self.table_eta_label.setText(f'[{passed_string}<{eta_string}, {self.total_files / passed_time:.2f}it/s]')
        else:
            self.table_eta_label.setText(f'[{passed_string}<{eta_string}, {speed}it/s]')
        self.name_eng_dict.update(result_dict.pop('Energies'))
        self.mdlname_pdbqtcombiner_map.update(result_dict.pop('Structure'))
        for k, v in result_dict.items():
            if k not in self.prop_dict: # CNN Affinity & CNN Score
                self.prop_dict[k] = []
            if isinstance(v, list):
                v = v[0]
            self.prop_dict[k].append(v)
    
    def update_final_table_result(self):
        self.show_dir_table_button.setEnabled(True)
        self.show_dir_table_button.setText('View')
        if 'CNN Score' in self.prop_dict:
            self.name_tree.setHeaderLabels(['Name', 'Score', 'CNN Score', 'CNN Aff.'])
        else:
            self.name_tree.setHeaderLabels(['Name', 'Score', 'RMSD L.B.', 'RMSD U.B.'])
        self.result_table.df = pd.DataFrame(self.prop_dict)
        self.result_table.reset_current_table()
        if self.result_table.bool_filter is not None and 'SMILES' in self.result_table.processing_df.columns:
            supplier_dict = self.result_table.processing_df.to_dict('list')
            for i in range(self.shopper_stacked_widget.count()):
                w: PubChemSupplierFinderWidget = self.shopper_stacked_widget.widget(i)
                w.add_new_mols_to_supplier_df(supplier_dict)
        self.plot_filter_dataframe.set_df(self.prop_dict)
        self.plot_filter_dict = self.plot_filter_dataframe.energy_outlier_detection(self.plot_filter_dict)
        self.plot_filter_dataframe.apply_filter(self.plot_filter_dict)
        self.auto_determine_button.setEnabled('QED' in self.result_table.df.columns)
        self.save_options_button.setEnabled(True)
        self.update_structure_tree_btn.setEnabled(True)
        self.save_structure_btn.setEnabled(True)
        self.reorient_structure_btn.setEnabled(True)
        self.show_ligplot_btn.setEnabled(True)
        self.fragment_button.setEnabled('Fragments' in self.result_table.df.columns)
        
        combo_texts = list(self.prop_dict)
        combo_texts.remove('File Path')
        combo_texts.remove('Name')
        if 'Fragments' in combo_texts:
            combo_texts.remove('Fragments')
        if 'SMILES' in combo_texts:
            combo_texts.remove('SMILES')
        combo_texts = [''] + combo_texts
        self.x_combo.clear()
        self.x_combo.addItems(combo_texts)
        self.y_combo.clear()
        self.y_combo.addItems(combo_texts)
        self.color_combo.clear()
        self.color_combo.addItems(combo_texts)
        
        self.allow_plot = True
        self.plot_and_view_distribution()
        if len(self.result_table.filtered_df) <= 100:
            self.update_structure_tree_with_filterd_df()
    
    def clear_table_data_and_other(self):
        self.result_table.clear_everything()
        self.plot_filter_dataframe.clear_everything()
        self.name_tree.clear_everything()
        self.contact_tabs.clear_tables()
        self.table_progress.setValue(0)
        self.auto_determine_button.setEnabled(False)
        self.save_options_button.setEnabled(False)
        self.update_structure_tree_btn.setEnabled(False)
        self.save_structure_btn.setEnabled(False)
        self.reorient_structure_btn.setEnabled(False)
        self.fragment_button.setEnabled(False)
        self.structure_browser_widget.clear_stage()
        self.allow_plot = False
        for i in range(self.shopper_stacked_widget.count()):
            w = self.shopper_stacked_widget.widget(i)
            w.clear_all()
        gc.collect()
    
    def apply_table_filters(self):
        dialog = TableFilterDialog(self.result_table.chem_filter_dict,
                                   self.result_table.chem_column_dict,
                                   self.result_table.between_chem_ops_dict)
        if dialog.exec():
            self.result_table.chem_filter_dict = copy.deepcopy(dialog.chem_filter_dict)
            self.result_table.chem_column_dict = copy.deepcopy(dialog.chem_column_dict)
            self.result_table.between_chem_ops_dict = copy.deepcopy(dialog.between_chem_ops_dict)
            if self.result_table.df is not None:
                self.result_table.reset_current_table()
                if len(self.result_table.filtered_df) <= 500:
                    self.update_structure_tree_with_filterd_df()
                    self.update_structure_tree_with_filterd_df()
    
    def update_structure_tree_with_filterd_df(self):
        self.structure_browser_widget.clear_stage()
        sorted_name_filepath_df = self.result_table.processing_df['Name']
        passed_eng_dict = {}
        for name in sorted_name_filepath_df:
            passed_eng_dict[name] = self.name_eng_dict[name]
        self.name_tree.populate_tree(passed_eng_dict)
        
        # Also update shopper too
        if self.result_table.bool_filter is not None and 'SMILES' in self.result_table.processing_df.columns:
            supplier_dict = self.result_table.processing_df.to_dict('list')
            for i in range(self.shopper_stacked_widget.count()):
                w = self.shopper_stacked_widget.widget(i)
                w.clear_all()
                w.add_new_mols_to_supplier_df(supplier_dict)        
    
    def add_ligand_to_browser(self, parent_name: str, mdl_name: str, block_contact: bool):
        name = f'{parent_name} {mdl_name}'
        combined_pdb = self.mdlname_pdbqtcombiner_map[name].get_combined_string()
        self.structure_browser_widget.load_protein_ligand_pdbqt_string(combined_pdb,
                                                                       name,
                                                                       block_contact)
    
    def rm_ligand_from_browser(self, parent_name: str, mdl_name: str):
        name = f'{parent_name} {mdl_name}'
        self.structure_browser_widget.remove_name_from_webpage(name)
    
    def update_contacts_of_selected(self, name: str):
        self.curr_struct_contact_name = name
        contact = self.structure_browser_widget.shown_contact_dict.get(name, None)
        self.contact_tabs.clear_tables()
        if contact is not None:
            for c_dict in contact.values():
                protein, ligand, dist, type = c_dict['atom1'], c_dict['atom2'], c_dict['distance'], c_dict['type']
                for aa_prefix in self.aminoAcids:
                    if ligand.startswith(aa_prefix):
                        protein, ligand = ligand, protein
                        break
                self.contact_tabs.all_tables_dict[type.capitalize()].add_contact_to_table(protein, ligand, dist)
                self.contact_tabs.change_button_color()
    
    def reorient_embedded_structure(self):
        self.structure_browser_widget.reorient()
    
    def save_current_selected_structure(self):
        dir = QFileDialog.getExistingDirectory(self, 'Select Output Directory', '')
        if dir:
            all_names: list[str] = list(self.structure_browser_widget.shown_sidechain_dict)
            for name in all_names:
                contact_dict = self.structure_browser_widget.shown_contact_dict.get(name, None)
                ligand_name, model_num = name.rsplit(' ', 1)
                model_num = model_num[1:]   # remove hash symbol
                parent_dir = os.path.join(dir, ligand_name)
                if not os.path.isdir(parent_dir):
                    os.mkdir(parent_dir)
                target_dir = os.path.join(parent_dir, f'Model_{model_num}')
                if not os.path.isdir(target_dir):
                    os.mkdir(target_dir)
                protein, ligand, complex = self.mdlname_pdbqtcombiner_map[name].get_all_pdb()
                if protein is not None:
                    with open(os.path.join(target_dir, f'{ligand_name}_{model_num}_protein.pdb'), 'w') as f:
                        f.write(protein)
                with open(os.path.join(target_dir, f'{ligand_name}_{model_num}_ligand.pdb'), 'w') as f:
                    f.write(ligand)
                if complex is not None:
                    with open(os.path.join(target_dir, f'{ligand_name}_{model_num}_complex.pdb'), 'w') as f:
                        f.write(complex)
                if contact_dict is not None:
                    contact_df = pd.DataFrame(contact_dict).T.drop(columns=['atom1Sel', 'atom2Sel'])
                    contact_df.to_csv(os.path.join(target_dir, f'{ligand_name}_{model_num}_contact.csv'), index=None)
                sdf_str, energy = self._extract_single_sdf_format(ligand_name, int(model_num))
                supp = Chem.SDMolSupplier()
                supp.SetData(sdf_str)
                with Chem.SDWriter(os.path.join(target_dir, f'{ligand_name}_{model_num}_ligand.sdf')) as w:
                    mol = next(supp)
                    mol = Chem.AddHs(mol, addCoords=True)
                    mol.SetProp('_Name', f'{ligand_name}_{model_num}')
                    mol.SetProp('VINA Energy', f'{energy[0]}')
                    if len(energy) > 1:
                        mol.SetProp('RMSD L.B.', f'{energy[1]}')
                        mol.SetProp('RMSD U.B.', f'{energy[2]}')
                    w.write(mol)
    
    def _extract_pdbqt_format(self, processed_df: pd.DataFrame):
        retrieved_text = ''
        for idx, row in processed_df.iterrows():
            name = row['Name']
            e    = row['Score']
            target_file = row['File Path']
            with open(target_file, 'r') as pdbqt_f:
                for i, l in enumerate(pdbqt_f):
                    if i == 1:
                        retrieved_text += f'REMARK VINA RESULT: {e} {name} rank_{idx+1}\n'
                    elif l.startswith(('MODEL', 'ROOT', 'ENDROOT', 'BRANCH', 'ENDBRANCH', 'ATOM', 'TORSDOF')):
                        retrieved_text += l
                    if l.startswith('ENDMDL'):
                        retrieved_text += l
                        break
        return retrieved_text
    
    def _extract_single_sdf_format(self, name: str, mdl_num: int):
        f_p = self.result_table.df[self.result_table.df['Name'] == name]['File Path'].to_list()[0]
        if f_p.endswith('.pdbqt'):
            with open(f_p, 'r') as f:
                ligand_pdbqt_str = f.read()
            models = re.findall(vina_mdl_compiled, ligand_pdbqt_str)
            submdl_str = models[mdl_num - 1][0]
            energy = [s.split()[3:6] for s in re.findall(vina_eng_compiled, ligand_pdbqt_str)][mdl_num - 1]
            pdbqt_mol = PDBQTMolecule(submdl_str, poses_to_read=1, skip_typing=True)
            output_string, _ = RDKitMolCreate.write_sd_string(pdbqt_mol)
        elif f_p.endswith('.sdf'):
            supp = Chem.SDMolSupplier(f_p, removeHs=False)
            for i, mol in enumerate(supp, start=1):
                if i == mdl_num:
                    final_mol = mol
                    break
            AllChem.AssignStereochemistryFrom3D(final_mol, replaceExistingTags=False)
            AllChem.AssignStereochemistry(final_mol, force=True, cleanIt=True)
            sdf_io = io.StringIO()
            with Chem.SDWriter(sdf_io) as writer:
                writer.write(final_mol)
            output_string = sdf_io.getvalue()
            energy = None
            for compiled_regex in sdf_regex_list:
                searched = re.search(compiled_regex, output_string)
                if searched is not None:
                    energy = re.findall(sdf_match_eng_rmsd_compiled, searched.group(0))
                    break
            if energy is None:  # smina format
                energy = re.findall(smina_eng_compiled, output_string)
                if not energy:
                    energy = None
        elif f_p.endswith('.mdm'):
            output_string, energy = self.mdlname_pdbqtcombiner_map[f'{name} #1'].get_sdf()
        return output_string, energy
    
    def _extract_sdf_format(self, processed_df: pd.DataFrame):
        retrieved_text_dict = {}
        for _, row in processed_df.iterrows():
            target_file = row['File Path']
            name = os.path.basename(target_file).rsplit('.', 1)[0]
            with open(target_file, 'r') as pdbqt_f:
                pdbqt_str = pdbqt_f.read()
            energy_rmsd_list = [s.split()[3:6] for s in re.findall(vina_eng_compiled, pdbqt_str)]
            pdbqt_mol = PDBQTMolecule(pdbqt_str, skip_typing=True)
            output_string, _ = RDKitMolCreate.write_sd_string(pdbqt_mol)
            retrieved_text_dict[name] = {'sdf': output_string, 'eng': energy_rmsd_list}
        return retrieved_text_dict
    
    def save_filtered_pdbqt(self):
        save_file, _ = QFileDialog.getSaveFileName(self, 'Save Filtered structure to pdbqt file',
                                                   '', 'AutoDock Format (*.pdbqt)')
        if save_file:
            pdbqt_str = self._extract_pdbqt_format(self.result_table.processing_df)
            with open(save_file, 'w') as f:
                f.write(pdbqt_str)
    
    def save_filtered_sdf(self):
        save_dir = QFileDialog.getExistingDirectory(self, 'Save Filtered structure to sdf files', '')
        if save_dir:
            name_sdf_dict = self._extract_sdf_format(self.result_table.processing_df)
            for name, d in name_sdf_dict.items():
                sdf_string, eng_rmsd_list = d['sdf'], d['eng']
                supp = Chem.SDMolSupplier()
                supp.SetData(sdf_string)
                with Chem.SDWriter(os.path.join(save_dir, name+'.sdf')) as w:
                    for idx, mol in enumerate(supp):
                        mol = Chem.AddHs(mol, addCoords=True)
                        mol.SetProp('_Name', f'{name}_{idx+1}')
                        mol.SetProp('VINA Energy', f'{eng_rmsd_list[idx][0]}')
                        mol.SetProp('RMSD L.B.', f'{eng_rmsd_list[idx][1]}')
                        mol.SetProp('RMSD U.B.', f'{eng_rmsd_list[idx][2]}')
                        w.write(mol)
    
    def save_filtered_mmgbsa(self): # Depracated...
        save_zip_pth, _ = QFileDialog.getSaveFileName(self, 'Save filtered structures to MM-GBSA-ready zip file.',
                                                      '', 'ZIP File (*.zip)')
        if save_zip_pth:
            all_names = self.result_table.processing_df['Name'].to_list()
            final_files = {}
            for name in all_names:
                pdbqt_combiner = self.mdlname_pdbqtcombiner_map[f'{name} #1']
                if pdbqt_combiner.complex is None:
                    pdbqt_combiner.get_combined_string()
                    if pdbqt_combiner.complex is None:
                        QMessageBox.critical(self, 'ProteinError', 'Protein not loaded!')
                        return
                protein, _, _ = pdbqt_combiner.get_all_pdb()
                ligand = pdbqt_combiner.get_mol_from_ligand_pdb()
                name = name.replace(' ', '_')   # replaces spaces with "_" since Uni-GBSA code does not account for spaces in file name
                final_files.update({os.path.join(name, f'{name}.mol' ): ligand,
                                    os.path.join(name, f'protein.pdb'): protein,})
            with zipfile.ZipFile(save_zip_pth, 'w') as zip_f:
                for f_pth, f_content in final_files.items():
                    zip_f.writestr(f_pth, f_content)
                    
    def save_for_openmm_minimize(self, format_type: str):
        if format_type == 'dir':
            save_dir = QFileDialog.getExistingDirectory(self, 'Save filtered structures to OpenMM-minimization ready format.', '')
            if save_dir:
                all_names = self.result_table.processing_df['Name'].to_list()
                for name in all_names:
                    pdbqt_combiner = self.mdlname_pdbqtcombiner_map[f'{name} #1']
                    if pdbqt_combiner.complex is None:
                        pdbqt_combiner.get_combined_string()
                    protein = pdbqt_combiner.get_protein_only()
                    sdf_str, energy = self._extract_single_sdf_format(name, 1)
                    name = name.replace(' ', '_')
                    target_dir = os.path.join(save_dir, name)
                    os.makedirs(target_dir, exist_ok=True)
                    with open(os.path.join(target_dir, f'protein.pdb'), 'w') as f:
                        f.write(protein)
                    supp = Chem.SDMolSupplier()
                    supp.SetData(sdf_str)
                    with Chem.SDWriter(os.path.join(target_dir, f'{name}.sdf')) as w:
                        mol = next(supp)
                        # mol = Chem.AddHs(mol, addCoords=True)
                        mol.SetProp('_Name', f'{name}')
                        mol.SetProp('Old Score', f'{energy[0]}')
                        w.write(mol)
        else:
            save_zip_pth, _ = QFileDialog.getSaveFileName(self, 'Save filtered structures to MM-GBSA-ready zip file.',
                                                          '', 'ZIP File (*.zip)')
            if save_zip_pth:
                final_files = {}
                all_names = self.result_table.processing_df['Name'].to_list()
                for name in all_names:
                    pdbqt_combiner = self.mdlname_pdbqtcombiner_map[f'{name} #1']
                    if pdbqt_combiner.complex is None:
                        pdbqt_combiner.get_combined_string()
                    protein = pdbqt_combiner.get_protein_only()
                    sdf_str, energy = self._extract_single_sdf_format(name, 1)
                    name = name.replace(' ', '_')
                    supp = Chem.SDMolSupplier()
                    supp.SetData(sdf_str)
                    sio = io.StringIO()
                    with Chem.SDWriter(sio) as w:
                        mol = next(supp)
                        mol = Chem.AddHs(mol, addCoords=True)
                        mol.SetProp('_Name', f'{name}')
                        mol.SetProp('Old Score', f'{energy[0]}')
                        w.write(mol)
                    final_files.update({os.path.join(name, f'{name}.sdf' ): sio.getvalue(),
                                        os.path.join(name, f'protein.pdb'): protein,})
                zip_progress_dialog = QProgressDialog('Zipping progress', 'Cancel', 0, len(final_files), self)
                zip_progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
                with zipfile.ZipFile(save_zip_pth, 'w', zipfile.ZIP_DEFLATED, compresslevel=5) as zip_f:
                    i = 0
                    zip_progress_dialog.show()
                    for f_pth, f_content in final_files.items():
                        zip_f.writestr(f_pth, f_content)
                        i += 1
                        zip_progress_dialog.setValue(i)
                        if zip_progress_dialog.wasCanceled():
                            return
    
    def save_filtered_df(self):
        available_formats = ['Comma-Separated Values (*.csv)',
                             'Tab-Separated Values (*.tsv)',
                             'Excel Format (*.xlsx)']
        save_file, _ = QFileDialog.getSaveFileName(self, 'Save Filtered Table', '', ';;'.join(available_formats))
        if save_file:
            if save_file.endswith('.tsv'):
                self.result_table.filtered_df.to_csv(save_file, '\t', index=None)
            elif save_file.endswith('.csv'):
                self.result_table.filtered_df.to_csv(save_file, index=None)
            elif save_file.endswith('.xlsx'):
                self.result_table.filtered_df.to_excel(save_file, index=None)
    
    def save_full_df(self):
        available_formats = ['Comma-Separated Values (*.csv)',
                             'Tab-Separated Values (*.tsv)',
                             'Excel Format (*.xlsx)']
        save_file, _ = QFileDialog.getSaveFileName(self, 'Save Full Table', '', ';;'.join(available_formats))
        if save_file:
            if save_file.endswith('.tsv'):
                self.result_table.df.to_csv(save_file, '\t', index=None)
            elif save_file.endswith('.csv'):
                self.result_table.df.to_csv(save_file, index=None)
            elif save_file.endswith('.xlsx'):
                self.result_table.df.to_excel(save_file, index=None)
    
    def show_plot_filter(self):
        dialog = PlotFilterDialog(self.plot_filter_dict)
        if dialog.exec():
            self.plot_filter_dict = copy.deepcopy(dialog.chem_filter_dict)
            if self.allow_plot:
                self.plot_filter_dataframe.apply_filter(self.plot_filter_dict)
                self.plot_and_view_distribution()
    
    def plot_and_view_distribution(self):
        if self.allow_plot:
            x = self.x_combo.currentText()
            y = self.y_combo.currentText()
            template_name = self.template_combo.currentText()
            bool_x, bool_y = bool(x), bool(y)
            if bool_x & bool_y: # both selected
                plot = self.browser_plot.plot_margin(self.plot_filter_dataframe.filtered_df,
                                                     x, y, self.color_combo.currentText(),
                                                     template_name, self.tmp_plotly_file)
            elif bool_x ^ bool_y:   # xor
                col = x if bool_x else y
                plot = self.browser_plot.plot_histogram(self.plot_filter_dataframe.filtered_df,
                                                        col, template_name, self.tmp_plotly_file)
            else:   # both empty, show correlation plot
                plot = self.browser_plot.plot_correlation(self.plot_filter_dataframe.filtered_df,
                                                          template_name, self.tmp_plotly_file)
            if plot:
                self.browser_plot.setup_html(self.tmp_plotly_file)
                self.save_plot_button.setEnabled(True)
            else:
                self.save_plot_button.setDisabled(True)
        else:
            self.save_plot_button.setDisabled(True)
    
    def save_plot_to_image(self, download: QWebEngineDownloadRequest):
        download.accept()
    
    def save_plot_to_html(self):
        save_file, _ = QFileDialog.getSaveFileName(self, 'Save Interactive Plot', '', 'Hypertext Markup Language (*.html)')
        if save_file:
            self.browser_plot.fig.write_html(save_file, self.browser_plot.plotly_html_config)
    
    def calculate_threshold_metrics(self):
        thres = self.energy_threshold_spinbox.value()
        true_bool = np.array(self.plot_filter_dataframe.filtered_df['Score'] <= thres)
        self.all_decisions = {k: None for k in self.plot_filter_dataframe.filtered_df.columns if k not in ['Score', 'Name']}
        for col in self.all_decisions:
            params = np.array(self.plot_filter_dataframe.filtered_df[col])
            min_x, max_x = min(params), max(params)
            if col in ['Hydrogen Bond Donors', 'Hydrogen Bond Acceptors', 'Rotatable Bonds',
                       'Number of Rings', 'Formal Charge', 'Number of Heavy Atoms', 'Number of Atoms']:
                thresholds_x = np.arange(min_x, max_x + 1)
            else:
                num = int((max_x - min_x) * 7.5)
                thresholds_x = np.linspace(min_x, max_x, num)
            ops_list = ['≥', '≤']
            _r = []
            for op in ops_list:
                if op == '≥':
                    predicted_bools = (params[:, None] >= thresholds_x)
                else:
                    predicted_bools = (params[:, None] <= thresholds_x)
                tpr_array, fpr_array = vec_calculate_tpr_fpr(true_bool, predicted_bools)
                youden_indices = abs(tpr_array - fpr_array)
                auc = abs(np.trapz(tpr_array, fpr_array))
                _r.append({'op': op, 'TPR': tpr_array,
                           'FPR': fpr_array, 'AUC': auc,
                           'Youden': youden_indices,
                           'Thresholds': thresholds_x})
            if _r[0]['AUC'] >= _r[1]['AUC']:
                self.all_decisions[col] = _r[0]
            else:
                self.all_decisions[col] = _r[1]
        self.allow_decision_plot = True
        self.decide_threshold()
        self.plot_and_view_scatter_or_roc()
        self.update_tpr_fpr_label()
    
    def decide_threshold(self):
        self.decision_threshold_dict = {}
        self.decision_threshold_bool = {}
        df = self.plot_filter_dataframe.filtered_df.select_dtypes('number')
        pearson_corr = self.browser_plot.calculate_pearson_corr(df)
        pearson_corr = abs(np.tril(pearson_corr[1:, 1:], -1))   # remove "Energy" then only preserve lower triangular value
        a, b = np.where(pearson_corr >= 0.5)    # find indices of abs(pearson) ≥ 0.85
        # corr_groups = []    # group correlated values together
        # for aa, bb in zip(a, b):
        #     in_group = False
        #     for group in corr_groups:
        #         if aa in group:
        #             if bb not in group:
        #                 group.append(bb)
        #             in_group = True
        #         elif bb in group:
        #             if aa not in group:
        #                 group.append(aa)
        #             in_group = True
        #     if not in_group:
        #         corr_groups.append([aa, bb])
        corr_groups = set(list(a) + list(b))
        partial_group_idx_syn_name_dict = {}
        tpr_threshold = self.tpr_threshold_spinbox.value()  # ≥
        fpr_threshold = self.fpr_threshold_spinbox.value()  # ≤
        partial_cnt = 0
        for idx, (col_name, metric_dict) in enumerate(self.all_decisions.items()):
            syn_name = self.full_name_to_chem_prop_map[col_name]
            tpr, fpr = metric_dict['TPR'], metric_dict['FPR']
            bool_both = (tpr >= tpr_threshold) & (fpr <= fpr_threshold)
            if np.any(bool_both):
                arg = np.argmax(abs(metric_dict['TPR'][bool_both] - metric_dict['FPR'][bool_both])) # use Youden's index to decide best thresholf
                threshold = float(f'{metric_dict['Thresholds'][bool_both][arg]:.2f}')
                self.decision_threshold_dict[syn_name] = [(metric_dict['op'], threshold)]
                if idx in corr_groups:
                    self.decision_threshold_bool[syn_name] = 'partial'
                    partial_cnt += 1
                # for g_idx, group in enumerate(corr_groups):
                #     if idx in group:
                #         self.decision_threshold_bool[syn_name] = 'partial'
                #         if g_idx not in partial_group_idx_syn_name_dict:
                #             partial_group_idx_syn_name_dict[g_idx] = [syn_name]
                #         else:
                #             partial_group_idx_syn_name_dict[g_idx] += [syn_name]
                #         break
                else:
                    self.decision_threshold_bool[syn_name] = True
            else:
                self.decision_threshold_dict[syn_name] = [()]
                self.decision_threshold_bool[syn_name] = False
        # c = 0
        # for g_idx, syn_names in partial_group_idx_syn_name_dict.items():
        #     if len(syn_names) == 1:
        #         self.decision_threshold_bool[syn_names[0]] = True
        #         c += 1
        if partial_cnt:
            tpr_fpr_numpy_array = []
            true_bool = np.array(self.result_table.df['Score'] <= self.energy_threshold_spinbox.value())
            for parital_threshold in range(1, partial_cnt + 1):
                self.decision_threshold_bool['partial_filter_threshold'] = parital_threshold
                tpr, fpr = self._calculate_overall_tpr_fpr(true_bool, self.decision_threshold_dict, self.decision_threshold_bool)
                tpr_fpr_numpy_array.append([tpr, fpr])
            tpr_fpr_numpy_array = np.array(tpr_fpr_numpy_array)
            tprs, fprs = tpr_fpr_numpy_array[:, 0], tpr_fpr_numpy_array[:, 1]
            matched_user_tpr_fpr_bool = (tprs >= tpr_threshold) & (fprs <= fpr_threshold)
            if np.any(matched_user_tpr_fpr_bool):
                matched_indices = np.where(matched_user_tpr_fpr_bool)[0]
                matched_youdens = abs(tprs[matched_user_tpr_fpr_bool] - fprs[matched_user_tpr_fpr_bool])
                best_youdens_threshold = matched_indices[np.argmax(matched_youdens)] + 1
            else:
                youdens = abs(np.diff(tpr_fpr_numpy_array, axis=-1).flatten())
                best_youdens_threshold = np.argmax(youdens) + 1
            self.decision_threshold_bool['partial_filter_threshold'] = best_youdens_threshold
        else:
            self.decision_threshold_bool['partial_filter_threshold'] = 0
        self.setup_filter_button.setEnabled(True)
        self.export_filter_button.setEnabled(True)
        
    def _calculate_overall_tpr_fpr(self, true_bool, decision_dict, decision_bool_dict):
        exact_bool = np.ones_like(true_bool, bool)
        partial_result = []
        for prop, prop_bool in decision_bool_dict.items():
            if prop != 'partial_filter_threshold':
                if prop_bool:
                    full_prop = self.result_table.chem_prop_to_full_name_map[prop]
                    full_prop_values = np.array(self.result_table.df[full_prop])
                    list_operations = decision_dict[prop]
                    predicted_bool = check_chemprop_matching(list_operations, full_prop_values)
                    if prop_bool == 'partial':
                        partial_result.append(predicted_bool)
                    else:
                        exact_bool &= predicted_bool
        if partial_result:
            partial_bool = np.stack(partial_result).sum(0) >= decision_bool_dict['partial_filter_threshold']
            final_predicted_bool = exact_bool & partial_bool
        else:
            final_predicted_bool = exact_bool
        return calculate_tpr_fpr(true_bool, final_predicted_bool)
        
    def update_tpr_fpr_label(self):
        thres = self.energy_threshold_spinbox.value()
        true_bool = np.array(self.result_table.df['Score'] <= thres)
        tpr, fpr = self._calculate_overall_tpr_fpr(true_bool, self.decision_threshold_dict, self.decision_threshold_bool)
        self.tpr_label.setText(f'Overall TPR: {tpr:.3f}')
        self.fpr_label.setText(f'Overall FPR: {fpr:.3f}')
        
    def apply_and_recalculate_overall_tpr_fpr(self, decision_dict, decision_bool):
        true_bool = np.array(self.result_table.df['Score'] <= self.energy_threshold_spinbox.value())
        tpr, fpr = self._calculate_overall_tpr_fpr(true_bool, decision_dict, decision_bool)
        self.tpr_label.setText(f'Overall TPR: {tpr:.3f}')
        self.fpr_label.setText(f'Overall FPR: {fpr:.3f}')
        self.tpr_fpr_applied = True
    
    def update_threshold_filter_dialog(self):
        self.tpr_fpr_applied = False
        dialog = AutoFilterDialog(self.decision_threshold_dict, self.decision_threshold_bool)
        dialog.apply_signal.connect(self.apply_and_recalculate_overall_tpr_fpr)
        if dialog.exec():
            curr_plot_row = self.column_selection_combo.currentText()
            syn_name = self.full_name_to_chem_prop_map[curr_plot_row]
            if dialog.chem_filter_dict[syn_name] == self.decision_threshold_dict[syn_name]:
                replot = False
            else:
                replot = True
            self.decision_threshold_dict = copy.deepcopy(dialog.chem_filter_dict)
            self.decision_threshold_bool = copy.deepcopy(dialog.chem_filter_bool)
            self.update_tpr_fpr_label()
            if replot:
                self.plot_and_view_scatter_or_roc()
        else:
            if self.tpr_fpr_applied:
                self.update_tpr_fpr_label()
                self.plot_and_view_scatter_or_roc()
    
    def _retrieve_scatter_overall_dict(self, list_operations, true_bool, target_numpy, template, x, y):
        color_dict = {'dark': {'energy_thres_color': 'white',
                               'y_thres_color': 'rgb(167, 167, 167)',
                               'legend_bgcolor': 'rgba(10, 10, 10, 0.7)',},
                      'light': {'energy_thres_color': 'black',
                                'y_thres_color': 'rgb(50, 50, 50)',
                                'legend_bgcolor': 'rgba(250, 250, 250, 0.7)',}}
        if list_operations[0]:
            predicted_bool = check_chemprop_matching(list_operations, target_numpy)
            tpr, fpr = calculate_tpr_fpr(true_bool, predicted_bool)
            tpr_str, fpr_str = f'{tpr:.4f}', f'{fpr:.4f}'
            true_positive_bool = predicted_bool & true_bool
            true_negative_bool = (~predicted_bool) & (~true_bool)
            fals_positive_bool = predicted_bool & (~true_bool)
            fals_negative_bool = (~predicted_bool) & true_bool
            if '_dark' in template:
                overall_dict = {'true positive' : {'bool': true_positive_bool, 'color': 'rgb(167, 250, 167)'},
                                'true negative' : {'bool': true_negative_bool, 'color': 'rgb(250, 250, 250)'},
                                'false positive': {'bool': fals_positive_bool, 'color': 'rgb(250, 167, 167)'},
                                'false negative': {'bool': fals_negative_bool, 'color': 'rgb(167, 167, 250)'},
                                'tpr_str': tpr_str, 'fpr_str': fpr_str}
                overall_dict.update(color_dict['dark'])
            else:
                overall_dict = {'true positive' : {'bool': true_positive_bool, 'color': 'rgb(83 , 167, 83 )'},
                                'true negative' : {'bool': true_negative_bool, 'color': 'rgb(50 , 50 , 50 )'},
                                'false positive': {'bool': fals_positive_bool, 'color': 'rgb(167, 83 , 83 )'},
                                'false negative': {'bool': fals_negative_bool, 'color': 'rgb(83 , 83 , 167)'},
                                'tpr_str': tpr_str, 'fpr_str': fpr_str}
                overall_dict.update(color_dict['light'])
        else:
            tpr_str, fpr_str = 'None', 'None'
            if '_dark' in template:
                overall_dict = {'positive' : {'bool':  true_bool, 'color': 'rgb(167, 250, 167)'},
                                'negative' : {'bool': ~true_bool, 'color': 'rgb(250, 250, 250)'},
                                'energy_thres_color': 'white',}
                overall_dict.update(color_dict['dark'])
            else:
                overall_dict = {'positive' : {'bool':  true_bool, 'color': 'rgb(83 , 167, 83 )'},
                                'negative' : {'bool': ~true_bool, 'color': 'rgb(50 , 50 , 50 )'},}
                overall_dict.update(color_dict['light'])
            overall_dict.update({'tpr_str': 'None', 'fpr_str': None})
        for plot_dict in overall_dict.values():
            if isinstance(plot_dict, dict):
                bool = plot_dict['bool']
                plot_dict.update({'x': x[bool], 'y': y[bool], 'sum': bool.sum()})
        overall_dict.update({'template': template, 'list_operations': list_operations})
        return overall_dict
    
    def plot_and_view_scatter_or_roc(self):
        if self.allow_decision_plot:
            col = self.column_selection_combo.currentText()
            prop = self.full_name_to_chem_prop_map[col]
            plot_type = self.plot_type_combo.currentText()
            energy = self.plot_filter_dataframe.filtered_df['Score']
            full_prop_values = np.array(self.plot_filter_dataframe.filtered_df[col])
            thres = self.energy_threshold_spinbox.value()
            true_bool = np.array(energy <= thres)
            if self.decision_threshold_bool[prop]:
                list_operations = self.decision_threshold_dict[prop]
                if plot_type == 'ROC Curve':
                    predicted_bool = check_chemprop_matching(list_operations, full_prop_values)
                    tpr, fpr = calculate_tpr_fpr(true_bool, predicted_bool)
                    combined_dict = {'Operations': list_operations, 'TPR': tpr, 'FPR': fpr}
                    self.browser_roc_plot.plot_roc_curve(self.all_decisions[col], col,
                                                         self.roc_template_combo.currentText(),
                                                         self.tmp_plotly_file,
                                                         combined_dict)
                else:
                    overall_dict = self._retrieve_scatter_overall_dict(list_operations,
                                                                       true_bool,
                                                                       full_prop_values,
                                                                       self.roc_template_combo.currentText(),
                                                                       energy,
                                                                       full_prop_values)
                    overall_dict.update({'energy_threshold': thres, 'Name': self.result_table.df['Name']})
                    self.browser_roc_plot.plot_scatter_threshold_plot(overall_dict, col, self.tmp_plotly_file)
            else:
                if plot_type == 'ROC Curve':
                    self.browser_roc_plot.plot_roc_curve(self.all_decisions[col], col,
                                                         self.roc_template_combo.currentText(),
                                                         self.tmp_plotly_file)
                else:
                    overall_dict = self._retrieve_scatter_overall_dict([()],
                                                                       true_bool,
                                                                       None,
                                                                       self.roc_template_combo.currentText(),
                                                                       energy,
                                                                       full_prop_values)
                    overall_dict.update({'energy_threshold': thres, 'Name': self.result_table.df['Name']})
                    self.browser_roc_plot.plot_scatter_threshold_plot(overall_dict, col, self.tmp_plotly_file)
                    
            self.browser_roc_plot.setup_html(self.tmp_plotly_file)
            
    def export_filter_dict_to_json(self):
        save_file, _ = QFileDialog.getSaveFileName(self, 'Save Filter Settings', '', 'JSON (*.json)')
        if save_file:
            combined_dict = {'Checkbox' : self.decision_threshold_bool,
                             'Decisions': self.decision_threshold_dict}
            with open(save_file, 'w') as f:
                json.dump(combined_dict, f, indent=4)
    
    def process_fragment_img_and_score(self):
        self.sorted_df = self.result_table.df.copy().sort_values('Score').reset_index(drop=True)
        score = -self.sorted_df['Score'].to_numpy()
        score = (score - score.mean()) / score.std()
        names = self.sorted_df['Name'].to_list()
        
        fragment_score_name_count_dict = {}
        
        for idx, frags in enumerate(self.sorted_df['Fragments']):
            for frag in frags:
                if frag in fragment_score_name_count_dict:
                    fragment_score_name_count_dict[frag]['Score'].append(score[idx])
                    fragment_score_name_count_dict[frag]['Names'].append(names[idx])
                else:
                    fragment_score_name_count_dict[frag] = {'Score': [score[idx]],
                                                            'Names': [names[idx]]}
        
        self.total_frags = len(fragment_score_name_count_dict)
        self.fragment_progress_bar.setValue(0)
        self.fragment_progress_bar.setMaximum(self.total_frags)
        self.str_length = len(str(self.total_frags))
        self.fragment_progress_text.setText(f'{0:{self.str_length}}/{self.total_frags}')
        
        self.frag_score_img_name_dict = {}
        self.thread = QThread()
        self.worker = MultiprocessFragmentScore(fragment_score_name_count_dict, self.total_frags, 10)
        self.worker.moveToThread(self.thread)
        self.worker.progress.connect(self.update_final_fragment_dict)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.finish_fragment_process)
        self.thread.started.connect(self.worker.run)
        self.thread.start()
        
    def update_final_fragment_dict(self, step: int, result_dict: dict, frag: str):
        self.fragment_progress_bar.setValue(step)
        self.fragment_progress_text.setText(f'{step:{self.str_length}}/{self.total_frags}')
        self.frag_score_img_name_dict[frag] = result_dict
        
    def finish_fragment_process(self):
        self.frag_score_img_name_dict = dict(sorted(self.frag_score_img_name_dict.items(), key=lambda item: item[1]['Score'], reverse=True))
        self.curr_page_lineedit.setText('1')
        self.total_fragment_page = math.ceil(len(self.frag_score_img_name_dict) / 10)
        self.page_total_label.setText(f'/ {self.total_fragment_page}')
        self.update_page()
        # for i in range(10):
        #     frag = list(self.frag_score_img_name_dict)[i]
        #     names = self.frag_score_img_name_dict[frag]['Names']
        #     name_energy_df = self.result_table.df[['Name', 'Score']].sort_values('Score').reset_index(drop=True)
        #     plot_energy_fragment(name_energy_df, names)
        
    def update_page(self):
        curr_page = max(1, min(self.total_fragment_page, int(self.curr_page_lineedit.text())))
        self.curr_page_lineedit.setText(str(curr_page))
        if curr_page == self.total_fragment_page:
            self.next_page_button.setDisabled(True)
        else:
            self.next_page_button.setEnabled(True)
        if curr_page == 1:
            self.prev_page_button.setDisabled(True)
        else:
            self.prev_page_button.setEnabled(True)
        while self.fragment_grid.count():
            item = self.fragment_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if curr_page == self.total_fragment_page:
            page_value = self.total_frags % 10
        else:
            page_value = 10
        populate_progress_dialog = QProgressDialog('Populating Grid...', 'Cancel', 0, page_value, self)
        populate_progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        for idx, frag in enumerate(self.frag_score_img_name_dict):
            if math.floor(idx / 10) + 1 == curr_page:
                populate_progress_dialog.show()
                row = idx % 10
                populate_progress_dialog.setValue(row + 1)
                if populate_progress_dialog.wasCanceled():
                    return
                
                img_label = ImageLabel(self.frag_score_img_name_dict[frag]['Image'], 200)
                
                smarts_lineedit = CopyLineEdit(self)
                smarts_lineedit.setText(frag)
                smarts_lineedit.setFixedWidth(300)
                smarts_lineedit.setCursorPosition(0)
                names_label = HiddenExpandLabel()
                names_label.set_initial_text(self.frag_score_img_name_dict[frag]['Names'])
                names_label.setWordWrap(True)
                
                score_label = ShowPlotLabel(self.sorted_df, self.frag_score_img_name_dict[frag]['Names'])
                score_label.setText(f'{self.frag_score_img_name_dict[frag]['Score']:.4f}')
                
                self.fragment_grid.addWidget(img_label, row, 0)
                self.fragment_grid.addWidget(smarts_lineedit, row, 1)
                self.fragment_grid.addWidget(names_label, row, 2)
                self.fragment_grid.addWidget(score_label, row, 3)
                if idx % 10 == page_value - 1:
                    return
    
    def fragment_prev_page(self):
        self.curr_page_lineedit.setText(str(int(self.curr_page_lineedit.text()) - 1))
        self.update_page()
    
    def fragment_next_page(self):
        self.curr_page_lineedit.setText(str(int(self.curr_page_lineedit.text()) + 1))
        self.update_page()
    
    def open_fpocket_web(self):
        if self.pdbqt_editor is not None:
            pdbqt_str = self.pdbqt_editor.convert_dict_to_pdbqt_text()
        else:
            pdbqt_str = None
        self.fpocket_browser = FPocketBrowser(self, pdbqt_str)
        self.fpocket_browser.signal.sendFPocketData.connect(self._retrieve_fpocket_pdb)
    
    def _retrieve_fpocket_pdb(self, fpocket_pdb: str, table_str: str):
        fpocket_boxes = {}
        table_start = False
        for l in table_str.splitlines():
            if l:
                if l.strip() == 'Details':
                    table_start = True
                elif table_start:   # skip "Details line"
                    if l.strip() == 'Clear Selection':
                        break
                    mdl, score, druggable, a_sphere, volume = l.strip().split()
                    fpocket_boxes[int(mdl)] = {'Score'        : float(score),
                                               'Druggability' : float(druggable),
                                               'Alpha Spheres': int(a_sphere),
                                               'Volume'       : float(volume)}
        hetatm_lines = re.findall(fpocket_hetatm_line, fpocket_pdb)
        mdl_pocket_xyz = {}
        for line in hetatm_lines:
            mdl, x, y, z = int(line[22:26]), float(line[30:38]), float(line[38:46]), float(line[46:54])
            if mdl not in mdl_pocket_xyz:
                mdl_pocket_xyz[mdl] = []
            mdl_pocket_xyz[mdl].append([x, y, z])
        for mdl, xyz in mdl_pocket_xyz.items():
            xyz = np.array(xyz)
            max_xyz, min_xyz = xyz.max(0), xyz.min(0)
            box = max_xyz - min_xyz
            center = np.round((max_xyz + min_xyz) / 2, 3)
            box = np.round(box + 8 / np.minimum(20, box), 3)
            fpocket_boxes[mdl].update({'Center': list(center),
                                       'Box'   : list(box)   ,})
        self.param_dict['fpocket'] = fpocket_boxes
        self.log_textedit.append(f'<b><span style="font-size:{int(self.default_fontsize)}px;'
                                 f'">{len(fpocket_boxes)} Pockets Found!</span></b>')
    
    def open_browser(self):
        BrowserWithTabs(self, None)
    
    def open_web_docking_programs(self, program: str, web_type: str):
        web_url = self.webserver_map[program][web_type]
        BrowserWithTabs(self, web_url)
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            if self.curr_page_lineedit.hasFocus():
                self.update_page()
            elif self.input_id_dict['PDB']['LineEdit'].hasFocus() and self.input_id_dict['PDB']['Load_Button'].isEnabled():
                if not self.is_docking:
                    self.input_id_dict['PDB']['Load_Button'].click()
            elif self.input_id_dict['AF Database']['LineEdit'].hasFocus() and self.input_id_dict['AF Database']['Load_Button'].isEnabled():
                if not self.is_docking:
                    self.input_id_dict['AF Database']['Load_Button'].click()
            elif self.docked_dir_line.hasFocus():
                if self.show_dir_table_button.isEnabled():
                    self.show_dir_table_button.click()
            else:
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
    
    def toggle_sidebar(self):
        max_width = self.swap_widget_button_frame.maximumWidth()
        self.sidebar_animation.setStartValue(max_width)
        
        if self.sidebar_visible:
            self.sidebar_animation.setEndValue(0)
            self.toggle_sidebar_button.setText('→')
            self.toggle_sidebar_button.setFont(self.sidebar_btn_hide_font)
            self.toggle_sidebar_button.setToolTip('Show tab')
        else:
            self.sidebar_animation.setEndValue(self.sidebar_minimum_width)
            self.toggle_sidebar_button.setText('≡')
            self.toggle_sidebar_button.setFont(self.sidebar_btn_expand_font)
            self.toggle_sidebar_button.setToolTip('Hide tab')
        
        self.sidebar_animation.start()
        self.sidebar_visible = not self.sidebar_visible
    
    def update_other_positions(self, value):
        self.toggle_sidebar_button.resize(max(20, value), 50)
    
    def open_directory_splitter(self):
        dialog = DirSplitterDialog(self.curr_display_mode)
        dialog.exec()
    
    def open_directory_combiner(self):
        dialog = DirCombinerDialog(self.curr_display_mode)
        dialog.exec()
    
    def open_db_dialog(self):
        dialog = DBSearchDialog()
        if dialog.exec():
            final_dict = {}
            for dbname_id in list(dialog.final_dict):
                s = dialog.final_dict[dbname_id]
                if dbname_id in self.db_small_molecule:
                    i = 1
                    while f'{dbname_id}_{i}' in self.db_small_molecule:
                        i += 1
                    dbname_id = f'{dbname_id}_{i}'
                item = QListWidgetItem(dbname_id)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                self.input_files_list.addItem(item)
                final_dict[dbname_id] = s
            self.db_small_molecule.update(dict(final_dict))
    
    def add_mol_name_smiles(self, smi: str, name: str):
        all_names = self.input_files_list.get_all_names()
        if name in all_names:
            i = 1
            new_name = f'{name}_{i}'
            while True:
                if new_name not in all_names:
                    break
                i += 1
                new_name = f'{name}_{i}'
            name = new_name
        self.db_small_molecule[name] = smi
        item = QListWidgetItem(name)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        self.input_files_list.addItem(item)
    
    def open_chemdraw_dialog(self):
        dialog = ChemEditorDialog(self, self.curr_display_mode == 'dark')
        dialog.smilesSignal.connect(self.add_mol_name_smiles)
        dialog.exec()
    
    def modify_dbname_to_url_map(self):
        db_map_pth = os.path.join(self.curr_dir, 'utilities', 'database', 'url_template.txt')
        dialog = ModifyNameURLTemplateDialog(db_map_pth)
        if dialog.exec():
            w: LocalDatabaseFinderWidget = self.shopper_stacked_widget.widget(2)
            for r in range(w.db_table.rowCount()):
                name = w.db_table.item(r, 1).text()
                if name in dialog.final_map:
                    w.db_table.item(r, 3).setText(dialog.final_map[name])
    
    def open_manual_browser(self):
        self.doc_dialog = DocumentationWindow(self)
        self.doc_dialog.show()
    
    def change_fp_setting_dialog(self):
        with open(os.path.join(curr_dir, 'utilities', 'fp_param', 'fp_param.json'), 'r') as f:
            fp_setting_dict = json.load(f)
        dialog = FingerprintSettingDialog(fp_setting_dict)
        if dialog.exec():
            with open(os.path.join(curr_dir, 'utilities', 'fp_param', 'fp_param.json'), 'w') as f:
                json.dump(dialog.new_settings, f, indent=4)
            self.shopper_stacked_widget.widget(2).db_table.reset_all_highlight()
    
    def display_2d_ligplot(self):
        contact_dict = self.structure_browser_widget.shown_contact_dict.get(self.curr_struct_contact_name, None)
        if contact_dict is None:
            return
        contact_df = pd.DataFrame(contact_dict).T.drop(columns=['atom1Sel', 'atom2Sel'])
        complex_pdb = self.mdlname_pdbqtcombiner_map[self.curr_struct_contact_name].complex
        ligplot_widget = LigPlotWidget(complex_pdb, contact_df, self)
        ligplot_widget.show()
    
    def closeEvent(self, event):
        shutil.rmtree(os.path.join(curr_dir, 'utilities', 'cookies'))
        os.mkdir(os.path.join(curr_dir, 'utilities', 'cookies'))
        if self.is_docking:
            self.dockStopSignal.emit()
        for i in range(self.shopper_stacked_widget.count()):
            w = self.shopper_stacked_widget.widget(i)
            if w.currently_searching:
                w.worker.stop()
        # kill_child_processes(os.getpid())
        super().closeEvent(event)

def compare_single_values(op, thres, target):
    if   op == '≥':
        return target >= thres
    elif op == '>':
        return target >  thres
    elif op == '<':
        return target <  thres
    elif op == '≤':
        return target <= thres

def check_chemprop_matching(list_operations: list[tuple], target):
    if len(list_operations) == 1:
        return compare_single_values(list_operations[0][0], list_operations[0][1], target)
    else:
        op1, thres1 = list_operations[0]
        op2, thres2 = list_operations[1]
        or_option = False
        if (op1 in ('≥', '>') and op2 in ('≤', '<')):
            if thres1 >= thres2:
                or_option = True
        elif (op1 in ('≤', '<') and op2 in ('≥', '>')):
            if thres2 >= thres1:
                or_option = True
        op1_result = compare_single_values(op1, thres1, target)
        op2_result = compare_single_values(op2, thres2, target)
        if or_option:
            return op1_result | op2_result
        else:
            return op1_result & op2_result

def calculate_tpr_fpr(true_bool, predicted_bool):
    tpr = ( true_bool & predicted_bool).sum() / ( true_bool).sum()
    fpr = (~true_bool & predicted_bool).sum() / (~true_bool).sum()
    return tpr, fpr

def vec_calculate_tpr_fpr(true_bool: np.array, predicted_bools: np.array):
    """
    Vectorized function for calculating TPR & FPR
    """
    true_bool = true_bool[:, None]
    tpr = ( true_bool & predicted_bools).sum(0) / ( true_bool).sum(0)
    fpr = (~true_bool & predicted_bools).sum(0) / (~true_bool).sum(0)
    return tpr, fpr

def recursive_read_all_files(input_dir: str):
    # Similar to os.walk()
    result = []
    for f_or_d in os.listdir(input_dir):
        f_or_d = os.path.join(input_dir, f_or_d)
        if os.path.isdir(f_or_d):
            files = recursive_read_all_files(f_or_d)
            result += files
        else:
            result.append(f_or_d)
    return result

curr_dir = os.path.dirname(__file__)
ledock_eng_compiled = re.compile(r'Score:\s*(-?\d+\.\d+)')
fpocket_hetatm_line = re.compile(r'HETATM.*')
os.environ["QT_LOGGING_RULES"] = "qt.pointer.dispatch=false;"
# os.environ["QT_DEBUG_PLUGINS"] = "1"
# os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--enable-logging --log-level=3"