from PySide6.QtWidgets import (QDialog, QVBoxLayout, QDialogButtonBox,
                               QMessageBox, QLabel, QLineEdit, QHBoxLayout, QWidget)
from PySide6.QtCore import Signal, Qt
from .rdeditor.rdEditor import MainWindow
from rdkit import Chem

class ChemEditorDialog(QDialog):
    smilesSignal = Signal(str, str)
    
    def __init__(self, parent=None, darkmode_stat=True):
        super().__init__(parent)
        
        self.setWindowTitle("Chemical Editor Dialog")
        screen = self.screen()
        screen_size = screen.availableGeometry()
        self.resize(screen_size.width() * 0.85, screen_size.height() * 0.75)
        
        layout = QVBoxLayout(self)
        
        name_layout = QHBoxLayout()
        name_widget = QWidget()
        name_widget.setLayout(name_layout)
        
        button_layout = QHBoxLayout()
        
        self.editorWindow = MainWindow(darkmode_stat=darkmode_stat)
        self.editorWindow.doneDrawingSignal.connect(self.update_smiles_string)
        layout.addWidget(self.editorWindow)
        
        name_label = QLabel('<b>Name :</b>')
        self.name_lineedit = QLineEdit()
        self.name_lineedit.setPlaceholderText('Molecule')
        self.name_lineedit.setMinimumWidth(300)
        smiles_label = QLabel('<b>SMILES :</b>')
        self.smiles_lineedit = QLineEdit()
        self.smiles_lineedit.setPlaceholderText('SMILES...')
        self.smiles_lineedit.setMinimumWidth(400)
        self.smiles_lineedit.textChanged.connect(self.update_mol_with_smiles)
        self.smiles_lineedit.setStyleSheet(f'color: #4CAF50;')
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.name_lineedit)
        name_layout.addWidget(smiles_label)
        name_layout.addWidget(self.smiles_lineedit)
        
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Close, self)
        button_layout.addWidget(name_widget, alignment=Qt.AlignmentFlag.AlignLeft)
        button_layout.addWidget(self.buttonBox, alignment=Qt.AlignmentFlag.AlignRight)
        
        layout.addLayout(button_layout)
        
        self.buttonBox.accepted.connect(self.add_smiles)
        self.buttonBox.rejected.connect(self.reject)
        
        self.setStyleSheet(self.parent().styleSheet())
        self.setPalette(self.parent().palette())
    
    def update_smiles_string(self):
        self.smiles_lineedit.blockSignals(True)
        smiles = Chem.MolToSmiles(self.editorWindow.editor.mol)
        self.smiles_lineedit.setText(smiles)
        self.smiles_lineedit.blockSignals(False)
    
    def update_mol_with_smiles(self, smi: str):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            self.editorWindow.editor.mol = mol
            self.smiles_lineedit.setStyleSheet(f'color: #4CAF50;')
        else:
            self.smiles_lineedit.setStyleSheet(f'color: #E57373;')
    
    def add_smiles(self):
        try:
            Chem.SanitizeMol(self.editorWindow.editor.mol)
            smiles = Chem.MolToSmiles(self.editorWindow.editor.mol)
            name = self.name_lineedit.text()
            if not name:
                name = 'Molecule'
            self.smilesSignal.emit(smiles, name)
        except Exception as e:
            QMessageBox.critical(self, 'Structure Error', f'Incorrect chemical structure: {e}')
