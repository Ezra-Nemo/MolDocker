import os, sys, platform
import importlib
from PySide6.QtWidgets import (QApplication, QLabel, QProgressBar, QVBoxLayout, QWidget, QMessageBox)
from PySide6.QtCore import Qt, Signal, QThread, QCoreApplication
from PySide6.QtGui import QGuiApplication, QIcon

all_modules = [
    ('io', 'io'),
    ('re', 're'),
    ('gc', 'gc'),
    ('csv', 'csv'),
    ('copy', 'copy'),
    ('lzma', 'lzma'),
    ('stat', 'stat'),
    ('copy', 'copy'),
    ('math', 'math'),
    ('time', 'time'),
    ('json', 'json'),
    ('psutil', 'psutil'),
    ('shutil', 'shutil'),
    ('platform', 'platform'),
    ('tempfile', 'tempfile'),
    ('requests', 'requests'),
    ('pickle', 'pickle'),
    ('sqlite3', 'sqlite3'),
    ('zipfile', 'zipfile'),
    ('platform', 'platform'),
    ('warnings', 'warnings'),
    ('qdarktheme', 'qdarktheme'),
    ('subprocess', 'subprocess'),
    ('concurrent.futures', 'concurrent.futures'),
    ('numpy', 'numpy'),
    ('pandas', 'pandas'),
    ('parmed', 'parmed'),
    ('shapely', 'shapely'),
    ('plotly.io', 'plotly.io'),
    ('plotly.express', 'plotly.express'),
    ('plotly.graph_objects', 'plotly.graph_objects'),
    ('plotly.figure_factory', 'plotly.figure_factory'),
    ('scipy.stats', 'scipy.stats'),
    ('rdkit.Chem', 'rdkit.Chem'),
    ('rdkit.Chem.AllChem', 'rdkit.Chem.AllChem'),
    ('openbabel.pybel', 'openbabel.pybel'),
    ('openmm', 'openmm'),
    ('openff.toolkit', 'openff.toolkit'),
    ('PySide6.QtWidgets', 'PySide6.QtWidgets'),
    ('PySide6.QtGui', 'PySide6.QtGui'),
    ('PySide6.QtCore', 'PySide6.QtCore'),
]

class ModuleLoader(QThread):
    progress = Signal(int, str)
    loadFinished = Signal()
    loadFailed = Signal(str)
    
    def run(self):
        c = 0
        total = len(all_modules) + 1
        percent = 0
        for alias, module_name in all_modules:
            try:
                message = f"Importing {module_name}..."
                self.progress.emit(percent, message)
                module = importlib.import_module(module_name)
                globals()[alias] = module
                if module_name == 'openbabel.pybel':
                    module.ob.obErrorLog.SetOutputLevel(0)
                message = f"Imported {module_name}"
            except ImportError as e:
                message = f"Failed to import {module_name}: {e}"
                self.loadFailed.emit(message)
                return
            c += 1
            percent = int((c / total) * 100)
            self.progress.emit(percent, message)
            
        try:
            module_name = __package__ + ".moldocker_gui" if __package__ else "moldocker.moldocker_gui"
            message = f"Starting MolDocker..."
            self.progress.emit(percent, message)
            moldocker_gui = importlib.import_module(module_name)
            globals()["MolDocker"] = moldocker_gui.MolDocker
        except ImportError as e:
            message = f"Failed to import MolDocker: {e}"
            self.loadFailed.emit(message)
            return
        self.progress.emit(100, message)
        self.loadFinished.emit()

class SplashScreen(QWidget):
    loadFailed = Signal()
    
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.SplashScreen)
        layout = QVBoxLayout()
        self.label = QLabel("Loading modules...")
        self.label.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.label)
        layout.addWidget(self.progress)
        self.setLayout(layout)
        
        self.loader = ModuleLoader()
        self.loader.progress.connect(self.update_progress)
        self.loader.loadFinished.connect(self.on_finished)
        self.loader.loadFailed.connect(self.on_failed)
        self.resize(400, 150)
        
    def start_loading(self):
        if platform.system() == 'Windows':
            import PySide6.QtWebEngineCore
        self.loader.start()
        
    def update_progress(self, value, message):
        self.progress.setValue(value)
        self.label.setText(message)
    
    def on_failed(self, failed_msg):
        QMessageBox.critical(self, 'Module Import Error', failed_msg)
        self.close()
        self.loadFailed.emit()
    
    def on_finished(self):
        self.close()
        start_main_gui()

def start_main_gui():
    stylehints = QGuiApplication.styleHints()
    theme = 'dark' if stylehints.colorScheme() == Qt.ColorScheme.Dark else 'light'
    qdarktheme.setup_theme(theme, custom_colors={'light': {'background': '#d4d4d4'}})   # type: ignore
    try:
        ex = MolDocker(theme)   # type: ignore
        ex.showMaximized()
    except Exception as e:
        print(e)
        QApplication.quit()

def main():
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    splash = SplashScreen()
    splash.loadFailed.connect(QApplication.quit)
    splash.show()
    
    curr_dir = os.path.dirname(__file__)
    icon_pth = os.path.join(curr_dir, 'icon', 'MolDocker_Icon.png')
    app.setWindowIcon(QIcon(icon_pth))
    
    splash.start_loading()
    sys.exit(app.exec())
