import os
import copy, json

import numpy as np
import pandas as pd

from PySide6.QtWidgets import (QVBoxLayout,
                               QDialog)
from PySide6.QtCore import QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView

class DocumentationWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Documentation")
        screen = self.screen()
        screen_size = screen.availableGeometry()
        self.resize(screen_size.width() * 0.8, screen_size.height() * 0.8)
        
        layout = QVBoxLayout()
        self.web_view = QWebEngineView()
        
        doc_pth = os.path.join(os.path.dirname(__file__), 'site', 'index.html')
        self.web_view.setUrl(QUrl.fromLocalFile(doc_pth))
        
        layout.addWidget(self.web_view)
        self.setLayout(layout)

