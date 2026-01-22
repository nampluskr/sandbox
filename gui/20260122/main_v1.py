"""
OLED Anomaly Detection Analysis System
Qt Designer + PySide6 based
"""

import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication, QMainWindow, QFileSystemModel
from PySide6.QtCore import QDir, Qt

# Import the generated UI class
from ui_oled_analyzer import Ui_MainWindow


class OLEDAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_data = None  # Currently loaded data
        
        # Setup UI from generated file
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.setWindowTitle("OLED Anomaly Detection Analysis System")
        self.resize(1400, 800)
        
        self.setup_file_tree()
        self.connect_signals()
        
        print("[OK] UI setup complete")
        
    def setup_file_tree(self):
        """Setup file tree view (like Windows Explorer)"""
        # Create QFileSystemModel
        self.file_model = QFileSystemModel()
        
        # Set data folder path
        data_path = Path("./data")
        if data_path.exists():
            root_path = str(data_path.absolute())
        else:
            root_path = QDir.currentPath()
        
        self.file_model.setRootPath(root_path)
        
        # Filter only .npz files
        self.file_model.setNameFilters(["*.npz"])
        self.file_model.setNameFilterDisables(False)
        
        # Connect model to tree view
        self.ui.fileTreeView.setModel(self.file_model)
        self.ui.fileTreeView.setRootIndex(self.file_model.index(root_path))
        
        # Hide unnecessary columns (size, type, date modified, etc.)
        self.ui.fileTreeView.setColumnWidth(0, 250)
        for i in range(1, 4):
            self.ui.fileTreeView.hideColumn(i)
        
        print(f"[OK] File tree setup complete: {root_path}")
            
    def connect_signals(self):
        """Connect signals and slots"""
        # File selection in tree view
        self.ui.fileTreeView.clicked.connect(self.on_file_selected)
        
        # Tab change
        self.ui.mainTabWidget.currentChanged.connect(self.on_tab_changed)
        
        print("[OK] Signal connections complete")
        
    def on_file_selected(self, index):
        """File selection event handler"""
        file_path = self.file_model.filePath(index)
        
        if file_path.endswith('.npz'):
            print(f"\n{'='*60}")
            print(f"[FILE] Selected: {Path(file_path).name}")
            print(f"{'='*60}")
            
            # TODO: Phase 3 - Implement npz loading and visualization
            # import numpy as np
            # data = np.load(file_path)
            # self.current_data = {
            #     'xyz': data['xyz'],
            #     'anomaly_map': data.get('anomaly_map', None),
            #     'file_path': file_path
            # }
            
    def on_tab_changed(self, index):
        """Tab change event handler"""
        tab_name = self.ui.mainTabWidget.tabText(index)
        print(f"[TAB] Changed to: {tab_name}")
        
        # TODO: Phase 4 - Implement matplotlib update


def main():
    app = QApplication(sys.argv)
    
    # Create data folder if not exists
    data_path = Path("./data")
    data_path.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("OLED Anomaly Detection Analysis System")
    print("="*60 + "\n")
    
    window = OLEDAnalyzer()
    window.show()
    
    print("\nUsage:")
    print("  1. Select .npz file from left file tree")
    print("  2. Choose analysis view from right tabs")
    print("  3. Adjust parameters using sliders\n")
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
