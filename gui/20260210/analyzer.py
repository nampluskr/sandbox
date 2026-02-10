## python -m PyQt5.uic.pyuic main_window.ui -o ui_main_window.py
## pip install pyinstaller
## pyinstaller --onefile --windowed --log-level=INFO analyzer.py

import sys
import os

import matplotlib
matplotlib.use('QtAgg')

from datetime import datetime
import numpy as np
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QHeaderView, QAbstractItemView,
                               QFileDialog, QVBoxLayout)
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt, QEvent

from ui_main_window import Ui_MainWindow
from colors import XYZ_to_RGB, XYZ_to_Luv, XYZ_to_xyuv


class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.canvas)
        self.setLayout(self.main_layout)

    def set_figure(self, fig):
        if self.layout().count() > 0:
            old_canvas = self.layout().takeAt(0).widget()
            if old_canvas:
                old_canvas.deleteLater()

        new_canvas = FigureCanvas(fig)
        self.layout().addWidget(new_canvas)
        new_canvas.draw_idle()
        
    def get_figure(self):
        return self.canvas.figure


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Window settings
        self.setWindowTitle("2D Data Analyzer v0.0.1")
        self.resize(1024, 768)

        # Default folder path
        self.current_dir = os.path.normpath(r"D:\Non_Documents\_github\image_analyzer\test_data")
        
        if not os.path.isdir(self.current_dir):
            self.current_dir = os.path.expanduser("~")
        self.log_message(f"[Folder selected] {self.current_dir}", 0)

        # Current image data
        self.current_image_xyz = None
        self.current_image_rgb = None
        self.current_image_luv = None
        self.current_image_luv = None

        # Setup widgets
        self.setup_tree()
        self.setup_table()
        self.setup_image_rgb()
        self.setup_image_xyz()
        self.setup_image_luv()
        self.setup_image_xyuv()

        # Load default folder
        self.load_tree()
        self.load_table()

        # Connect signals
        self.connect_signals()

        # Set default tab to Image tab
        self.ui.tabWidget.setCurrentIndex(0)

    def log_message(self, message, timeout=0):
        self.statusBar().showMessage(f">> {message}", timeout)

    # =========================================================
    # treeView setup and loading
    # =========================================================
    def setup_tree(self):
        self.tree_model = QStandardItemModel()
        self.ui.treeView.setModel(self.tree_model)
        self.ui.treeView.setHeaderHidden(True)

    def load_tree(self):
        if not os.path.isdir(self.current_dir):
            self.log_message(f"[ERROR] Folder not found: {self.current_dir}")
            return

        self.tree_model.clear()

        # Create root item
        root_item = QStandardItem(os.path.basename(self.current_dir))
        root_item.setData(self.current_dir, Qt.UserRole)
        root_item.setEditable(False)
        self.tree_model.appendRow(root_item)

        # Load immediate subfolders under root
        self.load_subfolders(root_item, self.current_dir)

        # Expand root item
        root_index = self.tree_model.index(0, 0)
        self.ui.treeView.expand(root_index)

    def load_subfolders(self, parent_item, target_dir):
        try:
            entries = sorted(os.listdir(target_dir), key=str.lower)
            for entry in entries:
                full_path = os.path.join(target_dir, entry)

                if not os.path.isdir(full_path):
                    continue

                # Create folder item
                item = QStandardItem(entry)
                item.setData(full_path, Qt.UserRole)
                item.setEditable(False)
                parent_item.appendRow(item)

                # Add placeholder if has subfolders (for lazy loading)
                if self.has_subfolders(full_path):
                    placeholder = QStandardItem("")
                    placeholder.setData("placeholder", Qt.UserRole)
                    item.appendRow(placeholder)

        except PermissionError:
            self.log_message(f"[ERROR] Permission denied: {target_dir}")
        except Exception as e:
            self.log_message(f"[ERROR] Failed to load subfolders: {e}")

    def has_subfolders(self, target_dir):
        try:
            for entry in os.listdir(target_dir):
                if os.path.isdir(os.path.join(target_dir, entry)):
                    return True
        except Exception:
            pass
        return False

    # =========================================================
    # tableView setup and loading
    # =========================================================
    def setup_table(self):
        self.table_model = QStandardItemModel()
        # self.table_model.setHorizontalHeaderLabels(["Filename", "Size", "Type", "Modified"])
        self.table_model.setHorizontalHeaderLabels(["Filename", "Size"])
        self.ui.tableView.setModel(self.table_model)

        # Selection settings
        self.ui.tableView.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.ui.tableView.setSelectionMode(QAbstractItemView.SingleSelection)
        self.ui.tableView.setAlternatingRowColors(True)

        # Column width settings
        header = self.ui.tableView.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)           # Filename
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Size
        # header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Type
        # header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Modified

    def load_table(self):
        # Clear rows only (keep headers)
        self.table_model.removeRows(0, self.table_model.rowCount())

        # Get selected extensions from checkboxes
        selected_extensions = []
        if self.ui.checkBox_npz.isChecked():
            selected_extensions.append('.npz')
        if self.ui.checkBox_mim.isChecked():
            selected_extensions.extend('.mim')
        if self.ui.checkBox_png.isChecked():
            selected_extensions.append('.png')

        try:
            entries = sorted(os.listdir(self.current_dir))
            for entry in entries:
                full_path = os.path.join(self.current_dir, entry)

                if not os.path.isfile(full_path):
                    continue

                # Filter by selected extensions
                file_ext = os.path.splitext(full_path)[1].lower()
                if selected_extensions and file_ext not in selected_extensions:
                    continue

                # Get file info
                file_name  = entry
                file_size  = os.path.getsize(full_path)
                # file_type  = os.path.splitext(full_path)[1].upper() or "Unknown"
                # file_mtime = datetime.fromtimestamp(os.path.getmtime(full_path)).strftime("%Y-%m-%d %H:%M")

                # Create row items
                name_item  = QStandardItem(file_name)
                size_item  = QStandardItem(self.format_file_size(file_size))
                # type_item  = QStandardItem(file_type)
                # mtime_item = QStandardItem(file_mtime)

                # Store full path in name item (hidden data)
                name_item.setData(full_path, Qt.UserRole)

                # Set all items as non-editable
                # for item in [name_item, size_item, type_item, mtime_item]:
                for item in [name_item, size_item]:
                    item.setEditable(False)

                # Add row to table
                # self.table_model.appendRow([name_item, size_item, type_item, mtime_item])
                self.table_model.appendRow([name_item, size_item])

        except PermissionError:
            self.log_message(f"[ERROR] Permission denied: {self.current_dir}")
        except Exception as e:
            self.log_message(f"[ERROR] Failed to load files: {e}")

    # =========================================================
    # Image RGB widget setup
    # =========================================================
    def setup_image_rgb(self):
        self.image_widget_rgb = MatplotlibWidget()

        layout = QVBoxLayout()
        layout.addWidget(self.image_widget_rgb)
        layout.setContentsMargins(0, 0, 0, 0)
        self.ui.widget_image_rgb.setLayout(layout)

    def display_image_rgb(self):
        if self.current_image_rgb is None:
            return

        rgb_vis = self.current_image_rgb

        # Create figure and axes
        fig = Figure(figsize=(12, 6))
        axes = fig.subplots(1, 4)

        # axes[0] - RGB image
        axes[0].imshow(rgb_vis)
        axes[1].imshow(rgb_vis[..., 0], cmap="Reds_r")
        axes[2].imshow(rgb_vis[..., 1], cmap="Greens_r")
        axes[3].imshow(rgb_vis[..., 2], cmap="Blues_r")

        axes[0].set_title('RGB')
        axes[1].set_title('R')
        axes[2].set_title('G')
        axes[3].set_title('B')

        for ax in axes.flatten():
            ax.axis("off")

        fig.tight_layout()
        self.image_widget_rgb.set_figure(fig)

    # =========================================================
    # Image XYZ widget setup
    # =========================================================
    def setup_image_xyz(self):
        self.image_widget_xyz = MatplotlibWidget()

        layout = QVBoxLayout()
        layout.addWidget(self.image_widget_xyz)
        layout.setContentsMargins(0, 0, 0, 0)
        self.ui.widget_image_xyz.setLayout(layout)

    def display_image_xyz(self):
        if self.current_image_xyz is None:
            return
        
        xyz_vis = np.clip(self.current_image_xyz, 0, 1.1) / 1.1

        # Create figure and axes
        fig = Figure(figsize=(12, 6))
        axes = fig.subplots(1, 4)

        axes[0].imshow(xyz_vis)
        axes[1].imshow(xyz_vis[..., 0], cmap="Grays_r")
        axes[2].imshow(xyz_vis[..., 1], cmap="Grays_r")
        axes[3].imshow(xyz_vis[..., 2], cmap="Grays_r")

        axes[0].set_title('XYZ')
        axes[1].set_title('X')
        axes[2].set_title('Y')
        axes[3].set_title('Z')

        for ax in axes.flatten():
            ax.axis("off")

        fig.tight_layout()
        self.image_widget_xyz.set_figure(fig)

    # =========================================================
    # Image Luv widget setup
    # =========================================================
    def setup_image_luv(self):
        self.image_widget_luv = MatplotlibWidget()

        layout = QVBoxLayout()
        layout.addWidget(self.image_widget_luv)
        layout.setContentsMargins(0, 0, 0, 0)
        self.ui.widget_image_luv.setLayout(layout)

    def display_image_luv(self):
        if self.current_image_luv is None:
            return
        
        L_luv = np.clip(self.current_image_luv[..., 0] / 100, 0, 1)
        u_luv = np.clip((self.current_image_luv[..., 1] + 50) / 100, 0, 1)
        v_luv = np.clip((self.current_image_luv[..., 2] + 50) / 100, 0, 1)
        luv_vis = np.stack([L_luv, u_luv, v_luv], axis=-1)

        # Create figure and axes
        fig = Figure(figsize=(12, 6))
        axes = fig.subplots(1, 4)

        axes[0].imshow(luv_vis)
        axes[1].imshow(luv_vis[..., 0], cmap="Grays_r")
        axes[2].imshow(luv_vis[..., 1], cmap="Grays_r")
        axes[3].imshow(luv_vis[..., 2], cmap="Grays_r")

        axes[0].set_title('L*u*v*')
        axes[1].set_title('L*')
        axes[2].set_title('u*')
        axes[3].set_title('v*')

        for ax in axes.flatten():
            ax.axis("off")

        fig.tight_layout()
        self.image_widget_luv.set_figure(fig)

    # =========================================================
    # Image xyuv widget setup
    # =========================================================
    def setup_image_xyuv(self):
        self.image_widget_xyuv = MatplotlibWidget()

        layout = QVBoxLayout()
        layout.addWidget(self.image_widget_xyuv)
        layout.setContentsMargins(0, 0, 0, 0)
        self.ui.widget_image_xyuv.setLayout(layout)

    def display_image_xyuv(self):
        if self.current_image_xyuv is None:
            return
        
        # Create figure and axes
        fig = Figure(figsize=(12, 6))
        axes = fig.subplots(1, 4)

        axes[0].imshow(self.current_image_xyuv[..., 0], cmap="Grays_r")
        axes[1].imshow(self.current_image_xyuv[..., 1], cmap="Grays_r")
        axes[2].imshow(self.current_image_xyuv[..., 2], cmap="Grays_r")
        axes[3].imshow(self.current_image_xyuv[..., 3], cmap="Grays_r")

        axes[0].set_title('x')
        axes[1].set_title('y')
        axes[2].set_title("u'")
        axes[3].set_title("v'")

        for ax in axes.flatten():
            ax.axis("off")

        fig.tight_layout()
        self.image_widget_xyuv.set_figure(fig)

    # =========================================================
    # Signal handlers
    # =========================================================
    def connect_signals(self):
        # pushButton clicked
        self.ui.pushButton_open.clicked.connect(self.on_open_folder)
        self.ui.pushButton_refresh.clicked.connect(self.on_refresh_folder)

        # checkBox state changed - filter tableView
        self.ui.checkBox_png.stateChanged.connect(self.load_table)
        self.ui.checkBox_npz.stateChanged.connect(self.load_table)
        self.ui.checkBox_mim.stateChanged.connect(self.load_table)

        # treeView event filter - Enter key
        self.ui.treeView.installEventFilter(self)

        # treeView clicked / doubleClicked - update tableView
        self.ui.treeView.clicked.connect(self.on_tree_clicked)
        self.ui.treeView.doubleClicked.connect(self.on_tree_clicked)

        # treeView expanded - lazy load subfolders
        self.ui.treeView.expanded.connect(self.on_tree_expanded)

        # tableView event filter - block Left/Right, allow Up/Down
        self.ui.tableView.installEventFilter(self)
        self.ui.tableView.selectionModel().currentChanged.connect(self.on_table_clicked)

        # tabWidget changed - update display
        self.ui.tabWidget.currentChanged.connect(self.on_tab_changed)

    def on_open_folder(self):
        selected_dir = QFileDialog.getExistingDirectory(self, "Select Folder", self.current_dir)
        selected_dir = os.path.normpath(selected_dir)

        if selected_dir:
            self.current_dir = selected_dir
            self.load_tree()
            self.load_table()
            self.log_message(f"[Folder opened] {self.current_dir}")

    def on_refresh_folder(self):
        self.load_tree()
        self.load_table()
        self.log_message(f"[Folder refreshed] {self.current_dir}")

    def on_tree_clicked(self, index):
        item = self.tree_model.itemFromIndex(index)
        if item is None:
            return

        selected_dir = item.data(Qt.UserRole)
        if selected_dir and selected_dir != "placeholder":
            self.current_dir = selected_dir
            self.load_table()
            self.log_message(f"[Folder selected] {self.current_dir}")

    def on_tree_expanded(self, index):
        item = self.tree_model.itemFromIndex(index)
        if item is None:
            return

        # Check if first child is placeholder
        if item.rowCount() == 1:
            child = item.child(0)
            if child and child.data(Qt.UserRole) == "placeholder":
                # Remove placeholder and load actual subfolders
                item.removeRow(0)
                target_dir = item.data(Qt.UserRole)
                if target_dir:
                    self.load_subfolders(item, target_dir)

    def on_table_clicked(self, current, previous):
        # Get file path from first column
        row = current.row()
        if row < 0 or row >= self.table_model.rowCount():
            return
        name_item = self.table_model.item(row, 0)
        if name_item is None:
            return

        file_path = name_item.data(Qt.UserRole)
        if file_path is None:
            return

        # Check if image file
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in ['.npz', '.mim', '.png']:
            return

        # Load image
        self.load_data(file_path)
        self.log_message(f"[Image loaded] {file_path}")

        # Display based on current tab
        tab_index = self.ui.tabWidget.currentIndex()
        if tab_index == 0:    # RGB tab
            self.display_image_rgb()
        elif tab_index == 1:  # XYZ tab
            self.display_image_xyz()
        elif tab_index == 2:  # L*u*v* tab
            self.display_image_luv()
        elif tab_index == 3:  # xy u'v' tab
            self.display_image_xyuv()

    def on_tab_changed(self, tab_index):
        if self.current_image_rgb is None:
            return
        if tab_index == 0:    # RGB tab
            self.display_image_rgb()
        elif tab_index == 1:  # XYZ tab
            self.display_image_xyz()
        elif tab_index == 2:  # L*u*v* tab
            self.display_image_luv()
        elif tab_index == 3:  # xy u'v' tab
            self.display_image_xyuv()

    # =========================================================
    # Qt override
    # =========================================================
    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress:
            if obj == self.ui.tableView:
                if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right):
                    self.handle_tab_switch(event.key())
                    return True
            elif obj == self.ui.treeView:
                if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                    index = self.ui.treeView.currentIndex()
                    self.on_tree_clicked(index)
                    return True
        return super().eventFilter(obj, event)

    def handle_tab_switch(self, key):
        count = self.ui.tabWidget.count()
        current = self.ui.tabWidget.currentIndex()
        if key == Qt.Key.Key_Right:
            self.ui.tabWidget.setCurrentIndex((current + 1) % count)
        elif key == Qt.Key.Key_Left:
            self.ui.tabWidget.setCurrentIndex((current - 1) % count)

    # =========================================================
    # Helper functions
    # =========================================================
    
    def load_data(self, file_path):
        try:
            data = np.load(file_path)['data']
            XYZ = data.astype(np.float32) / data[..., 1].max()
            RGB = XYZ_to_RGB(XYZ)
            Luv = XYZ_to_Luv(XYZ)
            
            self.current_image_xyz = XYZ
            self.current_image_rgb = RGB
            self.current_image_luv = Luv
            self.current_image_xyuv = np.stack(XYZ_to_xyuv(XYZ), axis=-1)

        except Exception as e:
            self.log_message(f"[ERROR] Failed to load image: {file_path}, {e}")
            self.current_image_xyz = None
            self.current_image_rgb = None
            self.current_image_luv = None
            self.current_image_xyuv = None

    def format_file_size(self, size_bytes):
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
