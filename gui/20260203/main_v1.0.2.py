import sys
import os
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow,
                                QHeaderView, QAbstractItemView, QFileDialog)
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtCore import Qt
from ui_main_window import Ui_MainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Window settings
        self.setWindowTitle("Image Viewer")
        self.resize(900, 600)

        # Default folder path
        self.current_dir = r"D:\Non_Documents\_github\image_analyzer"

        # Setup treeView and tableView
        self.setup_tree_view()
        self.setup_table_view()

        # Load default folder
        self.load_tree()
        self.load_table()

        # Connect signals
        self.connect_signals()

    # =========================================================
    # treeView setup and loading
    # =========================================================
    def setup_tree_view(self):
        self.tree_model = QStandardItemModel()
        self.ui.treeView.setModel(self.tree_model)
        self.ui.treeView.setHeaderHidden(True)

    def load_tree(self):
        if not os.path.isdir(self.current_dir):
            print(f"[ERROR] Folder not found: {self.current_dir}")
            return

        self.tree_model.clear()

        # Create root itemt
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
            print(f"[ERROR] Permission denied: {target_dir}")
        except Exception as e:
            print(f"[ERROR] Failed to load subfolders: {e}")

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
    def setup_table_view(self):
        self.table_model = QStandardItemModel()
        self.table_model.setHorizontalHeaderLabels(["Filename", "Size", "Type", "Modified"])
        self.ui.tableView.setModel(self.table_model)

        # Selection settings
        self.ui.tableView.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.ui.tableView.setSelectionMode(QAbstractItemView.SingleSelection)
        self.ui.tableView.setAlternatingRowColors(True)

        # Column width settings
        header = self.ui.tableView.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)           # Filename
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Size
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Type
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Modified

    def load_table(self):
        # Clear rows only (keep headers)
        self.table_model.removeRows(0, self.table_model.rowCount())

        # Get selected extensions from checkboxes
        selected_extensions = []
        if self.ui.checkBox_png.isChecked():
            selected_extensions.append('.png')
        if self.ui.checkBox_bmp.isChecked():
            selected_extensions.append('.bmp')

        try:
            entries = sorted(os.listdir(self.current_dir), key=str.lower)
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
                file_type  = os.path.splitext(full_path)[1].upper() or "Unknown"
                file_mtime = datetime.fromtimestamp(
                    os.path.getmtime(full_path)
                ).strftime("%Y-%m-%d %H:%M")

                # Create row items
                name_item  = QStandardItem(file_name)
                size_item  = QStandardItem(self.format_file_size(file_size))
                type_item  = QStandardItem(file_type)
                mtime_item = QStandardItem(file_mtime)

                # Store full path in name item (hidden data)
                name_item.setData(full_path, Qt.UserRole)

                # Set all items as non-editable
                for item in [name_item, size_item, type_item, mtime_item]:
                    item.setEditable(False)

                # Add row to table
                self.table_model.appendRow([name_item, size_item, type_item, mtime_item])

        except PermissionError:
            print(f"[ERROR] Permission denied: {self.current_dir}")
        except Exception as e:
            print(f"[ERROR] Failed to load files: {e}")

    # =========================================================
    # Signal handlers
    # =========================================================
    def connect_signals(self):
        # pushButton clicked
        self.ui.pushButton_open.clicked.connect(self.on_open_folder)
        self.ui.pushButton_refresh.clicked.connect(self.on_refresh_folder)
        # checkBox state changed - filter tableView
        self.ui.checkBox_png.stateChanged.connect(self.load_table)
        self.ui.checkBox_bmp.stateChanged.connect(self.load_table)
        self.ui.treeView.clicked.connect(self.on_tree_clicked)
        # treeView expanded - lazy load subfolders
        self.ui.treeView.expanded.connect(self.on_tree_expanded)

    def on_open_folder(self):
        selected_dir = QFileDialog.getExistingDirectory(
            self, "Select Folder", self.current_dir
        )
        if selected_dir:
            self.current_dir = selected_dir
            self.load_tree()
            self.load_table()
            print(f"[Folder opened] {self.current_dir}")

    def on_refresh_folder(self):
        self.load_tree()
        self.load_table()
        print(f"[Folder refreshed] {self.current_dir}")

    def on_tree_clicked(self, index):
        item = self.tree_model.itemFromIndex(index)
        if item is None:
            return

        selected_dir = item.data(Qt.UserRole)
        if selected_dir and selected_dir != "placeholder":
            self.current_dir = selected_dir
            self.load_table()
            print(f"[Folder selected] {self.current_dir}")

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

    # =========================================================
    # Utility functions
    # =========================================================
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
