"""
OLED Anomaly Detection Analysis System
Qt Designer + PySide6 based
"""

import sys
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QFileSystemModel, 
                               QVBoxLayout, QWidget)
from PySide6.QtCore import QDir, Qt

# Matplotlib imports
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import the generated UI class
from ui_oled_analyzer import Ui_MainWindow


class MplCanvas(FigureCanvas):
    """Matplotlib canvas for embedding in Qt"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)


class OLEDAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_data = None  # Currently loaded data
        
        # Setup UI from generated file
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        self.setWindowTitle("OLED Anomaly Detection Analysis System")
        self.resize(1400, 800)
        
        self.setup_matplotlib_canvases()
        self.setup_file_tree()
        self.connect_signals()
        
        print("[OK] UI setup complete")
        
    def setup_matplotlib_canvases(self):
        """Setup matplotlib canvases in each tab"""
        # RGB Canvas
        self.rgb_canvas = MplCanvas(self.ui.rgbCanvas, width=5, height=4, dpi=100)
        layout_rgb = QVBoxLayout(self.ui.rgbCanvas)
        layout_rgb.addWidget(self.rgb_canvas)
        layout_rgb.setContentsMargins(0, 0, 0, 0)
        
        # Luminance Canvas
        self.lum_canvas = MplCanvas(self.ui.lumCanvas, width=5, height=4, dpi=100)
        layout_lum = QVBoxLayout(self.ui.lumCanvas)
        layout_lum.addWidget(self.lum_canvas)
        layout_lum.setContentsMargins(0, 0, 0, 0)
        
        # Histogram Canvas
        self.hist_canvas = MplCanvas(self.ui.histCanvas, width=5, height=4, dpi=100)
        layout_hist = QVBoxLayout(self.ui.histCanvas)
        layout_hist.addWidget(self.hist_canvas)
        layout_hist.setContentsMargins(0, 0, 0, 0)
        
        # Anomaly Canvas
        self.anomaly_canvas = MplCanvas(self.ui.anomalyCanvas, width=5, height=4, dpi=100)
        layout_anomaly = QVBoxLayout(self.ui.anomalyCanvas)
        layout_anomaly.addWidget(self.anomaly_canvas)
        layout_anomaly.setContentsMargins(0, 0, 0, 0)
        
        # Initial placeholder text
        for canvas, title in [(self.rgb_canvas, 'RGB'),
                              (self.lum_canvas, 'Luminance'),
                              (self.hist_canvas, 'Histogram'),
                              (self.anomaly_canvas, 'Anomaly Map')]:
            canvas.axes.text(0.5, 0.5, f'{title}\n\nSelect a file to view',
                           horizontalalignment='center',
                           verticalalignment='center',
                           transform=canvas.axes.transAxes,
                           fontsize=14, color='gray')
            canvas.axes.set_xticks([])
            canvas.axes.set_yticks([])
            canvas.draw()
        
        print("[OK] Matplotlib canvases setup complete")
        
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
        
        # Hide unnecessary columns
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
        
        # Sliders (if they exist)
        try:
            self.ui.meanSlider.valueChanged.connect(self.update_luminance_plot)
            self.ui.stdSlider.valueChanged.connect(self.update_luminance_plot)
            self.ui.thresholdSlider.valueChanged.connect(self.update_anomaly_plot)
        except AttributeError:
            print("[INFO] Sliders not found in UI - skipping slider connections")
        
        print("[OK] Signal connections complete")
        
    def on_file_selected(self, index):
        """File selection event handler"""
        file_path = self.file_model.filePath(index)
        
        if file_path.endswith('.npz'):
            print(f"\n{'='*60}")
            print(f"[FILE] Selected: {Path(file_path).name}")
            
            # Load npz file
            try:
                data = np.load(file_path, allow_pickle=True)
                self.current_data = {
                    'xyz': data['xyz'],
                    'anomaly_map': data.get('anomaly_map', None),
                    'file_path': file_path,
                    'filename': Path(file_path).name
                }
                
                print(f"[OK] Loaded XYZ data: {self.current_data['xyz'].shape}")
                if self.current_data['anomaly_map'] is not None:
                    print(f"[OK] Loaded Anomaly Map: {self.current_data['anomaly_map'].shape}")
                
                # Update current tab
                self.update_current_tab()
                
            except Exception as e:
                print(f"[ERROR] Failed to load file: {e}")
                self.current_data = None
            
            print(f"{'='*60}")
    
    def xyz_to_rgb(self, xyz_image):
        """Convert XYZ color space to RGB"""
        # Normalize to 0-1 range
        xyz = xyz_image.astype(float) / 255.0
        
        # XYZ to RGB conversion matrix (sRGB)
        # RGB = M * XYZ
        M = np.array([
            [ 3.2406, -1.5372, -0.4986],
            [-0.9689,  1.8758,  0.0415],
            [ 0.0557, -0.2040,  1.0570]
        ])
        
        # Reshape for matrix multiplication
        h, w = xyz.shape[:2]
        xyz_flat = xyz.reshape(-1, 3)
        
        # Apply conversion
        rgb_flat = xyz_flat @ M.T
        
        # Gamma correction
        rgb_flat = np.where(rgb_flat > 0.0031308,
                           1.055 * np.power(rgb_flat, 1/2.4) - 0.055,
                           12.92 * rgb_flat)
        
        # Clip to valid range and reshape
        rgb = np.clip(rgb_flat, 0, 1).reshape(h, w, 3)
        
        return rgb
    
    def extract_luminance(self, xyz_image):
        """Extract Y channel (luminance) from XYZ image"""
        return xyz_image[:, :, 1]  # Y is the second channel
    
    def normalize_luminance(self, luminance, target_mean, target_std):
        """Normalize luminance to target mean and std"""
        current_mean = np.mean(luminance)
        current_std = np.std(luminance)
        
        # Z-score normalization then scale to target
        if current_std > 0:
            normalized = (luminance - current_mean) / current_std
            normalized = normalized * target_std + target_mean
        else:
            normalized = luminance
        
        return np.clip(normalized, 0, 255)
    
    def update_current_tab(self):
        """Update the currently visible tab"""
        if self.current_data is None:
            return
        
        current_index = self.ui.mainTabWidget.currentIndex()
        tab_name = self.ui.mainTabWidget.tabText(current_index)
        
        if tab_name == 'RGB':
            self.update_rgb_plot()
        elif tab_name == 'Luminance':
            self.update_luminance_plot()
        elif tab_name == 'Histogram':
            self.update_histogram_plot()
        elif tab_name == 'Anomaly Map':
            self.update_anomaly_plot()
    
    def update_rgb_plot(self):
        """Update RGB visualization"""
        if self.current_data is None:
            return
        
        self.rgb_canvas.axes.clear()
        
        # Convert XYZ to RGB
        rgb = self.xyz_to_rgb(self.current_data['xyz'])
        
        # Display
        self.rgb_canvas.axes.imshow(rgb)
        self.rgb_canvas.axes.set_title(f"RGB Conversion - {self.current_data['filename']}")
        self.rgb_canvas.axes.axis('off')
        self.rgb_canvas.draw()
        
        print("[UPDATE] RGB plot updated")
    
    def update_luminance_plot(self):
        """Update luminance visualization"""
        if self.current_data is None:
            return
        
        self.lum_canvas.axes.clear()
        
        # Extract luminance
        luminance = self.extract_luminance(self.current_data['xyz'])
        
        # Get slider values (if they exist)
        try:
            target_mean = self.ui.meanSlider.value()
            target_std = self.ui.stdSlider.value()
            
            # Normalize
            normalized = self.normalize_luminance(luminance, target_mean, target_std)
            
            title = f"Luminance (Mean={target_mean}, Std={target_std})"
        except AttributeError:
            # Sliders don't exist, use original
            normalized = luminance
            title = "Luminance (Y Channel)"
        
        # Display
        self.lum_canvas.axes.imshow(normalized, cmap='gray', vmin=0, vmax=255)
        self.lum_canvas.axes.set_title(title)
        self.lum_canvas.axes.axis('off')
        self.lum_canvas.draw()
        
        print("[UPDATE] Luminance plot updated")
    
    def update_histogram_plot(self):
        """Update histogram visualization"""
        if self.current_data is None:
            return
        
        self.hist_canvas.axes.clear()
        
        # Extract luminance
        luminance = self.extract_luminance(self.current_data['xyz'])
        
        # Plot histogram
        self.hist_canvas.axes.hist(luminance.flatten(), bins=256, range=(0, 255),
                                   color='blue', alpha=0.7, edgecolor='black')
        self.hist_canvas.axes.set_xlabel('Luminance Value')
        self.hist_canvas.axes.set_ylabel('Frequency')
        self.hist_canvas.axes.set_title(f"Luminance Histogram - {self.current_data['filename']}")
        self.hist_canvas.axes.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(luminance)
        std_val = np.std(luminance)
        self.hist_canvas.axes.axvline(mean_val, color='red', linestyle='--',
                                      label=f'Mean: {mean_val:.1f}')
        self.hist_canvas.axes.axvline(mean_val - std_val, color='orange',
                                      linestyle=':', alpha=0.7, label=f'Std: {std_val:.1f}')
        self.hist_canvas.axes.axvline(mean_val + std_val, color='orange',
                                      linestyle=':', alpha=0.7)
        self.hist_canvas.axes.legend()
        
        self.hist_canvas.draw()
        
        print("[UPDATE] Histogram plot updated")
    
    def update_anomaly_plot(self):
        """Update anomaly map visualization"""
        if self.current_data is None:
            return
        
        self.anomaly_canvas.axes.clear()
        
        if self.current_data['anomaly_map'] is None:
            # No anomaly map available
            self.anomaly_canvas.axes.text(0.5, 0.5,
                                         'No Anomaly Map\navailable for this file',
                                         horizontalalignment='center',
                                         verticalalignment='center',
                                         transform=self.anomaly_canvas.axes.transAxes,
                                         fontsize=14, color='gray')
            self.anomaly_canvas.axes.set_xticks([])
            self.anomaly_canvas.axes.set_yticks([])
        else:
            # Display anomaly map
            anomaly_map = self.current_data['anomaly_map']
            
            # Get threshold value (if slider exists)
            try:
                threshold = self.ui.thresholdSlider.value() / 100.0  # 0-1 range
                title = f"Anomaly Map (Threshold={threshold:.2f})"
            except AttributeError:
                threshold = 0.5
                title = "Anomaly Map"
            
            # Create classification mask
            mask = anomaly_map > threshold
            
            # Display
            im = self.anomaly_canvas.axes.imshow(anomaly_map, cmap='hot', vmin=0, vmax=1)
            self.anomaly_canvas.axes.contour(mask, colors='cyan', linewidths=2,
                                            levels=[0.5])
            
            # Add colorbar
            if not hasattr(self, '_anomaly_colorbar'):
                self._anomaly_colorbar = self.anomaly_canvas.fig.colorbar(im,
                                                                          ax=self.anomaly_canvas.axes)
            else:
                self._anomaly_colorbar.update_normal(im)
            
            self.anomaly_canvas.axes.set_title(title)
            self.anomaly_canvas.axes.axis('off')
            
            # Calculate statistics
            anomaly_ratio = np.sum(mask) / mask.size * 100
            print(f"[INFO] Anomaly region: {anomaly_ratio:.2f}%")
        
        self.anomaly_canvas.draw()
        
        print("[UPDATE] Anomaly map updated")
    
    def on_tab_changed(self, index):
        """Tab change event handler"""
        tab_name = self.ui.mainTabWidget.tabText(index)
        print(f"[TAB] Changed to: {tab_name}")
        
        # Update the newly selected tab
        self.update_current_tab()


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
