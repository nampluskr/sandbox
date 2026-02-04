## 참고

### 클립보드로 이미지 복사

```python
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPixmap
from io import BytesIO

def copy_figure_to_clipboard(self, fig):
    buf = BytesIO()
    try:
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)

        pixmap = QPixmap()
        success = pixmap.loadFromData(buf.getvalue(), "PNG")
        if success:
            clipboard = QApplication.clipboard()
            clipboard.setPixmap(pixmap, mode=QClipboard.Clipboard)
            self.log_message("[Copied] Figure copied to clipboard", 3000)
        else:
            self.log_message("[ERROR] Failed to load image into clipboard", 3000)
    except Exception as e:
        self.log_message(f"[ERROR] Clipboard copy failed: {e}", 5000)
    finally:
        buf.close()

def on_copy_current_figure(self):
    if self.current_image_rgb is None:
        self.log_message("[ERROR] No image loaded", 3000)
        return

    current_tab_index = self.ui.tabWidget.currentIndex()
    fig_to_copy = None

    if current_tab_index == 0:  # Image 탭
        fig_to_copy = self.image_widget.get_figure()
    elif current_tab_index == 1:  # Histogram 탭
        fig_to_copy = self.histogram_widget.get_figure()
    else:
        self.log_message("[INFO] Copy not supported in this tab", 3000)
        return

    if fig_to_copy is not None:
        self.copy_figure_to_clipboard(fig_to_copy)
    else:
        self.log_message("[ERROR] No figure to copy", 3000)
```

#### 예: 버튼 연결

```python
self.ui.pushButton_copy_image.clicked.connect(
    lambda: copy_figure_to_clipboard(self.image_widget.canvas.figure)
)
```
