## 참고

### 클립보드로 이미지 복사

```python
from PySide6.QtGui import QImage, QPixmap, QClipboard

def copy_figure_to_clipboard(fig):
    """Figure를 렌더링해 클립보드에 이미지로 복사"""
    # 메모리 내에 저장
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)

    # QImage로 변환
    img = QImage()
    img.loadFromData(buf.read(), "PNG")

    # 클립보드에 복사
    clipboard = QClipboard()
    clipboard.setImage(img)

    buf.close()
```

#### 예: 버튼 연결

```python
self.ui.pushButton_copy_image.clicked.connect(
    lambda: copy_figure_to_clipboard(self.image_widget.canvas.figure)
)
```
