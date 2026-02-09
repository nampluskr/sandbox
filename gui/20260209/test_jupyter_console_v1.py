import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="qtconsole")

from PyQt5.QtWidgets import QApplication, QMainWindow, QPlainTextEdit, QWidget, \
    QVBoxLayout, QTextEdit, QPushButton, QLabel, QMessageBox, QSplitter, QTextEdit, \
    QHBoxLayout
from PyQt5.QtCore import Qt

# qtconsole 위젯 및 커널 매니저
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager


# === 구문 강조, 라인 번호 등 코드 에디터 클래스 (이전과 동일) ===
from PyQt5.QtGui import (
    QColor, QTextCharFormat, QFont, QSyntaxHighlighter,
    QPainter, QTextFormat
)
from PyQt5.QtCore import QRegExp, QSize


def format(color, style=''):
    _color = QColor(color)
    _format = QTextCharFormat()
    _format.setForeground(_color)
    if 'bold' in style:
        _format.setFontWeight(QFont.Bold)
    if 'italic' in style:
        _format.setFontItalic(True)
    return _format


PYTHON_KEYWORDS = [
    'and', 'as', 'assert', 'break', 'class', 'continue', 'def',
    'del', 'elif', 'else', 'except', 'False', 'finally', 'for',
    'from', 'global', 'if', 'import', 'in', 'is', 'lambda',
    'None', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return',
    'True', 'try', 'while', 'with', 'yield'
]


class PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, document):
        super().__init__(document)
        self.styles = {
            'keyword': format('blue', 'bold'),
            'string': format('green'),
            'comment': format('gray', 'italic'),
            'number': format('brown'),
        }
        self.rules = []
        for word in PYTHON_KEYWORDS:
            pattern = r'\b' + word + r'\b'
            self.rules.append((QRegExp(pattern), self.styles['keyword']))
        self.rules.append((QRegExp(r'"[^"\\]*(\\.[^"\\]*)*"'), self.styles['string']))
        self.rules.append((QRegExp(r"'[^'\\]*(\\.[^'\\]*)*'"), self.styles['string']))
        self.rules.append((QRegExp(r'#[^\n]*'), self.styles['comment']))
        self.rules.append((QRegExp(r'\b[0-9]+\b'), self.styles['number']))

    def highlightBlock(self, text):
        for pattern, style in self.rules:
            index = pattern.indexIn(text)
            while index >= 0:
                length = pattern.matchedLength()
                self.setFormat(index, length, style)
                index = pattern.indexIn(text, index + length)


class LineNumberArea(QWidget):
    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self):
        return QSize(self.editor.lineNumberAreaWidth(), 0)

    def paintEvent(self, event):
        self.editor.lineNumberAreaPaintEvent(event)


class CodeEditor(QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self.lineNumberArea = LineNumberArea(self)
        self.blockCountChanged.connect(self.updateLineNumberAreaWidth)
        self.updateRequest.connect(self.updateLineNumberArea)
        self.cursorPositionChanged.connect(self.highlightCurrentLine)
        self.updateLineNumberAreaWidth()
        self.highlightCurrentLine()

    def lineNumberAreaWidth(self):
        digits = 1
        count = max(1, self.blockCount())
        while count >= 10:
            count //= 10
            digits += 1
        space = 3 + self.fontMetrics().horizontalAdvance('9') * digits
        return space

    def updateLineNumberAreaWidth(self):
        self.setViewportMargins(self.lineNumberAreaWidth(), 0, 0, 0)

    def updateLineNumberArea(self, rect, dy):
        if dy:
            self.lineNumberArea.scroll(0, dy)
        else:
            self.lineNumberArea.update(0, rect.y(), self.lineNumberArea.width(), rect.height())

    def lineNumberAreaPaintEvent(self, event):
        painter = QPainter(self.lineNumberArea)
        painter.fillRect(event.rect(), QColor(240, 240, 240))
        painter.setPen(Qt.black)
        block = self.firstVisibleBlock()
        blockNumber = block.blockNumber()
        top = self.blockBoundingGeometry(block).translated(self.contentOffset()).top()

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and top >= event.rect().top():
                number = str(blockNumber + 1)
                width = self.lineNumberArea.width()
                height = self.fontMetrics().height()
                painter.drawText(0, top, width, height, Qt.AlignRight, number)
            block = block.next()
            top += self.blockBoundingRect(block).height()
            blockNumber += 1

    def highlightCurrentLine(self):
        extraSelections = []
        if not self.isReadOnly():
            selection = QTextEdit.ExtraSelection()  # ← 수정된 부분
            selection.format.setBackground(QColor(255, 255, 220))
            selection.format.setProperty(QTextFormat.FullWidthSelection, True)
            selection.cursor = self.textCursor()
            selection.cursor.clearSelection()
            extraSelections.append(selection)
        self.setExtraSelections(extraSelections)


# === 메인 윈도우 ===
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OLEDi - 통합 파이썬 코딩 환경")
        self.setGeometry(100, 100, 1200, 800)

        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # 스플리터 (좌우 크기 조절 가능)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # === 왼쪽: 코드 에디터 ===
        editor_container = QWidget()
        editor_layout = QVBoxLayout(editor_container)
        editor_layout.setContentsMargins(0, 0, 0, 0)

        editor_title = QLabel("Python Code Editor")
        editor_title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")
        self.editor = CodeEditor()
        self.editor_highlighter = PythonHighlighter(self.editor.document())

        self.run_button = QPushButton("Run")
        self.run_button.setStyleSheet("font-size: 12px; padding: 8px;")
        self.run_button.clicked.connect(self.execute_code)

        editor_layout.addWidget(editor_title)
        editor_layout.addWidget(self.editor)
        editor_layout.addWidget(self.run_button)

        # === 오른쪽: Jupyter qtconsole 위젯 ===
        self.console = RichJupyterWidget()
        self.console.kernel_manager = QtInProcessKernelManager()
        self.console.kernel_manager.start_kernel(show_banner=True)
        self.console.kernel_client = self.console.kernel_manager.client()
        self.console.kernel_client.start_channels()

        console_container = QWidget()
        console_layout = QVBoxLayout(console_container)
        console_layout.setContentsMargins(0, 0, 0, 0)
        console_title = QLabel("Jupyter qtconsole")
        console_title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")
        console_layout.addWidget(console_title)
        console_layout.addWidget(self.console)

        # 위젯 추가
        splitter.addWidget(editor_container)
        splitter.addWidget(console_container)
        splitter.setSizes([600, 600])  # 초기 크기 비율

        # 예제 코드 삽입
        sample_code = '''%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 101)
y = np.sin(x)
plt.figure(figsize=(8, 4))
plt.plot(x, y, label='sin(x)', color='red')
plt.title("Graph")
plt.xlabel("x (radians)")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
'''
        self.editor.setPlainText(sample_code)

        # 종료 시 채널 종료
        app = QApplication.instance()
        app.aboutToQuit.connect(self.cleanup)

    def execute_code(self):
        code = self.editor.toPlainText()
        if not code.strip():
            return

        # prepared_code = "%matplotlib inline\n" + code
        self.console.execute(source=code)
        self.console.execute()


    def cleanup(self):
        """앱 종료 시 커널 정리"""
        if hasattr(self, 'console'):
            self.console.kernel_client.stop_channels()
            self.console.kernel_manager.shutdown_kernel()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
