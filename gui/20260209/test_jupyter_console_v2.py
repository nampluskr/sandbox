import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="qtconsole")

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, \
    QPushButton, QLabel, QSplitter, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont

# qtconsole 위젯 및 커널 매니저
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager

# QScintilla 관련 모듈
from PyQt5.Qsci import QsciScintilla, QsciLexerPython


# === 메인 윈도우 (QScintilla 사용) ===
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OLEDi - 통합 파이썬 코딩 환경 (QScintilla)")
        self.setGeometry(100, 100, 1200, 800)

        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # 스플리터 (좌우 크기 조절 가능)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # === 왼쪽: QScintilla 코드 에디터 ===
        editor_container = QWidget()
        editor_layout = QVBoxLayout(editor_container)
        editor_layout.setContentsMargins(0, 0, 0, 0)

        editor_title = QLabel("Python Code Editor")
        editor_title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")

        # QScintilla 설정
        self.editor = QsciScintilla()

        # 폰트: Consolas 또는 고정폭 폰트
        font = QFont("Consolas", 9)
        font.setFixedPitch(True)

        # Jupyter Notebook 테마 색상 (Solarized Light 기반)
        background = QColor("#f8f8f8")   # 매우 밝은 회백 배경
        default_color = QColor("#333333")  # 기본 텍스트 (검정에 가까움)

        # 렉서 생성 및 폰트 설정
        lexer = QsciLexerPython()
        lexer.setFont(font)

        # 배경 및 기본 색상
        lexer.setPaper(background)
        lexer.setDefaultPaper(background)
        lexer.setDefaultColor(default_color)

        # 구문 강조 색상 (Jupyter Notebook 스타일)
        lexer.setColor(QColor("#0086b3"), lexer.Keyword)              # 키워드: 파란 계열
        lexer.setColor(QColor("#1a1aa6"), lexer.ClassName)            # 클래스명
        lexer.setColor(QColor("#795e26"), lexer.FunctionMethodName)   # 함수명
        lexer.setColor(QColor("#a31515"), lexer.DoubleQuotedString)   # 문자열
        lexer.setColor(QColor("#a31515"), lexer.SingleQuotedString)
        lexer.setColor(QColor("#098658"), lexer.TripleDoubleQuotedString)  # docstring
        lexer.setColor(QColor("#098658"), lexer.TripleSingleQuotedString)
        lexer.setColor(QColor("#800080"), lexer.Number)               # 숫자: 보라
        lexer.setColor(QColor("#408080"), lexer.Comment)              # 주석: 녹색 계열
        lexer.setColor(QColor("#bb6688"), lexer.Decorator)            # 데코레이터 (@property 등)
        lexer.setColor(QColor("#000000"), lexer.Identifier)           # 변수명 (기본)

        # 선택 영역 색상 (옵션: Jupyter 느낌 살리기)
        self.editor.setSelectionBackgroundColor(QColor("#b3d9ff"))
        self.editor.setSelectionForegroundColor(default_color)

        # 렉서 적용
        self.editor.setLexer(lexer)

        # 나머지 에디터 설정
        self.editor.setMarginWidth(0, "000")
        self.editor.setMarginLineNumbers(0, True)
        self.editor.setBraceMatching(QsciScintilla.StrictBraceMatch)
        self.editor.setCaretLineVisible(True)
        self.editor.setCaretLineBackgroundColor(QColor("#efefef"))  # 커서 줄 연회색
        self.editor.setAutoIndent(True)
        self.editor.setIndentationGuides(True)

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
        self.editor.setText(sample_code)

        # 종료 시 채널 종료
        app = QApplication.instance()
        app.aboutToQuit.connect(self.cleanup)

    def execute_code(self):
        code = self.editor.text()
        if not code.strip():
            return
        self.console.execute(code)

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
