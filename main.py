"""
Entry point for the Hypothesis Consensus Analyzer application.
"""
import sys

from PySide6.QtCore import QCoreApplication, QSettings
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow
from gui.theme import get_stylesheet


def main():
    app = QApplication(sys.argv)

    # Read saved font size (default 10pt)
    settings = QSettings("Systes", "HypothesisConsensusAnalyzer")
    font_size = settings.value("appearance/font_size", 10, type=int)

    # Apply Win98-style theme with user font size
    app.setStyleSheet(get_stylesheet(font_size))

    # Set application-wide font
    font = QFont("Tahoma", font_size)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
