"""
Dialog windows for the Hypothesis Consensus Analyzer GUI.

Contains:
- ComponentEditor: Modal dialog for editing a hypothesis component
- SettingsDialog: Modal dialog for all application settings (moved from Settings tab)
- HelpDialog: Per-tab help dialog with methodology context
- AutoBackupOptionsDialog: Opt-in Google Drive/local-folder backup settings
- WelcomeDialog: First-run welcome dialog with pipeline overview
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.config import Config
from core.models import HypothesisComponent
from gui.project_manager import ProjectManager
from gui.theme import COLORS

if TYPE_CHECKING:
    from gui.main_window import MainWindow

logger = logging.getLogger(__name__)


def _default_google_drive_folder() -> str:
    """Return a likely local Google Drive folder, or the home directory."""
    return str(ProjectManager.default_auto_backup_destination())


class AutoBackupOptionsDialog(QDialog):
    """Configure lightweight automatic project backups."""

    def __init__(
        self,
        parent=None,
        default_enabled: bool | None = None,
        project_path: Path | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Automatic Project Backup")
        self.setMinimumWidth(520)
        self.project_path = Path(project_path).expanduser().resolve() if project_path else None

        settings = QSettings("Systes", "HypothesisConsensusAnalyzer")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        intro = QLabel(
            "Choose a local Google Drive folder or another synced folder. "
            "SystS will copy only changed review files by default."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        self.enabled = QCheckBox("Enable automatic project backup")
        saved_enabled = settings.value("backup/enabled", False, type=bool)
        self.enabled.setChecked(saved_enabled if default_enabled is None else default_enabled)
        outer.addWidget(self.enabled)

        destination_group = QGroupBox("Backup Destination")
        destination_layout = QHBoxLayout(destination_group)
        self.destination_edit = QLineEdit()
        self.destination_edit.setPlaceholderText("Google Drive / SystS Backups")
        self.destination_edit.setText(
            settings.value("backup/destination", _default_google_drive_folder(), type=str)
        )
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_destination)
        destination_layout.addWidget(self.destination_edit, 1)
        destination_layout.addWidget(browse_btn)
        outer.addWidget(destination_group)

        options_group = QGroupBox("What to Back Up")
        options_layout = QVBoxLayout(options_group)
        self.include_images = QCheckBox("Include generated plots/images")
        self.include_images.setChecked(settings.value("backup/include_images", True, type=bool))
        self.include_fulltext = QCheckBox("Include full-text PDFs and raw downloads")
        self.include_fulltext.setChecked(settings.value("backup/include_fulltext", False, type=bool))
        self.include_fulltext.setToolTip("Off by default to avoid large Drive sync jobs.")
        options_layout.addWidget(self.include_images)
        options_layout.addWidget(self.include_fulltext)
        outer.addWidget(options_group)

        timing_group = QGroupBox("When to Back Up")
        timing_layout = QFormLayout(timing_group)
        self.on_stage_complete = QCheckBox("After completed pipeline stages")
        self.on_stage_complete.setChecked(
            settings.value("backup/on_stage_complete", True, type=bool)
        )
        self.on_exit = QCheckBox("When exiting the app")
        self.on_exit.setChecked(settings.value("backup/on_exit", True, type=bool))
        self.min_interval = QSpinBox()
        self.min_interval.setRange(1, 1440)
        self.min_interval.setValue(int(settings.value("backup/min_interval_minutes", 10)))
        self.min_interval.setSuffix(" min")
        self.max_file_mb = QSpinBox()
        self.max_file_mb.setRange(1, 4096)
        self.max_file_mb.setValue(int(settings.value("backup/max_file_mb", 100)))
        self.max_file_mb.setSuffix(" MB")
        timing_layout.addRow(self.on_stage_complete)
        timing_layout.addRow(self.on_exit)
        timing_layout.addRow("Minimum interval:", self.min_interval)
        timing_layout.addRow("Skip files larger than:", self.max_file_mb)
        outer.addWidget(timing_group)

        note = QLabel(
            "Manual backups still create full timestamped zip files. Automatic backups "
            "update a lightweight latest/ folder for easy restore."
        )
        note.setWordWrap(True)
        outer.addWidget(note)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        save_btn = QPushButton("Save Settings")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._accept_if_valid)
        buttons.addWidget(cancel_btn)
        buttons.addWidget(save_btn)
        outer.addLayout(buttons)

    def _browse_destination(self) -> None:
        current = Path(self.destination_edit.text().strip() or _default_google_drive_folder())
        start_dir = current if current.exists() else current.parent
        path = QFileDialog.getExistingDirectory(
            self,
            "Choose Backup Destination",
            str(start_dir),
        )
        if path:
            self.destination_edit.setText(path)

    def _accept_if_valid(self) -> None:
        destination = self.destination_edit.text().strip()
        if self.enabled.isChecked() and not destination:
            QMessageBox.warning(self, "Backup Destination", "Choose a backup destination folder.")
            return
        if self.enabled.isChecked():
            destination_path = Path(destination).expanduser()
            try:
                destination_path.mkdir(parents=True, exist_ok=True)
                if self.project_path is not None:
                    ProjectManager._validate_backup_destination(
                        self.project_path,
                        destination_path,
                    )
            except Exception as exc:
                QMessageBox.warning(
                    self,
                    "Backup Destination",
                    f"Cannot use this backup destination:\n{exc}",
                )
                return
            self.destination_edit.setText(str(destination_path.resolve()))
        self.accept()

    def values(self) -> dict:
        return {
            "enabled": self.enabled.isChecked(),
            "destination": self.destination_edit.text().strip(),
            "include_images": self.include_images.isChecked(),
            "include_fulltext": self.include_fulltext.isChecked(),
            "on_stage_complete": self.on_stage_complete.isChecked(),
            "on_exit": self.on_exit.isChecked(),
            "min_interval_minutes": self.min_interval.value(),
            "max_file_mb": self.max_file_mb.value(),
        }


# ---------------------------------------------------------------------------
# ComponentEditor
# ---------------------------------------------------------------------------

_KEYWORD_ANALYSIS_LABEL = "Keyword analysis:"


def _split_keyword_analysis_description(description: str) -> tuple[str, str]:
    """Return editable description text and preserved keyword-analysis suffix."""
    if _KEYWORD_ANALYSIS_LABEL not in description:
        return description.strip(), ""

    base, analysis = description.split(_KEYWORD_ANALYSIS_LABEL, 1)
    return base.rstrip(" |"), f" | {_KEYWORD_ANALYSIS_LABEL}{analysis.rstrip()}"


def _parse_keyword_analysis_metrics(analysis_suffix: str) -> dict[str, str]:
    """Extract displayed metric values from a stored keyword-analysis suffix."""
    metrics = re.findall(
        r"\b(utility|coverage|relevance|confidence|direction)=(-?\d+(?:\.\d+)?)",
        analysis_suffix,
    )
    return {name: value for name, value in metrics}


class ComponentEditor(QDialog):
    """Modal dialog for editing a single hypothesis component."""

    def __init__(self, component: HypothesisComponent, parent=None):
        super().__init__(parent)
        self.component = component
        self._base_description, self._analysis_suffix = (
            _split_keyword_analysis_description(component.description)
        )
        self.setWindowTitle("Edit Component")
        self.setMinimumSize(720, 360)

        layout = QVBoxLayout(self)

        content_row = QHBoxLayout()
        content_row.setSpacing(12)

        form = QFormLayout()
        self.label_edit = QLineEdit(component.label)
        form.addRow("Label:", self.label_edit)

        self.desc_edit = QTextEdit()
        self.desc_edit.setPlainText(self._base_description)
        self.desc_edit.setMaximumHeight(100)
        form.addRow("Description:", self.desc_edit)

        self.keywords_edit = QTextEdit()
        self.keywords_edit.setPlainText(", ".join(component.keywords))
        self.keywords_edit.setMaximumHeight(60)
        form.addRow("Keywords:", self.keywords_edit)

        content_row.addLayout(form, 3)
        content_row.addWidget(self._build_keyword_feedback_panel(), 2)
        layout.addLayout(content_row)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(save_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    def _save(self):
        self.component.label = self.label_edit.text().strip()
        base_description = self.desc_edit.toPlainText().strip()
        self.component.description = (
            f"{base_description}{self._analysis_suffix}"
            if self._analysis_suffix and base_description
            else base_description or self._analysis_suffix.lstrip(" |")
        )
        kw_text = self.keywords_edit.toPlainText().strip()
        self.component.keywords = [
            k.strip() for k in kw_text.split(",") if k.strip()
        ]
        self.accept()

    def _build_keyword_feedback_panel(self) -> QGroupBox:
        group = QGroupBox("Keyword Analysis")
        layout = QVBoxLayout(group)

        metrics = _parse_keyword_analysis_metrics(self._analysis_suffix)
        if not metrics:
            note = QLabel("Run Keyword Analysis to populate feedback.")
            note.setWordWrap(True)
            note.setStyleSheet(f"color: {COLORS['text_secondary']};")
            layout.addWidget(note)
            layout.addStretch()
            return group

        metric_form = QFormLayout()
        rows = [
            (
                "Weight",
                f"{float(getattr(self.component, 'weight', 1.0) or 0.0):.3f}",
            ),
            ("Utility", metrics.get("utility", "-")),
            ("Coverage", metrics.get("coverage", "-")),
            ("Relevance", metrics.get("relevance", "-")),
            ("Confidence", metrics.get("confidence", "-")),
            ("Direction", metrics.get("direction", "-")),
        ]
        for label, value in rows:
            value_label = QLabel(value)
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            value_label.setStyleSheet("font-weight: bold;")
            metric_form.addRow(f"{label}:", value_label)
        layout.addLayout(metric_form)

        hint = QLabel("These values are read-only feedback from the last keyword analysis.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(hint)
        layout.addStretch()
        return group


# ---------------------------------------------------------------------------
# SettingsDialog  (new — replaces the old Settings tab)
# ---------------------------------------------------------------------------

class SettingsDialog(QDialog):
    """
    Modal dialog containing all pipeline settings.

    Previously these controls lived in tab #10 of MainWindow as
    ``_build_settings_tab``.  The dialog reads from and writes back to
    ``MainWindow.pipeline_config``, keeping the same widget attribute names
    so that existing helper methods (``_settings_apply``,
    ``_populate_settings_from_config``, etc.) continue to work.
    """

    def __init__(self, main_window: "MainWindow", parent=None):
        super().__init__(parent or main_window)
        self.main_window = main_window
        self.setWindowTitle("Systes Settings")
        self.setMinimumSize(620, 580)
        self.resize(680, 720)

        self._build_ui()
        self._populate_from_config(main_window.pipeline_config)

    # ── UI construction ────────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # Scrollable area for all settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(8)

        # --- Config file management ---
        layout.addWidget(self._build_config_group())

        # --- Appearance ---
        layout.addWidget(self._build_appearance_group())

        # --- Project root ---
        layout.addWidget(self._build_project_root_group())

        # --- API Keys ---
        layout.addWidget(self._build_api_keys_group())

        # --- LLM Provider Settings ---
        layout.addWidget(self._build_llm_group())

        # --- Processing Parameters ---
        layout.addWidget(self._build_processing_group())

        layout.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        # --- Bottom button row ---
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        apply_btn = QPushButton("Apply")
        apply_btn.setMinimumWidth(90)
        apply_btn.clicked.connect(self._on_apply)

        close_btn = QPushButton("Close")
        close_btn.setMinimumWidth(90)
        close_btn.clicked.connect(self.accept)

        btn_row.addWidget(apply_btn)
        btn_row.addWidget(close_btn)
        outer.addLayout(btn_row)

    def _build_config_group(self) -> QGroupBox:
        grp = QGroupBox("Configuration File")
        h = QHBoxLayout(grp)

        load_btn = QPushButton("Load Config...")
        load_btn.clicked.connect(self._on_load_config)
        save_btn = QPushButton("Save Config...")
        save_btn.clicked.connect(self._on_save_config)
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._on_reset_config)

        h.addWidget(load_btn)
        h.addWidget(save_btn)
        h.addWidget(reset_btn)
        h.addStretch()
        return grp

    def _build_appearance_group(self) -> QGroupBox:
        grp = QGroupBox("Appearance")
        form = QFormLayout(grp)

        self.settings_font_size = QSpinBox()
        self.settings_font_size.setRange(6, 24)
        self.settings_font_size.setValue(10)
        self.settings_font_size.setSuffix(" pt")
        self.settings_font_size.setToolTip(
            "Application font size (default 10pt). Changes take effect on Apply."
        )
        form.addRow("Font Size:", self.settings_font_size)

        return grp

    def _build_project_root_group(self) -> QGroupBox:
        grp = QGroupBox("Project Root")
        h = QHBoxLayout(grp)
        self.settings_project_root = QLineEdit()
        self.settings_project_root.setPlaceholderText("Project root directory")
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_project_root)
        h.addWidget(self.settings_project_root, 1)
        h.addWidget(browse_btn)
        return grp

    def _build_api_keys_group(self) -> QGroupBox:
        grp = QGroupBox("API Keys")
        form = QFormLayout(grp)

        self.settings_openai_key = QLineEdit()
        self.settings_openai_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.settings_openai_key.setPlaceholderText("sk-...")
        form.addRow("OpenAI API Key:", self.settings_openai_key)

        self.settings_longcat_key = QLineEdit()
        self.settings_longcat_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.settings_longcat_key.setPlaceholderText("LongCat API key")
        form.addRow("LongCat API Key:", self.settings_longcat_key)

        self.settings_anthropic_key = QLineEdit()
        self.settings_anthropic_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.settings_anthropic_key.setPlaceholderText("sk-ant-...")
        form.addRow("Anthropic API Key:", self.settings_anthropic_key)

        self.settings_s2_key = QLineEdit()
        self.settings_s2_key.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Semantic Scholar Key:", self.settings_s2_key)

        self.settings_pubmed_key = QLineEdit()
        self.settings_pubmed_key.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("PubMed API Key:", self.settings_pubmed_key)

        self.settings_pubmed_email = QLineEdit()
        self.settings_pubmed_email.setPlaceholderText("user@example.com")
        form.addRow("PubMed Email:", self.settings_pubmed_email)

        self.settings_elicit_key = QLineEdit()
        self.settings_elicit_key.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Elicit API Key:", self.settings_elicit_key)

        self.settings_scopus_key = QLineEdit()
        self.settings_scopus_key.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Scopus API Key:", self.settings_scopus_key)

        self.settings_scopus_insttoken = QLineEdit()
        self.settings_scopus_insttoken.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Scopus Inst Token:", self.settings_scopus_insttoken)

        self.settings_wos_key = QLineEdit()
        self.settings_wos_key.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Web of Science Key:", self.settings_wos_key)
        self.settings_wos_researcher_key = QLineEdit()
        self.settings_wos_researcher_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.settings_wos_researcher_key.setToolTip("Clarivate WoS Researcher API key (X-ApiKey header)")
        form.addRow("WoS Researcher Key:", self.settings_wos_researcher_key)

        # Opt-in toggle: only persist keys to QSettings when explicitly requested
        self.settings_remember_keys = QCheckBox(
            "Remember API keys between sessions (stored in plaintext by QSettings)"
        )
        self.settings_remember_keys.setToolTip(
            "When checked, API keys are saved to the OS application settings store "
            "(registry on Windows, plist on macOS, ini file on Linux) in plaintext. "
            "Uncheck on shared or multi-user machines. "
            "If these fields are pre-filled from configuration or environment variables, "
            "checking this option may also save those displayed values to QSettings."
        )
        form.addRow("", self.settings_remember_keys)

        return grp

    def _build_llm_group(self) -> QGroupBox:
        grp = QGroupBox("LLM Provider Settings")
        form = QFormLayout(grp)

        self.settings_provider = QComboBox()
        self.settings_provider.addItems(["ollama", "lmstudio", "openai", "longcat", "anthropic"])
        self.settings_provider.currentTextChanged.connect(self._on_provider_changed)
        form.addRow("Provider:", self.settings_provider)

        self.settings_ollama_model = QLineEdit()
        self.settings_ollama_model.setPlaceholderText("qwen2.5:32b-instruct-q4_K_M")
        form.addRow("Ollama Model:", self.settings_ollama_model)

        self.settings_ollama_url = QLineEdit()
        self.settings_ollama_url.setPlaceholderText("http://localhost:11434")
        form.addRow("Ollama Base URL:", self.settings_ollama_url)

        self.settings_lmstudio_model = QLineEdit()
        self.settings_lmstudio_model.setPlaceholderText("openai/gpt-oss-20b")
        form.addRow("LM Studio Model:", self.settings_lmstudio_model)

        self.settings_lmstudio_url = QLineEdit()
        self.settings_lmstudio_url.setPlaceholderText("http://localhost:1234")
        form.addRow("LM Studio Base URL:", self.settings_lmstudio_url)

        self.settings_openai_model = QLineEdit()
        self.settings_openai_model.setPlaceholderText("gpt-4o-mini")
        form.addRow("OpenAI Model:", self.settings_openai_model)

        self.settings_openai_url = QLineEdit()
        self.settings_openai_url.setPlaceholderText("https://api.openai.com/v1")
        form.addRow("OpenAI Base URL:", self.settings_openai_url)

        self.settings_longcat_model = QLineEdit()
        self.settings_longcat_model.setPlaceholderText("LongCat-2.0-Preview")
        form.addRow("LongCat Model:", self.settings_longcat_model)

        self.settings_longcat_url = QLineEdit()
        self.settings_longcat_url.setPlaceholderText("https://api.longcat.chat/openai/v1")
        form.addRow("LongCat Base URL:", self.settings_longcat_url)

        self.settings_anthropic_model = QLineEdit()
        self.settings_anthropic_model.setPlaceholderText("claude-haiku-4-5-20251001")
        form.addRow("Anthropic Model:", self.settings_anthropic_model)

        return grp

    def _build_processing_group(self) -> QGroupBox:
        grp = QGroupBox("Processing Parameters")
        form = QFormLayout(grp)

        self.settings_threshold = QSpinBox()
        self.settings_threshold.setRange(1, 10)
        self.settings_threshold.setValue(3)
        form.addRow("Relevance Threshold:", self.settings_threshold)

        self.settings_workers = QSpinBox()
        self.settings_workers.setRange(1, 32)
        self.settings_workers.setValue(3)
        form.addRow("Concurrent Workers:", self.settings_workers)

        self.settings_batch_size = QSpinBox()
        self.settings_batch_size.setRange(1, 50)
        self.settings_batch_size.setValue(1)
        form.addRow("Score Batch Size:", self.settings_batch_size)

        self.settings_score_max_tokens = QSpinBox()
        self.settings_score_max_tokens.setRange(256, 65536)
        self.settings_score_max_tokens.setSingleStep(512)
        self.settings_score_max_tokens.setValue(4096)
        form.addRow("Score Max Tokens:", self.settings_score_max_tokens)

        self.settings_stagger_delay = QDoubleSpinBox()
        self.settings_stagger_delay.setRange(0.0, 30.0)
        self.settings_stagger_delay.setSingleStep(0.5)
        self.settings_stagger_delay.setValue(2.0)
        form.addRow("Stagger Delay (s):", self.settings_stagger_delay)

        self.settings_timeout = QDoubleSpinBox()
        self.settings_timeout.setRange(10.0, 600.0)
        self.settings_timeout.setSingleStep(10.0)
        self.settings_timeout.setValue(120.0)
        form.addRow("Worker Timeout (s):", self.settings_timeout)

        # Stage 2 advanced-LLM toggle. Default: off, which locks Stage 2 to
        # the bespoke HouseCatBERT relevance model and hides the provider /
        # model dropdowns. Enable to expose Ollama / LM Studio / OpenAI /
        # Anthropic options for relevance scoring.
        self.settings_show_score_advanced = QCheckBox(
            "Show advanced LLM options for Stage 2 (relevance scoring)"
        )
        self.settings_show_score_advanced.setToolTip(
            "When unchecked (default), Stage 2 always uses the bespoke "
            "HouseCatBERT-SciBERT model. Check this to expose the LLM "
            "provider / model dropdowns and override the default."
        )
        form.addRow("", self.settings_show_score_advanced)

        return grp

    # ── Populate from config ───────────────────────────────────────────

    def _populate_from_config(self, cfg: Config):
        """Fill all dialog widgets from the given Config object.

        Priority for API keys / LLM settings:
        1. Value on the Config object (from loaded file or env var)
        2. Value persisted in QSettings (from a previous Apply)
        3. Empty string
        """
        s = QSettings("Systes", "HypothesisConsensusAnalyzer")

        # Appearance
        self.settings_font_size.setValue(
            s.value("appearance/font_size", 10, type=int)
        )

        self.settings_project_root.setText(str(cfg.project_root))

        # Helper: use config value if non-empty, else fall back to QSettings
        def _best(cfg_val: str, qsettings_key: str) -> str:
            if cfg_val and cfg_val.strip():
                return cfg_val.strip()
            saved = s.value(qsettings_key, "", type=str)
            return saved.strip() if saved else ""

        # API keys  (config → QSettings fallback)
        self.settings_openai_key.setText(_best(cfg.openai_api_key, "keys/openai_api_key"))
        self.settings_longcat_key.setText(_best(cfg.longcat_api_key, "keys/longcat_api_key"))
        self.settings_anthropic_key.setText(_best(cfg.anthropic_api_key, "keys/anthropic_api_key"))
        self.settings_s2_key.setText(_best(cfg.s2_api_key, "keys/s2_api_key"))
        self.settings_pubmed_key.setText(_best(cfg.pubmed_api_key, "keys/pubmed_api_key"))
        self.settings_pubmed_email.setText(_best(cfg.pubmed_email, "keys/pubmed_email"))
        self.settings_elicit_key.setText(_best(cfg.elicit_api_key, "keys/elicit_api_key"))
        self.settings_scopus_key.setText(_best(cfg.scopus_api_key, "keys/scopus_api_key"))
        self.settings_scopus_insttoken.setText(_best(cfg.scopus_insttoken, "keys/scopus_insttoken"))
        self.settings_wos_key.setText(_best(cfg.wos_api_key, "keys/wos_api_key"))
        self.settings_wos_researcher_key.setText(_best(getattr(cfg, "wos_researcher_api_key", ""), "keys/wos_researcher_api_key"))

        # Restore remember-keys preference (default: off for safety)
        remember = s.value("keys/remember_keys", False, type=bool)
        self.settings_remember_keys.setChecked(remember)

        # LLM provider  (config → QSettings fallback)
        provider = _best(cfg.llm_provider, "llm/provider") or "ollama"
        idx = self.settings_provider.findText(provider)
        if idx >= 0:
            self.settings_provider.setCurrentIndex(idx)

        self.settings_ollama_model.setText(_best(cfg.ollama_model, "llm/ollama_model"))
        self.settings_ollama_url.setText(_best(cfg.ollama_base_url, "llm/ollama_url"))
        self.settings_lmstudio_model.setText(_best(cfg.lmstudio_model, "llm/lmstudio_model"))
        self.settings_lmstudio_url.setText(_best(cfg.lmstudio_url, "llm/lmstudio_url") if hasattr(cfg, 'lmstudio_url') else _best(cfg.lmstudio_base_url, "llm/lmstudio_url"))
        self.settings_openai_model.setText(_best(cfg.openai_model, "llm/openai_model"))
        self.settings_openai_url.setText(_best(cfg.openai_base_url, "llm/openai_url"))
        self.settings_longcat_model.setText(_best(cfg.longcat_model, "llm/longcat_model"))
        self.settings_longcat_url.setText(_best(cfg.longcat_base_url, "llm/longcat_url"))
        self.settings_anthropic_model.setText(_best(cfg.anthropic_model, "llm/anthropic_model"))

        # Processing
        self.settings_threshold.setValue(cfg.relevance_threshold)
        self.settings_workers.setValue(cfg.concurrent_workers)
        self.settings_batch_size.setValue(cfg.score_batch_size)
        self.settings_score_max_tokens.setValue(cfg.score_max_tokens)
        self.settings_stagger_delay.setValue(cfg.stagger_delay)
        self.settings_timeout.setValue(cfg.worker_timeout_seconds)

        # Stage 2 advanced-LLM toggle (QSettings only — not part of Config)
        self.settings_show_score_advanced.setChecked(
            s.value("score/show_advanced_llm", False, type=bool)
        )

    # ── Write back to config ───────────────────────────────────────────

    def _apply_to_config(self, cfg: Config):
        """Write dialog widget values back into a Config object."""
        root = self.settings_project_root.text().strip()
        if root:
            cfg.project_root = Path(root)
            cfg.__post_init__()  # recalculate derived paths

        # API keys
        cfg.openai_api_key = self.settings_openai_key.text().strip()
        cfg.longcat_api_key = self.settings_longcat_key.text().strip()
        cfg.anthropic_api_key = self.settings_anthropic_key.text().strip()
        cfg.s2_api_key = self.settings_s2_key.text().strip()
        cfg.pubmed_api_key = self.settings_pubmed_key.text().strip()
        cfg.pubmed_email = self.settings_pubmed_email.text().strip()
        cfg.elicit_api_key = self.settings_elicit_key.text().strip()
        cfg.scopus_api_key = self.settings_scopus_key.text().strip()
        cfg.scopus_insttoken = self.settings_scopus_insttoken.text().strip()
        cfg.wos_api_key = self.settings_wos_key.text().strip()
        if hasattr(cfg, "wos_researcher_api_key"):
            cfg.wos_researcher_api_key = self.settings_wos_researcher_key.text().strip()

        # LLM provider
        cfg.llm_provider = self.settings_provider.currentText()
        cfg.ollama_model = self.settings_ollama_model.text().strip()
        cfg.ollama_base_url = self.settings_ollama_url.text().strip()
        cfg.lmstudio_model = self.settings_lmstudio_model.text().strip()
        cfg.lmstudio_base_url = self.settings_lmstudio_url.text().strip()
        cfg.openai_model = self.settings_openai_model.text().strip()
        cfg.openai_base_url = self.settings_openai_url.text().strip()
        cfg.longcat_model = self.settings_longcat_model.text().strip()
        cfg.longcat_base_url = self.settings_longcat_url.text().strip()
        cfg.anthropic_model = self.settings_anthropic_model.text().strip()

        # Processing
        cfg.relevance_threshold = self.settings_threshold.value()
        cfg.concurrent_workers = self.settings_workers.value()
        cfg.score_batch_size = self.settings_batch_size.value()
        cfg.score_max_tokens = self.settings_score_max_tokens.value()
        cfg.stagger_delay = self.settings_stagger_delay.value()
        cfg.worker_timeout_seconds = self.settings_timeout.value()

    # ── Button handlers ────────────────────────────────────────────────

    def _on_apply(self):
        """Apply current dialog values to MainWindow's pipeline_config."""
        self._apply_to_config(self.main_window.pipeline_config)

        # Persist API keys and LLM settings to QSettings so they survive restarts
        self._save_keys_to_qsettings()

        # Apply font size change live
        font_size = self.settings_font_size.value()
        settings = QSettings("Systes", "HypothesisConsensusAnalyzer")
        settings.setValue("appearance/font_size", font_size)

        # Persist Stage 2 advanced-LLM toggle
        show_advanced = self.settings_show_score_advanced.isChecked()
        settings.setValue("score/show_advanced_llm", show_advanced)

        from PySide6.QtGui import QFont
        from gui.theme import get_stylesheet
        app = QApplication.instance()
        if app:
            app.setStyleSheet(get_stylesheet(font_size))
            app.setFont(QFont("Tahoma", font_size))

        # Sync settings into all stage tab widgets
        if hasattr(self.main_window, '_sync_settings_to_tabs'):
            self.main_window._sync_settings_to_tabs()

        # Apply Stage 2 advanced-LLM visibility immediately
        if hasattr(self.main_window, '_apply_score_advanced_mode'):
            self.main_window._apply_score_advanced_mode(show_advanced)

        self.main_window._log("Settings applied.", level="INFO")
        if hasattr(self.main_window, 'statusBar'):
            self.main_window.statusBar().showMessage("Settings applied", 3000)

    def _save_keys_to_qsettings(self):
        """Persist API keys and LLM settings so they survive app restarts.

        API keys are only written to QSettings when the user has explicitly
        opted in via the "Remember API keys" checkbox.  LLM provider/model
        settings (non-secret) are always persisted.
        """
        s = QSettings("Systes", "HypothesisConsensusAnalyzer")
        remember = self.settings_remember_keys.isChecked()
        s.setValue("keys/remember_keys", remember)

        # Map QSettings key → widget accessor for all API key fields
        key_widgets = {
            "keys/openai_api_key":   self.settings_openai_key,
            "keys/longcat_api_key":  self.settings_longcat_key,
            "keys/anthropic_api_key": self.settings_anthropic_key,
            "keys/s2_api_key":       self.settings_s2_key,
            "keys/pubmed_api_key":   self.settings_pubmed_key,
            "keys/pubmed_email":     self.settings_pubmed_email,
            "keys/elicit_api_key":   self.settings_elicit_key,
            "keys/scopus_api_key":   self.settings_scopus_key,
            "keys/scopus_insttoken": self.settings_scopus_insttoken,
            "keys/wos_api_key":      self.settings_wos_key,
        }
        if remember:
            for qkey, widget in key_widgets.items():
                s.setValue(qkey, widget.text().strip())
        else:
            # Clear any previously stored keys when the user opts out
            for qkey in key_widgets:
                s.remove(qkey)

        s.setValue("llm/provider", self.settings_provider.currentText())
        s.setValue("llm/ollama_model", self.settings_ollama_model.text().strip())
        s.setValue("llm/ollama_url", self.settings_ollama_url.text().strip())
        s.setValue("llm/lmstudio_model", self.settings_lmstudio_model.text().strip())
        s.setValue("llm/lmstudio_url", self.settings_lmstudio_url.text().strip())
        s.setValue("llm/openai_model", self.settings_openai_model.text().strip())
        s.setValue("llm/openai_url", self.settings_openai_url.text().strip())
        s.setValue("llm/longcat_model", self.settings_longcat_model.text().strip())
        s.setValue("llm/longcat_url", self.settings_longcat_url.text().strip())
        s.setValue("llm/anthropic_model", self.settings_anthropic_model.text().strip())

    @staticmethod
    def _has_value(val) -> bool:
        """True if val is a non-empty, non-whitespace string (or non-default for other types)."""
        if isinstance(val, str):
            return bool(val.strip())
        if isinstance(val, list):
            return bool(val)
        return val is not None

    def _on_load_config(self):
        """Load a config file (YAML/JSON) and merge into current config.

        Rule: loaded values overwrite current values ONLY when the loaded
        value is non-empty.  Empty / whitespace-only values in the file
        do NOT erase what the user already has.
        """
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Configuration",
            str(self.main_window.pipeline_config.project_root),
            "Config files (*.yaml *.yml *.json);;All files (*)",
        )
        if not path:
            return
        try:
            path_obj = Path(path)
            suffix = path_obj.suffix.lower()
            cfg = self.main_window.pipeline_config

            if suffix == ".json":
                with open(path_obj) as f:
                    data = json.load(f)

                _RESEARCH_SECTIONS = {
                    "search", "relevance", "extraction", "synthesis",
                    "method_review", "audit",
                    "natural_language_question", "name", "description",
                    "query_families", "defaults", "cheap_triage", "full_text",
                    "axes", "overlap_rules", "tier_rules", "evidence_types",
                    "stopping", "seeds", "dive", "graph", "graph_analysis",
                    "llm_scoring", "committee_brief", "abct_tech_ai_infosheet",
                }
                for key, val in data.items():
                    if key in _RESEARCH_SECTIONS:
                        cfg._research_config[key] = val
                    elif key == "project_root" and self._has_value(val):
                        cfg.project_root = Path(val)
                        cfg.__post_init__()
                    elif hasattr(cfg, key) and self._has_value(val):
                        try:
                            expected_type = type(getattr(cfg, key))
                            setattr(cfg, key, expected_type(val))
                        except (ValueError, TypeError):
                            setattr(cfg, key, val)

            else:
                # YAML — load via Config.load_config, then merge non-empty fields
                loaded = Config.load_config(config_path=path,
                                            project_root=cfg.project_root)
                # Always take research config sections
                if loaded._research_config:
                    cfg._research_config = loaded._research_config
                # Merge other fields only if they differ from defaults
                # (i.e. the YAML actually set them)
                defaults = Config()
                for key, val in loaded.__dict__.items():
                    if key.startswith("_"):
                        continue
                    default_val = getattr(defaults, key, None)
                    if val != default_val and self._has_value(val) and hasattr(cfg, key):
                        setattr(cfg, key, val)

            # Now push saved Preferences into the loaded config. API keys keep
            # env/config priority, while LLM provider/model/base-url fields are
            # user preferences and override defaults such as OpenAI.
            s = QSettings("Systes", "HypothesisConsensusAnalyzer")
            from gui.settings_persistence import restore_saved_settings

            restore_saved_settings(cfg, s)

            self._populate_from_config(cfg)
            # Populate hypothesis components and sync settings to tabs
            if hasattr(self.main_window, '_populate_components_from_config'):
                self.main_window._populate_components_from_config()
            if hasattr(self.main_window, '_sync_settings_to_tabs'):
                self.main_window._sync_settings_to_tabs()
            self.main_window._log(f"Config loaded from {path}")
        except Exception as exc:
            logger.exception("Failed to load config")
            QMessageBox.warning(self, "Load Error", str(exc))

    def _on_save_config(self):
        """Save current dialog values to a JSON file."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration",
            str(self.main_window.pipeline_config.project_root / "pipeline_config.json"),
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            # Read current widget values directly
            data = {
                "project_root": self.settings_project_root.text().strip(),
                # LLM provider
                "llm_provider": self.settings_provider.currentText(),
                "ollama_model": self.settings_ollama_model.text().strip(),
                "ollama_base_url": self.settings_ollama_url.text().strip(),
                "lmstudio_model": self.settings_lmstudio_model.text().strip(),
                "lmstudio_base_url": self.settings_lmstudio_url.text().strip(),
                "openai_model": self.settings_openai_model.text().strip(),
                "openai_base_url": self.settings_openai_url.text().strip(),
                "longcat_model": self.settings_longcat_model.text().strip(),
                "longcat_base_url": self.settings_longcat_url.text().strip(),
                "anthropic_model": self.settings_anthropic_model.text().strip(),
                # API keys
                "openai_api_key": self.settings_openai_key.text().strip(),
                "longcat_api_key": self.settings_longcat_key.text().strip(),
                "anthropic_api_key": self.settings_anthropic_key.text().strip(),
                "s2_api_key": self.settings_s2_key.text().strip(),
                "pubmed_api_key": self.settings_pubmed_key.text().strip(),
                "pubmed_email": self.settings_pubmed_email.text().strip(),
                "elicit_api_key": self.settings_elicit_key.text().strip(),
                "scopus_api_key": self.settings_scopus_key.text().strip(),
                "scopus_insttoken": self.settings_scopus_insttoken.text().strip(),
                "wos_api_key": self.settings_wos_key.text().strip(),
                "wos_researcher_api_key": self.settings_wos_researcher_key.text().strip(),
                # Processing
                "relevance_threshold": self.settings_threshold.value(),
                "concurrent_workers": self.settings_workers.value(),
                "score_batch_size": self.settings_batch_size.value(),
                "score_max_tokens": self.settings_score_max_tokens.value(),
                "stagger_delay": self.settings_stagger_delay.value(),
                "worker_timeout_seconds": self.settings_timeout.value(),
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            self.main_window._log(f"Config saved to {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Save Error", str(exc))

    def _on_reset_config(self):
        """Reset all settings to defaults."""
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Reset all settings to defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            default_cfg = Config()
            self.settings_font_size.setValue(10)
            self._populate_from_config(default_cfg)
            self.main_window._log("Settings reset to defaults.")

    def _browse_project_root(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Project Root",
            self.settings_project_root.text() or str(Path.home()),
        )
        if path:
            self.settings_project_root.setText(path)

    def _on_provider_changed(self, provider: str):
        """Optionally highlight the relevant model/URL fields."""
        pass  # Could enable/disable fields based on provider selection


# ---------------------------------------------------------------------------
# HelpDialog — per-tab help with methodology context
# ---------------------------------------------------------------------------

class HelpDialog(QDialog):
    """Modal help dialog for a specific pipeline tab."""

    def __init__(self, tab_name: str, parent=None):
        super().__init__(parent)
        from gui.help_content import HELP_CONTENT

        self._tab_name = tab_name
        content = HELP_CONTENT.get(tab_name, {})

        self.setWindowTitle(f"Help \u2014 {content.get('title', tab_name)}")
        self.setMinimumSize(500, 550)
        self.resize(500, 550)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(6)

        # Scrollable content area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setSpacing(10)

        # Title
        icon = content.get("icon", "")
        title = content.get("title", tab_name)
        title_label = QLabel(f"{icon}  {title}")
        title_font = QFont("Tahoma", 12)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # Separator
        sep = QWidget()
        sep.setFixedHeight(2)
        sep.setStyleSheet(
            "border-top: 1px solid #808080; border-bottom: 1px solid #ffffff;"
        )
        layout.addWidget(sep)

        # What This Stage Does
        layout.addWidget(self._section_header("What This Stage Does"))
        what_label = QLabel(content.get("what", ""))
        what_label.setWordWrap(True)
        layout.addWidget(what_label)

        # Why This Matters
        layout.addWidget(self._section_header("Why This Matters"))
        why_label = QLabel(content.get("why", ""))
        why_label.setWordWrap(True)
        layout.addWidget(why_label)

        # Understanding the Outputs
        layout.addWidget(self._section_header("Understanding the Outputs"))
        out_label = QLabel(content.get("outputs", ""))
        out_label.setWordWrap(True)
        layout.addWidget(out_label)

        # Further Reading
        reading = content.get("reading", [])
        if reading:
            layout.addWidget(self._section_header("Further Reading"))
            for ref in reading:
                link_html = (
                    f'<a href="{ref["url"]}" '
                    f'style="color: #0000FF; text-decoration: underline;">'
                    f'{ref["text"]}</a>'
                )
                link_label = QLabel(link_html)
                link_label.setOpenExternalLinks(True)
                link_label.setWordWrap(True)
                layout.addWidget(link_label)

        layout.addStretch()
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        # Bottom row: checkbox + close button
        bottom = QHBoxLayout()
        self._auto_check = QCheckBox("Don't show automatically")
        settings = QSettings("Systes", "HypothesisConsensusAnalyzer")
        self._auto_check.setChecked(
            settings.value(f"help/{tab_name}/auto_show", False, type=bool)
        )
        bottom.addWidget(self._auto_check)
        bottom.addStretch()

        close_btn = QPushButton("Close")
        close_btn.setMinimumWidth(80)
        close_btn.clicked.connect(self._on_close)
        bottom.addWidget(close_btn)
        outer.addLayout(bottom)

    @staticmethod
    def _section_header(text: str) -> QLabel:
        label = QLabel(text)
        font = QFont("Tahoma", 9)
        font.setBold(True)
        label.setFont(font)
        label.setStyleSheet(f"color: {COLORS['text']}; margin-top: 4px;")
        return label

    def _on_close(self):
        settings = QSettings("Systes", "HypothesisConsensusAnalyzer")
        settings.setValue(
            f"help/{self._tab_name}/auto_show",
            self._auto_check.isChecked(),
        )
        self.accept()


# ---------------------------------------------------------------------------
# WelcomeDialog — first-run onboarding
# ---------------------------------------------------------------------------

class WelcomeDialog(QDialog):
    """Welcome dialog shown on first launch."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from gui.help_content import WELCOME_CONTENT

        self.setWindowTitle("Welcome to Systes")
        self.setMinimumSize(550, 600)
        self.resize(550, 600)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setSpacing(10)

        # Title
        title_label = QLabel(WELCOME_CONTENT["title"])
        title_font = QFont("Tahoma", 14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Separator
        sep = QWidget()
        sep.setFixedHeight(2)
        sep.setStyleSheet(
            "border-top: 1px solid #808080; border-bottom: 1px solid #ffffff;"
        )
        layout.addWidget(sep)

        # Overview
        overview = QLabel(WELCOME_CONTENT["overview"])
        overview.setWordWrap(True)
        layout.addWidget(overview)

        # Pipeline stages
        stages_header = QLabel("Pipeline Stages")
        hdr_font = QFont("Tahoma", 10)
        hdr_font.setBold(True)
        stages_header.setFont(hdr_font)
        layout.addWidget(stages_header)

        stages_html = "<ol style='margin-left: 16px;'>"
        for stage in WELCOME_CONTENT["pipeline_stages"]:
            stages_html += f"<li>{stage}</li>"
        stages_html += "</ol>"
        stages_browser = QTextBrowser()
        stages_browser.setHtml(stages_html)
        stages_browser.setOpenExternalLinks(False)
        stages_browser.setStyleSheet(
            f"background: {COLORS['surface']}; border: none; font-family: Tahoma; font-size: 8pt;"
        )
        stages_browser.setMaximumHeight(220)
        layout.addWidget(stages_browser)

        # Getting started
        gs_header = QLabel("Getting Started")
        gs_header.setFont(hdr_font)
        layout.addWidget(gs_header)

        for tip in WELCOME_CONTENT["getting_started"]:
            tip_label = QLabel(f"\u2022  {tip}")
            tip_label.setWordWrap(True)
            layout.addWidget(tip_label)

        layout.addStretch()
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        # Bottom row
        bottom = QHBoxLayout()
        self._dont_show = QCheckBox("Don't show again")
        bottom.addWidget(self._dont_show)
        bottom.addStretch()

        start_btn = QPushButton("Get Started")
        start_btn.setMinimumWidth(100)
        start_btn.clicked.connect(self._on_start)
        bottom.addWidget(start_btn)
        outer.addLayout(bottom)

    def _on_start(self):
        settings = QSettings("Systes", "HypothesisConsensusAnalyzer")
        if self._dont_show.isChecked():
            settings.setValue("welcome/shown", True)
        self.accept()
