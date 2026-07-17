"""Thread-safe GUI logging handler using a Qt signal bridge.

This module exposes :class:`GUILogBridge` and :class:`GUIHandler` which
together route Python ``logging`` records from any worker thread into the
Qt GUI thread via a ``QObject`` signal.

Usage (in MainWindow.__init__ or equivalent)::

    from gui.log_handler import install_gui_log_handler

    self._log_bridge = install_gui_log_handler(self._on_log_record)

Then define a slot::

    def _on_log_record(self, message: str, level: str) -> None:
        self._log(message, level)

This replaces the previous ``QueueHandler`` stub, which was never wired up.
"""

import logging
from typing import Callable

from PySide6.QtCore import QObject, Signal


class GUILogBridge(QObject):
    """QObject that carries a Qt signal for crossing thread boundaries."""

    #: Emitted with (formatted_message, levelname) from any thread.
    log_record = Signal(str, str)


class GUIHandler(logging.Handler):
    """Logging handler that emits records via a :class:`GUILogBridge` signal.

    Because Qt signals are thread-safe, this handler can be installed on
    any logger and used from worker threads — records are safely delivered
    to the connected GUI slot in the main thread.
    """

    def __init__(self, bridge: GUILogBridge) -> None:
        super().__init__()
        self.bridge = bridge

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.bridge.log_record.emit(msg, record.levelname)
        except Exception:
            self.handleError(record)


def install_gui_log_handler(
    slot: Callable[[str, str], None],
    package_names: tuple[str, ...] = ("gui", "process", "core", "stages", "acquire"),
    level: int = logging.DEBUG,
) -> GUILogBridge:
    """Create a bridge + handler and wire them to a GUI slot.

    Parameters
    ----------
    slot:
        A callable ``(message: str, level: str) -> None`` — typically a
        ``MainWindow`` method decorated with ``@Slot(str, str)``.
    package_names:
        Logger names to attach the handler to.
    level:
        Minimum log level to forward.

    Returns
    -------
    GUILogBridge
        Keep a reference to this in the parent object to prevent GC.
    """
    bridge = GUILogBridge()
    bridge.log_record.connect(slot)

    handler = GUIHandler(bridge)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(level)

    for name in package_names:
        log = logging.getLogger(name)
        log.addHandler(handler)
        log.setLevel(level)

    return bridge
