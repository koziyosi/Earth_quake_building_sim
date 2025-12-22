"""
Keyboard Shortcuts and Hotkeys Module.
Implements keyboard-driven interface improvements.
"""
import tkinter as tk
from typing import Dict, Callable, Optional
from dataclasses import dataclass


@dataclass
class Shortcut:
    """Keyboard shortcut definition."""
    key: str
    modifier: str  # 'Ctrl', 'Alt', 'Shift', 'Ctrl+Shift', ''
    description: str
    callback: Callable


class KeyboardManager:
    """
    Manages keyboard shortcuts for tkinter application.
    """
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.shortcuts: Dict[str, Shortcut] = {}
        self.enabled = True
        
        # Default shortcuts
        self._register_defaults()
        
    def _register_defaults(self):
        """Register default shortcuts."""
        defaults = [
            ('n', 'Ctrl', 'New Project', 'new_project'),
            ('o', 'Ctrl', 'Open Project', 'open_project'),
            ('s', 'Ctrl', 'Save Project', 'save_project'),
            ('s', 'Ctrl+Shift', 'Save As', 'save_as'),
            ('z', 'Ctrl', 'Undo', 'undo'),
            ('y', 'Ctrl', 'Redo', 'redo'),
            ('r', 'Ctrl', 'Run Simulation', 'run_sim'),
            ('Escape', '', 'Cancel/Stop', 'cancel'),
            ('F5', '', 'Run Simulation', 'run_sim'),
            ('F1', '', 'Help', 'show_help'),
            ('space', '', 'Play/Pause Animation', 'toggle_play'),
            ('Left', '', 'Previous Frame', 'prev_frame'),
            ('Right', '', 'Next Frame', 'next_frame'),
            ('Home', '', 'First Frame', 'first_frame'),
            ('End', '', 'Last Frame', 'last_frame'),
            ('plus', '', 'Zoom In', 'zoom_in'),
            ('minus', '', 'Zoom Out', 'zoom_out'),
            ('0', 'Ctrl', 'Reset View', 'reset_view'),
            ('g', 'Ctrl', 'Toggle Grid', 'toggle_grid'),
            ('Delete', '', 'Delete Selected', 'delete_selected'),
            ('a', 'Ctrl', 'Select All', 'select_all'),
            ('c', 'Ctrl', 'Copy', 'copy'),
            ('v', 'Ctrl', 'Paste', 'paste'),
            ('d', 'Ctrl', 'Duplicate', 'duplicate'),
        ]
        
        for key, mod, desc, action in defaults:
            self.register(key, mod, desc, action)
            
    def register(
        self,
        key: str,
        modifier: str,
        description: str,
        action: str
    ):
        """Register a keyboard shortcut."""
        binding = self._make_binding(key, modifier)
        shortcut = Shortcut(key, modifier, description, action)
        self.shortcuts[binding] = shortcut
        
    def _make_binding(self, key: str, modifier: str) -> str:
        """Convert to tkinter binding string."""
        parts = []
        
        if 'Ctrl' in modifier:
            parts.append('Control')
        if 'Alt' in modifier:
            parts.append('Alt')
        if 'Shift' in modifier:
            parts.append('Shift')
            
        parts.append(key)
        
        return '<' + '-'.join(parts) + '>'
        
    def bind_all(self, callbacks: Dict[str, Callable]):
        """
        Bind all shortcuts to callbacks.
        
        Args:
            callbacks: Dict mapping action names to callback functions
        """
        for binding, shortcut in self.shortcuts.items():
            if isinstance(shortcut.callback, str):
                action = shortcut.callback
                if action in callbacks:
                    self.root.bind(binding, 
                        lambda e, cb=callbacks[action]: cb() if self.enabled else None)
                    
    def enable(self):
        self.enabled = True
        
    def disable(self):
        self.enabled = False
        
    def get_shortcut_list(self) -> str:
        """Get formatted shortcut list for help dialog."""
        lines = ["Keyboard Shortcuts", "=" * 40, ""]
        
        categories = {
            'File': ['new_project', 'open_project', 'save_project', 'save_as'],
            'Edit': ['undo', 'redo', 'copy', 'paste', 'duplicate', 'delete_selected', 'select_all'],
            'Simulation': ['run_sim', 'cancel'],
            'View': ['zoom_in', 'zoom_out', 'reset_view', 'toggle_grid'],
            'Animation': ['toggle_play', 'prev_frame', 'next_frame', 'first_frame', 'last_frame']
        }
        
        for category, actions in categories.items():
            lines.append(f"\n{category}:")
            lines.append("-" * 20)
            
            for binding, shortcut in self.shortcuts.items():
                action = shortcut.callback if isinstance(shortcut.callback, str) else ''
                if action in actions:
                    key_str = shortcut.modifier + '+' + shortcut.key if shortcut.modifier else shortcut.key
                    lines.append(f"  {key_str:20} {shortcut.description}")
                    
        return '\n'.join(lines)


class ShortcutDialog(tk.Toplevel):
    """Dialog showing all keyboard shortcuts."""
    
    def __init__(self, parent, manager: KeyboardManager):
        super().__init__(parent)
        
        self.title("Keyboard Shortcuts")
        self.geometry("400x500")
        
        text = tk.Text(self, wrap=tk.WORD, font=('Consolas', 10))
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text.insert('1.0', manager.get_shortcut_list())
        text.config(state=tk.DISABLED)
        
        close_btn = tk.Button(self, text="Close", command=self.destroy)
        close_btn.pack(pady=10)


# ===== Tooltip System =====

class ToolTip:
    """Tooltip widget for adding help text."""
    
    def __init__(self, widget, text: str, delay: int = 500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tooltip_window = None
        self.id = None
        
        widget.bind('<Enter>', self._schedule)
        widget.bind('<Leave>', self._hide)
        widget.bind('<ButtonPress>', self._hide)
        
    def _schedule(self, event=None):
        self._hide()
        self.id = self.widget.after(self.delay, self._show)
        
    def _show(self):
        if self.tooltip_window:
            return
            
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                        background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                        font=('Helvetica', 9))
        label.pack()
        
    def _hide(self, event=None):
        if self.id:
            self.widget.after_cancel(self.id)
            self.id = None
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


def add_tooltips(widgets: Dict[tk.Widget, str]):
    """Add tooltips to multiple widgets."""
    for widget, text in widgets.items():
        ToolTip(widget, text)


# ===== Help System =====

GLOSSARY = {
    '層間変形角': 'Inter-story drift ratio. 各階の水平変位を階高で割った値。建物の損傷評価に使用。',
    '塑性率': 'Ductility factor (μ). 最大変位と降伏変位の比。μ=δmax/δy',
    '免震': 'Base isolation. 基礎と上部構造の間に免震装置を設置し、地震力を低減。',
    '制振': 'Damping. ダンパー等で振動エネルギーを吸収。',
    'オイルダンパー': 'Oil damper. 速度に比例した減衰力を発生。F=c*v',
    'Newmark-β法': '時刻歴解析の数値積分法。β=0.25, γ=0.5で等加速度法。',
    'Rayleigh減衰': '質量比例と剛性比例の和。C=αM+βK',
    'K-NET': '日本の強震観測網。防災科研が運営。',
    'プッシュオーバー': 'Static nonlinear analysis. 静的荷重を漸増させ耐力曲線を求める。',
    'Ds値': '構造特性係数。靱性に応じた低減係数。',
    'Ai分布': '層せん断力係数の高さ方向分布。',
    'ヒステリシス': 'Hysteresis. 荷重-変形の履歴挙動。エネルギー吸収を表す。',
    'Takeda': 'RC部材の履歴モデル。剛性劣化を考慮。',
    'Bilinear': '二折線型履歴。バイリニアモデル。',
    'P-Δ効果': '軸力と水平変位による付加曲げモーメント。',
}


def show_glossary_dialog(parent):
    """Show glossary dialog."""
    dialog = tk.Toplevel(parent)
    dialog.title("用語集 / Glossary")
    dialog.geometry("600x400")
    
    text = tk.Text(dialog, wrap=tk.WORD, font=('Yu Gothic', 10))
    text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    for term, definition in sorted(GLOSSARY.items()):
        text.insert(tk.END, f"【{term}】\n", 'term')
        text.insert(tk.END, f"  {definition}\n\n")
        
    text.tag_config('term', font=('Yu Gothic', 10, 'bold'))
    text.config(state=tk.DISABLED)
    
    scrollbar = tk.Scrollbar(text)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    text.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=text.yview)
