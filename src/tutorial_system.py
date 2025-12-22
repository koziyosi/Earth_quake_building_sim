"""
Tutorial System Module.
Interactive tutorials and guided learning.
"""
import tkinter as tk
from tkinter import ttk
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass, field


@dataclass
class TutorialStep:
    """Single step in a tutorial."""
    title: str
    description: str
    target_widget: str = ""  # Widget name to highlight
    action_required: str = ""  # Action user must take
    completed: bool = False
    

@dataclass
class Tutorial:
    """Complete tutorial sequence."""
    name: str
    description: str
    steps: List[TutorialStep] = field(default_factory=list)
    current_step: int = 0
    
    @property
    def is_complete(self) -> bool:
        return self.current_step >= len(self.steps)
        
    @property
    def progress(self) -> float:
        return self.current_step / len(self.steps) if self.steps else 1.0


# ===== Predefined Tutorials =====

TUTORIALS = {
    'getting_started': Tutorial(
        name="はじめてのシミュレーション",
        description="基本的なシミュレーション実行方法を学びます",
        steps=[
            TutorialStep(
                title="1. モデルタイプの選択",
                description="「Model Type」から「3D Frame」を選択してください。\n"
                           "これにより3次元フレーム構造の解析が可能になります。"
            ),
            TutorialStep(
                title="2. 建物パラメータ設定",
                description="「Floors」で階数を設定してください。\n"
                           "例: 5階建ての建物を解析する場合は「5」を入力"
            ),
            TutorialStep(
                title="3. 地震動パラメータ",
                description="「Max Acc」で最大加速度を設定します。\n"
                           "単位はgal (1gal = 0.01 m/s²)です。\n"
                           "例: 500 gal は震度6弱相当"
            ),
            TutorialStep(
                title="4. シミュレーション実行",
                description="「Run Simulation」ボタンをクリックしてシミュレーションを開始します。\n"
                           "計算には数秒〜数十秒かかります。"
            ),
            TutorialStep(
                title="5. 結果の確認",
                description="シミュレーション完了後、グラフに結果が表示されます。\n"
                           "・上: 時刻歴変位\n"
                           "・下: 各階の最大応答"
            ),
        ]
    ),
    
    'layout_editor': Tutorial(
        name="レイアウトエディタの使い方",
        description="カスタム建物平面の作成方法を学びます",
        steps=[
            TutorialStep(
                title="1. レイアウトエディタを開く",
                description="「Custom Layout」モデルを選択し、\n"
                           "「Edit Layout」ボタンをクリックします。"
            ),
            TutorialStep(
                title="2. グリッド設定",
                description="右パネルでグリッド間隔を設定します。\n"
                           "X方向・Y方向の間隔をメートル単位で指定。"
            ),
            TutorialStep(
                title="3. 柱の配置",
                description="グリッド交点をクリックして柱を配置します。\n"
                           "配置済みの柱は再クリックで削除できます。"
            ),
            TutorialStep(
                title="4. 階の追加",
                description="「Add Floor」ボタンで階を追加します。\n"
                           "各階で異なる柱配置が可能です。"
            ),
            TutorialStep(
                title="5. レイアウトの保存",
                description="「Save」ボタンでレイアウトをJSON形式で保存できます。\n"
                           "後で「Load」で読み込めます。"
            ),
        ]
    ),
    
    'analysis_types': Tutorial(
        name="解析タイプの理解",
        description="様々な解析手法の違いを学びます",
        steps=[
            TutorialStep(
                title="時刻歴解析",
                description="地震動を時間的に追跡する解析です。\n"
                           "Newmark-β法で運動方程式を解きます。\n"
                           "最も詳細な応答が得られます。"
            ),
            TutorialStep(
                title="応答スペクトル法",
                description="各振動モードの最大応答を重ね合わせます。\n"
                           "計算は高速ですが、非線形挙動は扱えません。"
            ),
            TutorialStep(
                title="プッシュオーバー解析",
                description="静的な水平力を漸増させる解析です。\n"
                           "建物の耐力と変形性能を評価します。\n"
                           "「Capacity Curve」が得られます。"
            ),
            TutorialStep(
                title="等価線形化法",
                description="非線形システムを等価な線形系に置換します。\n"
                           "有効周期と有効減衰を反復計算します。"
            ),
        ]
    ),
    
    'add_dampers': Tutorial(
        name="制振装置の設置",
        description="ダンパーを設置して応答を低減する方法",
        steps=[
            TutorialStep(
                title="1. ダンパー設置を有効化",
                description="「Add Dampers」チェックボックスをオンにします。"
            ),
            TutorialStep(
                title="2. ダンパータイプの選択",
                description="オイルダンパー: 速度比例の減衰力\n"
                           "粘弾性ダンパー: 変位と速度両方に依存\n"
                           "摩擦ダンパー: 一定の減衰力"
            ),
            TutorialStep(
                title="3. 効果の確認",
                description="ダンパーあり/なしで解析を比較してください。\n"
                           "最大変位や加速度の低減効果がわかります。"
            ),
        ]
    ),
}


class TutorialManager:
    """
    Manages tutorial display and progression.
    """
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.current_tutorial: Optional[Tutorial] = None
        self.overlay = None
        self.step_window = None
        
    def start_tutorial(self, name: str):
        """Start a tutorial by name."""
        if name not in TUTORIALS:
            return False
            
        self.current_tutorial = TUTORIALS[name]
        self.current_tutorial.current_step = 0
        self._show_step()
        return True
        
    def next_step(self):
        """Advance to next step."""
        if self.current_tutorial is None:
            return
            
        self.current_tutorial.current_step += 1
        
        if self.current_tutorial.is_complete:
            self._show_completion()
        else:
            self._show_step()
            
    def previous_step(self):
        """Go back to previous step."""
        if self.current_tutorial is None:
            return
            
        self.current_tutorial.current_step = max(0, self.current_tutorial.current_step - 1)
        self._show_step()
        
    def skip_tutorial(self):
        """Skip current tutorial."""
        self._cleanup()
        self.current_tutorial = None
        
    def _show_step(self):
        """Display current step."""
        if self.current_tutorial is None:
            return
            
        step = self.current_tutorial.steps[self.current_tutorial.current_step]
        
        self._cleanup()
        
        # Create step window
        self.step_window = tk.Toplevel(self.root)
        self.step_window.title("チュートリアル")
        self.step_window.geometry("400x250")
        self.step_window.resizable(False, False)
        self.step_window.transient(self.root)
        
        # Progress bar
        progress_frame = ttk.Frame(self.step_window)
        progress_frame.pack(fill=tk.X, padx=10, pady=5)
        
        progress = ttk.Progressbar(
            progress_frame,
            value=self.current_tutorial.progress * 100,
            length=380
        )
        progress.pack()
        
        # Step info
        ttk.Label(
            self.step_window,
            text=step.title,
            font=('Yu Gothic', 12, 'bold')
        ).pack(pady=10)
        
        text = tk.Text(self.step_window, height=6, width=45, wrap=tk.WORD)
        text.pack(padx=10, pady=5)
        text.insert('1.0', step.description)
        text.config(state=tk.DISABLED)
        
        # Buttons
        btn_frame = ttk.Frame(self.step_window)
        btn_frame.pack(pady=10)
        
        if self.current_tutorial.current_step > 0:
            ttk.Button(btn_frame, text="← 前へ", command=self.previous_step).pack(side=tk.LEFT, padx=5)
            
        ttk.Button(btn_frame, text="スキップ", command=self.skip_tutorial).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="次へ →", command=self.next_step).pack(side=tk.LEFT, padx=5)
        
    def _show_completion(self):
        """Show tutorial completion."""
        self._cleanup()
        
        self.step_window = tk.Toplevel(self.root)
        self.step_window.title("チュートリアル完了")
        self.step_window.geometry("300x150")
        self.step_window.transient(self.root)
        
        ttk.Label(
            self.step_window,
            text="🎉 チュートリアル完了！",
            font=('Yu Gothic', 14, 'bold')
        ).pack(pady=20)
        
        ttk.Label(
            self.step_window,
            text=f"「{self.current_tutorial.name}」を完了しました。"
        ).pack()
        
        ttk.Button(
            self.step_window,
            text="閉じる",
            command=self._cleanup
        ).pack(pady=20)
        
    def _cleanup(self):
        """Clean up tutorial windows."""
        if self.step_window:
            self.step_window.destroy()
            self.step_window = None
        if self.overlay:
            self.overlay.destroy()
            self.overlay = None


class TutorialMenuBuilder:
    """
    Builds tutorial menu for the application.
    """
    
    @staticmethod
    def create_menu(menubar: tk.Menu, manager: TutorialManager):
        """Add tutorial menu to menubar."""
        tutorial_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="チュートリアル", menu=tutorial_menu)
        
        for name, tutorial in TUTORIALS.items():
            tutorial_menu.add_command(
                label=tutorial.name,
                command=lambda n=name: manager.start_tutorial(n)
            )
            
        tutorial_menu.add_separator()
        tutorial_menu.add_command(label="すべてリセット", command=lambda: None)
        
        return tutorial_menu


def get_tutorial_names() -> List[str]:
    """Get list of available tutorial names."""
    return list(TUTORIALS.keys())


def get_tutorial_descriptions() -> Dict[str, str]:
    """Get tutorial descriptions."""
    return {name: t.description for name, t in TUTORIALS.items()}
