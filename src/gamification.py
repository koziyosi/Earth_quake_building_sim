"""
Gamification Module.
Implements educational interactive mode (#63).
"""
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import random


class DifficultyLevel(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3
    EXPERT = 4


@dataclass
class Challenge:
    """A single educational challenge."""
    id: str
    title: str
    description: str
    difficulty: DifficultyLevel
    target_drift: float      # Max acceptable drift
    target_cost: float       # Budget limit (arbitrary units)
    earthquake_level: float  # PGA in gal
    hints: List[str]
    success_message: str


@dataclass
class PlayerProgress:
    """Player progress tracking."""
    level: int = 1
    xp: int = 0
    challenges_completed: int = 0
    total_score: int = 0
    achievements: List[str] = None
    
    def __post_init__(self):
        if self.achievements is None:
            self.achievements = []


# Predefined challenges
CHALLENGES = [
    Challenge(
        id="intro_1",
        title="基礎を学ぼう",
        description="3階建ての建物を設計し、地震に耐えられるようにしましょう。層間変形角0.01以下を目指してください。",
        difficulty=DifficultyLevel.EASY,
        target_drift=0.01,
        target_cost=1000,
        earthquake_level=200,
        hints=[
            "柱を太くすると剛性が上がります",
            "1階の剛性が低いとsoft storyになります",
            "梁も重要な構造要素です"
        ],
        success_message="おめでとう！基本的な耐震設計を習得しました！"
    ),
    Challenge(
        id="soft_story",
        title="ピロティの罠",
        description="1階がピロティ（柔らかい層）の建物を安全にしましょう。",
        difficulty=DifficultyLevel.MEDIUM,
        target_drift=0.008,
        target_cost=1500,
        earthquake_level=300,
        hints=[
            "1階に壁やブレースを追加できます",
            "免震装置を検討してみましょう",
            "オイルダンパーも効果的です"
        ],
        success_message="素晴らしい！軟弱層対策をマスターしました！"
    ),
    Challenge(
        id="big_one",
        title="巨大地震に備えよ",
        description="M8クラスの巨大地震に耐える建物を設計してください。",
        difficulty=DifficultyLevel.HARD,
        target_drift=0.015,
        target_cost=3000,
        earthquake_level=600,
        hints=[
            "免震構造を検討しましょう",
            "制振装置を適切に配置してください",
            "建物の形状も重要です"
        ],
        success_message="驚異的！巨大地震対策の専門家になりました！"
    ),
    Challenge(
        id="budget_hero",
        title="限られた予算で",
        description="低予算で安全な建物を設計するチャレンジです。",
        difficulty=DifficultyLevel.EXPERT,
        target_drift=0.012,
        target_cost=800,
        earthquake_level=400,
        hints=[
            "どこにお金をかけるか戦略的に考えましょう",
            "高価な設備は本当に必要？",
            "シンプルな構造も強いことがあります"
        ],
        success_message="天才！コストパフォーマンスの達人です！"
    )
]

# Achievement definitions
ACHIEVEMENTS = {
    "first_design": ("初めての設計", "最初の建物を設計した", 10),
    "earthquake_survivor": ("地震サバイバー", "地震シミュレーションに成功した", 20),
    "perfect_score": ("パーフェクト", "目標を完全に達成した", 50),
    "budget_master": ("節約の達人", "予算内で設計を完了した", 30),
    "isolation_expert": ("免震マスター", "免震構造を使いこなした", 40),
    "damper_king": ("制振の王", "複数のダンパーを効果的に配置した", 40),
    "speed_runner": ("スピードランナー", "3分以内にクリアした", 25),
    "no_hints": ("自力クリア", "ヒントなしでクリアした", 35),
}


class GamificationManager:
    """
    Manages gamification features.
    """
    
    def __init__(self, save_dir: str = None):
        self.progress = PlayerProgress()
        self.current_challenge: Optional[Challenge] = None
        self.hints_used: int = 0
        self.start_time: float = 0
        
        self.save_dir = save_dir
        
    def start_challenge(self, challenge_id: str):
        """Start a challenge."""
        import time
        
        for c in CHALLENGES:
            if c.id == challenge_id:
                self.current_challenge = c
                self.hints_used = 0
                self.start_time = time.time()
                return c
        return None
        
    def get_hint(self) -> Optional[str]:
        """Get next hint for current challenge."""
        if not self.current_challenge:
            return None
            
        if self.hints_used < len(self.current_challenge.hints):
            hint = self.current_challenge.hints[self.hints_used]
            self.hints_used += 1
            return hint
        return "もうヒントはありません"
        
    def evaluate_result(
        self,
        max_drift: float,
        total_cost: float
    ) -> Dict:
        """
        Evaluate simulation result against challenge.
        
        Returns:
            Dict with success, score, new_achievements
        """
        import time
        
        if not self.current_challenge:
            return {'success': False, 'score': 0, 'new_achievements': []}
            
        c = self.current_challenge
        elapsed = time.time() - self.start_time
        
        # Check success conditions
        drift_ok = max_drift <= c.target_drift
        cost_ok = total_cost <= c.target_cost
        
        success = drift_ok and cost_ok
        
        # Calculate score
        score = 0
        new_achievements = []
        
        if success:
            # Base score
            score = 100 * c.difficulty.value
            
            # Bonus for being under targets
            drift_margin = (c.target_drift - max_drift) / c.target_drift
            score += int(drift_margin * 50)
            
            cost_margin = (c.target_cost - total_cost) / c.target_cost
            score += int(cost_margin * 30)
            
            # Hint penalty
            score -= self.hints_used * 5
            
            # Time bonus
            if elapsed < 180:  # 3 minutes
                score += 20
                if "speed_runner" not in self.progress.achievements:
                    new_achievements.append("speed_runner")
                    self.progress.achievements.append("speed_runner")
                    
            # Achievement checks
            if self.hints_used == 0 and "no_hints" not in self.progress.achievements:
                new_achievements.append("no_hints")
                self.progress.achievements.append("no_hints")
                
            if cost_margin > 0.2 and "budget_master" not in self.progress.achievements:
                new_achievements.append("budget_master")
                self.progress.achievements.append("budget_master")
                
            if drift_margin > 0.3 and "perfect_score" not in self.progress.achievements:
                new_achievements.append("perfect_score")
                self.progress.achievements.append("perfect_score")
                
            # Update progress
            self.progress.xp += score
            self.progress.challenges_completed += 1
            self.progress.total_score += score
            
            # Level up check
            xp_per_level = 500
            new_level = 1 + self.progress.xp // xp_per_level
            if new_level > self.progress.level:
                self.progress.level = new_level
                
        return {
            'success': success,
            'score': score,
            'new_achievements': new_achievements,
            'drift_ok': drift_ok,
            'cost_ok': cost_ok,
            'elapsed_time': elapsed,
            'message': c.success_message if success else "もう一度挑戦してみましょう！"
        }
        
    def get_available_challenges(self) -> List[Challenge]:
        """Get challenges available at current level."""
        available = []
        for c in CHALLENGES:
            # Unlock based on level
            if c.difficulty.value <= self.progress.level:
                available.append(c)
        return available
        
    def get_leaderboard_position(self, score: int) -> int:
        """Get position on leaderboard (mock)."""
        # Would connect to actual leaderboard
        return random.randint(1, 1000)


class GamificationPanel(ttk.Frame):
    """
    UI panel for gamification features.
    """
    
    def __init__(
        self, 
        parent,
        manager: GamificationManager,
        on_start_challenge: Callable
    ):
        super().__init__(parent)
        
        self.manager = manager
        self.on_start_challenge = on_start_challenge
        
        self._setup_ui()
        
    def _setup_ui(self):
        # Header
        header = ttk.Frame(self)
        header.pack(fill=tk.X, pady=10)
        
        ttk.Label(header, text="🎮 チャレンジモード", font=('', 14, 'bold')).pack(side=tk.LEFT)
        
        # Player info
        info_frame = ttk.LabelFrame(self, text="プレイヤー情報")
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.level_label = ttk.Label(info_frame, text=f"レベル: {self.manager.progress.level}")
        self.level_label.pack(anchor=tk.W)
        
        self.xp_label = ttk.Label(info_frame, text=f"経験値: {self.manager.progress.xp}")
        self.xp_label.pack(anchor=tk.W)
        
        # XP progress bar
        self.xp_bar = ttk.Progressbar(info_frame, length=200, mode='determinate')
        self.xp_bar.pack(fill=tk.X, padx=5, pady=5)
        self._update_xp_bar()
        
        # Challenge list
        challenges_frame = ttk.LabelFrame(self, text="チャレンジ一覧")
        challenges_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.challenge_list = ttk.Treeview(
            challenges_frame,
            columns=('difficulty', 'status'),
            height=6
        )
        self.challenge_list.pack(fill=tk.BOTH, expand=True)
        
        self.challenge_list.heading('#0', text='チャレンジ名')
        self.challenge_list.heading('difficulty', text='難易度')
        self.challenge_list.heading('status', text='状態')
        
        self.challenge_list.column('#0', width=150)
        self.challenge_list.column('difficulty', width=60)
        self.challenge_list.column('status', width=60)
        
        self._populate_challenges()
        
        # Start button
        self.start_btn = ttk.Button(
            self, 
            text="チャレンジ開始",
            command=self._start_selected
        )
        self.start_btn.pack(pady=10)
        
        # Achievements
        achiev_frame = ttk.LabelFrame(self, text="実績")
        achiev_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.achiev_label = ttk.Label(
            achiev_frame, 
            text=f"獲得: {len(self.manager.progress.achievements)} / {len(ACHIEVEMENTS)}"
        )
        self.achiev_label.pack()
        
    def _update_xp_bar(self):
        xp_per_level = 500
        current_level_xp = self.manager.progress.xp % xp_per_level
        self.xp_bar['value'] = (current_level_xp / xp_per_level) * 100
        
    def _populate_challenges(self):
        difficulty_names = {
            DifficultyLevel.EASY: "★☆☆☆",
            DifficultyLevel.MEDIUM: "★★☆☆",
            DifficultyLevel.HARD: "★★★☆",
            DifficultyLevel.EXPERT: "★★★★"
        }
        
        for c in self.manager.get_available_challenges():
            self.challenge_list.insert(
                '', 'end',
                text=c.title,
                values=(difficulty_names[c.difficulty], '未クリア'),
                tags=(c.id,)
            )
            
    def _start_selected(self):
        selection = self.challenge_list.selection()
        if selection:
            tags = self.challenge_list.item(selection[0], 'tags')
            if tags:
                challenge_id = tags[0]
                self.manager.start_challenge(challenge_id)
                self.on_start_challenge(challenge_id)
                
    def refresh_ui(self):
        """Refresh UI after game state change."""
        self.level_label.config(text=f"レベル: {self.manager.progress.level}")
        self.xp_label.config(text=f"経験値: {self.manager.progress.xp}")
        self._update_xp_bar()
        self.achiev_label.config(
            text=f"獲得: {len(self.manager.progress.achievements)} / {len(ACHIEVEMENTS)}"
        )
