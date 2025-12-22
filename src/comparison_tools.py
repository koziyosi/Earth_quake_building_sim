"""
Comparison Tools Module.
Compare multiple analysis results side by side.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import matplotlib.pyplot as plt


@dataclass
class AnalysisCase:
    """Single analysis case for comparison."""
    name: str
    description: str = ""
    
    # Results
    max_displacement: float = 0.0
    max_drift: float = 0.0
    max_acceleration: float = 0.0
    base_shear: float = 0.0
    period: float = 0.0
    
    # Time histories
    time: np.ndarray = None
    displacement_history: np.ndarray = None
    drift_history: np.ndarray = None
    
    # Additional data
    parameters: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    
    @property
    def key_metrics(self) -> Dict:
        return {
            'Max Disp (m)': self.max_displacement,
            'Max Drift': f"1/{int(1/self.max_drift) if self.max_drift > 0 else 'N/A'}",
            'Max Accel (g)': self.max_acceleration / 9.81,
            'Base Shear (kN)': self.base_shear / 1000,
            'Period (s)': self.period
        }


class ComparisonAnalyzer:
    """
    Analyzes and compares multiple analysis cases.
    """
    
    def __init__(self):
        self.cases: List[AnalysisCase] = []
        
    def add_case(self, case: AnalysisCase):
        """Add analysis case."""
        self.cases.append(case)
        
    def remove_case(self, name: str):
        """Remove case by name."""
        self.cases = [c for c in self.cases if c.name != name]
        
    def clear(self):
        """Clear all cases."""
        self.cases = []
        
    def get_case(self, name: str) -> Optional[AnalysisCase]:
        """Get case by name."""
        for case in self.cases:
            if case.name == name:
                return case
        return None
        
    def compare_metrics(self) -> Dict[str, Dict]:
        """Compare key metrics across all cases."""
        if not self.cases:
            return {}
            
        metrics = {}
        for case in self.cases:
            metrics[case.name] = case.key_metrics
            
        return metrics
        
    def find_best_case(self, metric: str = 'max_displacement', minimize: bool = True) -> AnalysisCase:
        """Find best performing case."""
        if not self.cases:
            return None
            
        key = lambda c: getattr(c, metric, 0)
        
        if minimize:
            return min(self.cases, key=key)
        else:
            return max(self.cases, key=key)
            
    def calculate_reductions(self, baseline_name: str) -> Dict[str, Dict[str, float]]:
        """Calculate % reduction relative to baseline."""
        baseline = self.get_case(baseline_name)
        if not baseline:
            return {}
            
        reductions = {}
        
        for case in self.cases:
            if case.name == baseline_name:
                continue
                
            case_red = {}
            
            if baseline.max_displacement > 0:
                case_red['displacement'] = (1 - case.max_displacement / baseline.max_displacement) * 100
            if baseline.max_drift > 0:
                case_red['drift'] = (1 - case.max_drift / baseline.max_drift) * 100
            if baseline.max_acceleration > 0:
                case_red['acceleration'] = (1 - case.max_acceleration / baseline.max_acceleration) * 100
                
            reductions[case.name] = case_red
            
        return reductions


def plot_comparison_bar(
    analyzer: ComparisonAnalyzer,
    metric: str = 'max_displacement',
    ax = None
):
    """Bar chart comparing metric across cases."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        
    if not analyzer.cases:
        return ax
        
    names = [c.name for c in analyzer.cases]
    values = [getattr(c, metric, 0) for c in analyzer.cases]
    
    # Color best case differently
    min_idx = values.index(min(values))
    colors = ['#4dabf7'] * len(values)
    colors[min_idx] = '#69db7c'
    
    bars = ax.bar(names, values, color=colors)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
               f'{val:.4f}', ha='center', va='bottom', fontsize=9)
               
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Comparison: {metric.replace("_", " ").title()}')
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax


def plot_comparison_time_history(
    analyzer: ComparisonAnalyzer,
    data_type: str = 'displacement',
    story: int = None,
    ax = None
):
    """Overlay time histories from multiple cases."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
        
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(analyzer.cases)))
    
    for i, case in enumerate(analyzer.cases):
        if case.time is None:
            continue
            
        if data_type == 'displacement':
            if case.displacement_history is not None:
                if story is not None and len(case.displacement_history.shape) > 1:
                    data = case.displacement_history[:, story]
                else:
                    data = np.max(np.abs(case.displacement_history), axis=1) if len(case.displacement_history.shape) > 1 else case.displacement_history
            else:
                continue
        elif data_type == 'drift':
            if case.drift_history is not None:
                if story is not None and len(case.drift_history.shape) > 1:
                    data = case.drift_history[:, story]
                else:
                    data = np.max(case.drift_history, axis=1) if len(case.drift_history.shape) > 1 else case.drift_history
            else:
                continue
        else:
            continue
            
        ax.plot(case.time, data, color=colors[i], label=case.name, linewidth=1.5)
        
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(data_type.title())
    ax.set_title(f'{data_type.title()} Time History Comparison')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_envelope_comparison(
    analyzer: ComparisonAnalyzer,
    story_heights: List[float],
    ax = None
):
    """Compare drift envelopes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 10))
        
    colors = plt.cm.tab10(np.linspace(0, 1, len(analyzer.cases)))
    
    for i, case in enumerate(analyzer.cases):
        if case.drift_history is None:
            continue
            
        # Get max drift per story
        if len(case.drift_history.shape) > 1:
            max_drifts = np.max(np.abs(case.drift_history), axis=0)
        else:
            max_drifts = [np.max(np.abs(case.drift_history))]
            
        n_stories = len(max_drifts)
        stories = np.arange(1, n_stories + 1)
        
        ax.plot([d * 100 for d in max_drifts], stories, 'o-', 
               color=colors[i], label=case.name, linewidth=2, markersize=8)
               
    # Limits
    ax.axvline(0.5, color='green', linestyle='--', alpha=0.7, label='1/200')
    ax.axvline(1.0, color='orange', linestyle='--', alpha=0.7, label='1/100')
    ax.axvline(2.0, color='red', linestyle='--', alpha=0.7, label='1/50')
    
    ax.set_xlabel('Max Inter-Story Drift (%)')
    ax.set_ylabel('Story')
    ax.set_title('Drift Envelope Comparison')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    return ax


def generate_comparison_table(analyzer: ComparisonAnalyzer) -> str:
    """Generate comparison table as text."""
    if not analyzer.cases:
        return "No cases to compare"
        
    metrics = analyzer.compare_metrics()
    
    # Get all metric names
    all_metrics = set()
    for case_metrics in metrics.values():
        all_metrics.update(case_metrics.keys())
    all_metrics = sorted(all_metrics)
    
    # Header
    lines = [
        "=" * 80,
        "ANALYSIS COMPARISON",
        "=" * 80,
        ""
    ]
    
    # Table header
    header = f"{'Metric':<20}"
    for case in analyzer.cases:
        header += f"{case.name:>15}"
    lines.append(header)
    lines.append("-" * 80)
    
    # Table rows
    for metric in all_metrics:
        row = f"{metric:<20}"
        for case in analyzer.cases:
            val = metrics[case.name].get(metric, 'N/A')
            if isinstance(val, float):
                row += f"{val:>15.4f}"
            else:
                row += f"{str(val):>15}"
        lines.append(row)
        
    lines.append("-" * 80)
    
    # Best case
    best = analyzer.find_best_case('max_displacement')
    if best:
        lines.append(f"Best Case (Min Displacement): {best.name}")
        
    lines.append("=" * 80)
    
    return '\n'.join(lines)


def plot_radar_comparison(
    analyzer: ComparisonAnalyzer,
    ax = None
):
    """Radar chart comparing normalized metrics."""
    if len(analyzer.cases) < 2:
        return None
        
    # Metrics to compare
    metric_names = ['max_displacement', 'max_drift', 'max_acceleration', 'period']
    labels = ['Displacement', 'Drift', 'Acceleration', 'Period']
    
    # Get values and normalize
    values_dict = {}
    for case in analyzer.cases:
        values_dict[case.name] = [getattr(case, m, 0) for m in metric_names]
        
    # Normalize (lower is better, so invert)
    max_vals = [max(values_dict[c][i] for c in values_dict) for i in range(len(metric_names))]
    
    for name, vals in values_dict.items():
        values_dict[name] = [1 - v/m if m > 0 else 0 for v, m in zip(vals, max_vals)]
        
    # Plot
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # Close the plot
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(analyzer.cases)))
    
    for i, (name, vals) in enumerate(values_dict.items()):
        vals_plot = vals + vals[:1]
        ax.plot(angles, vals_plot, 'o-', color=colors[i], label=name, linewidth=2)
        ax.fill(angles, vals_plot, alpha=0.1, color=colors[i])
        
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title('Performance Comparison (higher = better)')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    
    return ax
