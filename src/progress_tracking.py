"""
Progress Estimation Module.
Estimates remaining time and tracks simulation progress.
"""
import time
from typing import Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class ProgressInfo:
    """Progress tracking information."""
    current_step: int
    total_steps: int
    elapsed_time: float
    estimated_remaining: float
    progress_percent: float
    rate: float  # steps per second
    status: str
    substeps: List[str] = field(default_factory=list)


class ProgressTracker:
    """
    Tracks progress and estimates remaining time.
    """
    
    def __init__(
        self,
        total_steps: int,
        callback: Callable[[ProgressInfo], None] = None,
        update_interval: float = 0.1
    ):
        """
        Initialize progress tracker.
        
        Args:
            total_steps: Total number of steps
            callback: Function to call on progress update
            update_interval: Minimum seconds between updates
        """
        self.total_steps = max(1, total_steps)
        self.callback = callback
        self.update_interval = update_interval
        
        self.current_step = 0
        self.start_time = None
        self.last_update_time = 0
        self.status = "Initializing..."
        self.substeps = []
        
        # For smoothed rate estimation
        self._rate_history = []
        self._max_history = 10
        
    def start(self):
        """Start tracking."""
        self.start_time = time.time()
        self.current_step = 0
        self.status = "Running..."
        self._notify()
        
    def update(self, step: int = None, status: str = None, substep: str = None):
        """
        Update progress.
        
        Args:
            step: Current step (or None to increment by 1)
            status: Status message
            substep: Add substep detail
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
            
        if status:
            self.status = status
            
        if substep:
            self.substeps.append(substep)
            if len(self.substeps) > 5:
                self.substeps.pop(0)
                
        # Throttle updates
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self.last_update_time = current_time
            self._notify()
            
    def complete(self, status: str = "Complete"):
        """Mark as complete."""
        self.current_step = self.total_steps
        self.status = status
        self._notify()
        
    def _get_rate(self) -> float:
        """Calculate smoothed rate."""
        if self.start_time is None or self.current_step == 0:
            return 0
            
        elapsed = time.time() - self.start_time
        current_rate = self.current_step / elapsed if elapsed > 0 else 0
        
        self._rate_history.append(current_rate)
        if len(self._rate_history) > self._max_history:
            self._rate_history.pop(0)
            
        return sum(self._rate_history) / len(self._rate_history)
        
    def _estimate_remaining(self) -> float:
        """Estimate remaining time in seconds."""
        rate = self._get_rate()
        if rate <= 0:
            return float('inf')
            
        remaining_steps = self.total_steps - self.current_step
        return remaining_steps / rate
        
    def get_info(self) -> ProgressInfo:
        """Get current progress info."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        return ProgressInfo(
            current_step=self.current_step,
            total_steps=self.total_steps,
            elapsed_time=elapsed,
            estimated_remaining=self._estimate_remaining(),
            progress_percent=100.0 * self.current_step / self.total_steps,
            rate=self._get_rate(),
            status=self.status,
            substeps=self.substeps.copy()
        )
        
    def _notify(self):
        """Call callback with progress info."""
        if self.callback:
            try:
                self.callback(self.get_info())
            except:
                pass


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds == float('inf'):
        return "--:--"
    if seconds < 0:
        return "00:00"
        
    seconds = int(seconds)
    
    if seconds < 60:
        return f"0:{seconds:02d}"
    elif seconds < 3600:
        return f"{seconds // 60}:{seconds % 60:02d}"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{hours}:{mins:02d}:{seconds % 60:02d}"


def estimate_simulation_steps(
    duration: float,
    dt: float,
    n_elements: int,
    nonlinear: bool = True
) -> int:
    """
    Estimate total computation steps for simulation.
    
    Args:
        duration: Simulation duration (s)
        dt: Time step (s)
        n_elements: Number of elements
        nonlinear: Whether nonlinear analysis
        
    Returns:
        Estimated step count
    """
    time_steps = int(duration / dt)
    
    # Base: one step per time step
    steps = time_steps
    
    # Nonlinear iterations
    if nonlinear:
        avg_iterations = 3  # Average Newton iterations
        steps *= avg_iterations
        
    # Element updates (minor contribution)
    steps += time_steps * n_elements * 0.01
    
    return int(steps)


class SimulationTimer:
    """
    Timer for simulation performance tracking.
    """
    
    def __init__(self):
        self.timings = {}
        self._active_timers = {}
        
    def start(self, name: str):
        """Start a named timer."""
        self._active_timers[name] = time.perf_counter()
        
    def stop(self, name: str):
        """Stop a named timer and record duration."""
        if name in self._active_timers:
            duration = time.perf_counter() - self._active_timers[name]
            
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(duration)
            
            del self._active_timers[name]
            return duration
        return 0
        
    def get_summary(self) -> dict:
        """Get timing summary."""
        summary = {}
        
        for name, times in self.timings.items():
            summary[name] = {
                'total': sum(times),
                'count': len(times),
                'mean': sum(times) / len(times) if times else 0,
                'max': max(times) if times else 0,
                'min': min(times) if times else 0
            }
            
        return summary
        
    def print_summary(self):
        """Print timing summary."""
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print("TIMING SUMMARY")
        print("=" * 60)
        
        total_time = sum(s['total'] for s in summary.values())
        
        for name, stats in sorted(summary.items(), key=lambda x: -x[1]['total']):
            pct = 100 * stats['total'] / total_time if total_time > 0 else 0
            print(f"{name:30} {stats['total']:8.3f}s ({pct:5.1f}%) [{stats['count']:5d} calls]")
            
        print("=" * 60)
        print(f"{'TOTAL':30} {total_time:8.3f}s")


class MemoryTracker:
    """
    Track memory usage during simulation.
    """
    
    def __init__(self):
        self.peak_usage = 0
        self.snapshots = []
        
    def snapshot(self, label: str = ""):
        """Take memory snapshot."""
        try:
            import psutil
            process = psutil.Process()
            mem = process.memory_info().rss / 1024 / 1024  # MB
        except:
            import sys
            # Fallback - rough estimate
            mem = sys.getsizeof({}) / 1024 / 1024
            
        self.peak_usage = max(self.peak_usage, mem)
        self.snapshots.append((label, mem, datetime.now()))
        
        return mem
        
    def get_current_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except:
            return 0
            
    def print_summary(self):
        """Print memory usage summary."""
        print(f"\nPeak Memory Usage: {self.peak_usage:.1f} MB")
        
        if self.snapshots:
            print("\nSnapshots:")
            for label, mem, ts in self.snapshots[-10:]:
                print(f"  {ts.strftime('%H:%M:%S')} - {label:20} {mem:.1f} MB")
