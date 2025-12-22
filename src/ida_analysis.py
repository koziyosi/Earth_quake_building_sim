"""
Incremental Dynamic Analysis (IDA) module.
Performs IDA analysis to generate fragility curves.
"""
import numpy as np
from typing import List, Tuple, Optional, Callable, Dict
from dataclasses import dataclass
import copy


@dataclass
class IDAResult:
    """Result for a single IDA run."""
    scale_factor: float
    im_value: float  # Intensity measure value (e.g., Sa(T1))
    max_drift: float
    max_displacement: float
    collapse: bool
    damage_level: str


@dataclass
class IDACurve:
    """Complete IDA curve for one ground motion."""
    ground_motion_id: str
    results: List[IDAResult]
    
    def get_im_values(self) -> np.ndarray:
        return np.array([r.im_value for r in self.results])
    
    def get_drift_values(self) -> np.ndarray:
        return np.array([r.max_drift for r in self.results])


@dataclass
class FragilityResult:
    """Fragility curve result."""
    damage_state: str
    im_values: np.ndarray  # Intensity measure values
    probability: np.ndarray  # Probability of exceedance


class IDAAnalyzer:
    """
    Incremental Dynamic Analysis performer.
    
    IDA involves scaling ground motions incrementally and running
    nonlinear dynamic analysis at each intensity level to characterize
    structural response up to collapse.
    """
    
    def __init__(
        self,
        run_analysis: Callable,
        im_type: str = 'Sa_T1',
        scale_factors: Optional[List[float]] = None
    ):
        """
        Initialize IDA analyzer.
        
        Args:
            run_analysis: Function that runs analysis with signature:
                          (time_array, scaled_acc, scale_factor) -> (max_drift, max_disp, collapsed)
            im_type: Intensity measure type ('Sa_T1', 'PGA', 'PGV')
            scale_factors: List of scale factors to use
        """
        self.run_analysis = run_analysis
        self.im_type = im_type
        self.scale_factors = scale_factors or np.concatenate([
            np.arange(0.1, 0.5, 0.1),
            np.arange(0.5, 2.0, 0.25),
            np.arange(2.0, 5.0, 0.5)
        ]).tolist()
        
        self.curves: List[IDACurve] = []
        
    def run_ida_single(
        self,
        time_array: np.ndarray,
        acceleration: np.ndarray,
        gm_id: str,
        fundamental_period: float = 1.0
    ) -> IDACurve:
        """
        Run IDA for a single ground motion.
        
        Args:
            time_array: Time array
            acceleration: Base acceleration array (m/s²)
            gm_id: Ground motion identifier
            fundamental_period: Fundamental period for Sa(T1)
            
        Returns:
            IDACurve with all intensity levels
        """
        results = []
        
        for sf in self.scale_factors:
            scaled_acc = acceleration * sf
            
            # Calculate intensity measure
            im_value = self._calculate_im(
                time_array, scaled_acc, 
                fundamental_period, sf
            )
            
            try:
                # Run analysis
                max_drift, max_disp, collapsed = self.run_analysis(
                    time_array, scaled_acc, sf
                )
                
                # Classify damage
                damage_level = self._classify_damage(max_drift, collapsed)
                
                result = IDAResult(
                    scale_factor=sf,
                    im_value=im_value,
                    max_drift=max_drift,
                    max_displacement=max_disp,
                    collapse=collapsed,
                    damage_level=damage_level
                )
                results.append(result)
                
                # Stop if collapsed
                if collapsed:
                    break
                    
            except Exception as e:
                print(f"Analysis failed at SF={sf}: {e}")
                # Assume collapse
                results.append(IDAResult(
                    scale_factor=sf,
                    im_value=im_value,
                    max_drift=0.1,  # Placeholder
                    max_displacement=1.0,
                    collapse=True,
                    damage_level='Collapse'
                ))
                break
        
        curve = IDACurve(ground_motion_id=gm_id, results=results)
        self.curves.append(curve)
        return curve
    
    def _calculate_im(
        self,
        time_array: np.ndarray,
        acceleration: np.ndarray,
        T1: float,
        scale_factor: float
    ) -> float:
        """Calculate intensity measure value."""
        if self.im_type == 'PGA':
            return np.max(np.abs(acceleration))
        elif self.im_type == 'PGV':
            # Integrate to velocity
            dt = time_array[1] - time_array[0]
            velocity = np.cumsum(acceleration) * dt
            return np.max(np.abs(velocity))
        else:  # Sa(T1) default
            # Use response spectrum at T1
            from .response_spectrum import calculate_response_spectrum
            result = calculate_response_spectrum(
                time_array, acceleration,
                periods=np.array([T1])
            )
            return result.Sa[0]
    
    def _classify_damage(self, drift: float, collapsed: bool) -> str:
        """Classify damage state based on drift."""
        if collapsed or drift > 0.05:
            return 'Collapse'
        elif drift > 0.03:
            return 'Severe'
        elif drift > 0.015:
            return 'Moderate'
        elif drift > 0.005:
            return 'Minor'
        else:
            return 'None'
    
    def generate_fragility_curves(
        self,
        damage_states: Optional[Dict[str, float]] = None
    ) -> List[FragilityResult]:
        """
        Generate fragility curves from IDA results.
        
        Args:
            damage_states: Dictionary of {state_name: drift_threshold}
            
        Returns:
            List of FragilityResult for each damage state
        """
        if damage_states is None:
            damage_states = {
                'Minor': 0.005,
                'Moderate': 0.015,
                'Severe': 0.03,
                'Collapse': 0.05
            }
        
        if not self.curves:
            return []
        
        # Collect all IM values
        all_im = []
        for curve in self.curves:
            all_im.extend([r.im_value for r in curve.results])
        
        im_range = np.linspace(0, max(all_im) * 1.2, 50)
        
        fragility_results = []
        
        for state_name, drift_threshold in damage_states.items():
            # For each IM level, count how many ground motions exceed threshold
            probabilities = []
            
            for im in im_range:
                exceeds = 0
                total = 0
                
                for curve in self.curves:
                    total += 1
                    # Find if this ground motion exceeded threshold at this IM
                    for r in curve.results:
                        if r.im_value >= im * 0.9 and r.im_value <= im * 1.1:
                            if r.max_drift >= drift_threshold:
                                exceeds += 1
                            break
                
                prob = exceeds / max(total, 1)
                probabilities.append(prob)
            
            fragility_results.append(FragilityResult(
                damage_state=state_name,
                im_values=im_range,
                probability=np.array(probabilities)
            ))
        
        return fragility_results
    
    def plot_ida_curves(self):
        """Plot IDA curves (requires matplotlib)."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for curve in self.curves:
            im = curve.get_im_values()
            drift = curve.get_drift_values()
            ax.plot(drift, im, 'o-', alpha=0.7, label=curve.ground_motion_id)
        
        ax.set_xlabel('Maximum Inter-story Drift Ratio')
        ax.set_ylabel(f'Intensity Measure ({self.im_type})')
        ax.set_title('IDA Curves')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig
    
    def plot_fragility_curves(self, fragility_results: List[FragilityResult]):
        """Plot fragility curves."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        colors = ['green', 'yellow', 'orange', 'red']
        
        for i, fr in enumerate(fragility_results):
            color = colors[i % len(colors)]
            ax.plot(fr.im_values, fr.probability, color=color, 
                    linewidth=2, label=fr.damage_state)
        
        ax.set_xlabel(f'Intensity Measure ({self.im_type})')
        ax.set_ylabel('Probability of Exceedance')
        ax.set_title('Fragility Curves')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig
