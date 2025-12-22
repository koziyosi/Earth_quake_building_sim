"""
Element Response Analyzer Module.
Provides utilities for analyzing individual column and wall element responses.
"""
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ElementResponseSummary:
    """Summary of a single element's response."""
    element_id: int
    element_type: str
    peak_axial_force: float
    peak_shear: float
    peak_moment: float
    peak_rotation: float
    max_damage_index: float
    is_yielded: bool
    is_critical: bool  # True if damage > 1.5


class ElementResponseAnalyzer:
    """
    Analyzes element-level responses from simulation results.
    
    Provides separation of column, beam, and wall behavior for detailed analysis.
    """
    
    def __init__(self, elements: List[Any]):
        """
        Initialize analyzer with list of elements.
        
        Args:
            elements: List of BeamColumn3D or similar elements
        """
        self.elements = elements
        self._classify_elements()
    
    def _classify_elements(self):
        """Classify elements by type."""
        self.columns = []
        self.beams = []
        self.walls = []
        self.braces = []
        self.others = []
        
        for el in self.elements:
            if hasattr(el, 'element_type'):
                el_type = el.element_type
                if el_type == 'column':
                    self.columns.append(el)
                elif el_type == 'beam':
                    self.beams.append(el)
                elif el_type == 'wall':
                    self.walls.append(el)
                elif el_type == 'brace':
                    self.braces.append(el)
                else:
                    self.others.append(el)
            elif hasattr(el, 'custom_color') and el.custom_color == 'red':
                # Legacy wall detection
                self.walls.append(el)
            else:
                self.others.append(el)
    
    def get_summary_by_type(self) -> Dict[str, Dict[str, Any]]:
        """
        Get response summary grouped by element type.
        
        Returns:
            Dictionary with 'columns', 'beams', 'walls' keys containing statistics.
        """
        return {
            'columns': self._get_type_summary(self.columns),
            'beams': self._get_type_summary(self.beams),
            'walls': self._get_type_summary(self.walls),
            'braces': self._get_type_summary(self.braces),
        }
    
    def _get_type_summary(self, elements: List[Any]) -> Dict[str, Any]:
        """Get summary statistics for a list of elements."""
        if not elements:
            return {
                'count': 0,
                'max_damage_index': 0.0,
                'avg_damage_index': 0.0,
                'yielded_count': 0,
                'critical_count': 0,
                'max_axial_force': 0.0,
                'max_moment': 0.0,
                'max_rotation': 0.0,
            }
        
        damages = []
        axial_forces = []
        moments = []
        rotations = []
        
        for el in elements:
            if hasattr(el, 'max_damage_index'):
                damages.append(el.max_damage_index)
            elif hasattr(el, 'damage_index'):
                damages.append(el.damage_index)
            else:
                damages.append(0.0)
            
            if hasattr(el, 'peak_axial_force'):
                axial_forces.append(el.peak_axial_force)
            elif hasattr(el, 'current_N'):
                axial_forces.append(abs(el.current_N))
            else:
                axial_forces.append(0.0)
            
            if hasattr(el, 'peak_moment'):
                moments.append(el.peak_moment)
            else:
                moments.append(0.0)
            
            if hasattr(el, 'peak_rotation'):
                rotations.append(el.peak_rotation)
            else:
                rotations.append(0.0)
        
        yielded = sum(1 for d in damages if d >= 1.0)
        critical = sum(1 for d in damages if d >= 1.5)
        
        return {
            'count': len(elements),
            'max_damage_index': max(damages) if damages else 0.0,
            'avg_damage_index': np.mean(damages) if damages else 0.0,
            'yielded_count': yielded,
            'critical_count': critical,
            'max_axial_force': max(axial_forces) if axial_forces else 0.0,
            'max_moment': max(moments) if moments else 0.0,
            'max_rotation': max(rotations) if rotations else 0.0,
            'elements': elements,  # Reference to actual elements
        }
    
    def get_critical_elements(self, damage_threshold: float = 1.0) -> List[Any]:
        """Get all elements with damage index above threshold."""
        critical = []
        for el in self.elements:
            damage = getattr(el, 'max_damage_index', 0) or getattr(el, 'damage_index', 0)
            if damage >= damage_threshold:
                critical.append(el)
        return critical
    
    def get_element_responses(self) -> List[ElementResponseSummary]:
        """Get response summary for all elements."""
        summaries = []
        for el in self.elements:
            if hasattr(el, 'get_response_summary'):
                resp = el.get_response_summary()
                summaries.append(ElementResponseSummary(
                    element_id=resp['id'],
                    element_type=resp['type'],
                    peak_axial_force=getattr(el, 'peak_axial_force', 0),
                    peak_shear=getattr(el, 'peak_shear', 0),
                    peak_moment=getattr(el, 'peak_moment', 0),
                    peak_rotation=getattr(el, 'peak_rotation', 0),
                    max_damage_index=getattr(el, 'max_damage_index', 0),
                    is_yielded=getattr(el, 'max_damage_index', 0) >= 1.0,
                    is_critical=getattr(el, 'max_damage_index', 0) >= 1.5,
                ))
        return summaries
    
    def print_summary(self):
        """Print human-readable summary to console."""
        summary = self.get_summary_by_type()
        
        print("\n" + "="*60)
        print("ELEMENT RESPONSE SUMMARY")
        print("="*60)
        
        for el_type, stats in summary.items():
            if stats['count'] == 0:
                continue
            
            print(f"\n{el_type.upper()} ({stats['count']} elements)")
            print("-" * 40)
            print(f"  Max Damage Index: {stats['max_damage_index']:.3f}")
            print(f"  Avg Damage Index: {stats['avg_damage_index']:.3f}")
            print(f"  Yielded: {stats['yielded_count']} / {stats['count']}")
            print(f"  Critical (>1.5): {stats['critical_count']}")
            print(f"  Max Axial Force: {stats['max_axial_force']/1e3:.1f} kN")
            print(f"  Max Moment: {stats['max_moment']/1e3:.1f} kN*m")
            print(f"  Max Rotation: {stats['max_rotation']*1000:.2f} mrad")
        
        print("\n" + "="*60)
    
    def enable_tracking(self, element_types: Optional[List[str]] = None):
        """
        Enable history tracking for specified element types.
        
        Args:
            element_types: List of types to track ('column', 'beam', 'wall', 'brace')
                          If None, tracks all elements.
        """
        for el in self.elements:
            if not hasattr(el, 'track_history'):
                continue
            
            if element_types is None:
                el.track_history = True
            elif hasattr(el, 'element_type') and el.element_type in element_types:
                el.track_history = True
    
    def get_column_drift_ratios(self) -> Dict[int, float]:
        """
        Get story drift ratios for all columns.
        
        Returns:
            Dictionary mapping story number to max drift ratio.
        """
        story_drifts = {}
        
        for col in self.columns:
            # Estimate story from column bottom node Z
            story = int(col.node_i.z / 3.5) + 1  # Approximate story height
            
            # Get drift from rotation
            if hasattr(col, 'peak_rotation'):
                drift = col.peak_rotation
            else:
                drift = max(
                    abs(getattr(col, 'rot_my_i', 0)),
                    abs(getattr(col, 'rot_my_j', 0))
                )
            
            if story not in story_drifts or drift > story_drifts[story]:
                story_drifts[story] = drift
        
        return story_drifts


def analyze_element_responses(elements: List[Any], print_summary: bool = True) -> Dict:
    """
    Convenience function to analyze element responses.
    
    Args:
        elements: List of structural elements
        print_summary: Whether to print summary to console
    
    Returns:
        Dictionary with analysis results
    """
    analyzer = ElementResponseAnalyzer(elements)
    
    if print_summary:
        analyzer.print_summary()
    
    return {
        'summary_by_type': analyzer.get_summary_by_type(),
        'critical_elements': analyzer.get_critical_elements(),
        'column_drifts': analyzer.get_column_drift_ratios(),
        'analyzer': analyzer,
    }
