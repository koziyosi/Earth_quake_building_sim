"""
Undo/Redo Manager for Layout Editor.
Implements #40 from improvement list.
"""
from typing import Any, Callable, List, Optional
from dataclasses import dataclass
from copy import deepcopy


@dataclass
class UndoAction:
    """Represents a single undoable action."""
    name: str
    old_state: Any
    new_state: Any
    timestamp: float = 0.0


class UndoManager:
    """
    Manages undo/redo operations for the layout editor.
    
    Features:
    - Configurable undo stack size
    - Action grouping
    - State snapshots
    """
    
    def __init__(self, max_history: int = 50):
        self.max_history = max_history
        self.undo_stack: List[UndoAction] = []
        self.redo_stack: List[UndoAction] = []
        
        self.on_state_change: Optional[Callable[[Any], None]] = None
        
    def push(self, action: UndoAction):
        """Push a new action to the undo stack."""
        self.undo_stack.append(action)
        
        # Clear redo stack when new action is pushed
        self.redo_stack.clear()
        
        # Limit stack size
        if len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0)
            
    def record_change(self, name: str, old_state: Any, new_state: Any):
        """Record a state change as an undoable action."""
        import time
        
        action = UndoAction(
            name=name,
            old_state=deepcopy(old_state),
            new_state=deepcopy(new_state),
            timestamp=time.time()
        )
        self.push(action)
        
    def can_undo(self) -> bool:
        """Check if undo is possible."""
        return len(self.undo_stack) > 0
    
    def can_redo(self) -> bool:
        """Check if redo is possible."""
        return len(self.redo_stack) > 0
    
    def undo(self) -> Optional[Any]:
        """
        Undo the last action.
        
        Returns:
            The old state to restore, or None if no undo available
        """
        if not self.can_undo():
            return None
            
        action = self.undo_stack.pop()
        self.redo_stack.append(action)
        
        if self.on_state_change:
            self.on_state_change(action.old_state)
            
        return action.old_state
    
    def redo(self) -> Optional[Any]:
        """
        Redo the last undone action.
        
        Returns:
            The new state to apply, or None if no redo available
        """
        if not self.can_redo():
            return None
            
        action = self.redo_stack.pop()
        self.undo_stack.append(action)
        
        if self.on_state_change:
            self.on_state_change(action.new_state)
            
        return action.new_state
    
    def get_undo_description(self) -> str:
        """Get description of the next undo action."""
        if self.can_undo():
            return f"Undo: {self.undo_stack[-1].name}"
        return "Nothing to undo"
    
    def get_redo_description(self) -> str:
        """Get description of the next redo action."""
        if self.can_redo():
            return f"Redo: {self.redo_stack[-1].name}"
        return "Nothing to redo"
    
    def clear(self):
        """Clear all undo/redo history."""
        self.undo_stack.clear()
        self.redo_stack.clear()
        
    def get_history(self) -> List[str]:
        """Get list of action names in undo history."""
        return [action.name for action in self.undo_stack]


class LayoutUndoManager(UndoManager):
    """
    Specialized undo manager for building layout editor.
    Works with BuildingLayout objects.
    """
    
    def __init__(self, layout, max_history: int = 50):
        super().__init__(max_history)
        self.layout = layout
        
    def snapshot_before_edit(self, action_name: str):
        """Take a snapshot before making an edit."""
        self._pending_action = action_name
        self._pending_old_state = self.layout.to_dict()
        
    def commit_edit(self):
        """Commit the edit and record in undo stack."""
        if hasattr(self, '_pending_action'):
            new_state = self.layout.to_dict()
            self.record_change(
                self._pending_action,
                self._pending_old_state,
                new_state
            )
            del self._pending_action
            del self._pending_old_state
            
    def restore_state(self, state_dict: dict):
        """Restore layout from state dictionary."""
        # Clear current layout and rebuild from dict
        self.layout.grid = state_dict.get('grid', self.layout.grid)
        self.layout.floors = state_dict.get('floors', self.layout.floors)
        self.layout.sections = state_dict.get('sections', self.layout.sections)
