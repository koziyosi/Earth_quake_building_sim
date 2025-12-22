import numpy as np
from abc import ABC, abstractmethod

class Hysteresis(ABC):
    """
    Abstract base class for hysteresis models.
    Supports trial/commit interface for iterative solvers.
    """
    def __init__(self, k0):
        self.k0 = k0
        
        # Committed State
        self.disp = 0.0
        self.force = 0.0
        self.tangent = k0
        
        # Trial State
        self.trial_disp = 0.0
        self.trial_force = 0.0
        self.trial_tangent = k0
        
    def set_trial_displacement(self, disp):
        """
        Calculates response for a trial total displacement.
        Updates trial_force and trial_tangent.
        """
        self.trial_disp = disp
        self._calculate_trial_state()
        return self.trial_force, self.trial_tangent
        
    def commit(self):
        """
        Commits the trial state to the permanent state.
        """
        self.disp = self.trial_disp
        self.force = self.trial_force
        self.tangent = self.trial_tangent
        self._commit_history()

    def get_state(self):
        return self.force, self.tangent

    @abstractmethod
    def _calculate_trial_state(self):
        pass
        
    @abstractmethod
    def _commit_history(self):
        pass

class Bilinear(Hysteresis):
    """
    Bilinear Hysteresis (Kinematic Hardening).
    """
    def __init__(self, k0, fy, r):
        super().__init__(k0)
        self.fy = fy
        self.r = r
        self.kp = k0 * r
        
        # History: Plastic Displacement (center of yield surface)
        # We track 'u_p' (Back-stress / Plastic deformation equivalent)
        # Using parallel spring analogy:
        # Spring A (Linear): r*K0
        # Spring B (EP): (1-r)*K0, Yield=(1-r)Fy
        
        self.k_b = (1.0 - r) * k0
        self.fy_b = (1.0 - r) * fy
        self.k_a = r * k0
        
        self.u_p_b = 0.0 # Plastic disp of spring B
        self.u_p_b_commit = 0.0
        
    def _calculate_trial_state(self):
        # Force A
        f_a = self.k_a * self.trial_disp
        tan_a = self.k_a
        
        # Force B (Elastic-Plastic)
        # Trial elastic force relative to current plastic center
        f_b_trial = self.k_b * (self.trial_disp - self.u_p_b_commit)
        
        if f_b_trial > self.fy_b:
            f_b = self.fy_b
            tan_b = 0.0
            # New plastic center u_p such that K_b * (u - u_p) = Fy_b
            # u - u_p = Fy_b / K_b
            # u_p = u - Fy_b / K_b
            self.u_p_b = self.trial_disp - (self.fy_b / self.k_b)
        elif f_b_trial < -self.fy_b:
            f_b = -self.fy_b
            tan_b = 0.0
            self.u_p_b = self.trial_disp - (-self.fy_b / self.k_b)
        else:
            f_b = f_b_trial
            tan_b = self.k_b
            self.u_p_b = self.u_p_b_commit
            
        self.trial_force = f_a + f_b
        self.trial_tangent = tan_a + tan_b
        
    def _commit_history(self):
        self.u_p_b_commit = self.u_p_b

class Takeda(Hysteresis):
    """
    Takeda-like Model (Peak Oriented).
    Suitable for RC members.
    Degrades stiffness based on max historical deformation.
    """
    def __init__(self, k0: float, fy: float, r: float):
        """
        Initialize Takeda hysteresis model.
        
        Args:
            k0: Initial stiffness
            fy: Yield force  
            r: Post-yield stiffness ratio
        """
        super().__init__(k0)
        self.fy = fy
        self.r = r
        self.kp = k0 * r
        self.dy = fy / k0
        
        # History
        self.d_max = self.dy     # Max disp in positive
        self.d_min = -self.dy    # Max disp in negative
        
        # Committed history
        self.d_max_commit = self.dy
        self.d_min_commit = -self.dy
        
    def _calculate_trial_state(self):
        u = self.trial_disp
        d_max = self.d_max_commit
        d_min = self.d_min_commit
        
        # Envelope Functions
        f_pos_env = self.fy + self.kp * (u - self.dy)
        f_neg_env = -self.fy + self.kp * (u + self.dy)
        
        # 1. Check Envelopes
        if u >= d_max:
            # Pushing positive envelope
            self.trial_force = f_pos_env
            self.trial_tangent = self.kp
            return
            
        if u <= d_min:
            # Pushing negative envelope
            self.trial_force = f_neg_env
            self.trial_tangent = self.kp
            return
            
        # 2. Inside Loop (Unloading / Reloading)
        # Determine Target
        # If u > u_last_commit: Moving Positive -> Target d_max
        # If u < u_last_commit: Moving Negative -> Target d_min
        
        # Edge case: No displacement change - maintain current state
        # Keep tangent consistent with current position
        if abs(u - self.disp) < 1e-12:
            self.trial_force = self.force
            # Maintain tangent based on current position relative to envelope
            if u >= d_max - 1e-12:
                self.trial_tangent = self.kp  # On positive envelope
            elif u <= d_min + 1e-12:
                self.trial_tangent = self.kp  # On negative envelope
            else:
                self.trial_tangent = self.tangent  # Keep current tangent
            return
            
        if u > self.disp:
            # Target: Positive Peak
            target_u = d_max
            target_f = self.fy + self.kp * (d_max - self.dy)
        else:
            # Target: Negative Peak
            target_u = d_min
            target_f = -self.fy + self.kp * (d_min + self.dy)
            
        # Secant stiffness to target
        # K = (F_target - F_current) / (u_target - u_current)
        # NOTE: Using F_current (committed) and u_current (committed).
        
        delta_u = target_u - self.disp
        if abs(delta_u) < 1e-9:
            # Already at target? Should have been caught by Envelope check
            k_sec = self.kp
        else:
            k_sec = (target_f - self.force) / delta_u
            
        # Robustness: Stiffness should not be negative usually (unless softening)
        # And usually >= kp for Takeda
        if k_sec < self.kp: k_sec = self.kp
        
        # Upper bound to prevent extreme stiffness (max 10x initial stiffness)
        k_sec = min(k_sec, self.k0 * 10)
        
        # Use Large stiffness if unloading just started? 
        # Takeda unloading is K_un = K0 * (dy/dm)^alpha
        # Peak oriented simplifies unloading to point directly at opposite peak? 
        # No, that's "Origin Oriented" or specific "Peak Oriented".
        # Standard Takeda: Unloads with K_un, then aims at opposite crossing point.
        
        # For this implementation, "Peak Oriented" is the most robust simple degrading model.
        # It always aims at the peak.
        
        self.trial_tangent = k_sec
        self.trial_force = self.force + k_sec * (u - self.disp)
        
        # Bound force to prevent extreme values (max 10x yield force)
        max_force = self.fy * 10
        self.trial_force = np.clip(self.trial_force, -max_force, max_force)
            
    def _commit_history(self):
        # Update d_max/d_min
        if self.disp > self.d_max_commit:
            self.d_max_commit = self.disp
        if self.disp < self.d_min_commit:
            self.d_min_commit = self.disp
