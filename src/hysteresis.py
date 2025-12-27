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
    Takeda-like Model with proper unloading stiffness.
    Suitable for RC members.
    
    Key features:
    - Unloading stiffness starts at K0 and degrades with ductility
    - State tracking to prevent chattering in Newton-Raphson
    - Origin-oriented reloading after unloading
    """
    
    # State constants
    STATE_LOADING_POS = 1   # Loading toward positive
    STATE_LOADING_NEG = 2   # Loading toward negative  
    STATE_UNLOADING = 3     # Unloading (returning toward origin)
    
    def __init__(self, k0: float, fy: float, r: float, alpha: float = 0.4):
        """
        Initialize Takeda hysteresis model.
        
        Args:
            k0: Initial stiffness
            fy: Yield force  
            r: Post-yield stiffness ratio (typically 0.01-0.1)
            alpha: Unloading stiffness degradation exponent (typically 0.3-0.5)
        """
        super().__init__(k0)
        self.fy = fy
        self.r = r
        self.kp = k0 * r
        self.dy = fy / k0
        self.alpha = alpha  # Degradation exponent
        
        # History (peaks)
        self.d_max = self.dy     # Max disp in positive
        self.d_min = -self.dy    # Max disp in negative
        
        # Committed history
        self.d_max_commit = self.dy
        self.d_min_commit = -self.dy
        
        # State tracking to prevent chattering
        self.state = self.STATE_LOADING_POS
        self.state_commit = self.STATE_LOADING_POS
        
        # Reversal point (where unloading started)
        self.u_reversal = 0.0
        self.f_reversal = 0.0
        self.u_reversal_commit = 0.0
        self.f_reversal_commit = 0.0
        
    def _get_unloading_stiffness(self, d_max_abs: float) -> float:
        """
        Calculate unloading stiffness using Takeda degradation formula.
        K_unload = K0 * (dy / d_max)^alpha
        """
        if d_max_abs < self.dy:
            return self.k0
        
        # Takeda degradation: K_un = K0 * (dy / d_max)^alpha
        k_un = self.k0 * (self.dy / d_max_abs) ** self.alpha
        
        # Bound: at least post-yield stiffness, at most K0
        k_un = max(k_un, self.kp)
        k_un = min(k_un, self.k0)
        
        return k_un
        
    def _calculate_trial_state(self):
        u = self.trial_disp
        d_max = self.d_max_commit
        d_min = self.d_min_commit
        
        # Envelope forces
        f_pos_env = self.fy + self.kp * (u - self.dy)
        f_neg_env = -self.fy + self.kp * (u + self.dy)
        
        # Determine displacement direction
        delta_u = u - self.disp
        
        # Small displacement threshold
        if abs(delta_u) < 1e-12:
            self.trial_force = self.force
            self.trial_tangent = self.tangent
            return
        
        moving_positive = delta_u > 0
        
        # ============================================
        # 1. Check if hitting envelope
        # ============================================
        if u >= d_max:
            # Pushing positive envelope
            self.trial_force = f_pos_env
            self.trial_tangent = self.kp
            self.state = self.STATE_LOADING_POS
            self.u_reversal = u
            self.f_reversal = self.trial_force
            return
            
        if u <= d_min:
            # Pushing negative envelope
            self.trial_force = f_neg_env
            self.trial_tangent = self.kp
            self.state = self.STATE_LOADING_NEG
            self.u_reversal = u
            self.f_reversal = self.trial_force
            return
        
        # ============================================
        # 2. Inside loop: State-based behavior
        # ============================================
        
        # Detect reversal (change of direction)
        was_loading_pos = self.state_commit == self.STATE_LOADING_POS
        was_loading_neg = self.state_commit == self.STATE_LOADING_NEG
        was_unloading = self.state_commit == self.STATE_UNLOADING
        
        # Check for reversal
        if was_loading_pos and not moving_positive:
            # Reversal from positive loading -> start unloading
            self.state = self.STATE_UNLOADING
            self.u_reversal = self.disp
            self.f_reversal = self.force
            
        elif was_loading_neg and moving_positive:
            # Reversal from negative loading -> start unloading
            self.state = self.STATE_UNLOADING
            self.u_reversal = self.disp
            self.f_reversal = self.force
            
        elif was_unloading:
            # Continue unloading until we cross origin or re-reverse
            # Check if we've crossed the force=0 line
            if self.force > 0 and not moving_positive:
                # Still unloading from positive side
                self.state = self.STATE_UNLOADING
            elif self.force < 0 and moving_positive:
                # Still unloading from negative side
                self.state = self.STATE_UNLOADING
            elif self.force > 0 and moving_positive:
                # Re-loading toward positive
                self.state = self.STATE_LOADING_POS
            elif self.force < 0 and not moving_positive:
                # Re-loading toward negative
                self.state = self.STATE_LOADING_NEG
            else:
                self.state = self.STATE_UNLOADING
        else:
            # Default behavior
            if moving_positive:
                self.state = self.STATE_LOADING_POS
            else:
                self.state = self.STATE_LOADING_NEG
        
        # ============================================
        # 3. Calculate force based on state
        # ============================================
        
        if self.state == self.STATE_UNLOADING:
            # UNLOADING: Use degraded initial stiffness (NOT secant stiffness)
            # This is the key fix - unloading is "stiff" like K0
            d_max_abs = max(abs(d_max), abs(d_min))
            k_un = self._get_unloading_stiffness(d_max_abs)
            
            # Calculate force from reversal point
            self.trial_force = self.f_reversal + k_un * (u - self.u_reversal)
            self.trial_tangent = k_un
            
            # Check if force crossed zero (origin crossing)
            if (self.f_reversal > 0 and self.trial_force < 0) or \
               (self.f_reversal < 0 and self.trial_force > 0):
                # Crossed origin - transition to reloading toward opposite peak
                # But DON'T recalculate force here - let next iteration handle it
                # This prevents double-calculation in the same call
                if moving_positive:
                    self.state = self.STATE_LOADING_POS
                else:
                    self.state = self.STATE_LOADING_NEG
            # Note: Force already calculated above, no need to recalculate
            
        elif self.state == self.STATE_LOADING_POS:
            # RELOADING toward positive peak
            # Draw line from origin (or near it) to positive peak
            target_u = d_max
            target_f = self.fy + self.kp * (d_max - self.dy)
            
            # Secant stiffness from origin to peak
            k_reload = target_f / target_u if abs(target_u) > 1e-9 else self.k0
            k_reload = max(k_reload, self.kp)  # At least post-yield
            k_reload = min(k_reload, self.k0)  # At most initial
            
            self.trial_force = k_reload * u
            self.trial_tangent = k_reload
            
        elif self.state == self.STATE_LOADING_NEG:
            # RELOADING toward negative peak
            target_u = d_min
            target_f = -self.fy + self.kp * (d_min + self.dy)
            
            k_reload = target_f / target_u if abs(target_u) > 1e-9 else self.k0
            k_reload = max(abs(k_reload), self.kp)
            k_reload = min(k_reload, self.k0)
            
            self.trial_force = -k_reload * abs(u)
            self.trial_tangent = k_reload
        
        # ============================================
        # 4. Bound checks
        # ============================================
        
        # Don't exceed envelope
        self.trial_force = min(self.trial_force, f_pos_env)
        self.trial_force = max(self.trial_force, f_neg_env)
        
        # Sanity bounds
        max_force = self.fy * 10
        self.trial_force = np.clip(self.trial_force, -max_force, max_force)
        self.trial_tangent = max(self.trial_tangent, self.kp)
        self.trial_tangent = min(self.trial_tangent, self.k0 * 2)
            
    def _commit_history(self):
        # Update d_max/d_min
        if self.disp > self.d_max_commit:
            self.d_max_commit = self.disp
        if self.disp < self.d_min_commit:
            self.d_min_commit = self.disp
        
        # Commit state
        self.state_commit = self.state
        self.u_reversal_commit = self.u_reversal
        self.f_reversal_commit = self.f_reversal
