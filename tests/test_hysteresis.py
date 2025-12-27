import numpy as np
import matplotlib.pyplot as plt
from src.hysteresis import Bilinear, Takeda

def test_hysteresis():
    # Setup
    k0 = 100.0
    fy = 1000.0
    r = 0.05
    
    # Protocol: Cyclic displacement
    # 0 -> 2*dy -> 0 -> -2*dy -> 0 -> 4*dy ...
    dy = fy / k0
    peaks = [2*dy, -2*dy, 4*dy, -4*dy, 0]
    
    # Generate time history
    disp_history = []
    curr = 0.0
    for p in peaks:
        # Interpolate
        steps = 50
        disp_history.extend(np.linspace(curr, p, steps))
        curr = p
        
    # Test Bilinear
    bi_model = Bilinear(k0, fy, r)
    bi_force = []
    bi_disp = []
    
    for d in disp_history:
        f, k = bi_model.set_trial_displacement(d)
        bi_model.commit()
        bi_force.append(f)
        bi_disp.append(d)
        
    # Test Takeda/Clough
    tk_model = Takeda(k0, fy, r)
    tk_force = []
    tk_disp = []
    
    for d in disp_history:
        f, k = tk_model.set_trial_displacement(d)
        tk_model.commit()
        tk_force.append(f)
        tk_disp.append(d)
        
    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(bi_disp, bi_force, label='Bilinear')
    plt.title('Bilinear Model')
    plt.grid(True)
    plt.xlabel('Displacement')
    plt.ylabel('Force')
    
    plt.subplot(1, 2, 2)
    plt.plot(tk_disp, tk_force, label='Takeda/Clough', color='orange')
    plt.title('Takeda/Clough Model')
    plt.grid(True)
    plt.xlabel('Displacement')
    plt.ylabel('Force')
    
    plt.savefig('hysteresis_check.png')
    print("Hysteresis plot saved to hysteresis_check.png")

if __name__ == "__main__":
    test_hysteresis()
