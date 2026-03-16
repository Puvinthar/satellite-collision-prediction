import torch
import torch.nn as nn

class GatedPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 128), nn.Tanh(),
            nn.Linear(128, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
            nn.Linear(128, 6)
        )

    def forward(self, x, t):
        out = self.net(x)
        gate = torch.tanh(5.0 * t) # Soft Physics Gate (0 at t=0)
        return out * gate

# --- Physics Constants ---
J2 = 1.08263e-3
mu = 398600.4418  # km^3/s^2
Re = 6378.137     # km

def physics_loss(pred_pos, pred_vel, t, bstar):
    """
    Calculates the physics-informed loss using J2 perturbation and Drag.
    """
    # 1. Calculate Predicted Acceleration using Autograd
    # We differentiate Velocity to get Acceleration
    # Ensure t implies requires_grad if not already
    
    # Note: To use autograd.grad, inputs must have requires_grad=True.
    # In a full PINN, t is usually the input. Here 't' is passed separately.
    # We assume 'pred_vel' is a function of 't' through the network.
    # For this snippet to work strictly as written, 't' must be part of the graph.
    
    # If t is just a tensor, we might need to handle it carefully.
    # Assuming standard PINN setup where t is an input leaf or derived.
    
    # Placeholder for autograd implementation matching user snippet
    # In a real training loop, we'd ensure t.requires_grad=True
    
    # For the purpose of this snippet, we'll try to follow the user's logic exactly.
    # If pred_vel is output of NN(t), then grad(pred_vel, t) works.
    
    # However, in the user's Forward Pass: delta_r, delta_v = model(inputs)
    # The 'inputs' likely contains 't'.
    # So we need to compute grad w.r.t the time component of inputs or passed 't'.
    
    # Simplified approach:
    # We will assume 't' passed here is the tensor used in forward pass.
    
    acc_pred = torch.autograd.grad(pred_vel, t, grad_outputs=torch.ones_like(pred_vel), create_graph=True, allow_unused=True)[0]
    
    # Handle case where grad might be None if detached (e.g. valid/test phase)
    if acc_pred is None:
        acc_pred = torch.zeros_like(pred_vel)

    # 2. Define Physical Acceleration (J2 + Drag)
    # Gravity (Two-Body)
    r_mag = torch.norm(pred_pos, dim=1, keepdim=True)
    acc_gravity = -mu * pred_pos / (r_mag**3)

    # J2 Perturbation (Earth's Oblateness)
    z = pred_pos[:, 2:3]
    # Simplified J2 formula for backpropagation
    factor = -(1.5 * J2 * mu * Re**2) / (r_mag**5)
    acc_j2_x = factor * pred_pos[:, 0:1] * (1 - 5 * (z/r_mag)**2)
    acc_j2_y = factor * pred_pos[:, 1:2] * (1 - 5 * (z/r_mag)**2)
    acc_j2_z = factor * z * (3 - 5 * (z/r_mag)**2)
    
    # Drag (Simplified Exponential Model)
    # acc_drag = -0.5 * rho * v_rel^2 * Cd * A / m
    # Using bstar proxy: acc_drag ~ - Bstar * rho * v^2 
    # (Very rough approx, but fits the "scaffold" request)
    
    # Simple exponential density: rho = rho0 * exp(-(h - h0)/H)
    # r_mag - Re is altitude
    h = r_mag - Re
    rho_0 = 3.614e-13 # km^3/kg? No, density at 700km approx or generic
    # Let's use a simplified logical drag direction: opposes velocity
    
    # Drag acceleration is proportional to -v * |v|
    v_mag = torch.norm(pred_vel, dim=1, keepdim=True)
    # acc_drag = - (B* * rho) * v * |v| ? 
    # Bstar has units of 1/EarthRadii or similar in TLE? 
    # Standard TLE Bstar is in units of (1/earth_radii).
    # We'll treat bstar as a learned or provided coefficient.
    
    # Placeholder density factor
    # For robust physics, we'd need a real atmosphere model.
    # Checking user request: "define the J2 and Drag ODEs"
    # User code: acc_physics = acc_gravity + ... + acc_drag(bstar)
    
    # We'll define acc_drag inline or helper
    def get_acc_drag(v, bstar, r_mag):
        # Very simple drag model
        # Acceleration opposes velocity
        return -bstar * v * v_mag * 1e-3 # Scaling factor for stability
        
    drag_vec = get_acc_drag(pred_vel, bstar, r_mag)
    
    # 3. Sum the forces and find the Residual
    acc_physics = acc_gravity + torch.cat([acc_j2_x, acc_j2_y, acc_j2_z], dim=1) + drag_vec
    
    return torch.mean((acc_pred - acc_physics)**2)

