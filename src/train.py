import torch
import torch.nn as nn
import torch.optim as optim
import logging

logger = logging.getLogger(__name__)

try:
    from src.model import GatedPINN, physics_loss
except ModuleNotFoundError:
    from model import GatedPINN, physics_loss

def train_pinn(model, dataloader, epochs=50, learning_rate=1e-3, device=None):
    """
    Training loop for Project Zero GatedPINN.
    
    Parameters
    ----------
    model : GatedPINN
        PINN model to train
    dataloader : iterable
        Training data loader
    epochs : int
        Number of training epochs
    learning_rate : float
        Adam optimizer learning rate
    device : torch.device, optional
        Device to train on (default: GPU if available, else CPU)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    model.train()
    logger.info(f"Starting training: {epochs} epochs, lr={learning_rate}, device={device}")
    
    for epoch in range(epochs):
        total_epoch_loss = 0
        
        for batch in dataloader:
            # Unpack batch - assuming structure based on usage
            # inputs: [Batch, 9] (features)
            # t: [Batch, 1] (time query)
            # sgp4_pos: [Batch, 3] (SGP4 baseline position)
            # ground_truth_pos: [Batch, 3] (Actual position)
            # bstar: [Batch, 1] (Drag coefficient)
            inputs, t, sgp4_pos, sgp4_vel, ground_truth_pos, bstar = batch
            
            # Ensure t requires grad for physics loss (autograd)
            t.requires_grad = True
            
            optimizer.zero_grad()
            
            # Forward Pass - Model returns correction * gate internally now?
            # User snippet: delta_r, delta_v = model(inputs) -> apply gate manually?
            # In model.py, we implemented: return out * gate
            # So model(inputs, t) returns the GATED correction.
            
            # Wait, model.py signature is forward(x, t).
            # The user snippet implies manual gating in the loop: "Apply the Physics Gate (tanh(5t))"
            # But the model.py I modified DOES apply the gate inside forward.
            # I should align them. If model applies it, I don't need to apply it again.
            # However, user explicitly requested: "You must apply the tanh(5t) gate here..."
            # AND I already put it in model.py.
            # If I put it in model.py, it's safer.
            # I will use the model's output as the gated correction.
            
            # The model returns (delta_r, delta_v) tuple
            delta_r, delta_v = model(inputs, t)
            
            # Corrected State
            # sgp4_pos is likely shape [Batch, 3], delta_r is [Batch, 3]
            corrected_pos = sgp4_pos + delta_r
            corrected_vel = sgp4_vel + delta_v
            
            # Physics Gate Note:
            # Since check is inside model.forward(), delta_r is already 0 at t=0.
            # So corrected_pos = sgp4_pos + 0 = sgp4_pos at t=0. Correct.
            
            # Calculate Compound Loss
            # 1. Data Loss (MSE against Ground Truth)
            l_data = criterion(corrected_pos, ground_truth_pos)
            
            # 2. Physics Loss (J2 + Drag Consistency)
            # We need to pass 't' to physics_loss for autograd if we want d(vel)/dt
            # But here 'corrected_vel' is what we want to differentiate w.r.t 't'.
            # corrected_vel = sgp4_vel + delta_v.
            # delta_v comes from model(inputs, t).
            # So corrected_vel depends on 't'.
            l_phys = physics_loss(corrected_pos, corrected_vel, t, bstar)
            
            # Weighted Sum
            total_loss = l_data + 0.1 * l_phys
            
            total_loss.backward()
            optimizer.step()
            
            total_epoch_loss += total_loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_epoch_loss / len(dataloader):.6f}")

    print("Training Complete.")
    return model

if __name__ == "__main__":
    # Mock Data for standalone verification
    print("Running Training Loop Verification (Dry Run)...")
    
    # Mock Model
    model = GatedPINN()
    
    # Mock Batch: 
    # inputs: [Batch, 9], t: [Batch, 1], sgp4_pos: [Batch, 3], ...
    batch_size = 2
    inputs = torch.randn(batch_size, 9)
    t = torch.tensor([[0.1], [0.5]], requires_grad=True)
    sgp4_pos = torch.randn(batch_size, 3)
    sgp4_vel = torch.randn(batch_size, 3) # Added for complete state
    ground_truth_pos = torch.randn(batch_size, 3)
    bstar = torch.abs(torch.randn(batch_size, 1)) * 1e-4
    
    # Simple Dataloader-like list
    mock_loader = [(inputs, t, sgp4_pos, sgp4_vel, ground_truth_pos, bstar)]
    
    try:
        train_pinn(model, mock_loader, epochs=2)
        print("Dry Run Successful.")
    except Exception as e:
        print(f"Dry Run Failed: {e}")
        import traceback
        traceback.print_exc()
