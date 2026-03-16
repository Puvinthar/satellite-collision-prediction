"""
Training loop for GatedPINN v3.1.2.

Matches the notebook v3.1.2 training configuration:
- Optimizer: Adam(lr=0.002)
- Scheduler: ReduceLROnPlateau(factor=0.5, patience=15)
- Loss: MSELoss (pure data-driven, no physics loss)
- Epochs: 300, Batch size: 256
"""

import torch
import torch.nn as nn
import torch.optim as optim
import logging

logger = logging.getLogger(__name__)

try:
    from src.model import GatedPINN
except ModuleNotFoundError:
    from model import GatedPINN

# Training hyperparameters (matching notebook v3.1.2)
EPOCHS = 300
BATCH_SIZE = 256
LR = 0.002


def train_pinn(model, dataloader, epochs=EPOCHS, learning_rate=LR, device=None):
    """
    Training loop for Project Zero GatedPINN v3.1.2.

    Matches the notebook training loop exactly:
    - Pure MSE loss (no physics loss component)
    - ReduceLROnPlateau scheduler (factor=0.5, patience=15)
    - Adam optimizer

    Parameters
    ----------
    model : GatedPINN
        PINN model to train
    dataloader : iterable
        DataLoader yielding (x, y, t) batches where:
            x: [B, 10] scaled input features
            y: [B, 6]  scaled target residuals
            t: [B, 1]  normalized time (dt_minutes / 1440)
    epochs : int
        Number of training epochs (default: 300)
    learning_rate : float
        Adam optimizer learning rate (default: 0.002)
    device : torch.device, optional
        Device to train on (default: GPU if available, else CPU)

    Returns
    -------
    model : GatedPINN
        Trained model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15
    )
    loss_fn = nn.MSELoss()

    model.train()
    logger.info(
        f"Starting training: {epochs} epochs, lr={learning_rate}, device={device}"
    )
    logger.info(
        f"Training on {len(dataloader.dataset)} samples | Batches: {len(dataloader)}"
    )

    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0.0

        for x, y, t in dataloader:
            x, y, t = x.to(device), y.to(device), t.to(device)

            # Forward: model returns [B, 6] gated correction
            pred = model(x, t)
            loss = loss_fn(pred, y)

            # Backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)

        # Step scheduler based on average epoch loss
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.6f} | LR: {current_lr:.1e}"
            )

    logger.info("Training Complete.")
    return model


if __name__ == "__main__":
    # Mock Data for standalone verification
    from torch.utils.data import DataLoader, TensorDataset

    print("Running Training Loop Verification (Dry Run)...")

    model = GatedPINN()

    # Mock batch: x=[B, 10], y=[B, 6], t=[B, 1]
    batch_size = 8
    x = torch.randn(batch_size, 10)
    y = torch.randn(batch_size, 6)
    t = torch.rand(batch_size, 1)

    dataset = TensorDataset(x, y, t)
    mock_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    try:
        train_pinn(model, mock_loader, epochs=2)
        print("Dry Run Successful.")
    except Exception as e:
        print(f"Dry Run Failed: {e}")
        import traceback

        traceback.print_exc()
