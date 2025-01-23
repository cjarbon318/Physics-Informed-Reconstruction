import torch
import torch.optim as optim
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

def train_transformer_autoencoder(model, train_loader, num_epochs=10, learning_rate=0.01, accumulation_steps=1):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize mixed precision scaler
    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        print(f"Starting epoch {epoch + 1}/{num_epochs}...")
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device).float()

            # Enable mixed precision
            optimizer.zero_grad()
            with autocast():  # Enable mixed precision
                outputs = model(batch)
                loss = criterion(outputs, batch)

            # Scale the loss, backward pass and optimizer step
            scaler.scale(loss).backward()

            # Accumulate gradients over several batches if needed
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)  # Update weights
                scaler.update()  # Update the scaler to track the loss scaling

            # Accumulate loss for monitoring
            train_loss += loss.item()

            # Print progress for current batch
            if (batch_idx + 1) % 10 == 0:  # Print every 10 batches
                print(f"Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        # Print the average loss for the epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {train_loss / len(train_loader):.4f}")

    print("Training complete.")
    return model
