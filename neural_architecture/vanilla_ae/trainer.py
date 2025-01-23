import mlflow
import mlflow.pytorch

def train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, x in enumerate(train_loader):
            x = x.to(device).float()
            outputs = model(x)
            loss = criterion(outputs, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
        mlflow.log_metric("avg_loss", avg_loss, step=epoch)

    return model
