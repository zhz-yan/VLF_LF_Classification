import torch

def train(model, train_loader, criterion, optimizer, device, num_epochs=50):

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch +1}/{num_epochs}, Loss: {running_loss /len(train_loader)}")

def test(model, test_loader):

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)

            y_prob = torch.softmax(outputs, dim=1)
            y_pred = torch.argmax(y_prob, dim=1)
            total += labels.size(0)
            correct += (y_pred == labels).sum().item()

    print(f'Accuracy: {100 * correct / total:.3f}%')
