def train(model, loader, epochs, lr=1e-2, loss_fn=nn.CrossEntropyLoss()):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()                              # dropout on, BN uses batch statistics
        for xb, yb in loader:
            opt.zero_grad()                        # gradients accumulate otherwise
            loss = loss_fn(model(xb), yb)          # raw logits in, class indices as targets
            loss.backward()
            opt.step()
    return model

@torch.no_grad()                                   # no graph, less memory
def evaluate(model, X, Y):
    model.eval()                                   # dropout off, BN uses running stats
    return (model(X).argmax(1) == Y).float().mean().item()
