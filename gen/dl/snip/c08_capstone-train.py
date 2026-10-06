for step in range(STEPS):
    src, tgt = make_batch(64, g)
    logits = model(src, tgt[:, :-1])                       # teacher forcing: input is shifted target
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), tgt[:, 1:].reshape(-1), ignore_index=PAD)
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    losses.append(loss.item())
