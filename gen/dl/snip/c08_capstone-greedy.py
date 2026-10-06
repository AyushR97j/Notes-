@torch.no_grad()
def greedy_torch(model, src):
    mem, src_mask = model.encode(src)
    ys = torch.full((src.shape[0], 1), BOS)
    for _ in range(LMAX + 1):
        nxt = model.decode(ys, mem, src_mask)[:, -1].argmax(-1, keepdim=True)
        ys = torch.cat([ys, nxt], 1)
    return ys
