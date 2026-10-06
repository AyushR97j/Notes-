def clip_loss(img, txt, tau):
    logits = img @ txt.T / tau                     # (B, B): pair i with every caption j
    labels = torch.arange(len(img))                # the matching caption is on the diagonal
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
