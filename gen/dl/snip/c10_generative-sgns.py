def sgns_loss_grad(Win, Wout, c, o, negs):
    v = Win[c]                                         # centre word vector
    u_pos, u_neg = Wout[o], Wout[negs]                 # context and negative "output" vectors
    s_pos, s_neg = sig(u_pos @ v), sig(-u_neg @ v)
    loss = -np.log(s_pos) - np.log(s_neg).sum()
    g_v = -(1 - s_pos) * u_pos + ((1 - s_neg)[:, None] * u_neg).sum(0)
    g_pos = -(1 - s_pos) * v
    g_neg = (1 - s_neg)[:, None] * v[None, :]
    return loss, g_v, g_pos, g_neg
