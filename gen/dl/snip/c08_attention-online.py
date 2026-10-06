def blockwise_attention(Q, K, V, block=4):
    """Process keys/values in blocks, keeping a running max m, running
    normaliser l and running output O for each query (the FlashAttention idea)."""
    Tq, dk = Q.shape
    m = np.full(Tq, -np.inf); l = np.zeros(Tq); O = np.zeros((Tq, V.shape[1]))
    for s in range(0, K.shape[0], block):
        Sb = Q @ K[s:s + block].T / np.sqrt(dk)              # scores for this block only
        m_new = np.maximum(m, Sb.max(1))
        scale = np.exp(m - m_new)                            # rescale old partial sums
        P = np.exp(Sb - m_new[:, None])
        l = l * scale + P.sum(1)
        O = O * scale[:, None] + P @ V[s:s + block]
        m = m_new
    return O / l[:, None]
