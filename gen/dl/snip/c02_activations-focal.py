def focal(p_true, gamma=2.0, alpha=1.0):
    return -alpha * (1 - p_true) ** gamma * np.log(p_true)
