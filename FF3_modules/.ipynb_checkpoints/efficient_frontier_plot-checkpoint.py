def plot_efficient_frontier(mu, cov, points=50):
    n = len(mu)
    risks, rets = [], []

    for gamma in np.logspace(-1, 3, points):
        w = cp.Variable(n)
        objective = cp.Maximize(mu @ w - gamma * cp.quad_form(w, cov))
        constraints = [cp.sum(w) == 1, w >= 0]
        cp.Problem(objective, constraints).solve()
        w_val = w.value
        risks.append(np.sqrt(w_val.T @ cov @ w_val))
        rets.append(mu @ w_val)

    plt.figure(figsize=(8, 5))
    plt.plot(risks, rets, marker='o')
    plt.title("Efficient Frontier")
    plt.xlabel("Volatility (Risk)")
    plt.ylabel("Expected Return")
    plt.grid(True)
    plt.show()