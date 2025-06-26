def optimize_portfolio(expected_returns, cov_matrix, risk_aversion=1.0):
    """
    Mean-variance optimization using cvxpy.
    Ensures portfolio weights sum to 1 and are non-negative.
    """
    n = len(expected_returns)
    w = cp.Variable(n)
    mu = expected_returns.values
    Sigma = cov_matrix.values

    # Objective: maximize return - risk penalty
    objective = cp.Maximize(mu @ w - risk_aversion * cp.quad_form(w, Sigma))

    # Constraints: weights sum to 1, weights ≥ 0 (no shorting)
    constraints = [cp.sum(w) == 1, w >= 0]
    # Solve
    prob = cp.Problem(objective, constraints)
    prob.solve()
    weights = w.value
    threshold = 1e-4
    weights[np.abs(weights) < threshold] = 0 # eliminate tiny negative weights like -1e-20
    weights = weights / weights.sum()  # renormalize to sum to 1
    return weights