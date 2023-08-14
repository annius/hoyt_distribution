import numpy as np
from scipy.special import gammainc


def cumulative_density_hoyt(s1, s2, x, tol=1e-6):
    """
    Calculates the cumulative density of a Hoyt distribution (Nakagami-q distribution).

    The cumulative density function is used to determine the probability that a random variable is lesser than a threshold.

    The Hoyt distribution models the probability of r = sqrt(x^2 + y^2) where x and y are normally distributed with zero mean and standard deviations s1 and s2.
    Note that correlated 2D distributions with non-zero mean can be transformed into the form expected by this function.
    Since r is invariant to rotations, a correlated distribution can be rotated into a frame along the principal axes.

    Therefore this function is used to determine the following:

    prob(r <= t | s1, s2) = cdf(t, s1, s2).

    A numerical method to calculate the cumulative density of a random variable which follows the Hoyt distribution is described in [1].

    References:

    [1] Tavares, G. N., "Efficient Computation of the Hoyt Cumulative Distribution Function,"
        Electronic Letters, Vol. 46, No. 7, pp. 537--539, April 2010.
        DOI: 10.1049/el.2010.0189
    [2] Espinoza, P. R., "Analysis of Gaussian Quadratic Forms with Application to Statistical Channel Modeling,"
        Ph.D Dissertation, Universidad de Málaga, 2019.

    
    Args:
        s1: The first standard deviation.
        s2: The second standard deviation.
        x: Random variable.
        tol: Tolerance used for quadrature.

    Returns:
        Pair of values, where:
        * The first value is the cumulative density of the Hoyt distribution at x.
        * The second value is the number of terms required to meet the tolerance.
    
    """
    if s1 <= 0:
        raise ValueError("s1 must be greater than 0")
    if s2 <= 0:
        raise ValueError("s2 must be greater than 0")
    if np.any(x) < 0:
        raise ValueError("x must be greater than or equal to 0")
    if tol <= 0:
        raise ValueError("tol must be greater than 0")

    # omega is the mean of the Hoyt distribution. 
    # The mean of the root of the quadratic form is given by the sum of the variances.
    # See https://en.wikipedia.org/wiki/Quadratic_form_%28statistics%29#Expectation.
    omega = s1 * s1 + s2 * s2

    # The variable q is the Hoyt/Nakagami-q fading parameter defined in Eq. (1) of [1].
    # In Eq. (2.86) of [2] it is shown that q = sqrt(eta), where eta is the ratio of the smaller and larger variances.
    q = np.min([s1, s2]) / np.max([s1, s2])
    # Consequently, q2 is the square of q2.
    q2 = q * q

    # Eq. (6) of [1] can be used to determine how many terms are needed to reduce to the quadrature error to the specified tolerance.
    n_terms = int(np.ceil(np.log(2/tol - 1) / np.log((1+q)/(1-q)) / 2)) 

    cdfx = np.ones_like(x)
    for k in range(n_terms):
        # a, b, den are used to implement Eq. (5) in [1].
        den = (1 + (1 - q2) / (1 + q2) * np.cos(np.pi * (2 * k + 1) / 2 / n_terms))
        a = 2 * q / (n_terms) / (1 + q2) / den
        b = (1 + q2)**2/4/q2 * den / omega    
        cdfx -=  a * np.exp( -b * x**2)

    return cdfx, n_terms



def cumulative_density_hoyt_series(s1, s2, x, tol=1e-6):
    """
    Calculates the cumulative density of a Hoyt distribution (Nakagami-q distribution).

    The cumulative density function is used to determine the probability that a random variable is lesser than a threshold.

    The Hoyt distribution models the probability of r = sqrt(x^2 + y^2) where x and y are normally distributed with zero mean and standard deviations s1 and s2.
    Note that correlated 2D distributions with non-zero mean can be transformed into the form expected by this function.
    Since r is invariant to rotations, a correlated distribution can be rotated into a frame along the principal axes.

    Therefore this function is used to determine the following:

    prob(r <= t | s1, s2) = cdf(t, s1, s2).

    This implementation uses a 2D specialization of a more general series solution for quadratic forms on Gaussian random variables [1].
    Eq. (9) in [1] develops an infinite series solution which uses the incomplete gamma function.
    In the implementation presented in this function, recursive relationships are used in order to generate the coefficient of the incomplete gamma function.
      
    [1] Ropokis, G., Rontogiannis, A., and Mathiopoulos, P., "Quadratic Forms in Normal RVs: Theory and Applications to OSTBC Over Hoyt Fading Channels,"
        IEEE Transactions on Wireless Communications, Vol. 7, No. 12, pp. 5009--5019, December 2008.
        DOI: 10.1109/t-wc.2008.070830 

    Args:
        s1: The first standard deviation.
        s2: The second standard deviation.
        x: Random variable.
        tol: Tolerance used for quadrature.

    Returns:
        Pair of values, where:
        * The first value is the cumulative density of the Hoyt distribution at x.
        * The second value is the number of terms required to meet the tolerance.    
    """
    # Eigenvalues of a 2D uncorrelated distribution are the variances.
    # Without loss of generality it is assumed that l1 is the larger of the two eigenvalues, and l2 is the smaller of the two eigenvalues.
    l1 = np.max([s1, s2])**2
    l2 = np.min([s1, s2])**2

    # Eq. (10)
    beta = 2 * l1 * l2 / (l1 + l2)

    # The series solution uses two coefficients, d[k] and c[k].
    # It can be shown that 
    # d[k] = 0 when k is odd,
    # d[k] = 2 * eps^k when k is even, where 
    # eps = 1 - beta/l1 = -(1 - beta/l2) defined in Eq. (12).
    eps = (l1 - l2) / (l1 + l2)
    eps2 = eps * eps
    # Following this, we can use Eq. (11) to develop the following recurring relationship for c[k]:
    # (note, c[k] = 0 when k is odd)
    # c[k+1] = eps2 * (2 * k - 1) / (2 * k) * c[k]
    # c[0] = beta / sqrt(l1 * l2), see Eq. (12).
    ck = 2 * s1 * s2 / (s1**2 + s2**2)

    # Convenience variable.
    xp = x**2 / (2 * beta)

    # Start the iteration; the zeroth term is given by:
    #
    # cdf[0] = c[0] * gammainc(1, xp)
    #        = c[0] * (1 - exp(-xp))
    #
    # Note that gammainc here is the definition followed in scipy.special.gammainc, i.e. it is scaled by the Gamma function, Eq. (9).
    # The kth term of the CDF is given by
    #
    # cdf[k] = c[k] * gammainc(2 * k + 1, xp)
    #
    # It can be shown that
    #
    # gammainc(2 * k + 1, xp) = gammainc(2 * k - 1, xp) - xp^(2 * k - 1) / factorial(2 * k - 1) * exp(-xp) * (xp / (2 * k) + 1)
    #
    # However, this recurrence relation is numerically unstable because xp^(2 * k - 1) for large k can result in overflow errors.
    k = 1
    termk = gammainc(1, xp)
    cdf = ck * termk
    max_err = np.inf
    while max_err > tol:
        ck = eps2 * (2 * k - 1) / (2 * k) * ck
        termk = gammainc(2 * k + 1, xp)
        correction = ck * termk
        max_err = np.max(np.abs(correction))
        cdf += correction
        k += 1
    return cdf, k


def inverse_tail_probability_hoyt(s1, s2, probability, tol=1e-6, n_max=None, log=False):
    """
    Calculates the inverse of the tail probability of a Hoyt distribution (Nakagami-q distribution).
    Given a random variable x, the tail probability for a threshold t is given by
    
    prob(x > t) = tail_probability.
    
    This is equal to

    prob(x <= t) = 1 - tail_probability,

    or,

    cdf(t) = 1 - tail_probability.

    For a given tail probability, the threshold t can be obtained from the inverse of the cumulative density function of the random variable:
    t = cdf^{-1}(1 - tail_probability).

    The Hoyt distribution models the probability of r = sqrt(x^2 + y^2) where x and y are normally distributed with zero mean and standard deviations s1 and s2.
    Note that correlated 2D distributions with non-zero mean can be transformed into the form expected by this function.
    Since r is invariant to rotations, a correlated distribution can be rotated into a frame along the principal axes.

    A numerical method to calculate the cumulative density of a random variable which follows the Hoyt distribution is described in [1].
    This function wraps the numerical method in a Newton-Raphson iterative implementation in order to calculate the inverse of the cumulative density.

    References:

    [1] Tavares, G. N., "Efficient Computation of the Hoyt Cumulative Distribution Function,"
        Electronic Letters, Vol. 46, No. 7, pp. 537--539, April 2010.
        DOI: 10.1049/el.2010.0189
    [2] Espinoza, P. R., "Analysis of Gaussian Quadratic Forms with Application to Statistical Channel Modeling,"
        Ph.D Dissertation, Universidad de Málaga, 2019.

    
    Args:
        s1: The first standard deviation.
        s2: The second standard deviation.
        probability: Tail probability value (e.g. 1e-6)
        tol: Tolerance used for quadrature and Newton Raphson (note: Tolerance must be lesser than tail probability).
        n_max: Maximum number of iterations permitted for Newton-Raphson. Warning: setting this to too small a value may result in tolerance violation.
        log: True if iterated values are to printing, False for silent output.

    Returns:
        Pair of values, where:
        * The first value is threshold over which the tail density is equal the the user-specified probability
        * The second value is the number of iterations required to converge to the computed threshold.
    
    """
    if s1 <= 0:
        raise ValueError("s1 must be greater than 0")
    if s2 <= 0:
        raise ValueError("s2 must be greater than 0")
    if probability <= 0:
        raise ValueError("probability must be greater than 0")
    if probability >= 1:
        raise ValueError("probability must be lesser than 1")
    if tol <= 0:
        raise ValueError("tol must be greater than 0")
    if n_max is None:
        n_max = np.iinfo(np.uint64).max
    if n_max <= 0:
        raise ValueError("n_max must be greater than 0")
    if tol > probability:
        raise ValueError("tol must be lesser than probability")

    # omega is the mean of the Hoyt distribution. 
    # The mean of the root of the quadratic form is given by the sum of the variances.
    # See https://en.wikipedia.org/wiki/Quadratic_form_%28statistics%29#Expectation.
    omega = s1 * s1 + s2 * s2

    # The variable q is the Hoyt/Nakagami-q fading parameter defined in Eq. (1) of [1].
    # In Eq. (2.86) of [2] it is shown that q = sqrt(eta), where eta is the ratio of the smaller and larger variances.
    q = np.min([s1, s2]) / np.max([s1, s2])
    # Consequently, q2 is the square of q2.
    q2 = q * q

    # Eq. (6) of [1] can be used to determine how many terms are needed to reduce to the quadrature error to the specified tolerance.
    n_terms = int(np.ceil(np.log(2/tol - 1) / np.log((1+q)/(1-q)) / 2)) 

    # An initial guess for the threshold beyond which the probability is less than the specified value can be obtained from the cumulative density function of the Rayleigh distribution.
    # See https://en.wikipedia.org/wiki/Rayleigh_distribution.
    xk = np.sqrt(-2.0 * np.log(probability)) * np.max([s1, s2])

    # Use Newton-Raphson to calculate a correction to the initial guess.
    dx = np.inf
    ctr = 0
    while np.abs(dx) > tol and ctr <= n_max:
        if log:
            print(f"Iteration {ctr:03d}, Threshold = {xk:.10f}")
        sx = 0
        dsdx = 0
        for k in range(n_terms):
            # a, b, den are used to implement Eq. (5) in [1].
            den = (1 + (1 - q2) / (1 + q2) * np.cos(np.pi * (2 * k + 1) / 2 / n_terms))
            a = 2 * q / (n_terms) / (1 + q2) / den
            b = (1 + q2)**2/4/q2 * den / omega    
            sx +=  a * np.exp( -b * xk**2)
            dsdx += -2 * a * b * np.exp(-b * xk**2) * xk
        dx = (probability - sx) / dsdx
        xk += dx
        ctr += 1
    q_inv = xk

    if ctr >= n_max and log:
        print(f"Maximum number of iterations ({ctr}) exceed the specified value ({n_max}).")

    return q_inv, ctr



