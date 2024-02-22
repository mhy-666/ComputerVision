import functools
from autograd import elementwise_grad as gradient
import autograd.numpy as np
from matplotlib import pyplot as plt


class CallCounter:
    def __init__(self, f):
        functools.update_wrapper(self, f)
        self.f = f
        self.count = 0

    def __call__(self, *args, **kwargs):
        values = self.f(*args, **kwargs)
        self.count += values.size
        return values

    def calls(self):
        return self.count

    def reset(self):
        self.count = 0


def batch_index_generator(n_samples, batch_size):
    rg = np.random.default_rng(15)
    batch = rg.permutation(n_samples)
    start, stop = 0, batch_size
    while stop < n_samples:
        yield batch[start:stop]
        start += batch_size
        stop += batch_size
    stop = min(stop, n_samples)
    yield batch[start:stop]


def sgd(fct, grad, step, samples, z, batch_size, alpha, counter, budget, min_step=1.e-5):
    n_samples = samples.shape[0]
    assert batch_size <= n_samples, 'mini-batch size cannot exceed batch size'

    done = False
    while not done:
        z_before_epoch = z
        for batch_indices in batch_index_generator(n_samples, batch_size):
            batch = samples[batch_indices, :]
            z, f_batch = step(fct, grad, batch, z, alpha)
            call_count = counter()
            if call_count >= budget:
                print('budget of {} scalar calls exceeded'.format(budget))
                done = True
                break
        if np.linalg.norm(z - z_before_epoch) < min_step:
            break
    return z


def plot_fit(dots, f=None, title=None, sampling=100):
    px, py = dots[::sampling, 0], dots[::sampling, 1]
    plt.figure(figsize=(4, 6), tight_layout=True)
    plt.plot(px, py, '.', ms=5)
    if f is not None:
        x_range = (np.min(dots[:, 0]), np.max(dots[:, 0]))
        x = np.linspace(x_range[0], x_range[1], 2)
        y = f(x)
        plt.plot(x, y)
    plt.axis('equal')
    plt.axhline(0, c='gray', lw=0.5)
    plt.axvline(0, c='gray', lw=0.5)
    if title is None:
        pct = 100 / sampling
        plt.title('{:g} percent of all points'.format(pct))
    else:
        plt.title(title)
    plt.draw()


def step(fct, grad, samples, z, alpha):
    z_prime = z - alpha * grad(z, samples)
    z_prime = z_prime / np.sqrt(np.sum(z_prime[1:] ** 2))  # projection to unit sphere
    f_prime = fct(z_prime, samples)
    return z_prime, f_prime


def run_sgd(losses, xs, batch, rate, budget=int(3e6), tau=1.):
    def risk(z, x):
        return np.mean(losses(z, x, tau))

    risk_gradient = gradient(risk)
    losses.reset()
    z0 = np.array((0, 0, 1), dtype=float)
    z_hat = sgd(risk, risk_gradient, step, xs, z0, batch, rate, losses.calls, budget)
    evaluations = losses.calls()
    s2, s13 = np.sqrt(2), np.sqrt(13)
    true_lines = [np.array((s2, 1 / s2, -1 / s2)), np.array((2, 3, 2)) / s13]
    true_lines.extend([-ell for ell in true_lines])
    error = np.min([np.linalg.norm(z_hat - z) for z in true_lines])
    with np.printoptions(precision=3):
        plot_fit(xs, lambda x: (z_hat[0] - z_hat[1] * x) / z_hat[2],
                 title='line {}, error {:.3f}'.format(z_hat, error))
    fmt = 'batch size = {}, alpha = {}, evals = {}'
    with np.printoptions(precision=3):
        print(fmt.format(batch, rate, evaluations))
