import numpy as np
from scipy.signal import convolve2d as conv
from scipy.interpolate import RectBivariateSpline as interpolate
from ideal import show_image_pair


def gauss(h_win):
    win_size, sigma = 2 * h_win + 1, h_win / 2.5
    v = np.linspace(-h_win, h_win, win_size)
    g = np.exp(- np.power(v, 2) / np.power(sigma, 2))
    ramp = -v
    d = ramp * g
    return g / np.sum(g), d / np.dot(ramp, d), v


def row(vector):
    assert vector.ndim == 1, 'Input must be a 1-dimensional array'
    return np.expand_dims(vector, 0)


def column(vector):
    assert vector.ndim == 1, 'Input must be a 1-dimensional array'
    return np.expand_dims(vector, 1)


def gradient(img, h_win):
    g, d, _ = gauss(h_win)
    gr = conv(conv(img, row(g), 'same'), column(d), 'same')
    gc = conv(conv(img, row(d), 'same'), column(g), 'same')
    return gr, gc


# Default parameters for window selection and tracking
default_parameters = {
    'gaussian window size': 5,
    'feature window size': 31,
    'smallest acceptable step': 0.001,
    'maximum number of iterations': 100
}


def outside(p, shape, parms):
    margin = parms['gaussian window size'] + parms['feature window size']
    out = p[0] < margin or p[0] > shape[0] - margin
    return out or p[1] < margin or p[1] > shape[1] - margin


def lucas_kanade(f, g, p, parms):

    p = np.flip(p)
    
    def window(spline, p, h_gauss_win, h_feature_win):
        def crop(win, tail):
            return win[tail:-tail, tail:-tail]
        tail = h_gauss_win + h_feature_win
        win_size = 2 * tail + 1
        rows = np.linspace(p[0] - tail, p[0] + tail, win_size)
        cols = np.linspace(p[1] - tail, p[1] + tail, win_size)
        win = spline(rows, cols).astype(np.float32)
        gr, gc = gradient(win, h_gauss_win)
        gr, gc = crop(gr, h_gauss_win), crop(gc, h_gauss_win)
        win = crop(win, h_gauss_win)
        return win, gr, gc
        
    if outside(p, f.shape, parms):
        return None
            
    h_gauss_win, h_feature_win = parms['gaussian window size'], parms['feature window size']
    w, _, _ = gauss(h_feature_win)
    w = column(w) * row(w)

    rows = np.arange(f.shape[0], dtype='float')
    cols = np.arange(f.shape[1], dtype='float')
    f_spline = interpolate(rows, cols, f, kx=1, ky=1)
    g_spline = interpolate(rows, cols, g, kx=1, ky=1)
    
    f_win, _, _ = window(f_spline, p, h_gauss_win, h_feature_win)
    
    a, b = np.zeros((2, 2)), np.zeros(2)
    done, lost, iteration = False, False, 0
    q = p.copy()

    while not done and not lost:
        g_win, g_win_gr, g_win_gc = window(g_spline, q, h_gauss_win, h_feature_win)
        a[0][0] = np.sum(g_win_gr * g_win_gr * w)
        a[0][1] = np.sum(g_win_gr * g_win_gc * w)
        a[1][1] = np.sum(g_win_gc * g_win_gc * w)
        a[1][0] = a[0][1]
        diff = f_win - g_win
        b[0] = np.sum(g_win_gr * diff * w)
        b[1] = np.sum(g_win_gc * diff * w)
        s, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
        q += s
        iteration += 1
        
        done = np.linalg.norm(s) < parms['smallest acceptable step']
        lost = np.linalg.norm(q - p) > h_feature_win
        lost = lost or iteration > parms['maximum number of iterations']
        lost = lost or outside(q, g.shape, parms)
        
    if lost:
        q = None
    else:
        q = np.flip(q)
    
    return q


def show_parameters(parameters, which=''):
    print('Lucas-Kanade tracker {}parameters:'.format(which))
    for name, value in parameters.items():
        print('\t{}: {}'.format(name, value))


def track_and_compare(f, g, pts, true_displacement):
    print()
    qs = []
    parameters = default_parameters.copy()
    for k, (p, win) in enumerate(pts):
        parameters['feature window size'] = win
        pt = np.array(p).astype(float)
        q = lucas_kanade(f, g, pt, parameters)
        qs.append(q)
        if q is None:
            print('tracking of point {}: {} with window size {} failed'.format(k, p, win))
        else:
            computed_d = q - p
            true_d = true_displacement[p[1], p[0], :]
            epe = np.linalg.norm(computed_d - true_d)
            with np.printoptions(precision=3):
                print('point {}: {} -> {}, window size {}'.format(k, p, q, win))
                print('\tcomputed displacement: {}'.format(computed_d))
                print('\ttrue displacement: {}'.format(true_d))
                print('\tend-point error: {:.3f} pixels'.format(epe))
    show_image_pair(f, g, points_and_wins_0=pts, points1=qs)
