from autograd import numpy as np
from autograd import elementwise_grad as gradient
from autograd import make_jvp
from torchvision.utils import flow_to_image
import torch
from matplotlib import pyplot as plt
import pickle


small = 1.e-8


def choose_rows(array, condition, if_true, if_false):
    assert array.ndim == 2, 'array must be two-dimensional'
    rep = np.tile(condition, (array.shape[1], 1)).T
    result = np.choose(rep, (if_false, if_true))
    return result


def is_positive_float(x):
    return isinstance(x, float) and x > 0.


def is_point(p):
    return p.shape == (3,)


def is_rotation_matrix(r):
    if r.shape != (3, 3):
        return False
    if np.linalg.norm(np.dot(r.T, r) - np.eye(3)) > small:
        return False
    if np.linalg.det(r) < 0.:
        return False
    return True


# A single transformation for all the points
def transform(p, t, r):
    d = p - t if p.ndim == 1 else p - t[None, :]
    return np.dot(d, r.T)


# One transformation per point
def multi_transform(p, t, r):
    d = p - t
    rt = np.transpose(r, [0, 2, 1])
    return np.sum(np.stack((d, d, d), axis=2) * rt, axis=1)


def inverse(t, r):
    return - np.dot(r, t), r.T


def distances(a, b):
    differences = a - b
    d = np.linalg.norm(differences, axis=1)
    return d


def texture_1d(x, parameters):
    t = np.zeros_like(x)
    for amplitude, frequency, phase in parameters:
        t += amplitude * (1. + np.cos(2. * np.pi * (frequency * x + phase)))
    return t


def texture(x, y, parameters):
    t = texture_1d(x, parameters['x']) + texture_1d(y, parameters['y'])
    b = parameters['base']
    t = (t + b) / (1. + b)
    return t


class Points:
    def __init__(self, points=None, indices=None, colors=None):
        if points is None:
            assert indices is None, 'cannot specify indices without points'
            assert colors is None, 'cannot specify colors without points'
            self.points = np.empty((0, 3))
            self.colors = np.empty(0)
            self.indices = np.empty(0, dtype=int)
        else:
            assert points.ndim == 2 and points.shape[1] == 3, 'need an n by 3 array'
            self.points = points
            n = points.shape[0]
            if colors is None:
                self.colors = np.full(n, np.nan)
            else:
                assert colors.shape[0] == n, 'number of colors must match number of points'
                self.colors = colors
            if indices is None:
                self.indices = np.arange(n)
            else:
                assert indices.shape[0] == n, 'number of indices must match number of points'
                self.indices = indices

    def transform(self, t, r):
        self.points = transform(self.points, t, r)

    def retain(self, predicate):
        self.points = self.points[predicate, :]
        self.colors = self.colors[predicate]
        self.indices = self.indices[predicate]

    def add(self, new_points, new_indices, new_colors=None):
        assert np.sum(np.intersect1d(self.indices, new_indices)).astype(int) == 0,\
            'cannot join point sets if new and old indices have common elements'
        indices = np.concatenate((self.indices, new_indices))
        order = np.argsort(indices)
        self.indices = indices[order]

        points = np.concatenate((self.points, new_points))
        self.points = points[order, :]

        if new_colors is not None:
            assert new_points.shape[0] == new_colors.shape[0],\
                'need as many colors as there are points'
            colors = np.concatenate((self.colors, new_colors))
            self.colors = colors[order]

    def replace(self, predicate, new_points, new_colors=None):
        old_indices = self.indices[~predicate]
        new_indices = self.indices[predicate]
        indices = np.concatenate((old_indices, new_indices))
        order = np.argsort(indices)
        self.indices = indices[order]

        old_points = self.points[~predicate, :]
        points = np.concatenate((old_points, new_points))
        self.points = points[order, :]

        if new_colors is not None:
            assert new_points.shape[0] == new_colors.shape[0],\
                'need as many colors as there are points'
            old_colors = self.colors[~predicate]
            colors = np.concatenate((old_colors, new_colors))
            self.colors = colors[order]

    def modify_z(self, values):
        p = self.points
        values = np.zeros(p.shape[0]) + values  # Broadcast
        self.points = np.column_stack((p[:, 0], p[:, 1], values))


class Disk:
    def __init__(self, radius, pattern, lift=0.):
        assert is_positive_float(radius), 'radius must be positive'
        self.radius = radius
        self.lift = lift
        self.alpha = lambda x: (x + radius) / (2 * radius)
        self.texture = lambda p: texture(self.alpha(p[:, 0]), self.alpha(p[:, 1]), pattern)

    def intersections(self, rays, tips):
        tips = np.column_stack((tips[:, 0], tips[:, 1], tips[:, 2] - self.lift))
        tips_z, rays_z = tips[:, 2], rays[:, 2]
        horizontal = rays_z == 0.
        tilted = ~horizontal
        rays, rays_z = rays[tilted, :], rays_z[tilted]
        tips, tips_z = tips[tilted, :], tips_z[tilted]
        alphas = -tips_z / rays_z
        p = Points(tips + alphas[:, None] * rays)
        # Just to clean up numerical noise
        p.modify_z(0.)
        norms = np.linalg.norm(p.points, axis=1)
        inside = (alphas > 0) & (norms <= self.radius)
        p.retain(inside)
        p.modify_z(p.points[:, 2] + self.lift)
        p.colors = self.texture(p.points)
        return p


class Ring:
    def __init__(self, major_radius, minor_radius, pattern, lift=0.):
        assert minor_radius < major_radius, 'minor radius muse be smaller than major radius'
        self.outer = Disk(major_radius, pattern, lift=lift)
        self.inner = Disk(minor_radius, pattern, lift=lift)

    def intersections(self, rays, tips):
        points = self.outer.intersections(rays, tips)
        inner_points = self.inner.intersections(rays, tips)
        points.retain(~np.isin(points.indices, inner_points.indices))
        return points


class CylinderSide:
    def __init__(self, radius, height, pattern, lift=0.):
        assert is_positive_float(radius), 'radius must be positive'
        assert is_positive_float(height), 'height must be positive'
        self.radius = radius
        self.height = height
        self.lift = lift
        self.alpha = lambda x, y: (np.arctan2(x, y) + np.pi) / (2. * np.pi)
        self.beta = lambda z: z / height
        self.texture = lambda p: texture(self.alpha(p[:, 0], p[:, 1]), self.beta(p[:, 2]), pattern)

    def intersections(self, rays, tips):
        tips = np.column_stack((tips[:, 0], tips[:, 1], tips[:, 2] - self.lift))
        tips_xy, rays_xy = tips[:, :2], rays[:, :2]
        a = np.sum(rays_xy ** 2, axis=1)
        b = np.sum(tips_xy * rays_xy, axis=1)
        c = np.sum(tips_xy ** 2, axis=1) - self.radius ** 2
        delta_2 = b ** 2 - a * c
        near = (a > 0.) & (delta_2 >= 0.)
        delta = np.sqrt(delta_2[near])
        alphas = np.column_stack(((-b[near] - delta) / a[near],
                                  (-b[near] + delta) / a[near]))
        p = tips[near, :] + alphas[:, 0][:, None] * rays[near, :]
        q = tips[near, :] + alphas[:, 1][:, None] * rays[near, :]
        p_cuts = (alphas[:, 0] > 0.) & (0 <= p[:, 2]) & (p[:, 2] <= self.height)
        q_cuts = (alphas[:, 1] > 0.) & (0 <= q[:, 2]) & (q[:, 2] <= self.height)
        zero, one = p_cuts, q_cuts & ~p_cuts
        alpha = np.concatenate((alphas[zero, 0], alphas[one, 1]))
        near_indices = np.argwhere(near).flatten()
        zero_indices = near_indices[zero]
        one_indices = near_indices[one]
        used_indices = np.concatenate((zero_indices, one_indices))
        order = np.argsort(used_indices)
        used_indices = used_indices[order]
        alpha = alpha[order]
        p = tips[used_indices, :] + alpha[:, None] * rays[used_indices, :]
        points = Points(p, indices=used_indices)
        points.modify_z(p[:, 2] + self.lift)
        points.colors = self.texture(p)
        return points


class Body:
    def __init__(self, parts, base_centroid, translation=None, rotation=None):
        self.parts = parts
        if translation is None:
            translation = np.zeros(3)
        if rotation is None:
            rotation = np.eye(3)
        self.translation = translation
        self.rotation = rotation
        self.base_centroid = base_centroid

    def intersections(self, rays, tips):
        t, r = inverse(self.translation, self.rotation)
        rays, tips = transform(rays, np.zeros(3), r), transform(tips, t, r)
        points = Points()
        for surface in self.parts:
            p = surface.intersections(rays, tips)
            if len(p.indices):
                if len(points.indices) == 0:
                    points.add(p.points, p.indices, new_colors=p.colors)
                else:
                    new_only = np.setdiff1d(p.indices, points.indices)
                    new_only_in_p = np.isin(p.indices, new_only)

                    both = np.intersect1d(p.indices, points.indices)
                    if len(both):
                        both_in_p = np.isin(p.indices, both)
                        both_in_points = np.isin(points.indices, both)
                        new_closer = both[distances(tips[both, :], p.points[both_in_p, :]) <
                                          distances(tips[both, :], points.points[both_in_points, :])]

                    if np.sum(new_only_in_p):
                        points.add(p.points[new_only_in_p, :], new_only, new_colors=p.colors[new_only_in_p])

                    if len(both):
                        new_closer_in_p = np.isin(p.indices, new_closer)
                        new_closer_in_points = np.isin(points.indices, new_closer)

                        points.replace(new_closer_in_points, p.points[new_closer_in_p, :], p.colors[new_closer_in_p])

        points.transform(self.translation, self.rotation)
        return points


def bolt_surfaces(lower_radius, lower_height, upper_radius, upper_height, patterns):
    parts = [
        Disk(lower_radius, patterns['bottom']),
        CylinderSide(lower_radius, lower_height, patterns['lower side']),
        Ring(lower_radius, upper_radius, patterns['ring'], lift=lower_height),
        CylinderSide(upper_radius, upper_height, patterns['upper side'], lift=lower_height),
        Disk(upper_radius, patterns['top'], lift=lower_height + upper_height)
    ]
    centroid = np.array([0., 0., (lower_height + upper_height) / 2.])
    return parts, centroid


def room_surfaces(radius, height, floor_level, patterns):
    parts = [
        Disk(radius, patterns['background'], lift=floor_level),
        CylinderSide(radius, height, patterns['background'], lift=floor_level),
        Disk(radius, patterns['background'], lift=floor_level + height)
    ]
    centroid = floor_level + height / 2.
    return parts, centroid


def camera_angle(t, round_trip_time, frame_rate, phase):
    return 2. * np.pi * (t / round_trip_time / frame_rate) + phase


def orbit_axes(tilt):
    return np.array([[1., 0., 0.], [0., np.cos(tilt), np.sin(tilt)]])


def camera_position(t, distance, orbit_tilt, round_trip_time, frame_rate, phase):
    theta = camera_angle(t, round_trip_time, frame_rate, phase)
    axes = orbit_axes(orbit_tilt)
    unit = axes[0] * np.cos(theta) + axes[1] * np.sin(theta)
    p = distance * unit
    return p


# Version of camera_position with one time per point, to make autograd.element_wise work
def camera_positions(t, distance, orbit_tilt, round_trip_time, frame_rate, phase):
    theta = camera_angle(t, round_trip_time, frame_rate, phase)
    axes = orbit_axes(orbit_tilt)
    unit = np.outer(np.cos(theta), axes[0]) + np.outer(np.sin(theta), axes[1])
    p = distance * unit
    return p


# Batch version of the cross product. The inputs a and b are n x 3 arrays,
# and the function computes an n x 3 array c c such that
# c[k] = np.cross(a[k], b[k]), but without  using for loops.
def cross_batch(a, b):
    return np.column_stack((
        a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
        a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
        a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    ))


def fixation_rotation(target, cam):
    k = target - cam
    k_norm = np.sqrt(np.sum(k * k))
    assert k_norm > 0., 'fixation point cannot coincide with viewpoint'
    k /= k_norm
    vertical = np.array([0., 0., 1.])
    i = np.cross(k, vertical)
    i_norm = np.sqrt(np.sum(i * i))
    if i_norm == 0:
        i = np.array([1., 0., 0.])
    else:
        i /= i_norm
    j = np.cross(k, i)
    r = np.row_stack((i, j, k))
    return r


# Version of fixation_rotation with one camera per point, to make autograd.elementwise_grad work
def fixation_rotations(target, cams):
    k = target - cams
    k_norms = np.sqrt(np.sum(k * k, axis=1))
    assert np.all(k_norms > 0.), 'fixation point cannot coincide with viewpoint'
    k /= k_norms[:, None]
    n = cams.shape[0]
    zero, one = np.zeros(n), np.ones(n)
    vertical = np.column_stack((zero, zero, one))
    i = cross_batch(k, vertical)
    i_norm = np.sqrt(np.sum(i * i, axis=1))
    nonzero = i_norm != 0.
    nonzero_indices = np.argwhere(nonzero).flatten()
    zero_indices = np.argwhere(~nonzero).flatten()
    i_nonzero = i[nonzero, :] / i_norm[nonzero, None]
    z = len(zero_indices)
    i_zero = np.column_stack((one[:z], zero[:z], zero[:z]))
    i = np.concatenate((i_nonzero, i_zero), axis=0)
    indices = np.concatenate((nonzero_indices, zero_indices))
    i = i[np.argsort(indices), :]
    j = cross_batch(k, i)
    r = np.stack((i, j, k), axis=1)
    return r


def transformation(t, view_parms):
    viewpoint = camera_position(t, view_parms['distance'],
                                view_parms['orbit tilt'],
                                view_parms['round trip time'],
                                view_parms['frame rate'],
                                view_parms['orbit phase'])
    orientation = fixation_rotation(view_parms['fixation point'], viewpoint)
    return viewpoint, orientation


def transformations(t, view_parms):
    viewpoints = camera_positions(t, view_parms['distance'],
                                  view_parms['orbit tilt'],
                                  view_parms['round trip time'],
                                  view_parms['frame rate'],
                                  view_parms['orbit phase'])
    orientations = fixation_rotations(view_parms['fixation point'], viewpoints)
    return viewpoints, orientations


class Camera:
    def __init__(self, cam_parms, view_parms):
        self.transformation = lambda t: transformation(t, view_parms)
        self.transformations = lambda t: transformations(t, view_parms)
        self.f = cam_parms['focal distance']
        self.fov = cam_parms['field of view']
        self.resolution = cam_parms['resolution']
        self.max_x = self.f * np.tan(self.fov / 2.)
        self.principal_point = (self.resolution - 1.) / 2. * np.ones(2)
        self.pixel_scale = (self.resolution - 1.) / (2. * self.max_x) if self.resolution > 1 else 1.

    def image_grid(self, canonical=False, x_range=None, y_range=None):
        if canonical:
            samples = np.linspace(-self.max_x, self.max_x, num=self.resolution)
        else:
            samples = np.arange(self.resolution, dtype=float)
        x_samples = samples if x_range is None else samples[(x_range[0] <= samples) & (samples <= x_range[1])]
        y_samples = samples if y_range is None else samples[(y_range[0] <= samples) & (samples <= y_range[1])]
        p = np.meshgrid(x_samples, y_samples)
        return np.stack(p, axis=2)

    def canonical_to_image(self, p):
        return self.pixel_scale * p + self.principal_point[None, :]

    def image_to_canonical(self, p):
        return (p - self.principal_point[None, :]) / self.pixel_scale

    # x (horizontal left to right) and y (vertical down) are in an infinite image.
    # The finite sub-image has top left corner at x = y = 0 and bottom right corner
    # at x = y = self.pixel - 1.
    def pixel_rays(self, image_points):
        c = self.image_to_canonical(image_points[:, :2])
        rays = np.column_stack([c[:, 0], c[:, 1], np.full(c.shape[0], self.f)])
        translations, rotations = self.transformations(image_points[:, 2])
        rays = np.sum(np.stack((rays, rays, rays), axis=2) * rotations, axis=1)
        return rays, translations

    def project(self, world_points, t, image_frame=True):
        # From object to camera coordinates
        translation, rotation = self.transformation(t)
        world_points = transform(world_points, translation, rotation)
        in_front = world_points[:, 2] > 0.
        x, y, z = world_points[in_front, 0], world_points[in_front, 1], world_points[in_front, 2]
        q = np.column_stack((self.f * x / z, self.f * y / z))
        if image_frame:
            q = self.canonical_to_image(q)
        return q

    def project_multi_time(self, world_points, ts, image_frame=True):
        # From object to camera coordinates
        translations, rotations = self.transformations(ts)
        world_points = multi_transform(world_points, translations, rotations)
        in_front = world_points[:, 2] > 0.
        x, y, z = world_points[in_front, 0], world_points[in_front, 1], world_points[in_front, 2]
        q = np.column_stack((self.f * x / z, self.f * y / z))
        if image_frame:
            q = self.canonical_to_image(q)
        return q


def image_points_and_times(grid, t, cam):
    if grid is None:
        grid = cam.image_grid()
    shape = grid.shape[:-1]
    k = np.prod(shape)
    t_vector = t * np.ones((k, 1))
    points = np.concatenate((np.reshape(grid, (k, 2)), t_vector), axis=1)
    return points, shape


# Assumes that the scene is closed and the camera is inside it
def make_image(t, cam, scene, grid=None, grad=False):

    def values(image_points):
        rays, tips = cam.pixel_rays(image_points)
        world_points = scene.intersections(rays, tips)
        return world_points.colors

    points, shape = image_points_and_times(grid, t, cam)
    colors = values(points)
    colors = np.reshape(colors, shape)
    if grad:
        g_colors = gradient(values)(points)
        g_colors = np.reshape(g_colors, (*shape, 3))
        return colors, g_colors
    else:
        return colors


def pixel_value(x, y, t, cam, scene, grad=False):
    grid = np.array([[x, y]])
    v, g = make_image(t, cam, scene, grid=grid, grad=grad)
    return v, g if grad else v


def motion_field(t, cam, scene, grid=None):
    image_points, shape = image_points_and_times(grid, t, cam)
    rays, tips = cam.pixel_rays(image_points)
    world_points = scene.intersections(rays, tips).points

    def projections(time):
        return cam.project(world_points, time)

    # Forward-mode Jacobian computation autograd.maka_jvp is much faster than
    # reverse-mode computation autograd.jacobian for this R -> R^m function
    field = make_jvp(projections)(t)(np.array(1))[1]
    return np.reshape(field, (*shape, 2))


def displacement(t0, t1, cam, scene, grid=None):
    i0, shape = image_points_and_times(grid, t0, cam)
    rays, tips = cam.pixel_rays(i0)
    world_points = scene.intersections(rays, tips).points
    i1 = cam.project(world_points, t1)
    return np.reshape(i1 - i0[:, :2], (*shape, 2))


def stats(scalar, scalar_name=''):
    flat = scalar.flatten()
    s = {
        'minimum': np.min(flat),
        'maximum': np.max(flat),
        'mean absolute value': np.mean(np.abs(scalar)),
        'standard deviation': np.std(flat)
    }

    if len(scalar_name):
        print('Statistics of the {}:'.format(scalar_name))
    strings = ['{}: {:.4g}'.format(key, val) for key, val in s.items()]
    print(', '.join(strings))


def show_image_pair(image0, image1=None, name0='image 0', name1='image 1',
                    points_and_wins_0=None, points1=None):
    if points_and_wins_0 is not None:
        msg = 'points must be given in either both images or neither'
        assert points1 is not None, msg
        show_points = True
        pw = list(zip(*points_and_wins_0))
        points0, wins = pw[0], pw[1]
    else:
        show_points = False
        points0, wins = None, None
    plt.figure(figsize=(10, 5.5), tight_layout=True)
    for plot, img, pts, title in (
            (1, image0, points0, name0),
            (2, image1, points1, name1)
    ):
        if img is not None:
            if image1 is not None:
                plt.subplot(1, 2, plot)
            plt.imshow(img, cmap='gray')
            if show_points:
                for k in range(len(pts)):
                    if pts[k] is not None:
                        x, y = pts[k][0], pts[k][1]
                        plt.plot(x, y, 'or', ms=5)
                        win = wins[k]
                        if win is not None:
                            h = (win - 1) / 2.
                            wdx = np.array([-h, h, h, -h, -h])
                            wdy = np.array([-h, -h, h, h, -h])
                            wx = x + wdx
                            wy = y + wdy
                            plt.plot(wx, wy, 'r')
            plt.axis('off')
            plt.title(title)
    plt.show()


def show_field(field, name=None):
    tensor = torch.from_numpy(field.astype(np.float32)).permute((2, 0, 1))
    image = flow_to_image(tensor).permute((1, 2, 0)).numpy()
    plt.figure(figsize=(8, 8), tight_layout=True)
    plt.imshow(image)
    plt.axis('off')
    if name is not None:
        plt.title(name)
    plt.show()
    return image


def save_all(base_name, t0, t1, frames, i0, i1, g0, field, dis, field_img, dis_img):
    frame_pair = {
        'time 0': t0, 'time 1': t1, 'frames': frames,
        'image 0': i0, 'image 1': i1,
        'gradient of image 0': g0,
        'motion field': field, 'displacement': dis,
        'motion field image': field_img,
        'displacement image': dis_img
    }
    file_name = '{}.pkl'.format(base_name)
    with open(file_name, 'wb') as file:
        pickle.dump(frame_pair, file)
    print('saved all information to file {}'.format(file_name))


textures = {
    'background': {
        'x': [(0.6, 5.0, 0.)],
        'y': [(0.7, 3.0, 0.0)],
        'base': 2.2
    },
    'bottom': {
        'x': [(1., 3., 0.)],
        'y': [(0.3, 5., 0.3)],
        'base': 2.0
    },
    'lower side': {
        'x': [(1., 6., 0.5)],
        'y': [(0.6, 1.3, 0.1)],
        'base': 2.4
    },
    'ring': {
        'x': [(0., 3.5, 0.)],
        'y': [(0., 2.4, 0.7)],
        'base': 6.7
    },
    'upper side': {
        'x': [(0.3, 6.5, 0.)],
        'y': [(0.9, 1.3, 0.5)],
        'base': 2.0
    },
    'top': {
        'x': [(0.7, 1.5, 0.)],
        'y': [(0.0, 2., 0.)],
        'base': 2.1
    }
}

view_distance = 25.

bottom_radius, top_radius = 8., 5.
bottom_height, top_height = 3., 5.
bolt_parts, bolt_centroid = bolt_surfaces(
    bottom_radius, bottom_height, top_radius, top_height, textures
)

room_radius = 2. * view_distance
room_height = 2. * room_radius
room_floor = -room_height / 2.
room_parts, room_centroid = room_surfaces(
    room_radius, room_height, room_floor, textures
)

world = Body(room_parts + bolt_parts, bolt_centroid)

camera_parameters = {
    'focal distance': 2.,
    'field of view': np.pi / 3.,
    'resolution': 301,
}
viewing_parameters = {
    'distance': view_distance,
    'orbit tilt': np.pi / 3.,
    'orbit phase': 0.83 * np.pi,
    'round trip time': 60.,  # in seconds
    'frame rate': 30.,  # frames per second
    'fixation point': bolt_centroid
}

camera = Camera(camera_parameters, viewing_parameters)
