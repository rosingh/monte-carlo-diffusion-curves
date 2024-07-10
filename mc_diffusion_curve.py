import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import cv2
from numba import njit, prange


def parse_control_points(control_points_element):
    points = []
    for point in control_points_element:
        x = float(point.get('x'))
        y = float(point.get('y'))
        points.append((x, y))
    return np.array(points)

def parse_colors(color_set_element):
    colors = []
    max_global_id = 0
    for color in color_set_element:
        B = float(color.get('R')) / 255
        G = float(color.get('G')) / 255
        R = float(color.get('B')) / 255
        t = float(color.get('globalID'))
        max_global_id = max(max_global_id, t)

        colors.append(np.array((R, G, B, t)))

    colors = np.array(colors, dtype=np.float64)
    
    colors[:, 3] /= max_global_id

    return np.array(colors, dtype=np.float64)

@njit
def get_color_at(color_points, t):
    # Linearly interpolate the color given color points on the curve
    positions = color_points[:, -1]
    colors = color_points[:, :-1]
    # Find the interval containing t
    indices = np.where((positions[:-1] <= t) & (t <= positions[1:]))[0]
    if len(indices) == 0:
        # Return black if t is out of bounds
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)

    i = indices[0]
    s = (t - positions[i]) / (positions[i + 1] - positions[i])
    interpolated_color = colors[i] + s * (colors[i + 1] - colors[i])

    return interpolated_color

@njit
def bezier_curve(control_points, t):
    n = len(control_points) - 1
    point = np.zeros(2)
    for i in range(n + 1):
        bin_coeff = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        point += bin_coeff * control_points[i]
    return point

@njit
def comb(n, k):
    if k > n:
        return 0
    if k == 0 or n == k:
        return 1
    k = min(k, n - k)
    c = 1
    for i in range(k):
        c = c * (n - i) // (i + 1)
    return c

@njit
def approximate_bezier_as_lines(control_points, num_segments=16):
    lines = np.zeros((num_segments, 2, 2))
    t_values = np.linspace(0, 1, num_segments + 1)
    for i in range(num_segments):
        lines[i, 0] = bezier_curve(control_points, t_values[i])
        lines[i, 1] = bezier_curve(control_points, t_values[i + 1])
    # Return the line segments, and t_values for start of each segment
    return lines, t_values[:-1]  

@njit
def find_closest_point_on_segments(segments, t_values, point):
    min_distance = np.inf
    closest_point = None
    best_t = 0
    best_segment = None
    for i, segment in enumerate(segments):
        p0, p1 = segment
        v = p1 - p0
        w = point - p0
        c1 = np.dot(w, v)
        if c1 <= 0:
            distance = np.linalg.norm(w)
            nearest = p0
            local_t = 0
        else:
            c2 = np.dot(v, v)
            if c2 <= c1:
                distance = np.linalg.norm(point - p1)
                nearest = p1
                local_t = 1
            else:
                b = c1 / c2
                pb = p0 + b * v
                distance = np.linalg.norm(point - pb)
                nearest = pb
                local_t = b
        if distance < min_distance:
            min_distance = distance
            closest_point = nearest
            best_segment = segment
            best_t = t_values[i] + local_t * (t_values[i + 1] - t_values[i])

    return closest_point, best_t, min_distance, best_segment

@njit
def find_closest_curve_among_curves(curve_data, point):
    global_min_distance = np.inf
    global_closest_point = None
    global_best_t = 0
    curve_index = -1
    global_best_segment = None
    for i,val in enumerate(curve_data):
        segments, t_vals = val
        closest_point, best_t, min_distance, best_segment = find_closest_point_on_segments(segments, t_vals, point)
        if min_distance < global_min_distance:
            global_min_distance = min_distance
            global_closest_point = closest_point
            global_best_t = best_t
            curve_index = i
            global_best_segment = best_segment
    return global_closest_point, global_best_t, curve_index, global_min_distance, global_best_segment

@njit
def check_left_of_segment(point, segment):
    p1 = segment[0]
    p2 = segment[1]
    vector_segment = p2 - p1
    vector_point = point - p1
    cross_product = vector_segment[0] * vector_point[1] - vector_segment[1] * vector_point[0]

    # Point is to the left
    if cross_product > 0:
        return 1
    # Point is to the right
    elif cross_product < 0:
        return -1
    # Point is collinear
    return 0


@njit
def solve(x0, all_control_points, all_left_colors, all_right_colors):
    eps = 0.001
    nWalks = 4
    maxSteps = 16

    color = np.array([0.0, 0.0, 0.0])
    for i in range(nWalks):
        x = x0
        steps = 0
        R = np.inf
        while R > eps and steps < maxSteps:
            # Get the distance to the closest curve, and whether you're on the left or right side
            global_closest_point, global_best_t, curve_index, global_min_distance, global_best_segment = find_closest_curve_among_curves(all_control_points, x)
            is_left = check_left_of_segment(x, global_best_segment)
            R = min(R, global_min_distance)

            theta = np.random.uniform(low=0, high=2*np.pi)
            inc_vec = np.array((R*np.cos(theta), R*np.sin(theta)))
            x = x + inc_vec
            steps += 1

        if is_left == 1:
            color = color + get_color_at(all_left_colors[curve_index], global_best_t)
        else:
            color = color + get_color_at(all_right_colors[curve_index], global_best_t)

    return color/nWalks

@njit(parallel=True)
def create_image(all_control_points, all_left_colors, all_right_colors, dimensions):
    image_array = np.zeros((dimensions[0], dimensions[1], 3))
    for i in prange(dimensions[0]):
        for j in prange(dimensions[1]):
            x0 = np.array((i, j), dtype=np.float64)
            image_array[i][j] = solve(x0, all_control_points, all_left_colors, all_right_colors)
    return image_array

@njit(parallel=True)
def create_crop_image(all_control_points, all_left_colors, all_right_colors, dimensions):
    image_array = np.zeros((dimensions[0] // 2, dimensions[1], 3))
    for i in prange(dimensions[0] // 2, dimensions[0]):
        for j in prange(dimensions[1]):
            x0 = np.array((i, j), dtype=np.float64)
            image_array[i - (dimensions[0] // 2)][j] = solve(x0, all_control_points, all_left_colors, all_right_colors)
    return image_array

tree = ET.parse('flower.xml')
root = tree.getroot()

# Parse each curve in the XML
curves = []
for curve in root.findall('curve'):
    control_points_set = curve.find('control_points_set')
    control_points = parse_control_points(control_points_set)

    left_colors_set = curve.find('left_colors_set')
    left_colors = parse_colors(left_colors_set)

    right_colors_set = curve.find('right_colors_set')
    right_colors = parse_colors(right_colors_set)

    curve_data = {
        'control_points': control_points,
        'left_colors': left_colors,
        'right_colors': right_colors
    }
    curves.append(curve_data)

image_width = int(root.attrib['image_width'])
image_height = int(root.attrib['image_height'])
all_control_points = [curve['control_points'] for curve in curves]
all_left_colors = [curve['left_colors'] for curve in curves]
all_right_colors = [curve['right_colors'] for curve in curves]

precomputed_data = [approximate_bezier_as_lines(cp) for cp in all_control_points]

crop_array = create_crop_image(precomputed_data, all_left_colors, all_right_colors, (image_height, image_width))
image_array = create_image(precomputed_data, all_left_colors, all_right_colors, (image_height, image_width))

normalized_zoom = np.empty_like(crop_array, dtype=np.uint8)
normalized_image = np.empty_like(image_array, dtype=np.uint8)

cv2.normalize(image_array, normalized_image, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite('flower.png', normalized_image[..., ::-1])
cv2.normalize(crop_array, normalized_zoom, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imwrite('flower_cropped.png', normalized_zoom[..., ::-1])

plt.imshow(crop_array)
plt.show()
plt.imshow(image_array)
plt.show()