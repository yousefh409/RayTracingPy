import numpy as np


def normalize(vec):
    return vec / np.linalg.norm(vec)

################## SPHERE ##################
def sphere_normal(sphere, point):
    return normalize(point - sphere['center'])

def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c # a = 1 since the direction is a unit vector
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2) # We return the nimimum since we only care about the first interesction
    return None
################## ###### ##################

object_intersects = {
    'sphere': sphere_intersect
}

object_normals = {
    'sphere': sphere_normal
}

object_largest_rad = {
    'sphere': (lambda s: s["radius"])
}
