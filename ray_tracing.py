import numpy as np
import matplotlib.pyplot as plt
import sys
from surface_defs import *
import random

##################### CONSTANTS/SETUP #####################
width = 1200 # Width of screen
height = 800 # Height of screen
ratio = width / height # Aspect Ratio

max_reflections = 5 # Max number of reflections

camera = np.array([0, 1.1, 3]) # Position of Camera

screen = (-camera[1] + -1, camera[1] +  + 1 / ratio, camera[1] + 1, -camera[1] - 1 / ratio) # Screen: left, top, right, bottom

image = np.zeros([height, width, 3]) # Image, initially black

light = { 'position': np.array([10, 10, 10]), 'ambient': np.array([1, 1, 1]), 'diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1]) } # Light Info

num_random_objects = 75
##################### ############### #####################

#################### INITIAL OBJECTS ####################
objects = [
    { "object_type": "sphere", 'center': np.array([-0.2, 0.7, -1]), 'radius': 0.7, 'ambient': np.array([0.1, 0, 0]), 'diffuse': np.array([0.7, 0, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100 , 'reflection': 0.5 },
    { "object_type": "sphere", 'center': np.array([0.1, 0.1, 0]), 'radius': 0.1, 'ambient': np.array([0.1, 0, 0.1]), 'diffuse': np.array([0.7, 0, 0.7]), 'specular': np.array([1, 1, 1]), 'shininess': 100 , 'reflection': 0.5 },
    { "object_type": "sphere", 'center': np.array([-0.3, 0.15, 0]), 'radius': 0.15, 'ambient': np.array([0, 0.1, 0]), 'diffuse': np.array([0, 0.6, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5  },
    { "object_type": "sphere", 'center': np.array([0, -9000, 0]), 'radius': 9000, 'ambient': np.array([0.1, 0.1, 0.1]), 'diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 }
]
#################### ############### ####################


################### RANDOMIZE OBJECTS ###################
rand = random.random
for i in range(num_random_objects):
    objects.append({
     "object_type": "sphere",
     'center': np.array([-2.5 + rand() * 5, 0.1, -2.5 + rand() * 5]),
     'radius': 0.1,
     'ambient': np.array([rand(), rand(), rand()]),
     'diffuse': np.array([rand(), rand(), rand()]),
     'specular': np.array([1, 1, 1]),
     'shininess': 100 ,
      'reflection': 0.5
    })
#################### ############### ####################


####################### FUNCTIONS #######################
def normalize(vec):
    return vec / np.linalg.norm(vec)

def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis

def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = [object_intersects[obj["object_type"]](obj['center'], obj['radius'], ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance
####################### ######### #######################



for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
    for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
        pixel = np.array([x, y, 0])
        origin = camera
        direction = normalize(pixel - origin)

        color = np.zeros((3))
        total_reflection = 1

        for _ in range(max_reflections):
            nearest_object, min_distance = nearest_intersected_object(objects, origin, direction)
            if nearest_object is None:
                break

            intersection = origin + min_distance * direction # Origin + Distance(time)
            normal_to_surface = object_normals[nearest_object["object_type"]](nearest_object, intersection) # Normal of object toward intersection point
            shifted_point = intersection + 1e-5 * normal_to_surface # Shift point out, as to not mistake the object we are at, for another

            intersection_to_light = normalize(light['position'] - shifted_point) # Vector from light to intersection
            _, min_distance = nearest_intersected_object(objects, shifted_point, intersection_to_light) # Distance from intersection to object in between light
            intersection_to_light_distance = np.linalg.norm(light['position'] - intersection) # Distance from light to intersection
            is_shadowed = min_distance < intersection_to_light_distance # Wether the intersected object is between the light, or not

            if is_shadowed:
                break # Leave color as black

            illumination = np.zeros((3))

            ################### PHONG MODEL ###################
            ### Used to compute the illumination of a point ###
            illumination += nearest_object['ambient'] * light['ambient'] # Ka * Ia

            illumination += nearest_object['diffuse'] * light['diffuse'] * np.dot(intersection_to_light, normal_to_surface) # Ka * Ia * dot(L, N)

            intersection_to_camera = normalize(camera - intersection)
            H = normalize(intersection_to_light + intersection_to_camera) # ||L + V||
            illumination += nearest_object['specular'] * light['specular'] * np.dot(normal_to_surface, H) ** (nearest_object['shininess'] / 4) # Ks * Is * dot(N , (L + V) / ||L + V||) ^ ()α / 4)
            ################### ########### ###################


            ################### REFLECTION COLOR ###################
            ### Formula Used: Color = I0 + (R0)(I1) + (R0)(R1)(I2) + .... ###
            color += total_reflection * illumination
            total_reflection *= nearest_object['reflection']
            ################### ################ ###################

            origin = shifted_point # Reflection originates at intersection(shifted to avoid any issues)
            direction = reflected(direction, normal_to_surface) # Get direction of reflected ray, which then becomes our new ray

        image[i, j] = np.clip(color, 0, 1)

    sys.stdout.write(f"###########    Rendering Image   [{round(float(i) * 100 / height)}%]     ###########\r")
    sys.stdout.flush()

print("###########    Rendering Image   [100%]    ###########")
print("Rendering 'image.png' complete.")

plt.imsave('image.png', image)
