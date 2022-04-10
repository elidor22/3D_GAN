# blender -b --python test.py -- /home/elidor/Documents/1a9b552befd6306cc8f2d5fe7449af61/models/model_normalized.obj
# Use previous command to run script  # -- means that blender ignores the argument after the double dash, use it for custom argumetns

import bpy, sys
from math import *
from mathutils import *
import os


def enable_gpus(device_type, use_cpus=False):
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cycles_preferences.refresh_devices()
    devices = cycles_preferences.devices

    if not devices:
        raise RuntimeError("Unsupported device type")

    activated_gpus = []
    for device in devices:
        if device.type == "CPU":
            device.use = use_cpus
        else:
            device.use = True
            activated_gpus.append(device.name)
            print('activated gpu', device.name)

    cycles_preferences.compute_device_type = device_type
    bpy.context.scene.cycles.device = "GPU"

    return activated_gpus


# enable_gpus("CUDA")





# Import obj from the path given
bpy.ops.import_scene.obj(filepath=sys.argv[-1])

# Remove cube

objs = bpy.data.objects
objs.remove(objs["Cube"], do_unlink=True)


#set your own target here
target = bpy.data.objects['model_normalized']
cam = bpy.data.objects['Camera']
t_loc_x = target.location.x
t_loc_y = target.location.y
cam_loc_x = cam.location.x
cam_loc_y = cam.location.y

#dist = sqrt((t_loc_x-cam_loc_x)**2+(t_loc_y-cam_loc_y)**2)
dist = (target.location.xy-cam.location.xy).length
#ugly fix to get the initial angle right
init_angle  = (1-2*bool((cam_loc_y-t_loc_y)<0))*acos((cam_loc_x-t_loc_x)/dist)-2*pi*bool((cam_loc_y-t_loc_y)<0)

num_steps = 8 #how many rotation steps
for x in range(num_steps):
    alpha = init_angle + (x+1)*2*pi/num_steps
    cam.rotation_euler[2] = pi/2+alpha
    cam.location.x = t_loc_x+cos(alpha)*dist
    cam.location.y = t_loc_y+sin(alpha)*dist
    file = f'/home/elidor/Videos/renders/{x}'
    bpy.context.scene.render.filepath = file
    bpy.ops.render.render( write_still=True ) 