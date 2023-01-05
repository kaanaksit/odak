import math
import bpy
import mathutils
from mathutils import Vector
import numpy as np


def set_rotation(input_obj, angle):
    input_obj.rotation_euler[0] = np.radians(angle[0])
    input_obj.rotation_euler[1] = np.radians(angle[1])
    input_obj.rotation_euler[2] = np.radians(angle[2])
    return input_obj


def set_location(input_obj, location):
    input_obj.location[0] = location[0]
    input_obj.location[1] = location[1]
    input_obj.location[2] = location[2]
    return input_obj


def import_ply(filename, location=[0, 0, 0], angle=[0, 0, 0], scale=[1, 1, 1]):
    """
    Definition to import a PLY object.

    Parameters
    ----------
    filename       : str
                     Path of the PLY file to be imported.
    location       : list
                     location of the imported object along X,Y, and Z axes.
    angle          : list
                     Euler angles for rotating the imported object along X,Y, and Z axes.
    scale          : list
                     Scale of the imported object.

    Returns
    ----------
    ply_object     : blender object
                     Created blender object.
    """
    bpy.ops.import_mesh.ply(filepath=filename)
    ply_object = bpy.context.view_layer.objects.active
    ply_object.scale[0] = scale[0]
    ply_object.scale[1] = scale[1]
    ply_object.scale[2] = scale[2]
    ply_object.location[0] = location[0]
    ply_object.location[1] = location[1]
    ply_object.location[2] = location[2]
#    ply_object                   = set_location(ply_object,location)
#    ply_object                   = set_rotation(ply_object,angle)
    print('Imported name: ', ply_object.name)
    return ply_object


def intersect_a_ray(p0, p1, ob):
    """
    Definition to create a ray between two points.

    Parameters
    ----------
    p0             : list
                     Starting point of a ray.
    p1             : list
                     Ending point of a ray.
    ob             : blender object
                     Object to be tested for intersection with a ray.

    Returns
    ----------
    result         : list
                     Result of ray intersection test containing state, coalision ray, surface normal.
    loc            : bool/float
                     Location of the intersection. If ray isn't intersecting returns False.
    dist           : float
                     Distance from the starting point of the ray to the intersection point.
    """
    mwi = ob.matrix_world.inverted()
    ray_begin = mwi @ Vector((p0[0], p0[1], p0[2]))
    ray_end = mwi @ Vector((p1[0], p1[1], p1[2]))
    ray_direction = (ray_end-ray_begin).normalized()
    result = ob.ray_cast(origin=ray_begin, direction=ray_direction)
    loc = False
    dist = 0
    if result[0] == True:
        mw = ob.matrix_world
        loc = mw @ result[1]
        dist = ((loc[0]-p0[0])**2+(loc[1]-p0[1])**2+(loc[2]-p0[2])**2)**0.5
    return result, loc, dist


def create_ray_from_two_points(x0y0z0, x1y1z1):
    # Because of Python 2 -> Python 3.
    x0, y0, z0 = x0y0z0
    x1, y1, z1 = x1y1z1
    # Create a vector from two given points.
    point = np.array([x0, y0, z0], dtype=np.float32)
    # Distance between two points.
    s = np.sqrt(pow((x0-x1), 2) + pow((y0-y1), 2) + pow((z0-z1), 2))
    if s != 0:
        alpha = (x1-x0)/s
        beta = (y1-y0)/s
        gamma = (z1-z0)/s
    elif s == 0:
        alpha = float('nan')
        beta = float('nan')
        gamma = float('nan')
    # Cosines vector
    cosines = np.array([alpha, beta, gamma], dtype=np.float32)
    # Returns vector and the distance.
    ray = np.array([point, cosines], dtype=np.float32)
    return ray, s


def reflect_ray(input_ray, normal, intersection):
    mu = 1
    div = pow(normal[0], 2) + pow(normal[1], 2) + pow(normal[2], 2)
    a = mu * (input_ray[1][0]*normal[0]
              + input_ray[1][1]*normal[1]
              + input_ray[1][2]*normal[2]) / div
    output_ray = input_ray.copy()
    output_ray[0][0] = intersection[0]
    output_ray[0][1] = intersection[1]
    output_ray[0][2] = intersection[2]
    output_ray[1][0] = input_ray[1][0] - 2*a*normal[0]
    output_ray[1][1] = input_ray[1][1] - 2*a*normal[1]
    output_ray[1][2] = input_ray[1][2] - 2*a*normal[2]
    p0 = output_ray[0]
    p1 = [
        output_ray[0][0]+10*output_ray[1][0],
        output_ray[0][1]+10*output_ray[1][1],
        output_ray[0][2]+10*output_ray[1][2],
    ]
    return p0, p1


def quit():
    """
    Definition to quit blender.
    """
    bpy.ops.wm.quit_blender()


def render(fn, exit=False):
    """
    Definition to render a scene, and save it as a PNG file.

    Parameters
    ----------
    fn             : str
                     Filename.
    exit           : bool
                     When set to True blender exits upon rendering completion.
    """
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = fn
    print(fn)
    bpy.data.scenes["Scene"].render.resolution_percentage = 100
    print('Device: %s' % bpy.data.scenes["Scene"].cycles.device)
    bpy.ops.render.render(use_viewport=True, write_still=True)
    if exit == True:
        quit()
    return True


def set_render_type(render_type='rgb'):
    """
    Definition to set the render type.

    Parameters
    ----------
    render_type    : str
                     Set it to `rgb` or `depth` to get either color or depth images.
    """
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    for node in tree.nodes:
        tree.nodes.remove(node)
    node_composite = tree.nodes.new('CompositorNodeComposite')
    node_norm = tree.nodes.new('CompositorNodeNormalize')
    node_layers = tree.nodes.new('CompositorNodeRLayers')
    node_composite.location = 800, 0
    node_norm.location = 400, -100
    node_layers.location = 0, 0
    link0 = links.new(node_layers.outputs[2], node_norm.inputs[0])
    if render_type == 'rgb':
        linkr = links.new(node_layers.outputs[0], node_composite.inputs[0])
    elif render_type == 'depth':
        linkr = links.new(node_norm.outputs[0], node_composite.inputs[0])
    return True


def prepare(resolution=[1920, 1080], camera_fov=40.0, camera_location=[0., 0., -15.], camera_lookat=[0., 0., 0.], clip=[0.1, 100000000.], device='GPU', intensity=2., world_color=[0.3, 0.3, 0.3]):
    """
    Definition to prepare Blender for renderings.
    """
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.objects["Camera"].data.lens_unit = 'FOV'
    bpy.context.scene.objects["Camera"].data.angle = np.radians(camera_fov)
    bpy.context.scene.objects["Camera"].data.clip_start = clip[0]
    bpy.context.scene.objects["Camera"].data.clip_end = clip[1]
    bpy.context.scene.objects["Camera"].location[0] = camera_location[0]
    bpy.context.scene.objects["Camera"].location[1] = camera_location[1]
    bpy.context.scene.objects["Camera"].location[2] = camera_location[2]

    dx = camera_location[0]-camera_lookat[0]
    dy = camera_location[1]-camera_lookat[1]
    dz = camera_location[2]-camera_lookat[2]
    dist = math.sqrt(dx**2+dy**2+dz**2)
    phi = math.atan2(dy, dx)
    theta = math.acos(dz/dist)
    camera_angles = [
        0,
        np.degrees(theta),
        np.degrees(phi)
    ]
    bpy.context.scene.objects["Camera"].rotation_euler[0] = np.radians(
        camera_angles[0])
    bpy.context.scene.objects["Camera"].rotation_euler[1] = np.radians(
        camera_angles[1])
    bpy.context.scene.objects["Camera"].rotation_euler[2] = np.radians(
        camera_angles[2])
    bpy.data.scenes["Scene"].render.resolution_x = resolution[0]
    bpy.data.scenes["Scene"].render.resolution_y = resolution[1]
    bpy.data.scenes["Scene"].cycles.device = device
    bpy.data.worlds["World"].use_nodes = True
    bpy.data.worlds["World"].node_tree.nodes['Background'].inputs[0].default_value[:3] = world_color
    bpy.data.worlds["World"].node_tree.nodes['Background'].inputs[1].default_value = intensity
    try:
        delete_object('Cube')
        delete_object('Light')
    except:
        pass
    return camera_angles


def delete_object(label):
    """
    Definition to delete an object from the scene.

    Parameters
    ----------
    label          : str
                     String that identifies the object to be deleted.
    """
    bpy.data.objects.remove(bpy.context.scene.objects[label], do_unlink=True)

# Clear all nodes in a mat


def clear_material(material):
    if material.node_tree:
        material.node_tree.links.clear()
        material.node_tree.nodes.clear()


def assign_color(fn, obj, color=[0., 0., 0.], label='color'):
    mat = bpy.data.materials.new(label)
    mat.use_nodes = True
    clear_material(mat)
    matnodes = mat.node_tree.nodes
    links = mat.node_tree.links
    colors = matnodes.new('ShaderNodeRGB')
    output = matnodes.new('ShaderNodeOutputMaterial')
    link = links.new(colors.outputs['Color'], output.inputs['Surface'])
    obj.data.materials.append(mat)
    colors.outputs['Color'].default_value = (color[0], color[1], color[2], 0)


def assign_texture(fn, obj, label):
    mat = bpy.data.materials.new(label)
    mat.use_nodes = True
    clear_material(mat)
    matnodes = mat.node_tree.nodes
    links = mat.node_tree.links
    diffuse = matnodes.new('ShaderNodeBsdfDiffuse')
    output = matnodes.new('ShaderNodeOutputMaterial')
    tex = matnodes.new('ShaderNodeTexImage')
    bpy.ops.image.open(filepath=fn)
#    bpy.data.images[0].pack(as_png=True)
    bpy.data.images[0].pack()
    tex.image = bpy.data.images[0]
    link = links.new(tex.outputs['Color'], diffuse.inputs['Color'])
    link = links.new(diffuse.outputs['BSDF'], output.inputs['Surface'])
#    disp           = matnodes['Material Output'].inputs['Displacement']
#    mat.node_tree.links.new(disp, tex.outputs['Color'])
    obj.data.materials.append(mat)
    bpy.data.objects[obj.name].select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.uv.unwrap()
    bpy.ops.object.mode_set(mode='OBJECT')
#    obj.material_slots[0].material.node_tree.nodes[“main image”].image = tex


def create_plane(objname, location, size=[1., 1.]):
    """
    Definition to create a plane.

    Parameters
    ----------
    objname        : str
                     Label of the object to be created.
    location       : list
                     Center of the plane in X,Y,Z coordinates.
    size           : list
                     Size of the plane to be created in X and Y axes.

    Returns
    ----------
    plane          : blender object
                     Created blender object.
    """
    ob = bpy.ops.mesh.primitive_plane_add(
        size=size[0],
        enter_editmode=False,
        location=location
    )
    plane = bpy.context.render_layer.obkects.active
    plane.name = objname
    plane.scale[1] = size[1]/size[0]
    return plane


def create_circle(objname, location, radius=1.):
    """
    Definition to create a circle.

    Parameters
    ----------
    objname        : str
                     Label of the object to be created.
    location       : list
                     Center of the plane in X,Y,Z coordinates.
    radius         : float
                     Radius of the circle.

    Returns
    ----------
    circle         : blender object
                     Created blender object.
    """
    ob = bpy.ops.mesh.primitive_circle_add(
        radius=radius,
        enter_editmode=False,
        location=location,
        fill_type='TRIFAN'
    )
    circle = bpy.context.render_layer.obkects.active
    circle.name = objname
    return circle

# Taken from https://blender.stackexchange.com/questions/67517/possible-to-add-a-plane-using-vertices-via-python


def create_plane_from_meshes(objname, location, size=[1., 1.]):
    """
    Definition to create a plane using simple meshes.

    Parameters
    ----------
    objname        : str
                     Label of the object to be created.
    location       : list
                     Center of the plane in X,Y,Z coordinates.
    size           : list
                     Size of the plane to be created in X and Y axes.

    Returns
    ----------
    myobject       : blender object
                     Created blender object.
    """
    # Define arrays for holding data.
    myvertex = []
    myfaces = []
    # Create all Vertices
    # vertex 0
    mypoint = [(-size[0]/2., -size[1]/2., 0.0)]
    myvertex.extend(mypoint)
    # vertex 1
    mypoint = [(size[0]/2., -size[1]/2., 0.0)]
    myvertex.extend(mypoint)
    # vertex 2
    mypoint = [(-size[0]/2., size[1]/2., 0.0)]
    myvertex.extend(mypoint)
    # vertex 3
    mypoint = [(size[0]/2., size[1]/2., 0.0)]
    myvertex.extend(mypoint)
    # -------------------------------------
    # Create all Faces
    # -------------------------------------
    myface = [(0, 1, 3, 2)]
    myfaces.extend(myface)
    mymesh = bpy.data.meshes.new(objname)
    myobject = bpy.data.objects.new(objname, mymesh)
    bpy.context.collection.objects.link(myobject)
    # Generate mesh data
    mymesh.from_pydata(myvertex, [], myfaces)
    # Calculate the edges
    mymesh.update(calc_edges=True)
    # Set Location
    myobject = set_location(
        myobject,
        location
    )
    return myobject

# Taken from https://blender.stackexchange.com/questions/5898/how-can-i-create-a-cylinder-linking-two-points-with-python


def cylinder_between(start_loc, end_loc, r=0.1, objname='cylinder', color=[0., 0.5, 0., 0.]):
    """
    Definition to create a cylinder in between two points with a certain radius.

    Parameters
    ----------
    stat_loc       : list
                     List that contains X,Y and Z start location.
    end_loc        : list
                     List that contains X,Y and Z end location.
    r              : float
                     Radius of the cylinder.
    objname        : str
                     Label of the object to be created.
    color          : list
                     Color of the cylinder (RGBA)

    Returns
    ----------
    obj            : blender object
                     Created cylinder.
    """
    x1 = float(start_loc[0])
    y1 = float(start_loc[1])
    z1 = float(start_loc[2])
    x2 = float(end_loc[0])
    y2 = float(end_loc[1])
    z2 = float(end_loc[2])
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    dist = math.sqrt(dx**2 + dy**2 + dz**2)
    obj = bpy.ops.mesh.primitive_cylinder_add(
        radius=r,
        depth=dist,
        location=(dx/2 + x1, dy/2 + y1, dz/2 + z1)
    )
    cylinder = bpy.context.view_layer.objects.active
    cylinder.name = objname
    phi = math.atan2(dy, dx)
    theta = math.acos(dz/dist)
    cylinder.rotation_euler[1] = theta
    cylinder.rotation_euler[2] = phi
    mat = bpy.data.materials.new('diffuse_texture')
    mat.use_nodes = True
    clear_material(mat)
    matnodes = mat.node_tree.nodes
    links = mat.node_tree.links
    diffuse = matnodes.new('ShaderNodeBsdfDiffuse')
    diffuse.inputs[0].default_value = [color[0], color[1], color[2], color[3]]
    output = matnodes.new('ShaderNodeOutputMaterial')
    link = links.new(diffuse.outputs['BSDF'], output.inputs['Surface'])
    cylinder.data.materials.append(mat)
    return cylinder
