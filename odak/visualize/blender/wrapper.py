"""
Wrappers for blender client calling definitions from `odak/visualize/blender/libblend.py` at the server side.
"""
import socket

host = 'localhost'
port = 8082


def send_msg(cmds):
    """
    Definition to send command to blender server started by `odak/visualize/blender/server.py`.Make sure that you call init() definition from `odak/visualize/blender/__init__.py` to start the server first.

    Parameters
    ----------
    cmds           : list
                     List of pythonic commands to be transmittted to the blender server.

    Returns
    ----------
    state          : bool
                     Returns True if command is successfully sent.
    """
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except:
        return False
    client.connect((host, port))
    for cmd in cmds:
        client.sendall(cmd.encode('utf-8') + b'\x00')
    client.shutdown(0)
    client.close()
    return True


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
    cmd             : str
                      Command sent by a client.
    """
    cmd = [
        'import_ply(filename="%s",location=%s,angle=%s,scale=%s)' % (
            filename, location, angle, scale),
    ]
    print(cmd)
    send_msg(cmd)
    return cmd


def prepare(resolution=[1920, 1080], camera_fov=40.0, camera_location=[0., 0., -15.], camera_lookat=[0., 0., 0.], clip=[0.1, 100000000.], device='GPU', intensity=2., world_color=[0.3, 0.3, 0.3]):
    """
    Definition to prepare the viewport in Blender.

    Parameters
    ----------
    resolution      : list
                      Resolution of the viewport.
    camera_fov      : float
                      Viewport field of view in angles.
    camera_location : list
                      Viewport location in XYZ.
    clip            : list
                      Clipping planes near and far.
    device          : str
                      Device to be used for rendering CPU or GPU.
    intensity       : float
                      World's illumination intensity.
    world_color     : list
                      RGB color of the world's illumination.

    Returns
    ----------
    cmd             : str
                      Command sent by a client.
    """
    cmd = [
        'prepare(resolution=%s,camera_fov=%s,camera_location=%s,camera_lookat=%s,clip=%s,device="%s",intensity=%s,world_color=%s)' % (
            resolution, camera_fov, camera_location, camera_lookat, clip, device, intensity, world_color),
    ]
    print(cmd)
    send_msg(cmd)
    return cmd


def delete_the_cube():
    """
    Definitioni to delete the default cube that appears every time one starts blender.

    Returns
    ----------
    cmd             : str
                      Command sent by a client.   Returns
    """
    cmd = [
        'delete_object("Cube")'
    ]
    print(cmd)
    send_msg(cmd)
    return cmd


def render(fn, exit=False):
    """
    Definition to render a scene, and save it as a PNG file.

    Parameters
    ----------
    fn             : str
                     Filename.

    Returns
    ----------
    cmd            : str
                     Command sent by a client.   
    exit           : bool                      
                     When set to True blender exits upon rendering completion.                      
    """
    cmd = [
        'render("%s",exit=%s)' % (fn, exit),
    ]
    print(cmd)
    send_msg(cmd)
    return cmd


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
    cmd = [
        'cylinder_between(%s,%s,r=%s,objname="%s",color=%s)' % (
            start_loc, end_loc, r, objname, color)
    ]
    print(cmd)
    send_msg(cmd)
    return cmd


def quit():
    """
    Definition to quit blender.

    Returns
    ----------
    cmd             : str
                      Command sent by a client.   
    """
    cmd = [
        'quit',
    ]
    print(cmd)
    send_msg(cmd)
    return cmd
