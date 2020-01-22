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
    client.connect((host,port))
    for cmd in cmds:
        client.sendall(cmd.encode('utf-8') + b'\x00')
    client.shutdown(0)
    client.close()
    return True

def import_ply(filename,location=[0,0,0],angle=[0,0,0],scale=[1,1,1]):
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
    cmd = [
           'import_ply(filename="%s",location=%s,angle=%s,scale=%s)' % (filename,location,angle,scale),
          ]
    print(cmd)
    send_msg(cmd)
    return cmd
