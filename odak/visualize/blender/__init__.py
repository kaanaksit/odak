"""
``odak.visualize.blender``
==========================
This is your adapter to visualize things on blender.

"""
import odak.tools as tools
import os

def init(headless=False):
    """
    Definition to start a blender server. This has been tested with a Blender 2.8. Blender server runs on port 8082 on your local machine.

    Parameters
    ----------
    headless     : bool
                   To set blender to headless mode set it to True.

    Returns
    ----------
    proc         : subprocess.Popen
                   Generated process. Make sure to kill the process with a `proc.kill()` once you are done with it.
    outs         : str
                   Outputs of the executed command, returns None when check is set to False.
    errs         : str
                   Errors of the executed command, returns None when check is set to False.

                   
    """
    directory = os.path.dirname(__file__)
    server_fn = '%s/server.py' % directory
    if headless == False:
        cmd = [
               'blender',
               '-P',
               server_fn 
              ]
    elif headless == True:
        cmd = [
               'blender',
               '-b',
               '-P',
               server_fn
              ]
    proc,outs,errs = tools.shell_command(cmd,check=False)
    return proc,outs,errs

