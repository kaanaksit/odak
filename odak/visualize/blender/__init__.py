"""
``odak.visualize.blender``
==========================
This is your adapter to visualize things on blender.

"""
import odak.tools as tools
import os
import time
from subprocess import Popen, PIPE
from .wrapper import *


def init(headless=False, blend_fn=''):
    """
    Definition to start a blender server. This has been tested with a Blender 2.8. Blender server runs on port 8082 on your local machine.

    Parameters
    ----------
    headless      : bool
                    To set blender to headless mode set it to True.
    blend_fn      : str
                    Filename.

    Returns
    ----------
    proc          : subprocess.Popen
                    Generated process. Make sure to kill the process with a `proc.kill()` once you are done with it.
    outs          : str
                    Outputs of the executed command, returns None when check is set to False.
    """
    directory = os.path.dirname(__file__)
    server_fn = '%s/server.py' % directory
    if headless == False:
        cmd = [
            'blender',
            blend_fn,
            '-P',
            server_fn,
        ]
    elif headless == True:
        cmd = [
            'blender',
            blend_fn,
            '-b',
            '-P',
            server_fn,
        ]
    proc, outs, errs = tools.shell_command(cmd, check=False)
    time.sleep(2)
    return proc


def check(proc):
    """
    Definition to check if blender is still alive and wait for completion. Once completed, this definition kills the process.

    Parameters
    ----------
    proc         : subprocess.Popen
                   Process to be monitored, see init() in odak.visualizer.blender to create one.

    """
    while proc.poll() == None:
        pass
    proc.kill()
