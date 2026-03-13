from libblend import *
import bpy
import queue
import threading
import os
import socket
import traceback
import sys

sys.path.append("%s" % os.path.dirname(os.path.realpath(__file__)))

execution_queue = queue.Queue()

ALLOWED_FUNCTIONS = {
    "import_ply",
    "prepare",
    "delete_the_cube",
    "render",
    "cylinder_between",
    "quit",
    "delete_object",
}


class whisper2blender(threading.Thread):
    def __init__(self, port=8082, host="127.0.0.1", path_max=4096):
        threading.Thread.__init__(self)
        self.port = port
        self.host = host
        self.path_max = path_max

    def run(self):
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serversocket.bind((self.host, self.port))
        serversocket.listen(1)
        print("Listening on %s:%s" % (self.host, self.port))
        while True:
            connection, address = serversocket.accept()
            buf = connection.recv(self.path_max)
            for cmd in buf.split(b"\x00"):
                if cmd:
                    try:
                        cmd_str = cmd.decode()
                        validate_and_queue_command(cmd_str)
                    except Exception as e:
                        print("Command validation failed:", str(e))
                        traceback.print_exc()


def run_in_main_thread(function_tuple):
    execution_queue.put(function_tuple)


def validate_and_queue_command(cmd_str):
    for func_name in ALLOWED_FUNCTIONS:
        prefix = f"{func_name}("
        if cmd_str.strip().startswith(prefix):
            run_in_main_thread((func_name, cmd_str))
            return
    raise ValueError(
        "Unauthorized function call. Only whitelisted functions are allowed."
    )


def execute_queued_functions():
    while not execution_queue.empty():
        function_tuple = execution_queue.get()
        if function_tuple:
            func_name, cmd_str = function_tuple
            local_namespace = {}
            global_namespace = globals().copy()
            if func_name in globals():
                exec(cmd_str, global_namespace, local_namespace)
    return 0.1


def main():
    messenger = whisper2blender()
    messenger.start()
    bpy.app.timers.register(execute_queued_functions)
    return True


if __name__ == "__main__":
    main()
