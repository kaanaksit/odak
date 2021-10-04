from libblend import *
import bpy
import queue
import threading
import os
import socket
import traceback
import sys
sys.path.append('%s' % os.path.dirname(os.path.realpath(__file__)))

execution_queue = queue.Queue()


class whisper2blender(threading.Thread):
    def __init__(self, port=8082, host='127.0.0.1', path_max=4096):
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
            for cmd in buf.split(b'\x00'):
                if cmd:
                    try:
                        cmd = cmd.decode()
#                        print(cmd)
                        run_in_main_thread(cmd)
    #                    exec(cmd)
                    except:
                        traceback.print_exc()


def run_in_main_thread(function):
    execution_queue.put(function)


def execute_queued_functions():
    while not execution_queue.empty():
        function = execution_queue.get()
        if type(function) != type(None):
            exec(function)
    return 0.1


def main():
    messenger = whisper2blender()
    messenger.start()
    bpy.app.timers.register(execute_queued_functions)
    return True


if __name__ == "__main__":
    main()
