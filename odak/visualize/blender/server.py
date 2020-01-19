# Script to run from blender:
# blender --python blender_server.py

import bpy
import threading
import os,socket,traceback

class whisper2blender(threading.Thread):
    def __init__(self,port=8082,host='127.0.0.1',path_max=4096):
        threading.Thread.__init__(self)
        self.port     = port
        self.host     = host
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
                        cmd.decode()
                        print(cmd)
                        exec(cmd)
                    except:
                        traceback.print_exc()

def main():
    print(__file__)
    messenger = whisper2blender()
    messenger.start()
    return True

if __name__ == "__main__":
    main()
