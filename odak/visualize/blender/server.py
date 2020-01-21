import bpy
import functools
import socket

def process(server,path_max):
    connection, address = server.accept()
    buf                 = connection.recv(path_max)
    for cmd in buf.split(b'\x00'):
        if cmd:
            try:
                cmd = cmd.decode()
                print(cmd)
                exec(cmd)
            except:
                import traceback
                traceback.print_exc()
                server.close()
    return 1.0

port     = 8082
host     = 'localhost'
path_max = 4096
server   = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
server.bind((host,port))
server.listen(1)
print("Listening on %s:%s" % (host, port))

bpy.app.timers.register(functools.partial(process,server,path_max), first_interval=1.0)
