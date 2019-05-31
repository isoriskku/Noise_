"""
This python program is for the android application
"""
# Import socket module
import socket                   

# Create a socket object
s = socket.socket() 
# Ip address that the TCPServer is there            
host = "203.252.43.96"
# Reserve a port for your service every new transfer wants a new port 
#   or you must wait.
port = 50000                     

s.connect((host, port))
s.send(b"Hello, Server!")

# The file you want to tranfser must be in the same folder or path
#   with this file running 
filename = 'recorded_file.wav'
f = open(filename, 'rb')
buf = f.read(1024)
while buf:
    s.send(buf)
    # print('Sent ', repr(buf))
    buf = f.read(1024)
f.close()
print('Done sending')

answer = s.recv(1024)
if answer == b"1":
    # generate warning
    print("WARNING!")

s.close()
print('connection closed')
