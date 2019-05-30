"""
This python program is for the android application
"""
# Import socket module
import socket                   

# Create a socket object
s = socket.socket() 
# Ip address that the TCPServer is there            
host = "1somehing.11somehing."
# Reserve a port for your service every new transfer wants a new port 
#   or you must wait.
port = 50000                     

s.connect((host, port))
s.send("Hello, Server!")

# The file you want to tranfser must be in the same folder or path
#   with this file running 
filename='recorded_file.wav' 
f = open(filename,'rb')
l = f.read(1024)
while (l):
   s.send(l)
   # print('Sent ',repr(l))
   l = f.read(1024)
f.close()
print('Done sending')

answer = s.recv(1024)
if answer[0] == 1:
   # generate warning

s.close()
print('connection closed')