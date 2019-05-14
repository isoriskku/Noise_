"""
This python program is for the noise classifier server
"""
from model.CNN import CNN
import os
# Import socket module
import socket

# Set the model to classify
model = CNN()
model.load_model()
path = os.path.join('./data', 'received/received_file.wav')

# Reserve a port for your service every new transfer wants a new port or you must wait.
port = 50000

# Create a socket object
s = socket.socket()
# Get local machine name
host = ""
# Bind to the port
s.bind((host, port))
# Now wait for client connection. 
s.listen(5)

print('Server listening....')


while True:
    # Establish connection with client.
    conn, addr = s.accept()
    print('Got connection from', addr)
    data = conn.recv(1024)
    print('Server received', repr(data))
    
    with open(path, 'wb') as f:
    print('file opened')
    while True:
        print('receiving data...')
        data = conn.recv(1024)
        if not data:
            break
        # write data to a file
        f.write(data)
    f.close()
    print('Successfully get the file')
    
    x = model.preproc_test(path)
    y = model.predict(x)
    
    if y == 1:
      conn.send("1")
    else:
      conn.send("0")
    
    conn.send('Thank you for connecting')
    conn.close()