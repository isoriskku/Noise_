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
host = "203.252.43.96"
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
    print('Server received', data.decode())
    
    with open(path, 'wb') as f:
        print('file opened')
        while True:
            print('receiving data...')
            data = conn.recv(1024)
            # print("Recv ", repr(data))
            if len(data) < 1024 or not data:
                print("Done Receiving")
                break
            f.write(data)
    f.close()

    x = model.preprocess(path)
    y = model.predict(x)
    print("Classification result :", y)
    
    if y in [0, 2, 5, 9]:
        conn.send(b"0")
    else:
        conn.send(b"1")
    
    conn.close()
