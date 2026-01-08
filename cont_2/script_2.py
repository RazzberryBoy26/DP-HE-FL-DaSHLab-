#Importing the basic libraries.
#We will be using Flask API for the connection setup.
from flask import Flask, request, jsonify
import random

#Initialize the flash connection.
app = Flask(__name__)

#This functions as an if-else statement concerning the flask connection.
#If the path through which object is received is the /add path, and the method of sending is 'POST';
#Then execute the receiver adder and sender function.
@app.route('/add', methods = ['POST'])
def rec_add_send_num():
    #Convert the JSON object into its key-value pair and store it in data.
    data = request.get_json()
    #Get the value from that pair.
    num_1 = data.get('num')
    num_2 = random.randint(1, 100)
    #We use the flush command so as to let the print statement execute as well.
    print("generated", num_2, flush = True)
    #Add the two numbers, package them into a JSON object, and send it through the port logged in by the flask app.
    result = num_1 + num_2
    return jsonify({"result" : result})

#Run the code.
if __name__ == "__main__":
    #We use the 0.0.0.0 host since flask is a self-contained application.
    #Mentioning this particular host address allows the script to interact with other containers.
    #Login to the 5000 port via the flask app.
    app.run(host = '0.0.0.0', port = 5000)
