#Importing the basic libraries
import random
import requests
import time

#We define a URL that our container 1 will access. 
#We use the http networking protocol for our purpose.
#We use the server name of the second container as defined in the .yml file. Connect it to port 5000.
#We follow the /add path for our URL.
URL = "http://cont_2:5000/add"

#Define the function that will generate, send and receive our numbers.
def gen_send_num():
    #Generate the number.
    num_1 = random.randint(1, 100)
    #A safety sleep time for allowing the connections to properly setup.
    time.sleep(5)
    #We essentially want 11 rounds to start.
    for round in range(1,11):
        #We use the try/except commands for safety in debugging.
        try:
            print("sent", num_1)
            #Open up the pipeline through the URL, and send a response to the other script through the URL.
            #We package up the randomly geenrated number into a key-value JSON object and send it.
            #This line also works as the receiver, and does not proceed to the next line unless until it received a JSON object from the other script.
            response = requests.post(URL, json = {"num": num_1})
            #Stores the value from the key-value pair packaged inside the JSON object.   
            num_1 = response.json()['result']
            num_2 = random.randint(1, 100)
            print("received", num_1)
            num_1 = num_1 + num_2
            #Print the received result and add another random integer to received result. Loop repeats.
        except Exception as e:
            print(e)
        #A safety timer for getting and receiving the objects.
        time.sleep(1)

#Run the code.
if __name__ == "__main__":
    gen_send_num()
