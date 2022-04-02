import pymongo
import datetime
from interaction import dbConnection
import requests


def save_interaction(children, startTime, interaction, duration):
    now = datetime.datetime.now()
    endTime = now.strftime("%H:%M")
    date = now.strftime("%Y-%m-%d")
    db = dbConnection.get_connection()
    event = db["events"]
    if interaction > 0:
        interaction_type = "Positive"
    else:
        interaction_type = "Negative"
    child1 = int(children[0])
    child2 = int(children[1])
    x= event.insert_one(
        {"_date": date,
         "_startTime": startTime,
         "_endTime": endTime,
         "_duration": duration,
         "_eventType": interaction_type,
         "_child1": child1,
         "_child2": child2,
         "_videoUrl": ""}
    )
    print(interaction_type)
    url = "http://localhost:8000/events/"
    data1 = {'_date': date,
             '_startTime': startTime,
             '_endTime': endTime,
             '_duration': duration,
             '_eventType': interaction_type,
             '_child1': child1,
             '_child2': child2,
             '_videoUrl': ''}
    request = requests.post(url, json=data1)
    print(request.text)


def determine_interaction(prediction):
    print("in prediction function")
    for pred in prediction:

        if pred == "empty" or pred == "Neutral":
            return 0
        elif pred == "Happy" or pred == "Happy":
            return 1
        else:
            return -1
