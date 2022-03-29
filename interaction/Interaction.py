import pymongo
import datetime
from interaction import dbConnection


def save_interaction(children, startTime, interaction):
    now = datetime.datetime.now()
    endTime = now.strftime("%H:%M:%S")
    date = now.strftime("%d-%m-%Y")
    db = dbConnection.get_connection()
    event = db["events"]
    if interaction > 0:
        interaction_type = "positive"
    else:
        interaction_type = "negative"
    child1 = int(children[0])
    child2 = int(children[1])
    event.insert_one(
        {"_date": date,
         "_startTime": startTime,
         "_endTime": endTime,
         "_eventType": interaction_type,
         "_child1": child1,
         "_child2": child2,
         "_videoUrl": ""}
    )
    print(interaction_type)
