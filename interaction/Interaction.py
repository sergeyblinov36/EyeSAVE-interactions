import datetime
import requests


def save_interaction(children, startTime, interaction, duration):
    now = datetime.datetime.now()
    endTime = now.strftime("%H:%M")
    date = now.strftime("%Y-%m-%d")
    if interaction > 0:
        interaction_type = "Positive"
    else:
        interaction_type = "Negative"
    child1 = int(children[0])
    child2 = int(children[1])
    url = "https://eyesave.herokuapp.com/events/"
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
    value = 0
    for pred in prediction:

        if pred == "empty" or pred == "Neutral":
            value += 1
        elif pred == "Happy" or pred == "Happy":
            value += 1
        else:
            value += -1
    return value
