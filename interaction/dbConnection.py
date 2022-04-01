import pymongo
from interaction import config

client = pymongo.MongoClient(config.mongoConnection)


def get_connection():
    db = client["EyeSAVE_DB"]
    return db
