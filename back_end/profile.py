'''
This file handles the logic of PyMongo DB, including all read/write operations, etc.

Other backend details are handled in other files.
'''
from pymongo import *
from bson.objectid import ObjectId
# To avoid exposing MONGO DB credentials, the following will be blanked. Please enter your own cluster id.
CONNECTION_STRING = 'mongodb+srv://Andrew:MongoDB@cluster0.zxjai.mongodb.net/NoCovidConfusion?retryWrites=true&w=majority'
# Connect to Mongo DB database
client = MongoClient(CONNECTION_STRING)
db = client['NoCovidConfusion']
table = db['Profile']
class DataBase:
    # Helper Class to R/W to database
    @classmethod
    def retrieve(cls, username):
        entry = {'_id': username}
        all_entries = table.find(entry)
        for entry in all_entries:
            if entry is not None:
                return entry
        raise Exception()
    @classmethod
    def register(cls, username, password):
        entry = {'_id': username,
        'password': password,
        'visited': [],
        'quarantine': 0}
        table.insert_one(entry)
    @classmethod
    def update_profile(cls, username, quarantine = None, visited = None):
        query = {'_id': username}
        change = {}
        if quarantine is not None:
            change['quarantine'] = quarantine
        if visited is not None:
            change['visited'] = []
        change = {'$set': change}
        table.update_one(query, change)
    @classmethod
    def exists(cls, username):
        entry = {'_id': username}
        all_entries = table.find(entry)
        for entry in all_entries:
            if entry is not None:
                return True
        return False
    @classmethod
    def verify(cls, username, password):
        entry = {'_id': username, 'password': password}
        all_entries = table.find(entry)
        for entry in all_entries:
            if entry is not None:
                return True
        return False

def main():
    Database.verify("andrew", "ff")
