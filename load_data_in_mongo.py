from pymongo import MongoClient
import pandas as pd



client = MongoClient('mongodb://0.0.0.0:27017/')

db = client["my_db"]

db.my_data.drop()

df = pd.read_csv("data/data.csv", low_memory=False)

i = 0

for _ ,row in df.iterrows():
    i += 1
    db.my_data.insert_one(row.to_dict())
    print(i)


future_df = []

headers_dict = dict(db.my_data.find_one({}))

headers_dict.pop("_id")

headers = list(headers_dict.keys())

with open("data/data_.csv", "w") as csv_file:

    csv_file.write("\t".join(headers))
    csv_file.write("\n")

    for doc in db.my_data.find({}):
        dict_doc = dict(doc)
        dict_doc.pop("_id")

        csv_file.write("\t".join(str(value) for value in dict_doc.values()))
        csv_file.write("\n")
