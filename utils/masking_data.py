import json
import pandas as pd
import uuid


def create_json(load_from_file) -> list:

    dict_all = []
    if(not load_from_file):

        df = pd.read_csv("./data/noisy_imagenette_extended.csv")
        df = df.sample(frac=1).reset_index(drop=True)
        df = df.sort_values(by="noisy_labels_0")

        for i in range(0, 10):
            res = df.loc[df['noisy_labels_0'] == get_label_for_int(i)]
            res = res[:100]

            intermed = {}
            for index, row in res.iterrows():
                d = {create_uuid(): {
                "path": row['path'],
                "class": i,
                "probab": 0
                }}
                intermed.update(d)

            dict_all.append(intermed)
    else:
        with open('./benchmarks/example_data.json') as json_file:
            json_data = json.load(json_file)

            for key in json_data:
                intermed = {}
                for i in json_data[key]:
                    d = {create_uuid(): i}
                    intermed.update(d)

                dict_all.append(intermed)
    return dict_all


def get_int_for_label(string):

    imagenette_map = {
        "n01440764": 0,
        "n02102040": 1,
        "n02979186": 2,
        "n03000684": 3,
        "n03028079": 4,
        "n03394916": 5,
        "n03417042": 6,
        "n03425413": 7,
        "n03445777": 8,
        "n03888257": 9,
    }
    return imagenette_map[string]


def get_label_for_int(int):

    imagenette_map = {
        0: "n01440764",
        1: "n02102040",
        2: "n02979186",
        3: "n03000684",
        4: "n03028079",
        5: "n03394916",
        6: "n03417042",
        7: "n03425413",
        8: "n03445777",
        9: "n03888257",
    }
    return imagenette_map[int]


def create_uuid():
    return str(uuid.uuid4().hex[:10])
