itimport matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm

# iterate through results, load and parse all jsons and put them into a dataframe
results = []
df = pd.DataFrame()
for file in tqdm(os.listdir('results')):
    if file.endswith('.json'):
        with open(os.path.join('results', file)) as json_file:
            data = json.load(json_file)
            data_dict = {
                "mc1": data['results']['TruthfulQAMultipleChoiceRole']['mc1'],
                "mc2": data['results']['TruthfulQAMultipleChoiceRole']['mc2'],
                "model": data['config']['model'],
                "role.occupation": data['role']['occupation'],
                "role.education": data['role']['education'],
                "role.gender": data['role']['gender'],
                "role.age": data['role']['age'],
                "role.nationality": data['role']['nationality'],
            }
            results.append(data_dict)

df = pd.DataFrame(results)
df.to_excel('results_all.xlsx')

print(df)
