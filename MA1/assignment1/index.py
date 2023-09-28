import pandas as pd


data = pd.read_csv('./data/SpotifyFeatures.csv')

dt = pd.DataFrame(data)

df=dt.loc[
   ( dt['genre']=='Ska') |
    ( dt['genre']=='Opera')
]

df['label']

print(df)