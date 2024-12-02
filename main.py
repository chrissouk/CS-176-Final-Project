import kagglehub
import pandas as pd
import os
import matplotlib.pyplot as plt

### get datasets via kagglehub
path = kagglehub.dataset_download('finkleiseinhorn/mlb-every-player-in-mlb-history')
files = os.listdir(path)

pitchers_csv = [f for f in files if f.endswith('csv')][0]
pitchers_path = os.path.join(path, pitchers_csv)
pitchers_df = pd.read_csv(pitchers_path)

hitters_csv = [f for f in files if f.endswith('csv')][1]
hitters_path = os.path.join(path, hitters_csv)
hitters_df = pd.read_csv(hitters_path)


### augment datasets

## create extra columns to calculate relevant experience criteria
pitchers_df['Years Played'] = pitchers_df.groupby('name').cumcount() + 1
pitchers_df['Total IP'] = pitchers_df.groupby('name')['IP'].transform('sum')

hitters_df['Years Played'] = hitters_df.groupby('name').cumcount() + 1
hitters_df['Total AB'] = hitters_df.groupby('name')['AB'].transform('sum')

## filter players who didn't play
# remove 'Awards' column
pitchers_df = pitchers_df.drop(columns=['Awards'])
hitters_df = hitters_df.drop(columns=['Awards'])
# filter nan amount of games
pitchers_df = pitchers_df.dropna(subset=['G'])
hitters_df = hitters_df.dropna(subset=['G'])

### create rookie and vet sub-datasets
## rookie criteria via mlb.com/glossary/rules/rookie-eligibility

# no more than 50 innings pitched for pitchers
rookie_pitchers_df = pitchers_df[(pitchers_df['Total IP'] <= 50)]
print(rookie_pitchers_df.head())

# no more than 150 at bats for hitters
rookie_hitters_df = hitters_df[(hitters_df['Total AB'] <= 130)]
print(rookie_hitters_df.head())


## vet criteria via baseball-almanac.com

# at least 5 seasons played
vet_pitchers_df = pitchers_df[pitchers_df['Years Played'] >= 5]
print(vet_pitchers_df.head())

vet_hitters_df = hitters_df[hitters_df['Years Played'] >= 5]
print(vet_hitters_df.head())


### compare

## hitters compared with OBP
# according to 'Advanced Baseball Analytics to Measure a Great Hitter' on thehittingvault.com, the best metric for evaluating hitter performance is a version of OBP.

## pitchers compared with 