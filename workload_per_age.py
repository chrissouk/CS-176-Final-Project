import kagglehub
import pandas as pd
import numpy as np
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


### filter datasets

# filter nan games played
pitchers_df = pitchers_df.dropna(subset=['G'])
hitters_df = hitters_df.dropna(subset=['G'])
# filter no games played
pitchers_df = pitchers_df[pitchers_df['G'] != 0]
hitters_df = hitters_df[hitters_df['G'] != 0]
# filter nan at bats and nan innings pitched
pitchers_df = pitchers_df.dropna(subset=['IP'])
hitters_df = hitters_df.dropna(subset=['PA'])
# filter no at bats and no innings pitched
pitchers_df = pitchers_df[pitchers_df['IP'] != 0]
hitters_df = hitters_df[hitters_df['PA'] != 0]
# remove infinite ERA values
pitchers_df.replace(np.inf, np.nan, inplace=True)
pitchers_df.dropna(subset=['ERA'], inplace=True)


### prepare data for comparisons

## merge for workload/age comparison
all_players_df = pd.merge(pitchers_df, hitters_df, how='outer')

## find the average workload for all ages
avg_workload_per_age = all_players_df.groupby('Age')['G'].mean().reset_index()


### compare & visualize

plt.figure(figsize=(12, 6))
plt.scatter(avg_workload_per_age['Age'], avg_workload_per_age['G'], alpha=0.7)
plt.plot(avg_workload_per_age['Age'], avg_workload_per_age['G'], color='red', linestyle='--')
plt.title('Average Games Played by Age', fontsize=14)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Average Games Played', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig('Visualizations/workload_per_age.png', dpi=300)
plt.close()