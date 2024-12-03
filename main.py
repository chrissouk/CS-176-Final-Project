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
# filter nan games
pitchers_df = pitchers_df.dropna(subset=['G'])
hitters_df = hitters_df.dropna(subset=['G'])
# filter no games
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

### create rookie and vet sub-datasets
## rookie criteria via mlb.com/glossary/rules/rookie-eligibility

# no more than 50 innings pitched for pitchers
rookie_pitchers_df = pitchers_df[(pitchers_df['Total IP'] <= 50)].copy()

# no more than 150 at bats for hitters
rookie_hitters_df = hitters_df[(hitters_df['Total AB'] <= 130)].copy()


## vet criteria via baseball-almanac.com
 
# at least 5 seasons played
vet_pitchers_df = pitchers_df[pitchers_df['Years Played'] >= 5].copy()

vet_hitters_df = hitters_df[hitters_df['Years Played'] >= 5].copy()




### compare
# comparison stats selected to highlight key performance metrics, normalizing for playing time and career stage, to fairly compare rookie and veteran performance.

hitting_comparison_columns = [
    'H',   # Hits
    '2B',  # Doubles
    '3B',  # Triples
    'HR',  # Home Runs
    'RBI', # Runs Batted In
    'BB',  # Walks
    'SO',  # Strikeouts
    'OBP', # On-Base Percentage
    'SLG', # Slugging Percentage
    'OPS'  # On-Base Plus Slugging
]

pitching_comparison_columns = [
    'ERA',  # Earned Run Average
    'WHIP', # Walks + Hits per Inning Pitched
    'H9',   # Hits per 9 innings
    'HR9',  # Home Runs per 9 innings
    'BB9',  # Walks per 9 innings
    'SO9',  # Strikeouts per 9 innings
    'SO/W', # Strikeout-to-Walk Ratio
]

rookie_hitters_average_stats = rookie_hitters_df[hitting_comparison_columns].mean()
vet_hitters_average_stats = vet_hitters_df[hitting_comparison_columns].mean()

rookie_pitchers_average_stats = rookie_pitchers_df[pitching_comparison_columns].mean()
vet_pitchers_average_stats = vet_pitchers_df[pitching_comparison_columns].mean()

print(rookie_hitters_average_stats)
print(vet_hitters_average_stats)
# print(rookie_pitchers_average_stats)
# print(vet_pitchers_average_stats)


### visualize:

def save_hitting_comparison(rookie_stats, vet_stats, columns, title, filename):
    x = np.arange(len(columns))
    width = 0.35

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(x - width/2, rookie_stats, width, label='Rookies', color='skyblue')
    ax.bar(x + width/2, vet_stats, width, label='Veterans', color='orange')

    ax.set_xlabel('Hitting Stats')
    ax.set_ylabel('Average')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(columns, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_pitching_comparison(rookie_stats, vet_stats, columns, title, filename):
    x = np.arange(len(columns))
    width = 0.35

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(x - width/2, rookie_stats, width, label='Rookies', color='lightgreen')
    ax.bar(x + width/2, vet_stats, width, label='Veterans', color='purple')

    ax.set_xlabel('Pitching Stats')
    ax.set_ylabel('Average')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(columns, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Normalize hitting stats per plate appearance
def normalize_batter_stats(df, hitting_columns):
    for col in hitting_columns:
        if col not in ['OBP', 'SLG', 'OPS']:
            df[col + '_per_PA'] = df[col] / df['PA']
    return df

# Apply normalization
rookie_hitters_df = normalize_batter_stats(rookie_hitters_df, hitting_comparison_columns)
vet_hitters_df = normalize_batter_stats(vet_hitters_df, hitting_comparison_columns)

# Recalculate average stats for normalized columns
rookie_hitters_normalized_stats = rookie_hitters_df[[col + '_per_PA' if col not in ['OBP', 'SLG', 'OPS'] else col for col in hitting_comparison_columns]
].mean()
vet_hitters_normalized_stats = vet_hitters_df[[col + '_per_PA' if col not in ['OBP', 'SLG', 'OPS'] else col for col in hitting_comparison_columns]
].mean()

print(rookie_hitters_normalized_stats)
print(vet_hitters_normalized_stats)
# print(rookie_pitchers_average_stats)
# print(vet_pitchers_average_stats)

# Visualize comparison with normalized stats
save_hitting_comparison(
    rookie_hitters_normalized_stats.values,
    vet_hitters_normalized_stats.values,
    [col + '_per_PA' if col not in ['OBP', 'SLG', 'OPS'] else col for col in hitting_comparison_columns],
    'Rookie vs Veteran Hitters: Normalized by Plate Appearance',
    'hitting_comparison_normalized.png'
)

save_pitching_comparison(
    rookie_pitchers_average_stats.values,
    vet_pitchers_average_stats.values,
    pitching_comparison_columns,
    'Rookie vs Veteran Pitchers',
    'pitching_comparison_normalized.png'
)