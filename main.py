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
# comparison stats selected to highlight key performance metrics, normalizing for playing time, to fairly compare rookie and veteran performance.

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

# Normalize hitting stats per plate appearance
def normalize_batter_stats(df, hitting_columns):
    for col in hitting_columns:
        if col not in ['OBP', 'SLG', 'OPS']:
            df[col + '_per_PA'] = df[col] / df['PA']
    return df

rookie_hitters_df = normalize_batter_stats(rookie_hitters_df, hitting_comparison_columns)
vet_hitters_df = normalize_batter_stats(vet_hitters_df, hitting_comparison_columns)


### visualize

def save_hitting_comparison(df_rookies, df_vets, columns, title, filename):
    fig, ax = plt.subplots(figsize=(20, 10))
    
    plot_data = []
    labels = []
    for col in columns:
        rookie_col = pd.to_numeric(df_rookies[col], errors='coerce').dropna()
        vet_col = pd.to_numeric(df_vets[col], errors='coerce').dropna()
        
        plot_data.append(rookie_col)
        plot_data.append(vet_col)
        
        labels.append(f'{col} (R)')
        labels.append(f'{col} (V)')
    
    bp = ax.boxplot(plot_data, 
                    patch_artist=True,
                    labels=labels,
                    showfliers=False)
    
    colors = ['skyblue', 'orange'] * len(columns)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, color='skyblue', label='Rookies'),
        plt.Rectangle((0,0), 1, 1, color='orange', label='Veterans')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Hitting Stats', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_pitching_comparison(df_rookies, df_vets, columns, title, filename):
    fig, ax = plt.subplots(figsize=(20, 10))
    
    plot_data = []
    labels = []
    for col in columns:
        rookie_col = pd.to_numeric(df_rookies[col], errors='coerce').dropna()
        vet_col = pd.to_numeric(df_vets[col], errors='coerce').dropna()
        
        plot_data.append(rookie_col)
        plot_data.append(vet_col)
        
        labels.append(f'{col} (R)')
        labels.append(f'{col} (V)')
    
    bp = ax.boxplot(plot_data, 
                    patch_artist=True,
                    labels=labels,
                    showfliers=False)
    
    colors = ['lightgreen', 'purple'] * len(columns)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, color='lightgreen', label='Rookies'),
        plt.Rectangle((0,0), 1, 1, color='purple', label='Veterans')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Pitching Stats', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

save_hitting_comparison(
    rookie_hitters_df,
    vet_hitters_df,
    [col + '_per_PA' if col not in ['OBP', 'SLG', 'OPS'] else col for col in hitting_comparison_columns],
    'Rookie vs Veteran Hitters: Normalized by Plate Appearance',
    'hitting_comparison_normalized.png'
)

save_pitching_comparison(
    rookie_pitchers_df,
    vet_pitchers_df,
    pitching_comparison_columns,
    'Rookie vs Veteran Pitchers',
    'pitching_comparison_normalized.png'
)