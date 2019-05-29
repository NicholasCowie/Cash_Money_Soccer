import pandas as pd
import numpy as np
import seaborn as sns
import scipy as sp
import os
import matplotlib.pyplot as plt

%cd ~/Documents/Cash_Money_Soccer/

Events = pd.read_csv('data/events.csv', low_memory=False)

Events = Events[Events['competition_name'] == 'NWSL']

Events.possession

# Preliminary Analysis

Shots = Events.groupby('match_id').agg({'shot_outcome_name': 'count','shot_statsbomb_xg': np.sum})

hist_shots = sns.distplot(Shots.shot_outcome_name, kde=False)
hist_shots_fig = hist_shots.get_figure()


hist_shots_fig.savefig('Shot_Hist.png')


hist_shots_plot = hist_shots.get_figure()

hist_shots_plot.savefig('Histogram_Shots_Taken.png')

Goals = Events[Events.shot_outcome_name=='Goal']

Goal_Matches = Goals.groupby('match_id').agg({'shot_outcome_name': 'count', 'shot_statsbomb_xg': np.sum})

Shot_Goal_Match = pd.merge(Goal_Matches, Shots, how='inner', on='match_id')

Shot_Goal_Match.plot.scatter(y='shot_statsbomb_xg_x', x='shot_statsbomb_xg_y')

Shot_Goal_Match.columns = ['Goals', 'Sum_Goal_Score', 'Shots', 'Sum_Shot_Score']


Shot_Reg = sns.regplot(y='Goals', x='Sum_Shot_Score', data=Shot_Goal_Match)

Shot_Reg_fig = Shot_Reg.get_figure()


Shot_Reg_fig.savefig('Goal_Shot_Corr.png')
