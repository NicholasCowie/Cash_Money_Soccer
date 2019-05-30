# Predictive performance assessment

The predicted share of possessions ended with shots for every team in every match they played against each other can be used to assess the model's predictive power. Setting a threshold for the predicted share of shots allows to make a classifier that predicts goals and related outcomes. The questions that can be answered are f.e. if the team will win the game or if the team will score at least once.

The results of the analysis are presented below. The ability-distance model has predictive power which is shown by AUC being around 0.7 for both target variables. It is not extremely high though which shows that due to the complexity of the task more complex models with more variables should be used.

![Figure]{score.png}
![Figure]{win.png}


# Conclusions

Both hierarchical models reveal the differences in teams abilities to attack and defend. Both models show similar team strategy profiles. However, the shot distance is strongly correlated with the probability of possession ending with a shot, so the attack and defense abilities differences between teams are less visible in the model that does not take the distance into account. It is also clear from the figure that the ability-only model is not able to make up for the lack of this information and the means of the predicted probabilities of the shots grouped by possession distances is far from the true value. 

![Figure]{predicted_poster_ability_dist.png}

The pair plot for the teams attacking abilities shows that the joint probabilities show a slight positive correlations, and the teams attack vs defend abilities shows a negative correlation which follows the general assumptions in the model's concepts.

Even though the information about goals and matches outcomes hasn't been explicitely used in the analysis the inferred team abilities for attack and defence with the correction on the shot distance have predictive power for teams scoring during the match. The classifier can be constructed with predicted probability of a shots during the game.

The following analysis explores the factors influencing the fooball matches outcome and contributes to ranking their importance.


# Problems and discussion

Even though the teams strategy profiles appear more clear with the model that takes distances to the account the ability coefficients are very close to zero for about half of the teams which suggest those factors not contributing much to the teams performance. It suggests the results could be improved by adding more variables to the generalised linear model. Just the distance impacted the model's performance, it could be that such factors as individual players, duration of a possession or a home/away stadium should be added into the analysis.

Even though the model shows predictive power for the matches outcome it was not validated on any dataset that wasn't used in the training which means it most likely overfitted. Either extra dataset for the same team matches should be added as a test set or the part of the data for the current competition must be separated and used as a validation set.

Only one competition was used for running the analysis which might suggest the results could be biased towards the few dozens matches and only eight teams. However, the number of possessions in all of those games is probably large enough to come to the relevant conclusions.

Since the analysis of the possessions and shots in the game does not predict the match outcome with a high accuracy it can not be considered the final model but only as a stepping stone in a more complex model.


