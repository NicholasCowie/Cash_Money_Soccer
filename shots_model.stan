data {
  int<lower=1> N;
  int<lower=1> team_attack[N];
  int<lower=1> team_defend[N];
  int<lower=1> k;
  int<lower=0> shot[N];
  int<lower=1> L;
  int<lower=1> team_1[L];
  int<lower=1> team_2[L];

}
parameters {
  vector[k] ability_Az;
  vector[k] ability_Dz;
  real mu;
  real<lower=0> sigma_ability_A;
  real<lower=0> sigma_ability_D;
}
transformed parameters {
    vector[k] ability_A = ability_Az * sigma_ability_A;
    vector[k] ability_D = ability_Dz * sigma_ability_D;
}
model {
  vector[N] eta;
  mu ~ normal(0, 1);
  sigma_ability_A ~ student_t(6, 0, 1);
  sigma_ability_D ~ student_t(6, 0, 1);
  ability_Az ~ normal(0, 1);
  ability_Dz ~ normal(0, 1);

  eta = mu + ability_A[team_attack] - ability_D[team_defend];
  shot ~ bernoulli_logit(eta);

}
generated quantities {
  int successes_pred[L];
  vector[N] pred_shot;
  vector[N] log_like;
  vector[N] eta_ll =  mu + ability_A[team_attack] - ability_D[team_defend];
  
  for (l in 1:L){
    real eta_gq = mu + (ability_A[team_1[l]] - ability_D[team_2[l]]);
    successes_pred[l] = bernoulli_rng(inv_logit(eta_gq));
  }
  
  for (n in 1:N){
      log_like[n] = bernoulli_logit_lpmf(shot[n]|eta_ll[n]);
      pred_shot[n] = bernoulli_rng(inv_logit(eta_ll[n]));
  }
}
