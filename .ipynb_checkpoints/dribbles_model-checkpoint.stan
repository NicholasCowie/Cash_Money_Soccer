data {
  int<lower=1> N;
  int<lower=1> attempts[N];
  int<lower=0> successes[N];
}
parameters {
  vector[N] ability;
  real mu;
  real<lower=0> sigma_ability;
}
model {
  vector[N] eta =  mu + ability * sigma_ability;
  mu ~ normal(0, 1);
  sigma_ability ~ student_t(6, 0, 1);
  ability ~ normal(0, 1);
  successes ~ binomial_logit(attempts, eta);
}
generated quantities {
  int successes_pred[N];
  vector[N] log_lik;
  for (n in 1:N){
    real eta_gq = mu + ability[n] * sigma_ability;
    successes_pred[n] = binomial_rng(attempts[n], inv_logit(eta_gq));
    log_lik[n] = binomial_logit_lpmf(successes[n]| attempts[n], eta_gq);
  }
}