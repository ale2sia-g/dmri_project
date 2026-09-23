from dmri_project import *

y, point_estimate, gtab = get_preprocessed_data()
S0_init, evals_init, evecs_init = point_estimate

prior = frozen_prior()
print("prior logpdf:", round(prior.logpdf(S0_init, evals_init), 3))

lik = frozen_likelihood(gtab, y)
lp = lik.logpdf(np.array([S0_init]), evecs_init[None], evals_init[None])
print("likelihood logpdf:", round(float(lp), 3))