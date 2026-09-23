from dmri_project import *

np.random.seed(0)

y, point_estimate, gtab = get_preprocessed_data()
S0_init, evals_init, evecs_init = point_estimate
print("S0_init:", S0_init)
print("evals_init:", evals_init)
print("MD_init:", evals_init.mean())
print("FA_init:", __import__('dipy').reconst.dti.fractional_anisotropy(evals_init))

# Run VI (first run takes a few minutes, then cached)
posterior_vi = variational_inference(force_recompute=True)

# Draw samples and plot
y, point_estimate, gtab = get_preprocessed_data()
_, _, evecs_init = point_estimate
evec_principal = evecs_init[:, 0]

S0_vi, evals_vi, evecs_vi = posterior_vi.rvs(size=10000)
plot_results(S0_vi, evals_vi, evecs_vi, evec_principal, method="vi")

print("Done. See results_vi.png")