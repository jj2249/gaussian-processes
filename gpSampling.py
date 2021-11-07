import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared
import matplotlib.pyplot as plt

x_axis = np.linspace(-5, 5, 200).reshape(-1, 1)

kernel = 1.0*ExpSineSquared(length_scale=np.exp(-0.5), periodicity=np.exp(0))*RBF(length_scale=np.exp(0))
gpr = GaussianProcessRegressor(kernel=kernel)
print(gpr.kernel)
fig, ax = plt.subplots()
y_samp = gpr.sample_y(x_axis, random_state=None)
ax.plot(x_axis, y_samp)



y_mean, y_std = gpr.predict(x_axis, return_std=True)

# ax.plot(x_axis, y_mean, c='black', label='Predictive Mean')
# ax.plot(x_axis, y_mean-1.96*y_std, c='b', alpha=.6, label='95% CI')
# ax.plot(x_axis, y_mean+1.96*y_std, c='b', alpha=.6)
# ax.fill_between(x_axis.flatten(), (y_mean-1.96*y_std).flatten(), (y_mean+1.96*y_std).flatten(), color='orange', alpha=.3)
# ax.legend()
fig.suptitle("Sample from Product Kernel")
ax.set_xlabel('Input Feature')
ax.set_ylabel('Output')
plt.show()