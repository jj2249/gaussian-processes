import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ExpSineSquared
import matplotlib.pyplot as plt

with open("./cw1a.npy", 'rb') as f:
	data = np.load(f, allow_pickle=True).item() # 0 dimensional array of object dtype
	x = data['x']
	y = data['y']
	f.close()

# prefactor is the signal std_dev, length_scale is the RBF scale, noiselevel measures the noise variance
# kernel = np.exp(0)*RBF(length_scale=np.exp(-1), length_scale_bounds=(1e-5,1e5)) + WhiteKernel(noise_level=np.exp(0), noise_level_bounds=(1e-10, 1e3))
kernel = np.exp(0) * RBF(length_scale=np.exp(-1), length_scale_bounds=(1e-5,1e5))# + WhiteKernel(noise_level=np.exp(-10), noise_level_bounds=(1e-10, 1e3))
# kernel = np.exp(-10) * RBF(length_scale=np.exp(10), length_scale_bounds=(1e-5,1e5)) + WhiteKernel(noise_level=np.exp(0), noise_level_bounds=(1e-10, 1e3))

# x_axis = np.linspace(-5, 5, 1000).reshape(-1, 1)
# fig, ax = plt.subplots()

gpr = GaussianProcessRegressor(kernel=kernel)
# for i in range(5):
# 	y_samp = gpr.sample_y(x_axis, random_state=None)
# 	plt.plot(x_axis, y_samp, label='Sample {}'.format(i))

# prior_m, prior_std = gpr.predict(x_axis, return_std=True)
# ax.plot(x_axis, prior_m, label='Mean')
# ax.plot(x_axis, prior_m-1.96*prior_std, c='b', alpha=.6, label='95% CI')
# ax.plot(x_axis, prior_m+1.96*prior_std, c='b', alpha=.6)
# ax.fill_between(x_axis.flatten(), (prior_m-1.96*prior_std).flatten(), (prior_m+1.96*prior_std).flatten(), color='orange', alpha=.3)
# ax.set_xlabel('Input Feature')
# ax.set_ylabel('Output')
# plt.legend()
# fig.suptitle('Constant: 10, Length Scale: exp{1}')
# plt.show()

gpr.fit(x, y)

x_plot = np.linspace(-3.5, 3.5, 1000).reshape(-1, 1)
y_mean, y_std = gpr.predict(x_plot, return_std=True)
y_std = y_std.reshape(-1, 1)
# print(y_std.reshape(-1, 1).shape, y_mean.shape)

fig, ax = plt.subplots()
# for i in range(3):
# 	y_samp = gpr.sample_y(x_plot, random_state=None)
# 	plt.plot(x_plot, y_samp.flatten(), alpha=0.5, linewidth=1)
ax.scatter(x, y, marker='x', c='r', linewidths=1.0, label='Training Data')
ax.plot(x_plot, y_mean, c='black', label='Predictive Mean')
ax.plot(x_plot, y_mean-1.96*y_std, c='b', alpha=.6, label='95% CI')
ax.plot(x_plot, y_mean+1.96*y_std, c='b', alpha=.6)
ax.fill_between(x_plot.flatten(), (y_mean-1.96*y_std).flatten(), (y_mean+1.96*y_std).flatten(), color='orange', alpha=.3)
ax.set_xlabel('Input Feature')
ax.set_ylabel('Output')
# ax.set_xlim(-1, 1)
ax.legend()
fig.suptitle('Kernel: RBF, HyperParameters: {}, LML: {}'.format(np.exp(gpr.kernel_.theta), gpr.log_marginal_likelihood_value_))
print(gpr.kernel_)
plt.show()
