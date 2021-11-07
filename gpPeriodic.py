import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, RBF
import matplotlib.pyplot as plt

with open("./cw1a.npy", 'rb') as f:
	data = np.load(f, allow_pickle=True).item() # 0 dimensional array of object dtype
	x = data['x']
	y = data['y']
	f.close()

kernel = 1.0*ExpSineSquared(length_scale=np.exp(0), periodicity=np.exp(0))+ 1.0*RBF(length_scale=np.exp(0)) + WhiteKernel(noise_level=np.exp(100), noise_level_bounds=(1e-10, 1e3))

gpr = GaussianProcessRegressor(kernel=kernel)
gpr.fit(x, y)

x_plot = np.linspace(-4, 4, 1000).reshape(-1, 1)
y_mean, y_std = gpr.predict(x_plot, return_std=True)
y_std = y_std.reshape(-1, 1)
# print(y_std.reshape(-1, 1).shape, y_mean.shape)

fig, ax = plt.subplots()
ax.scatter(x, y, marker='x', c='r', linewidths=1.0)
ax.plot(x_plot, y_mean, c='black')
ax.plot(x_plot, y_mean-1.96*y_std, c='b', alpha=.6)
ax.plot(x_plot, y_mean+1.96*y_std, c='b', alpha=.6)
ax.fill_between(x_plot.flatten(), (y_mean-1.96*y_std).flatten(), (y_mean+1.96*y_std).flatten(), color='orange', alpha=.3)
print(gpr.kernel_)
# ax.set_xlim(1.75, 3.75)
fig.suptitle('Kernel: Periodic, HyperParameters: {}, LML: {}'.format(np.exp(gpr.kernel_.theta), gpr.log_marginal_likelihood_value_))
plt.show()
