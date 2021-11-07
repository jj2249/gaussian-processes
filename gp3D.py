import numpy as np
# import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, WhiteKernel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
# import plotly.express as px
# import plotly.graph_objects as go
# import plotly.graph_objs
# from scipy.interpolate import griddata


with open("./cw1e.npy", 'rb') as f:
	data = np.load(f, allow_pickle=True).item() # 0 dimensional array of object dtype
	x1 = data['x'][:,0].flatten()
	x2 = data['x'][:,1].flatten()
	y = data['y'].flatten()
	f.close()

# X1 = x1.reshape((11, 11))
# X2 = x2.reshape((11, 11))
# Y = y.reshape((11, 11))
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.plot_surface(X1, X2, Y)
# # cax = ax.contourf(X1, X2, Y, 10, lw=1.5)
# ax.set_xlabel('x0')
# ax.set_ylabel('x1')
# # cbar = fig.colorbar(cax)
# fig.suptitle('3D Viewer')
# plt.show()


# kernel1 = 1.0*RBF(length_scale=np.random.rand(2)) + WhiteKernel(noise_level=np.exp(0), noise_level_bounds=(1e-10, 1e3))
kernel1 = 1.11**2 * RBF(length_scale=[1.51, 1.29]) + WhiteKernel(noise_level=0.0105)
# kernel2 = 1.0*RBF(length_scale=np.random.rand(2)) + 1.0*RBF(length_scale=np.random.rand(2)) + WhiteKernel(noise_level=np.exp(0), noise_level_bounds=(1e-10, 1e3))
kernel2 = 0.71**2 * RBF(length_scale=[4.28e+04, 0.986]) + 1.11**2 * RBF(length_scale=[1.45, 1e+05]) + WhiteKernel(noise_level=0.00957)
gpr1 = GaussianProcessRegressor(kernel=kernel1)
gpr2 = GaussianProcessRegressor(kernel=kernel2)

inputs = np.array([x1, x2]).T
outputs = y

gpr1.fit(inputs, outputs)
gpr2.fit(inputs, outputs)

print(gpr1.kernel_)
print(gpr2.kernel_)
print(gpr1.log_marginal_likelihood())
print(gpr2.log_marginal_likelihood())



# # mesh = np.meshgrid(axis, axis)
# input_axis = np.array(list(zip(x1_plot.flatten(), x2_plot.flatten()))).reshape(10, 2, 10)
# print(input_axis.shape)
# print(input_axis)
# print(mean1)

def cartesian(arrays, out=None):

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    #m = n / arrays[0].size
    m = int(n / arrays[0].size) 
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
        #for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out

sams_p_ax=50
axis = np.linspace(-10, 10, sams_p_ax)
input_axis = cartesian([axis, axis])
x2_plot, x1_plot = np.meshgrid(axis, axis)

mean1, std1 = gpr1.predict(input_axis, return_std=True)
mean2, std2 = gpr2.predict(input_axis, return_std=True)
mean1 = mean1.reshape(sams_p_ax, sams_p_ax)
mean2 = mean2.reshape(sams_p_ax, sams_p_ax)
std1 = std1.reshape(sams_p_ax, sams_p_ax)
std2 = std2.reshape(sams_p_ax, sams_p_ax)

fig1 = plt.figure()
ax1 = fig1.add_subplot(projection='3d')
ax1.plot_surface(x1_plot, x2_plot, mean1, alpha=0.7)
ax1.plot_surface(x1_plot, x2_plot, mean1+1.96*std1, alpha=0.3, color='orange')
ax1.plot_surface(x1_plot, x2_plot, mean1-1.96*std1, alpha=1., color='orange')

ax1.scatter(x1, x2, y, c='r', linewidths=.1, alpha=0.4)
ax1.set_xlabel('x0')
ax1.set_ylabel('x1')
fig1.suptitle('Product Kernel')

fig2 = plt.figure()
ax2 = fig2.add_subplot(projection='3d')
fig2.suptitle('Sum Kernel')
ax2.set_xlabel('x0')
ax2.set_ylabel('x1')
ax2.plot_surface(x1_plot, x2_plot, mean2, alpha=0.7)
ax2.plot_surface(x1_plot, x2_plot, mean2+1.96*std2, alpha=0.3, color='orange')
ax2.plot_surface(x1_plot, x2_plot, mean2-1.96*std2, alpha=1., color='orange')

ax2.scatter(x1, x2, y, c='r', linewidths=.1, alpha=0.4)
plt.show()
index = 25
plt.subplot(121)
plt.title('Product Kernel')
plt.plot(axis, mean1[index], color='black', lw=1.0)
plt.plot(axis, mean1[index]+1.96*std1[index], color='b', alpha=0.6)
plt.plot(axis, mean1[index]-1.96*std1[index], color='b', alpha=0.6)
plt.fill_between(axis, (mean1[index]-1.96*std1[index]).flatten(), (mean1[index]+1.96*std1[index]).flatten(), color='orange', alpha=.3)
plt.scatter(x2.reshape(11, 11)[5], y.reshape(11, 11)[5], marker='x', c='r', linewidths=1.0)
plt.subplot(122)
plt.title('Sum Kernel')
plt.plot(axis, mean2[index], color='black', lw=1.0)
plt.plot(axis, mean2[index]+1.96*std2[index], color='b', alpha=0.6)
plt.plot(axis, mean2[index]-1.96*std2[index], color='b', alpha=0.6)
plt.fill_between(axis, (mean2[index]-1.96*std2[index]).flatten(), (mean2[index]+1.96*std2[index]).flatten(), color='orange', alpha=.3)
plt.scatter(x2.reshape(11, 11)[5], y.reshape(11, 11)[5], marker='x', c='r', linewidths=1.0)

plt.show()
