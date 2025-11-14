import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt

'''
trials = ['Base', 'Half length scale', 'Double length scale', 'Double length scale, 1.5 resolution', 'Double length scale, half resolution',
          'Double length scale, half x resolution', 'Double length scale, half t resolution', '1.5 resolution', 'Half resolution']
means = [9.47e-3, 1.09e-2, 6.99e-3, 6.88e-3, 7.71e-3, 1.78e-2, 8.63e-3, 8.19e-3, 1.10e-2]
stddevs = [4.56e-3, 3.78e-3, 3.56e-3, 3.67e-3, 4.05e-3, 7.79e-3, 4.37e-3, 4.23e-3, 5.61e-3]

n = 100
df = 198

def t_test(mean1, stddev1, mean2, stddev2):
    s = np.sqrt((stddev1**2 + stddev2**2)/2)
    t = (mean1 - mean2)/(s * np.sqrt(2/n))
    return 2 * (1 - stat.t.cdf(abs(t), df))

def compare(ind1, ind2):
    p = t_test(means[ind1], stddevs[ind1], means[ind2], stddevs[ind2])
    if (p < 0.05):
        print('Difference between ', trials[ind1], ' and ', trials[ind2], ' is statistically significant.')
    else:
        print('Not statistically significant')

compare(0,1)
compare(0,2)
compare(0,7)
compare(0,8)
compare(5,6)
compare(2,7)
compare(1,8)
'''
'''
S_i = np.load("s_iter.npy")
S_sol = np.load("s_diff.npy")
E_n = np.zeros(100)
n = np.linspace(1,50)

for i in range(100):
    S_in = S_i[i]
    E_n[i] = np.max(np.abs(S_in - S_sol)/S_sol)

plt.plot(E_n)
plt.yscale("log")
plt.show()

S_fin_ft = np.loadtxt("Dullemond4/A_00/s_final.out")
plt.plot(S_fin_ft)
plt.show()

S_fin = np.load("s_final.npy")
plt.plot(S_fin)
plt.show()
'''

taui_ft = np.loadtxt("Dullemond4/A_00/taui.out")
taui_py = np.load("taui.npy")
print(taui_ft - taui_py)

s_diff_ft = np.loadtxt("Dullemond4/A_00/s_diff.out")
s_diff_py = np.load("s_diff.npy")
print(s_diff_ft - s_diff_py)

taui_ft = np.loadtxt("Dullemond4/A_00/taui.out")
taui_py = np.load("taui.npy")
print(taui_ft - taui_py)

s_final_ft = np.loadtxt("Dullemond4/A_00/s_final.out")
s_final_py = np.load("s_final.npy")
print(s_final_ft - s_final_py)

s_iter_ft = np.loadtxt("Dullemond4/A_00/s_iter.out")
s_iter_ft = s_iter_ft.reshape(50, -1)
s_iter_py = np.load("s_iter.npy")
print(s_iter_ft.shape, s_iter_py.shape)