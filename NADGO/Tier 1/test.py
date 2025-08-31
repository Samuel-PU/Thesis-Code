import math, json
from components import LeakageMonitor, epsilon_est
m = LeakageMonitor(delta_budget=0.1, delta_kill=0.2, alpha=1e-3, q_min=1e-3, eps_sync=0.0)
print("k =", len(m._create_feature_alphabet()), "log2(k)=", math.log(len(m._create_feature_alphabet()),2))
print("epsilon_est( n=1   ) =", epsilon_est(1,   len(m.feature_alphabet), 1e-3, 1e-3))
print("epsilon_est( n=10  ) =", epsilon_est(10,  len(m.feature_alphabet), 1e-3, 1e-3))
print("epsilon_est( n=300 ) =", epsilon_est(300, len(m.feature_alphabet), 1e-3, 1e-3))