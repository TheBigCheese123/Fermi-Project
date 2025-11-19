# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 16:13:19 2025

@author: charl
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import pyro
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
import seaborn as sns
import pandas as pd

pyro.set_rng_seed(1)

assert issubclass(PyroModule[nn.Linear], nn.Linear)
assert issubclass(PyroModule[nn.Linear], PyroModule)

#####  y = 3x + 2  #####
x_data = torch.linspace(0, 100, 101)
y_data = 3 * x_data + 2

##### Gaussian noise profile #####
master_sigma = 5
y_data = y_data + master_sigma*torch.randn_like(y_data)

#print(y_data - (3*x_data)+2)

iterations = 5000
'''
class BayesianRegression(PyroModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        self.linear.weight = PyroSample(dist.Normal(0., 5.).expand([out_features, in_features]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(0., 10.).expand([out_features]).to_event(1))

    def forward(self, x, y=None):
        sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        mean = self.linear(x).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean
'''

def BayesianModel(x_data, y_data=None):
    grad0 = pyro.sample("grad0", dist.Normal(0, 10))
    intercept0 = pyro.sample("intercept0", dist.Normal(0, 10))
    sigma = pyro.sample("sigma", dist.Uniform(0., 50.))
    
    mean = intercept0 + grad0 * x_data
    
    with pyro.plate("data", len(x_data)):
        return pyro.sample("obs", dist.Normal(mean, sigma), obs=y_data)

pyro.render_model(BayesianModel, model_args=(x_data, y_data), render_distributions=True)

def train(model, loss_fn, optim):
    y_pred = model(x_data)#.squeeze(-1)
    loss = loss_fn(y_pred, y_data)
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss

total_runs = 1
run_results = np.zeros((total_runs,2))

for i in range(total_runs):
    x_data = x_data.unsqueeze(1)
    y_data = y_data.unsqueeze(1)
    
    linear_regression_model = PyroModule[nn.Linear](1,1)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optim = torch.optim.Adam(linear_regression_model.parameters(), lr=0.05)
    
    for j in range(iterations):
        loss = train(linear_regression_model, loss_fn, optim)
        #if (j + 1) % 1000 == 0:
            #print("[iteration %04d] loss: %.4f" % (j + 1, loss.item()))
    
    # Inspect learned parameters
    print("Learned parameters:")
    num=0
    for name, param in linear_regression_model.named_parameters():
        print(name, param.data.numpy())
        run_results[i, num] = param.data.numpy()
        num = num+1
        
print("Final parameters:")
print("Weight (gradient) mean & std.:", np.mean(run_results[:, 0]), np.std(run_results[:, 0]))
print("Bias (y-intercept) mean & std.:", np.mean(run_results[:, 1]), np.std(run_results[:, 1]))

#0.008302377344580227 0.5534771154007225
#Weight (gradient) mean & std.: 2.964349286556244 0.00984933347506304
#Bias (y-intercept) mean & std.: 2.0607680678367615 0.6566041662854095

#fig = plt.figure()
#ax = fig.gca()
#ax.scatter(x_data, y_data, marker='x')
#ax.plot(x_data, linear_regression_model(x_data).detach().cpu().numpy(), color='black')
#plt.show()

print("BAYESIAN TIME OH YEAH")

x_data = x_data.squeeze(1)
y_data = y_data.squeeze(1)

pyro.clear_param_store()

#model = BayesianModel(x_data, y_data)
guide = pyro.infer.autoguide.AutoDiagonalNormal(BayesianModel)

adam = pyro.optim.Adam({"lr": 0.005})
svi = SVI(BayesianModel, guide, adam, loss=Trace_ELBO())

for j in range(iterations):
    loss = svi.step(x_data, y_data)
    if j % 100 == 0:
        print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(x_data)))

guide.requires_grad_(False)
for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name).data.cpu().numpy())#
    
with pyro.plate("samples", 800, dim=-1):
    samples = guide(x_data)
    
grads = samples["grad0"]
intercepts = samples["intercept0"]
errors = samples["sigma"]
    
fig = plt.figure(figsize=(10, 6))
sns.histplot(grads.detach().cpu().numpy(), kde=True, stat="density", label="Gradients")
plt.title("Predicted probability distribution for the gradient")
plt.legend()

fig = plt.figure(figsize=(10, 6))
sns.histplot(intercepts.detach().cpu().numpy(), kde=True, stat="density", color='orange', label="y-intercepts")
plt.title("Predicted probability distribution for the y-intercept")
plt.legend()

fig = plt.figure(figsize=(10, 6))
sns.histplot(errors.detach().cpu().numpy(), kde=True, stat="density", color='green', label="Noise")
plt.title("Predicted probability distribution for the noise")
plt.legend()

predictive = pyro.infer.Predictive(BayesianModel, guide=guide, num_samples=800)
svi_samples = predictive(x_data, y_data=None)
svi_y_data = svi_samples["obs"]

predictions = pd.DataFrame({
    "x_data": x_data,
    "y_mean": svi_y_data.mean(0).detach().cpu().numpy(),
    "y_perc_2.5": svi_y_data.kthvalue(int(len(svi_y_data) * 0.025), dim=0)[0].detach().cpu().numpy(),
    "y_perc_97.5": svi_y_data.kthvalue(int(len(svi_y_data) * 0.975), dim=0)[0].detach().cpu().numpy(),
    "true_y_data": y_data,
})

fig = plt.figure()
ax = fig.gca()
ax.scatter(x_data, y_data, marker='x', label = 'Actual data points')
ax.plot(predictions["x_data"], predictions["y_mean"], color='black', label='Line of best fit')
ax.fill_between(predictions["x_data"], predictions["y_perc_2.5"], predictions["y_perc_97.5"], label = '95% confidence interval', alpha=0.5)
plt.title("Bayesian Regression applied to a y=3x+2 line with Gaussian noise $\sigma = 10$")
plt.show()
