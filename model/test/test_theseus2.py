# This file is part of EquivAlign.
# 
# Copyright [2024] [Authors of Paper: Correspondence-free SE(3) point cloud registration in RKHS via unsupervised equivariant learning]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Author Email: <Ray Zhang rzh@umich.edu>
import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)

def generate_data(num_points=100, a=1, b=0.5, noise_factor=0.01):
    # Generate data: 100 points sampled from the quadratic curve listed above
    data_x = torch.rand((1, num_points))
    noise = torch.randn((1, num_points)) * noise_factor
    data_y = a * data_x.square() + b + noise
    return data_x, data_y

def generate_learning_data(num_points, num_models):
    a, b = 3, 1
    data_batches = []
    for i in range(num_models):
        b = b + 2
        data = generate_data(num_points, a, b)
        data_batches.append(data)
    return data_batches

num_models = 10
data_batches = generate_learning_data(100, num_models)

fig, ax = plt.subplots()
for i in range(num_models):
    ax.scatter(data_batches[i][0], data_batches[i][1])
ax.set_xlabel('x');
ax.set_ylabel('y');



import theseus as th

data_x, data_y = generate_data()

# data is of type Variable
x = th.Variable(data_x, name="x")
y = th.Variable(data_y, name="y")
print(x.shape, y.shape)
# optimization variables are of type Vector with 1 degree of freedom (dof)
a = th.Vector(1, name="a")
b = th.Vector(1, name="b")

def quad_error_fn(optim_vars, aux_vars):
    a, b = optim_vars 
    x, y = aux_vars
    est = a.tensor * x.tensor.square() + b.tensor
    err = y.tensor - est
    return err

optim_vars = a, b
aux_vars = x, y
cost_function = th.AutoDiffCostFunction(
    optim_vars, quad_error_fn, 100, aux_vars=aux_vars, name="quadratic_cost_fn"
)
objective = th.Objective()
objective.add(cost_function)
optimizer = th.GaussNewton(
    objective,
    max_iterations=15,
    step_size=0.5,
)
theseus_optim = th.TheseusLayer(optimizer)

theseus_inputs = {
"x": data_x,
"y": data_y,
"a": 2 * torch.ones((1, 1)),
"b": torch.ones((1, 1))
}
with torch.no_grad():
    updated_inputs, info = theseus_optim.forward(
        theseus_inputs, optimizer_kwargs={"track_best_solution": True, "verbose":True})
print("Best solution:", info.best_solution)

# Plot the optimized function
fig, ax = plt.subplots()
ax.scatter(data_x, data_y);

a = info.best_solution['a'].squeeze()
b = info.best_solution['b'].squeeze()
x = torch.linspace(0., 1., steps=100)
y = a*x*x + b
ax.plot(x, y, color='k', lw=4, linestyle='--',
        label='Optimized quadratic')
ax.legend()

ax.set_xlabel('x');
ax.set_ylabel('y');
