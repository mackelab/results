import numpy as np
import time
from sbi.inference import base
import torch
from torch import nn, optim
import sbibm
import pickle
from torch.utils import data
from torch.distributions import Bernoulli
import matplotlib.pyplot as plt
from sbi import inference
from sbi.utils import likelihood_nn
import pandas as pd
from sbibm.tasks.ddm.utils import (
    BernoulliMN,
    MixedModelSyntheticDDM,
    run_mcmc,
    train_choice_net,
)
from joblib import Parallel, delayed

task = sbibm.get_task("ddm")
prior = task.get_prior_dist()
simulator = task.get_simulator()

# overall data set
num_examples = 100000
nxos = 1000
nthos = 10000
l_lower_bound = 1e-7
num_workers = 18

# xos = simulator(prior.sample((nxos,)))
# test_thetas = [prior.sample((nthos,)) for _ in range(nxos)]

# # data
# theta = prior.sample((num_examples,))
# x = simulator(theta)

# with open(f"data_{num_examples}.p", "wb") as fh:
#     pickle.dump(dict(theta=theta, x=x, xos=xos, test_thetas=test_thetas), fh)

# assert False

with open(f"data_{num_examples}.p", "rb") as fh:
    theta, x, xos, test_thetas = pickle.load(fh).values()

rts = abs(x)
choices = torch.ones_like(x)
choices[x < 0] = 0
theta_and_choices = torch.cat((theta, choices), dim=1)
validation_fraction = 0.1
stop_after_epochs = 20
batch_size = 100

# define experiments functions
def huberloss(y, yhat):
    diff = abs(y - yhat)

    err = np.zeros(y.numel())
    err[diff <= 1.0] = 0.5 * diff[diff <= 1.0] ** 2
    err[diff > 1.0] = 0.5 + diff[diff > 1.0]
    return err.mean()


def mse(y, yhat):
    return torch.mean((y - yhat) ** 2)


def run(
    use_log_rts,
    num_bins,
    num_transforms,
    base_distribution,
    tails,
    tail_bound,
    tail_bound_eps,
    hidden_features,
    n_hidden_layers,
    training_batch_size=100,
    stop_after_epochs=30,
    idx=-1,
    repetitions=2,
):
    results = []
    for repi in range(repetitions):
        try:
            # train choice net
            choice_net, vallp = train_choice_net(
                theta,
                choices,
                BernoulliMN(
                    n_hidden_layers=n_hidden_layers, n_hidden_units=hidden_features
                ),
                validation_fraction=validation_fraction,
                stop_after_epochs=stop_after_epochs,
                batch_size=training_batch_size,
            )
            # train flow
            density_estimator_fun = likelihood_nn(
                model="nsf",
                num_transforms=num_transforms,
                hidden_features=hidden_features,
                num_bins=num_bins,
                base_distribution=base_distribution,
                tails=tails,
                tail_bound=tail_bound,
                tail_bound_eps=tail_bound_eps,
                num_hidden_spline_context_layers=n_hidden_layers,
            )

            inference_method = inference.SNLE(
                density_estimator=density_estimator_fun,
                prior=prior,
            )
            inference_method = inference_method.append_simulations(
                theta=theta_and_choices,
                x=torch.log(rts) if use_log_rts else rts,
                from_round=0,
            )
            rt_flow = inference_method.train(
                training_batch_size=training_batch_size,
                show_train_summary=False,
                stop_after_epochs=stop_after_epochs,
            )

            mm = MixedModelSyntheticDDM(choice_net, rt_flow, use_log_rts=use_log_rts)

            with open(f"models/largesweep/mm_{idx}_{repi}.p", "wb") as fh:
                pickle.dump(mm, fh)

            # evaluate
            errs = []
            for xo, thos in zip(xos, test_thetas):

                # Sample test thetas from prior.
                xo = xo.reshape(-1, 1)
                # Extract positive RTs and choices for mixed model.
                rs = abs(xo)
                cs = torch.ones_like(rs)
                cs[xo < 0] = 0
                assert thos.shape == (nthos, 4)
                assert xo.shape == (1, 1)

                # Evaluate
                tic = time.time()
                lp_mm = mm.log_prob(
                    rs, cs, thos, ll_lower_bound=np.log(l_lower_bound)
                ).squeeze()
                lp_true = task.get_log_likelihood(
                    thos, data=xo.reshape(1, -1), l_lower_bound=l_lower_bound
                )
                errs.append(
                    [huberloss(lp_mm, lp_true), mse(lp_mm, lp_true), time.time() - tic]
                )

        except:
            print("failed")
            errs = np.zeros((nxos, 2))
        results.append(errs)
    return np.array(results)


use_log_rts = [False, True]
num_bins = [3, 5, 10]
num_transforms = [1, 2, 3]
base_distribution = ["normal", "lognormal"]
tail_bound = [5.0, 10.0]
tails = ["linear", "rectified"]
tail_bound_eps = [1e-7]
training_batch_size = 100
n_hidden_layers = [1, 2, 3]
hidden_features = [10, 25, 50]

import itertools

combos = itertools.product(
    use_log_rts,
    num_bins,
    num_transforms,
    base_distribution,
    tails,
    tail_bound,
    tail_bound_eps,
    hidden_features,
    n_hidden_layers,
)

arglist = [c for c in combos]
print(len(arglist))

# with open(f"models/choicenet{num_examples}.p", "wb") as fh:
#     pickle.dump(choice_net, fh)

results = Parallel(n_jobs=num_workers)(
    delayed(run)(*arg, idx=idx) for idx, arg in enumerate(arglist)
)

with open(f"results_{len(arglist)}_largesweep.p", "wb") as fh:
    pickle.dump(dict(results=results, arglist=arglist), fh)

# with open(f"results_{len(arglist)}.p", "rb") as fh:
#     print(pickle.load(fh))
