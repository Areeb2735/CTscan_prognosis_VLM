from math import ceil, sqrt
from typing import Optional, Union

import numpy as np
import torch


TensorOrArray = Union[torch.Tensor, np.ndarray]


def encode_survival(time: Union[float, int, TensorOrArray],
                    event: Union[int, bool, TensorOrArray],
                    bins: TensorOrArray) -> torch.Tensor:
    """Encodes survival time and event indicator in the format
    required for MTLR training.

    For uncensored instances, one-hot encoding of binned survival time
    is generated. Censoring is handled differently, with all possible
    values for event time encoded as 1s. For example, if 5 time bins are used,
    an instance experiencing event in bin 3 is encoded as [0, 0, 0, 1, 0], and
    instance censored in bin 2 as [0, 0, 1, 1, 1]. Note that an additional
    'catch-all' bin is added, spanning the range `(bins.max(), inf)`.

    Parameters
    ----------
    time
        Time of event or censoring.
    event
        Event indicator (0 = censored).
    bins
        Bins used for time axis discretisation.

    Returns
    -------
    torch.Tensor
        Encoded survival times.
    """
    # TODO this should handle arrays and (CUDA) tensors
    if isinstance(time, (float, int, np.ndarray)):
        time = np.atleast_1d(time)
        time = torch.tensor(time)
    if isinstance(event, (int, bool, np.ndarray)):
        event = np.atleast_1d(event)
        event = torch.tensor(event)

    if isinstance(bins, np.ndarray):
        bins = torch.tensor(bins)

    try:
        device = bins.device
    except AttributeError:
        device = "cpu"

    time = np.clip(time, 0, bins.max())
    # add extra bin [max_time, inf) at the end
    y = torch.zeros((time.shape[0], bins.shape[0] + 1),
                    dtype=torch.float,
                    device=device)
    # For some reason, the `right` arg in torch.bucketize
    # works in the _opposite_ way as it does in numpy,
    # so we need to set it to True
    bin_idxs = torch.bucketize(time, bins, right=True)
    for i, (bin_idx, e) in enumerate(zip(bin_idxs, event)):
        if e == 1:
            y[i, bin_idx] = 1
        else:
            y[i, bin_idx:] = 1
    return y.squeeze()


def reset_parameters(model: torch.nn.Module) -> torch.nn.Module:
    """Resets the parameters of a PyTorch module and its children."""
    for m in model.modules():
        try:
            m.reset_parameters()
        except AttributeError:
            continue
    return model


def make_time_bins(times: TensorOrArray,
                   num_bins: Optional[int] = None,
                   use_quantiles: bool = True,
                   event: Optional[TensorOrArray] = None
                   ) -> torch.Tensor:
    """Creates the bins for survival time discretisation.

    By default, sqrt(num_observation) bins corresponding to the quantiles of
    the survival time distribution are used, as in https://github.com/haiderstats/MTLR.

    Parameters
    ----------
    times
        Array or tensor of survival times.
    num_bins
        The number of bins to use. If None (default), sqrt(num_observations)
        bins will be used.
    use_quantiles
        If True, the bin edges will correspond to quantiles of `times`
        (default). Otherwise, generates equally-spaced bins.
    event
        Array or tensor of event indicators. If specified, only samples where
        event == 1 will be used to determine the time bins.

    Returns
    -------
    torch.Tensor
        Tensor of bin edges.
    """
    # TODO this should handle arrays and (CUDA) tensors
    if event is not None:
        times = times[event == 1]
    if num_bins is None:
        num_bins = ceil(sqrt(len(times)))
    if use_quantiles:
        # NOTE we should switch to using torch.quantile once it becomes
        # available in the next version
        bins = np.unique(np.quantile(times, np.linspace(0, 1, num_bins)))
    else:
        bins = np.linspace(times.min(), times.max(), num_bins)
    bins = torch.tensor(bins, dtype=torch.float)
    return bins


def make_optimizer(opt_cls: torch.optim.Optimizer,
                   model: torch.nn.Module,
                   **kwargs) -> torch.optim.Optimizer:
    """Creates a PyTorch optimizer for MTLR training.

    This is a helper function to instantiate an optimizer with no weight decay
    on biases (which shouldn't be regularized) and MTLR parameters (which have
    a separate regularization mechanism). Note that the `opt_cls` argument
    should be the optimizer class, not an instantiated object (e.g. optim.Adam
    instead of optim.Adam(model.parameters(), ...)).

    Parameters
    ----------
    opt_cls
        The optimizer class to instantiate.
    model
        The PyTorch module whose parameters should be optimized.
    kwargs
        Additional keyword arguments to optimizer constructor.

    Returns
    -------
    torch.optim.Optimizer
        The instantiated optimizer object.

    """
    params_dict = dict(model.named_parameters())
    weights = [v for k, v in params_dict.items() if "mtlr" not in k and "bias" not in k]
    biases = [v for k, v in params_dict.items() if "bias" in k]
    mtlr_weights = [v for k, v in params_dict.items() if "mtlr_weight" in k]
    # Don't use weight decay on the biases and MTLR parameters, which have
    # their own separate L2 regularization
    optimizer = opt_cls([
        {"params": weights},
        {"params": biases, "weight_decay": 0.},
        {"params": mtlr_weights, "weight_decay": 0.},
    ], **kwargs)
    return optimizer

from typing import Union

import numpy as np
from scipy.interpolate import interp1d
import torch
import torch.nn as nn


class MTLR(nn.Module):
    """Multi-task logistic regression for individualised
    survival prediction.

    The MTLR time-logits are computed as:
    `z = sum_k x^T w_k + b_k`,
    where `w_k` and `b_k` are learnable weights and biases for each time
    interval.

    Note that a slightly more efficient reformulation is used here, first
    proposed in [2]_.

    References
    ----------
    ..[1] C.-N. Yu et al., ‘Learning patient-specific cancer survival
    distributions as a sequence of dependent regressors’, in Advances in neural
    information processing systems 24, 2011, pp. 1845–1853.
    ..[2] P. Jin, ‘Using Survival Prediction Techniques to Learn
    Consumer-Specific Reservation Price Distributions’, Master's thesis,
    University of Alberta, Edmonton, AB, 2015.
    """

    def __init__(self, in_features: int, num_time_bins: int):
        """Initialises the module.

        Parameters
        ----------
        in_features
            Number of input features.
        num_time_bins
            The number of bins to divide the time axis into.
        """
        super().__init__()
        if num_time_bins < 1:
            raise ValueError("The number of time bins must be at least 1")
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
        self.in_features = in_features
        self.num_time_bins = num_time_bins + 1 # + extra time bin [max_time, inf)

        self.mtlr_weight = nn.Parameter(torch.Tensor(self.in_features,
                                                     self.num_time_bins - 1))
        self.mtlr_bias = nn.Parameter(torch.Tensor(self.num_time_bins - 1))

        # `G` is the coding matrix from [2]_ used for fast summation.
        # When registered as buffer, it will be automatically
        # moved to the correct device and stored in saved
        # model state.
        self.register_buffer(
            "G",
            torch.tril(
                torch.ones(self.num_time_bins - 1,
                           self.num_time_bins,
                           requires_grad=True)))
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass on a batch of examples.

        Parameters
        ----------
        x : torch.Tensor, shape (num_samples, num_features)
            The input data.

        Returns
        -------
        torch.Tensor, shape (num_samples, num_time_bins)
            The predicted time logits.
        """
        out = torch.matmul(x, self.mtlr_weight) + self.mtlr_bias
        return torch.matmul(out, self.G)

    def reset_parameters(self):
        """Resets the model parameters."""
        nn.init.xavier_normal_(self.mtlr_weight)
        nn.init.constant_(self.mtlr_bias, 0.)

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_features={self.in_features},"
                f" num_time_bins={self.num_time_bins})")


def masked_logsumexp(x: torch.Tensor,
                     mask: torch.Tensor,
                     dim: int = -1) -> torch.Tensor:
    """Computes logsumexp over elements of a tensor specified by a mask
    in a numerically stable way.

    Parameters
    ----------
    x
        The input tensor.
    mask
        A tensor with the same shape as `x` with 1s in positions that should
        be used for logsumexp computation and 0s everywhere else.
    dim
        The dimension of `x` over which logsumexp is computed. Default -1 uses
        the last dimension.

    Returns
    -------
    torch.Tensor
        Tensor containing the logsumexp of each row of `x` over `dim`.
    """
    max_val, _ = (x * mask).max(dim=dim)
    max_val = torch.clamp_min(max_val, 0)
    return torch.log(
        torch.sum(torch.exp(x - max_val.unsqueeze(dim)) * mask,
                  dim=dim)) + max_val


def mtlr_neg_log_likelihood(logits: torch.Tensor,
                            target: torch.Tensor,
                            model: torch.nn.Module,
                            C1: float,
                            average: bool = False) -> torch.Tensor:
    """Computes the negative log-likelihood of a batch of model predictions.

    Parameters
    ----------
    logits : torch.Tensor, shape (num_samples, num_time_bins)
        Tensor with the time-logits (as returned by the MTLR module) for one
        instance in each row.
    target : torch.Tensor, shape (num_samples, num_time_bins)
        Tensor with the encoded ground truth survival.
    model
        PyTorch Module with at least `MTLR` layer.
    C1
        The L2 regularization strength.
    average
        Whether to compute the average log likelihood instead of sum
        (useful for minibatch training).

    Returns
    -------
    torch.Tensor
        The negative log likelihood.
    """
    censored = target.sum(dim=1) > 1
    nll_censored = masked_logsumexp(logits[censored], target[censored]).sum() if censored.any() else 0
    nll_uncensored = (logits[~censored] * target[~censored]).sum() if (~censored).any() else 0

    # the normalising constant
    norm = torch.logsumexp(logits, dim=1).sum()

    nll_total = -(nll_censored + nll_uncensored - norm)
    if average:
        nll_total = nll_total / target.size(0)

    # L2 regularization
    for k, v in model.named_parameters():
        if "mtlr_weight" in k:
            nll_total += C1/2 * torch.sum(v**2)

    return nll_total


def mtlr_survival(logits: torch.Tensor) -> torch.Tensor:
    """Generates predicted survival curves from predicted logits.

    Parameters
    ----------
    logits
        Tensor with the time-logits (as returned by the MTLR module) for one
        instance in each row.

    Returns
    -------
    torch.Tensor
        The predicted survival curves for each row in `pred` at timepoints used
        during training.
    """
    # TODO: do not reallocate G in every call
    G = torch.tril(torch.ones(logits.size(1),
                              logits.size(1))).to(logits.device)
    density = torch.softmax(logits, dim=1)
    return torch.matmul(density, G)


def mtlr_survival_at_times(logits: torch.Tensor,
                           train_times: Union[torch.Tensor, np.ndarray],
                           pred_times: np.ndarray) -> np.ndarray:
    """Generates predicted survival curves at arbitrary timepoints using linear
    interpolation.

    Notes
    -----
    This function uses scipy.interpolate internally and returns a Numpy array,
    in contrast with `mtlr_survival`.

    Parameters
    ----------
    logits
        Tensor with the time-logits (as returned by the MTLR module) for one
        instance in each row.
    train_times
        Time bins used for model training. Must have the same length as the
        first dimension of `pred`.
    pred_times
        Array of times used to compute the survival curve.

    Returns
    -------
    np.ndarray
        The survival curve for each row in `pred` at `pred_times`. The values
        are linearly interpolated at timepoints not used for training.
    """
    train_times = np.pad(train_times, (1, 0))
    surv = mtlr_survival(logits).detach().cpu().numpy()
    interpolator = interp1d(train_times, surv)
    return interpolator(np.clip(pred_times, 0, train_times.max()))


def mtlr_hazard(logits: torch.Tensor) -> torch.Tensor:
    """Computes the hazard function from MTLR predictions.

    The hazard function is the instantenous rate of failure, i.e. roughly
    the risk of event at each time interval. It's computed using
    `h(t) = f(t) / S(t)`,
    where `f(t)` and `S(t)` are the density and survival functions at t,
    respectively.

    Parameters
    ----------
    logits
        The predicted logits as returned by the `MTLR` module.

    Returns
    -------
    torch.Tensor
        The hazard function at each time interval in `y_pred`.
    """
    return torch.softmax(
        logits, dim=1)[:, :-1] / (mtlr_survival(logits) + 1e-15)[:, 1:]


def mtlr_risk(logits: torch.Tensor) -> torch.Tensor:
    """Computes the overall risk of event from MTLR predictions.

    The risk is computed as the time integral of the cumulative hazard,
    as defined in [1]_.

    Parameters
    ----------
    logits
        The predicted logits as returned by the `MTLR` module.

    Returns
    -------
    torch.Tensor
        The predicted overall risk.
    """
    hazard = mtlr_hazard(logits)
    return torch.sum(hazard.cumsum(1), dim=1)

def train_mtlr(model, data_train, time_bins,
               num_epochs=1000, lr=.01, weight_decay=0.,
               C1=1., batch_size=None,
               verbose=True, device="cpu"):
    x = torch.tensor(data_train.drop(["time", "event"], axis=1).values, dtype=torch.float)
    y = encode_survival(data_train["time"].values, data_train["event"].values, time_bins)
    optimizer = make_optimizer(Adam, model, lr=lr, weight_decay=weight_decay)
    reset_parameters(model)
    model = model.to(device)
    model.train()
    train_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    
    pbar =  trange(num_epochs, disable=not verbose)
    for i in pbar:
        for xi, yi in train_loader:
            xi, yi = xi.to(device), yi.to(device)
            y_pred = model(xi)
            loss = mtlr_neg_log_likelihood(y_pred, yi, model, C1=C1, average=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pbar.set_description(f"[epoch {i+1: 4}/{num_epochs}]")
        pbar.set_postfix_str(f"loss = {loss.item():.4f}")
    model.eval()
    return model