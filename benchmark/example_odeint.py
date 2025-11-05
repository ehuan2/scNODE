from torchdiffeq import odeint
import torch


# let's do something simple, where we approximate the e^y
def f(t, y):
    return y


# then this should approximate e^t, should return e^0, e^1, e^2, etc.
print(odeint(f, torch.tensor(1.0), torch.tensor([0.0, 1.0, 2.0])))
# then this should approximate 2 * e^t, should return 2 * e^0, 2 * e^2, 2 * e^4, etc.
print(odeint(f, torch.tensor(2.0), torch.tensor([1.0, 3.0, 5.0])))
