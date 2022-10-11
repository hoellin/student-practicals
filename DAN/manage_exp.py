from os import mkdir, path
import torch
import sys
import filters
import os
import pickle as pkl
import matplotlib.pyplot as plt

_v0 = None 
_x0 = None

def get_x0(b_size, x_dim, sigma):
    global _x0
    if _x0 is None:
        _x0 = 3*torch.ones(b_size, x_dim)\
             + sigma * torch.randn(b_size, x_dim)
    x0 = _x0
    return x0

def get_x0_test(b_size, x_dim, sigma):
    _x0_test = 3*torch.ones(b_size, x_dim)\
               + sigma * torch.randn(b_size, x_dim)
    x0 = _x0_test
    return x0

def get_ha0(b_size, h_dim):
    global _v0
    if _v0 is None:
        _v0 = torch.zeros(1, h_dim)
    ha0 = torch.zeros(b_size, h_dim)
    for b in range(b_size):
        ha0[b, :] = _v0
    return ha0

def set_tensor_type(tensor_type, cuda):
    print('use gpu', cuda)
    print('use tensor_type', tensor_type)
    if (tensor_type == "double") and cuda:
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    elif (tensor_type == "float") and cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    elif (tensor_type == "double") and (not cuda):
        torch.set_default_tensor_type(torch.DoubleTensor)
    elif (tensor_type == "float") and (not cuda):
        torch.set_default_tensor_type(torch.FloatTensor)
    else:
        raise NameError("Unknown tensor_type")
    
def pre_train_full(net, b_size, h_dim, x_dim, sigma0,
                   optimizer_classname, optimizer_kwargs):
    """
    Pre-train c at t=0
    # learn the parameters in net.c using ha0 and x0
    # by minimizing the L_0(q_0^a) loss
    """
    
    print('Pre-train c at t=0')
    
    # Generate x0 of batch size b_size at t=0
    # print out the emprical mean of x0
    x0 = get_x0(b_size, x_dim, sigma0)
    print(f"Empirical mean of x0: {x0.mean().item()}")
    
    # create an optimizer optimizer0 for the paramerters in c
    optimizer0 = eval(optimizer_classname)(net.c.parameters(), **optimizer_kwargs)

    # minimize L_0(q_0^a), check how small is the loss 
    ite = 0
    
    # Initlize h0
    ha0 = get_ha0(b_size, h_dim)
    print("###########################")
    print("Initial loss:", net.c.forward(ha0).log_prob(x0))
    
    # Use closure0 to compute the loss and gradients
    def closure0():
        # first use optimizer0 to set all the gradients to zero
        optimizer0.zero_grad()
        # then compute the loss logpdf_a0 = L_0(q_0^a), by using x0, h0, and c
        logpdf_a0 = -torch.mean(net.c.forward(ha0).log_prob(x0))
        # print(logpdf_a0.size())
        # perform back-propogation of the loss
        logpdf_a0.backward()

        # a counter of number of evaluations
        nonlocal ite
        ite += 1

        print(f"iteration {ite}: loss = {logpdf_a0}")
         # return the loss logpdf_a0
        return logpdf_a0

    # run optimizer
    optimizer0.step(closure0)

    # print out the final mean and covariance of q_0^a
    pdf_a0 = net.c(ha0)
    print('## INIT a0 mean', pdf_a0.mean[0, :])  # first sample
    print('## INIT a0 var', pdf_a0.variance[0, :])  # first sample
    print('## INIT a0 covar', pdf_a0.covariance_matrix[0, :, :]) # first sample

def train_full(net, b_size, h_dim, x_dim,
               T, checkpoint, direxp,
               prop, obs, sigma0,
               optimizer_classname, optimizer_kwargs):
    
    """
    Train over full time 0..T with BPTT
    # learn the parameters in net.a, net.b, net.c using t=0..T
    # by minimizing the total loss
    """
    if not path.exists(direxp):
        mkdir(direxp)          
    
    print('Train over full time 0..T with BPTT')

    # generate training data seq for t=0..T
    x0 = get_x0(b_size, x_dim, sigma0)
    xt = []
    yt = []
    x = x0
    for t in range(T):
        x = prop(x).sample(sample_shape=torch.Size([1])).squeeze(0)
        y = obs(x).sample(sample_shape=torch.Size([1])).squeeze(0)
        xt.append(x)
        yt.append(y)
    
    # Train net using xt and yt, t = 1 .. T and x0
    # miminize total loss, by constructing optimizer and rewriting closure    
    ite = 0
    
    # create an optimizer optimizer0 for the paramerters in c
    optimizer = eval(optimizer_classname)(net.parameters(), **optimizer_kwargs)
    ha0 = get_ha0(b_size, h_dim)
    
    def closure():
        ha = ha0
        optimizer.zero_grad()
        # then compute the loss logpdf_a0 = L_0(q_0^a), by using x0, h0, and c
        loss0 = -torch.mean(net.c.forward(ha0).log_prob(x0))
        total_loss = torch.zeros_like(loss0)
        for t in range(T):
            loss, ha = net.forward(ha, xt[t], yt[t])
            total_loss = total_loss+loss
        
        total_loss = total_loss/T + loss0
        
        total_loss.backward()
        
        # Checkpoint
        nonlocal ite
        if ite == 1 or (ite % checkpoint == 0):
            print("## Train Ite "+str(ite)+" ##")
            save_dict(direxp, scores=net.scores)
            print_scores(net.scores)
            
        ite += 1
        
        return total_loss
    
    # run optimizer
    optimizer.step(closure)    
    
def train_online(net, b_size, h_dim, x_dim,
                 T, checkpoint, direxp,
                 prop, obs, sigma0,
                 optimizer_classname, optimizer_kwargs, 
                 scheduler_classname, scheduler_kwargs):
    """
    Train functions for the DAN, online and truckated BPTT
    """
    if not path.exists(direxp):
        mkdir(direxp)
        
    # construct optimizer and scheduler
    assert(optimizer_classname != "NONE")
    print(' optimizer_classname', optimizer_classname)

    assert(scheduler_classname != "NONE")
    print(' scheduler_classname', scheduler_classname)
    
    x0 = get_x0(b_size, x_dim, sigma0)
    ha = get_ha0(b_size, h_dim)

    for t in range(1, T+1):
        # on the fly data generation
        # Truncated back propagation through time
        ha = None
        
        # Checkpoint
        if (t % checkpoint == 0) or (t == T):
            if ha is not None:
                print("## Train Cycle " + str(t)+" ##")
                save_dict(direxp,
                          net=net.state_dict(),
                          ha=ha,
                          x=x,
                          scores=net.scores,
                          optimizer=optimizer.state_dict())
                print_scores(net.scores)

@torch.no_grad()
def test(net, b_size, h_dim, x_dim, T, checkpoint, direxp, prop, obs, sigma0):
    
    # We generate samples for testing (we generate them "on the fly")
    # and we do the forward propagation
    ha = get_ha0(b_size, h_dim)
    x0 = get_x0(b_size, x_dim, sigma0)
    xt = []
    yt = []
    x = x0
    analysis = [net.c.forward(ha)]
    
    #tf = T # first test
    tf = 2*T # second test
    
    for t in range(tf):
       x = prop(x).sample(sample_shape=torch.Size([1])).squeeze(0)
       y = obs(x).sample(sample_shape=torch.Size([1])).squeeze(0)
       xt.append(x)
       yt.append(y)

       # Evaluates the loss
       _, ha = net(ha, x, y)
        
       qt = net.c.forward(ha)
       analysis.append(qt)
        
       # Checkpoint
       if (t % checkpoint == 0) or (t == T):
           print("## Test Cycle " + str(t)+" ##")
           save_dict(direxp, test_scores=net.scores)
           print_scores(net.scores)
             
    torch.save(net, direxp + "net.pt")
    
    test_sample_x = torch.mean(torch.stack(xt), dim=1).detach().cpu().numpy()
    test_sample_y = torch.mean(torch.stack(yt), dim=1).detach().cpu().numpy()
    
    pkl.dump(test_sample_x, open(direxp + "test_sample_x.pkl", "wb"))
    pkl.dump(test_sample_y, open(direxp + "test_sample_y.pkl", "wb"))
    pkl.dump(analysis, open(direxp + "analysis.pkl", "wb"))


def experiment(tensor_type, seed,
               net_classname, net_kwargs,
               sigma0, prop_kwargs, obs_kwargs,
               train_kwargs, test_kwargs,
               optimizer_classname, optimizer_kwargs,
               scheduler_classname, scheduler_kwargs,
               directory, nameexp):

    # CPU or GPU tensor
    cuda = torch.cuda.is_available()
    set_tensor_type(tensor_type,cuda)

    # Reproducibility
    torch.manual_seed(seed)

    net = eval(net_classname)(**net_kwargs)
    prop = filters.Constructor(**prop_kwargs)
    obs = filters.Constructor(**obs_kwargs)
    b_size = train_kwargs['b_size']
    h_dim = train_kwargs['h_dim']
    x_dim = train_kwargs['x_dim']
    T = train_kwargs['T']
    checkpoint = train_kwargs['checkpoint']
    direxp = directory + nameexp
    
    if train_kwargs["mode"] == "full":
        pre_train_full(net,b_size,h_dim,x_dim,sigma0,
                       optimizer_classname,optimizer_kwargs)        
        train_full(net, b_size, h_dim, x_dim,
                   T, checkpoint, direxp,
                   prop, obs, sigma0,
                   optimizer_classname, optimizer_kwargs)
    else:
        train_online(net, b_size, h_dim, x_dim,
                     T, checkpoint, direxp,
                     prop, obs, sigma0,
                     optimizer_classname, optimizer_kwargs, 
                     scheduler_classname, scheduler_kwargs)

    # Clear scores
    net.clear_scores()

    # Testing
    b_size = test_kwargs['b_size']
    h_dim = test_kwargs['h_dim']
    x_dim = test_kwargs['x_dim']
    T = test_kwargs['T']
    checkpoint = test_kwargs['checkpoint']
    test(net, b_size, h_dim, x_dim,
         T, checkpoint, direxp,
         prop, obs, sigma0)    


def save_dict(prefix, **kwargs):
    """
    saves the arg dict val with name "prefix + key + .pt"
    """
    for key, val in kwargs.items():
        torch.save(val, prefix + key + ".pt")


def print_scores(scores):
    for key, val in scores.items():
        if len(val) > 0:
            print(key+"= "+str(val[-1]))


def update(k_default, k_update):
    """Update a default dict with another dict
    """
    for key, value in k_update.items():
        if isinstance(value, dict):
            k_default[key] = update(k_default[key], value)
        else:
            k_default[key] = value
    return k_default


def update_and_save(k_default, list_k_update, name_fun):
    """update and save a default dict for each dict in list_k_update,
    generates a name for the exp with name_fun: dict -> string
    returns the exp names on stdout
    """
    out, directory = "", k_default["directory"]
    for k_update in list_k_update:
        nameexp = name_fun(k_update)
        if not os.path.exists(nameexp):
            os.mkdir(nameexp)
        k_default["nameexp"] = nameexp + "/"
        torch.save(update(k_default, k_update), nameexp + "/kwargs.pt")
        out += directory + "," + nameexp

    # return the dir and nameexp
    sys.stdout.write(out)


if __name__ == "__main__":
    """
    the next argument is the experiment name
    - launch the exp
    """
    torch.autograd.set_detect_anomaly(True)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    experiment(**torch.load(sys.argv[1] + "/kwargs.pt",
                            map_location=torch.device(device)))
