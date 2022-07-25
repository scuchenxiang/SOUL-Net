import ipdb
import torch

def geometric_approximation_1(s):  
    ba, chan = s.shape
    dtype = s.dtype

    I = torch.ones([ba, chan], device=s.device).type(dtype)
    temp = 1e-8 * I
    s = s + temp
    I = torch.diag_embed(I)
    # I = torch.eye(s.shape[0], device=s.device).type(dtype)
    p = s.unsqueeze(-1) / s.unsqueeze(-2) - I
    p = torch.where(p < 1., p, 1. / p) 

    a1 = s.unsqueeze(-1).repeat(1, 1, chan).permute(0, 2,1)
    # a1 = s.repeat(s.shape[0], 1).t()
    a1_t = a1.permute(0, 2,1)
    lamiPluslamj = 1. / ((s.unsqueeze(-1) + s.unsqueeze(-2)))  # do not need to sub I,because have been * a1

    a1 = 1. / torch.where(a1 >= a1_t, a1, - a1_t)
    # a1 *= torch.ones(s.shape[0], s.shape[0], device=s.device).type(dtype) - I
    a1 *= torch.ones_like(I, device=s.device).type(dtype) - I
    p_app = torch.ones_like(p)
    p_hat = torch.ones_like(p)
    for i in range(9):
        p_hat = p_hat * p
        p_app += p_hat
    a1 = lamiPluslamj * a1 * p_app

    return a1


class svdv2_1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M):
        try:
            U, S, V = torch.svd(M,some=True,compute_uv=True)
        except:#avoid cond() too large
            print(M.max())
            print(M.min())
            ipdb.set_trace()
            U, S, V = torch.svd(M+1e-3*M.mean()*torch.rand_like(M),some=True,compute_uv=True)
        dtype = M.dtype
        S[S <= torch.finfo(dtype).eps] = torch.finfo(dtype).eps
        ctx.save_for_backward(M, U, S,V)
        return U,S,V

    @staticmethod
    def backward(ctx, dL_du, dL_ds,dL_dv):
        M, U,S,V = ctx.saved_tensors
        k= geometric_approximation_1(S)
        k[k == float('inf')] = k[k != float('inf')].max()
        k[k == float('-inf')] = k[k != float('-inf')].min()
        k[k != k] = k.max()
        K_t=k.permute(0,2,1)
        diag_s=torch.diag_embed(S)
        VT=torch.permute(V,[0,2,1])
        tt=2*torch.matmul(diag_s,K_t*torch.matmul(VT,dL_dv))
        grad_input=tt+torch.diag_embed(dL_ds)
        US=torch.matmul(U, grad_input)
        grad_input = torch.matmul(US, VT)
        return grad_input

