import ipdb
import torch.nn as nn
import torch
import ctlib
from torch.autograd import Function

class prj_module(nn.Module):
    def __init__(self, weight,options):
        super(prj_module, self).__init__()
        self.weight = weight
        self.options = nn.Parameter(options, requires_grad=False)

    def forward(self, input_data, proj):
        return prj_fun.apply(input_data, self.weight, proj, self.options)
class prj_fun(Function):
    @staticmethod
    def forward(self, input_data, weight, proj, options):
        b, c, h, w = input_data.shape
        input_data = input_data.contiguous().view(b, c, h, w)
        b_sino, c_sino, h_sino, w_sino = proj.shape
        proj = proj.contiguous().view(b_sino, c_sino, h_sino, w_sino)
        temp=ctlib.projection(input_data,options,0)-proj
        intervening_res = ctlib.backprojection(temp, options,0)
        self.save_for_backward(intervening_res, weight, options)
        out = input_data - weight * intervening_res
        return out

    @staticmethod
    def backward(self, grad_output):
        intervening_res, weight, options = self.saved_tensors
        b,c,h,w=grad_output.shape
        grad_output = grad_output.contiguous().view(b,c,h,w)
        temp = ctlib.projection(grad_output, options,0)
        t_b,t_c,t_h,t_w=temp.shape
        temp = temp.contiguous().view(t_b,t_c,t_h,t_w)
        temp = ctlib.backprojection(temp, options,0)
        grad_input = grad_output - weight * temp
        temp = intervening_res * grad_output
        grad_weight = - temp.sum().view(-1)
        return grad_input, grad_weight, None, None