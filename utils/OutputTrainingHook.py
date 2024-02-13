import torch
import torch.nn as nn
import torch.autograd.forward_ad as fwAD

class OutputTrainingHook(nn.Module):
    #This training hook captures and handles the gradients at the output of the network
    def __init__(self, train_mode):
        super(OutputTrainingHook, self).__init__()
        self.train_mode = train_mode

    def forward(self, input, dir_der_at_output, grad_at_output):
        out = OutputHookFunction.apply(input, dir_der_at_output, grad_at_output, self.train_mode)

        return out

class OutputHookFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dir_der_at_output, grad_at_output, train_mode):
        ctx.grad_out = grad_at_output
        ctx.dir_der_at_output = dir_der_at_output
        ctx.train_mode = train_mode

        return input.clone()

    @staticmethod
    def jvp(ctx, input, dir_der_at_output, grad_at_output, train_mode):
        ctx.dir_der = input

        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_at_output = ctx.grad_out

        grad_at_output[:grad_output.shape[0], :].data.copy_(grad_output.data)

        if ctx.train_mode == 'FDFA':
            dir_der = ctx.dir_der
            dir_der_at_output = ctx.dir_der_at_output
            dir_der_at_output[:dir_der_at_output.shape[0], :].data.copy_(dir_der.data)

        return grad_output, None, None, None