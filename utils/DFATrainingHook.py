import numpy as np
import torch
import torch.nn as nn
import torch.autograd.forward_ad as fwAD


class DFATrainingHook(nn.Module):
    # This training hook calculates and injects the gradients made by DFA
    def __init__(self, train_mode, device):
        super(DFATrainingHook, self).__init__()
        self.device = device
        self.train_mode = train_mode
        self.is_not_initialized = True
        self.feedbacks = nn.Parameter(requires_grad=True)

    def init_weights(self, dim):
        self.feedbacks = nn.Parameter(torch.Tensor(torch.Size(dim)).to(self.device))
        if self.train_mode in ['FDFA', 'DKP']:
            self.feedbacks.requires_grad = True

            torch.nn.init.zeros_(self.feedbacks)
            #torch.nn.init.kaiming_uniform_(self.feedbacks)
        elif self.train_mode == 'DFA':
            self.feedbacks.requires_grad = False
            torch.nn.init.kaiming_uniform_(self.feedbacks)

    def forward(self, input, dir_der_at_output, grad_at_output):
        if self.is_not_initialized and self.train_mode in ['FDFA', 'DKP', 'DFA']:
            if len(input.shape) > 2:
                dim = [grad_at_output.shape[1], input.shape[1], input.shape[2], input.shape[3]]
            else:
                dim = [grad_at_output.shape[1], input.shape[1]]
            self.init_weights(dim)
            self.is_not_initialized = False

        return DFAHookFunction.apply(input, self.feedbacks, dir_der_at_output, grad_at_output, self.train_mode)


class DFAHookFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, feedbacks, dir_der_at_output, grad_at_output, train_mode):
        ctx.save_for_backward(input, feedbacks)
        ctx.grad_at_output = grad_at_output
        ctx.dir_der_at_output = dir_der_at_output
        ctx.train_mode = train_mode

        return input.clone()

    @staticmethod
    def jvp(ctx, input, feedbacks, dir_der_at_output, grad_at_output, train_mode):
        perturbations = torch.randn_like(input, device=input.device)
        ctx.perturbations = perturbations

        return input + perturbations

    @staticmethod
    def backward(ctx, grad_output):
        input, feedbacks = ctx.saved_variables
        grad_at_output = ctx.grad_at_output
        dir_der_at_output = ctx.dir_der_at_output
        train_mode = ctx.train_mode

        grad_at_output = grad_at_output[:grad_output.shape[0], :]

        if train_mode == 'DFA':
            feedbacks_view = feedbacks.view(-1, np.prod(feedbacks.shape[1:]))
            grad_output_est = grad_at_output.mm(feedbacks_view).view(grad_output.shape)
            return grad_output_est, None, None, None, None

        elif train_mode == 'DKP':
            layer_out_view = input.view(-1, np.prod(input.shape[1:]))
            feedbacks_view = feedbacks.view(-1, np.prod(feedbacks.shape[1:]))

            grad_output_est = grad_at_output.mm(feedbacks_view)
            grad_feedback = grad_at_output.t().mm(layer_out_view)

            return grad_output_est.view(grad_output.shape), grad_feedback.view(feedbacks.shape), None, None, None

        elif train_mode == 'FDFA':
            perturbations = ctx.perturbations
            feedbacks_view = feedbacks.view(-1, np.prod(feedbacks.shape[1:]))
            perturbations_view = perturbations.view(-1, np.prod(perturbations.shape[1:]))

            grad_output_est = grad_at_output.mm(feedbacks_view)

            dir_grad = torch.unsqueeze(dir_der_at_output, 2) * torch.unsqueeze(perturbations_view, 1)
            grad_feedback = (feedbacks_view - torch.mean(dir_grad, 0))

            return grad_output_est.view(grad_output.shape), grad_feedback.view(feedbacks.shape), None, None, None

        return grad_output, None, None, None, None
