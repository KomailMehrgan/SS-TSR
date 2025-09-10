# dynamic_weights.py

import torch
import torch.nn as nn
import torch.optim as optim



class DynamicLossWeighter(nn.Module):
    """
    Learns to balance multiple loss terms using uncertainty weighting.
    L_total = 0.5 * exp(-log_var_1) * L_1 + 0.5 * exp(-log_var_2) * L_2 + log_var_1 + log_var_2
    """

    def __init__(self, num_losses=2):
        super().__init__()
        # We learn the log of the variance for numerical stability
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, img_loss, ocr_loss):
        # Image loss term
        precision_img = torch.exp(-self.log_vars[0])
        weighted_img_loss = precision_img * img_loss + self.log_vars[0]

        # OCR loss term
        precision_ocr = torch.exp(-self.log_vars[1])
        weighted_ocr_loss = precision_ocr * ocr_loss + self.log_vars[1]

        # We use a 0.5 factor for stability, common in implementations
        return 0.5 * (weighted_img_loss + weighted_ocr_loss)


class DTP(nn.Module):
    """
    Method 3: Dynamic Task Prioritization (DTP)
    Philosophy: Prioritizes tasks based on their learning progress (Curriculum Learning).
    This is a simplified implementation that adjusts weights based on the ratio
    of the current loss to its moving average.
    """

    def __init__(self, alpha=0.8, num_losses=2):
        super().__init__()
        self.alpha = alpha
        # We store a moving average for each task's loss
        self.register_buffer('running_loss_avg', torch.zeros(num_losses))
        self.register_buffer('steps', torch.tensor(0.0))

    def forward(self, img_loss, ocr_loss):
        losses = torch.stack([img_loss, ocr_loss])

        # On the first step, initialize the moving average
        if self.steps == 0:
            self.running_loss_avg = losses.detach()

        # Calculate a simplified progress rate
        progress_rate = losses.detach() / self.running_loss_avg

        # Calculate weights
        # The task with more progress (smaller ratio) gets a smaller weight
        # to focus attention on the harder task.
        weights = torch.pow(progress_rate, self.alpha)
        weights = weights / weights.sum() * 2  # Normalize so that the sum of weights is 2

        # Apply weights
        total_loss = weights[0] * img_loss + weights[1] * ocr_loss

        # Update the moving average
        self.running_loss_avg = 0.9 * self.running_loss_avg + 0.1 * losses.detach()
        self.steps += 1

        return total_loss


class PCGrad():
    """
    Method 2: Projecting Conflicting Gradients (PCGrad)
    Philosophy: Geometrically remove the conflicting components of gradients.
    This class wraps a standard optimizer and implements the PCGrad logic.
    """

    def __init__(self, optimizer):
        self._optimizer = optimizer
        self.zero_grad = optimizer.zero_grad

    def step(self, losses):
        """
        Performs a single optimization step using the PCGrad logic.
        Args:
            losses (list): A list of losses for each task. e.g., [img_loss, ocr_loss]
        """
        # 1. Calculate gradients for each task separately
        task_grads = []
        for i, loss in enumerate(losses):
            self._optimizer.zero_grad()
            loss.backward(retain_graph=True)

            grad_vec = []
            for param in self._optimizer.param_groups[0]['params']:
                if param.grad is not None:
                    grad_vec.append(param.grad.clone().detach())
                else:
                    # Handle case where a parameter doesn't receive a gradient
                    grad_vec.append(torch.zeros_like(param).to(param.device))
            task_grads.append(grad_vec)

        self._optimizer.zero_grad()

        # 2. Project conflicting gradients if necessary
        projected_grads = self._project_conflicting(task_grads)

        # 3. Apply the final gradients and update the model
        self._apply_grads(projected_grads)
        self._optimizer.step()

    def _project_conflicting(self, task_grads):
        """The core PCGrad logic to resolve gradient conflicts."""
        num_tasks = len(task_grads)
        projected_grads = [g for g in task_grads]

        for i in range(num_tasks):
            for j in range(num_tasks):
                if i == j:
                    continue

                grad_i = task_grads[i]
                grad_j = task_grads[j]

                # Calculate the dot product to detect conflict
                dot_product = sum([torch.dot(g_i.view(-1), g_j.view(-1)) for g_i, g_j in zip(grad_i, grad_j)])

                # If gradients conflict (negative dot product)
                if dot_product < 0:
                    # Project grad_i onto grad_j and remove the conflicting component
                    norm_j_sq = sum([torch.norm(g_j) ** 2 for g_j in grad_j])
                    proj_component = (dot_product / norm_j_sq)

                    for g_i, g_j in zip(projected_grads[i], grad_j):
                        g_i -= proj_component * g_j

        return projected_grads

    def _apply_grads(self, grads):
        """Applies the final, processed gradients to the model's parameters."""
        final_grads = [torch.zeros_like(p) for p in self._optimizer.param_groups[0]['params']]

        # Sum the modified gradients from all tasks
        for task_grad in grads:
            for i, grad_comp in enumerate(task_grad):
                final_grads[i] += grad_comp

        # Assign the final gradient to the parameters
        for param, grad in zip(self._optimizer.param_groups[0]['params'], final_grads):
            param.grad = grad