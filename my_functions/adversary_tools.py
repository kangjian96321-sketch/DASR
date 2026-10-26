import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class NativeAdversary:
    @staticmethod
    def get_adv_images(model, images, labels, attack_type, device, norm_fn=None,
                       eps=8/255, alpha=2/255, iters=10, patch_size=8):
        """
        统一的对抗样本生成入口。
        为保证通用性，将 eps, alpha, iters 提取为参数，而非依赖外部全局变量。
        """
        attack_type = attack_type.lower()
        if attack_type == 'clean':
            return images
        elif attack_type == 'fgsm':
            return NativeAdversary.fgsm(model, images, labels, eps=eps, norm_fn=norm_fn)
        elif attack_type == 'bim':
            return NativeAdversary.bim(model, images, labels, eps=eps, alpha=alpha, iters=iters, norm_fn=norm_fn)
        elif attack_type == 'mifgsm':
            return NativeAdversary.mifgsm(model, images, labels, eps=eps, alpha=alpha, iters=iters, norm_fn=norm_fn)
        elif attack_type == 'pgd':
            return NativeAdversary.pgd(model, images, labels, eps=eps, alpha=alpha, iters=iters, norm_fn=norm_fn)
        elif attack_type == 'pgd_l2':
            return NativeAdversary.pgd_l2(model, images, labels, eps=eps, alpha=alpha, iters=iters, norm_fn=norm_fn)
        elif attack_type == 'pgd_cw':
            return NativeAdversary.pgd_cw(model, images, labels, eps=eps, alpha=alpha, iters=iters, norm_fn=norm_fn)
        elif attack_type == 'patch':
            return NativeAdversary.patch(model, images, labels, patch_size=patch_size, iters=iters, norm_fn=norm_fn)
        elif attack_type == 'cw_l2':
            # 标准 C&W L2 攻击
            return NativeAdversary.cw_l2(model, images, labels, c=1.0, kappa=0, iters=iters, lr=0.01, norm_fn=norm_fn)
        else:
            return images

    @staticmethod
    def fgsm(model, images, labels, eps=8/255, norm_fn=None):
        was_training = model.training
        model.eval()
        with torch.enable_grad():
            images = images.clone().detach().to(images.device)
            images.requires_grad = True
            model_input = norm_fn(images) if norm_fn else images
            outputs = model(model_input)
            loss = F.cross_entropy(outputs, labels)
            grad = torch.autograd.grad(loss, images)[0]
            adv_images = images + eps * grad.sign()
        model.train(was_training)
        return torch.clamp(adv_images, 0, 1).detach()

    @staticmethod
    def pgd(model, images, labels, eps=8/255, alpha=2/255, iters=10, norm_fn=None):
        was_training = model.training
        model.eval()
        with torch.enable_grad():
            adv_images = images.clone().detach() + torch.empty_like(images).uniform_(-eps, eps)
            adv_images = torch.clamp(adv_images, 0, 1)
            for _ in range(iters):
                adv_images.requires_grad = True
                model_input = norm_fn(adv_images) if norm_fn else adv_images
                outputs = model(model_input)
                loss = F.cross_entropy(outputs, labels)
                grad = torch.autograd.grad(loss, adv_images)[0]
                adv_images = adv_images.detach() + alpha * grad.sign()
                delta = torch.clamp(adv_images - images, min=-eps, max=eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        model.train(was_training)
        return adv_images

    @staticmethod
    def bim(model, images, labels, eps=8/255, alpha=2/255, iters=10, norm_fn=None):
        was_training = model.training
        model.eval()
        with torch.enable_grad():
            adv_images = images.clone().detach()
            for _ in range(iters):
                adv_images.requires_grad = True
                model_input = norm_fn(adv_images) if norm_fn else adv_images
                outputs = model(model_input)
                loss = F.cross_entropy(outputs, labels)
                grad = torch.autograd.grad(loss, adv_images)[0]
                adv_images = adv_images.detach() + alpha * grad.sign()
                delta = torch.clamp(adv_images - images, min=-eps, max=eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        model.train(was_training)
        return adv_images

    @staticmethod
    def mifgsm(model, images, labels, eps=8/255, alpha=2/255, iters=10, decay=1.0, norm_fn=None):
        was_training = model.training
        model.eval()
        with torch.enable_grad():
            adv_images = images.clone().detach()
            momentum = torch.zeros_like(images).detach()
            for _ in range(iters):
                adv_images.requires_grad = True
                model_input = norm_fn(adv_images) if norm_fn else adv_images
                outputs = model(model_input)
                loss = F.cross_entropy(outputs, labels)
                grad = torch.autograd.grad(loss, adv_images)[0]
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), p=1, dim=1).view(-1, 1, 1, 1)
                grad_norm = torch.max(grad_norm, torch.tensor(1e-8, device=images.device))
                grad_normalized = grad / grad_norm
                momentum = decay * momentum + grad_normalized
                adv_images = adv_images.detach() + alpha * momentum.sign()
                delta = torch.clamp(adv_images - images, min=-eps, max=eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        model.train(was_training)
        return adv_images

    @staticmethod
    def pgd_l2(model, images, labels, eps=0.5, alpha=0.1, iters=10, norm_fn=None):
        was_training = model.training
        model.eval()
        with torch.enable_grad():
            delta = torch.empty_like(images).normal_()
            d_flat = delta.view(delta.shape[0], -1)
            n = d_flat.norm(p=2, dim=1).view(-1, 1, 1, 1)
            r = torch.empty_like(n).uniform_(0, 1)
            delta = delta * r / n * eps
            adv_images = torch.clamp(images + delta, 0, 1).detach()

            for _ in range(iters):
                adv_images.requires_grad = True
                model_input = norm_fn(adv_images) if norm_fn else adv_images
                outputs = model(model_input)
                loss = F.cross_entropy(outputs, labels)
                grad = torch.autograd.grad(loss, adv_images)[0]
                grad_norms = torch.norm(grad.view(grad.shape[0], -1), p=2, dim=1) + 1e-8
                grad_normalized = grad / grad_norms.view(-1, 1, 1, 1)
                adv_images = adv_images.detach() + alpha * grad_normalized
                delta = adv_images - images
                delta_norms = torch.norm(delta.view(delta.shape[0], -1), p=2, dim=1)
                factor = torch.min(torch.tensor(1.0).to(images.device), eps / (delta_norms + 1e-8))
                delta = delta * factor.view(-1, 1, 1, 1)
                adv_images = torch.clamp(images + delta, 0, 1).detach()
        model.train(was_training)
        return adv_images

    @staticmethod
    def pgd_cw(model, images, labels, eps=8/255, alpha=2/255, iters=10, norm_fn=None):
        was_training = model.training
        model.eval()
        with torch.enable_grad():
            adv_images = images.clone().detach() + torch.empty_like(images).uniform_(-eps, eps)
            adv_images = torch.clamp(adv_images, 0, 1)
            for _ in range(iters):
                adv_images.requires_grad = True
                model_input = norm_fn(adv_images) if norm_fn else adv_images
                outputs = model(model_input)
                one_hot_labels = torch.eye(outputs.shape[1], device=outputs.device)[labels]
                real_logits = torch.sum(one_hot_labels * outputs, dim=1)
                other_logits = torch.max((1 - one_hot_labels) * outputs - one_hot_labels * 10000, dim=1)[0]
                loss = torch.sum(other_logits - real_logits)
                grad = torch.autograd.grad(loss, adv_images)[0]
                adv_images = adv_images.detach() + alpha * grad.sign()
                delta = torch.clamp(adv_images - images, min=-eps, max=eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        model.train(was_training)
        return adv_images

    @staticmethod
    def patch(model, images, labels, patch_size=8, iters=10, norm_fn=None):
        was_training = model.training
        model.eval()
        b, c, h, w = images.size()
        adv_images = images.clone().detach()
        mask = torch.zeros_like(images)
        for i in range(b):
            start_x = np.random.randint(0, w - patch_size + 1)
            start_y = np.random.randint(0, h - patch_size + 1)
            mask[i, :, start_y:start_y + patch_size, start_x:start_x + patch_size] = 1.0

        adv_images = torch.where(mask == 1.0, torch.rand_like(adv_images), adv_images).detach()
        with torch.enable_grad():
            for _ in range(iters):
                adv_images.requires_grad = True
                model_input = norm_fn(adv_images) if norm_fn else adv_images
                outputs = model(model_input)
                loss = F.cross_entropy(outputs, labels)
                grad = torch.autograd.grad(loss, adv_images)[0]
                adv_images = adv_images.detach() + 0.1 * grad.sign()
                adv_images = torch.where(mask == 1.0, adv_images, images)
                adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        model.train(was_training)
        return adv_images

    @staticmethod
    def cw_l2(model, images, labels, c=10.0, kappa=0, iters=15, lr=0.05, norm_fn=None, bound_eps=None):
        was_training = model.training
        model.eval()
        eps_clamp = 1e-6
        images_clamped = torch.clamp(images, min=eps_clamp, max=1.0 - eps_clamp)
        w = (torch.log(images_clamped) - torch.log(1.0 - images_clamped)) / 2.0
        w = w.detach().requires_grad_(True)
        optimizer = optim.Adam([w], lr=lr)

        with torch.enable_grad():
            for _ in range(iters):
                optimizer.zero_grad()
                adv_images = 0.5 * (torch.tanh(w) + 1.0)
                loss_l2 = torch.norm((adv_images - images).view(images.size(0), -1), p=2, dim=1).mean()
                model_input = norm_fn(adv_images) if norm_fn else adv_images
                outputs = model(model_input)
                one_hot_labels = torch.eye(outputs.shape[1], device=outputs.device)[labels]
                real_logits = torch.sum(one_hot_labels * outputs, dim=1)
                other_logits = torch.max((1 - one_hot_labels) * outputs - one_hot_labels * 10000, dim=1)[0]
                loss_margin = torch.clamp(real_logits - other_logits, min=-kappa).mean()
                loss = loss_l2 + c * loss_margin
                loss.backward()
                optimizer.step()

        adv_images = 0.5 * (torch.tanh(w) + 1.0)
        if bound_eps is not None:
            delta = adv_images - images
            delta_norms = torch.norm(delta.view(delta.shape[0], -1), p=2, dim=1)
            factor = torch.min(torch.tensor(1.0).to(images.device), bound_eps / (delta_norms + 1e-8))
            delta = delta * factor.view(-1, 1, 1, 1)
            adv_images = torch.clamp(images + delta, min=0, max=1)

        model.train(was_training)
        return adv_images.detach()
