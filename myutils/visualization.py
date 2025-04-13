import matplotlib.pyplot as plt
import torch


def plot_loss(
        history: dict,
        loss: str='BPR LOSS',
        figsize: tuple=(8,5),
        ):
    plt.figure(figsize=figsize)
    plt.plot(history['trn'], label='TRN')
    plt.plot(history['val'], label='VAL')
    plt.xlabel('EPOCH')
    plt.ylabel(loss)
    plt.title('TRN vs. VAL')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_kl(
    history: dict,
    loss: str='KL DIV',
    figsize: tuple=(8,5),
    ):
    plt.figure(figsize=figsize)
    plt.plot(history['trn'], label='TRN')
    plt.plot(history['val'], label='VAL')
    plt.xlabel('EPOCH')
    plt.ylabel(loss)
    plt.title('TRN vs. VAL')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compare_prior_approx(
    prior_shape: torch.Tensor, 
    approx_shape: torch.Tensor, 
    idx: int, 
    type: str="user", 
    figsize: tuple=(8, 5),
    ):
    prior_user = prior_shape[idx].detach().cpu().numpy()
    approx_user = approx_shape[idx].detach().cpu().numpy()

    plt.figure(figsize=figsize)
    plt.plot(prior_user, label='Prior', marker='o')
    plt.plot(approx_user, label='Approx', marker='x')
    plt.title(f'{type} {idx}: Prior vs. Approx Attention Shape')
    plt.xlabel('Key Index')
    plt.ylabel('Shape Parameter Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()