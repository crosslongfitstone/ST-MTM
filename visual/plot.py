import matplotlib.pyplot as plt



def plot_reconstruction(x_rec, x_true, epoch, iter, loss, data):
    b, l, d = x_rec.shape    
    batch_idx = 0
    title = f'ep{epoch} - it{iter} - b{batch_idx} reconstruction plot (loss = {loss:.4f})'
    fname = f'outputs/reconstruction/{data}/rec_plot_it{iter}_ep{epoch}_b{batch_idx}.png'
    x_rec, x_true = x_rec[batch_idx], x_true[batch_idx]

    fig, ax = plt.subplots(d, 1, sharex=True, figsize=(8, 12))
    if d==1:
        ax.plot(range(l), x_rec, c='maroon', label='rec', linewidth=0.8)
        ax.plot(range(l), x_true, c='royalblue', label='true', linewidth=0.8)
        handles, labels = ax.get_legend_handles_labels()
    else:
        for i in range(d):
            ax[i].plot(range(l), x_rec[:, i], c='maroon', label='rec', linewidth=0.8)
            ax[i].plot(range(l), x_true[:, i], c='royalblue', label='true', linewidth=0.8)
        handles, labels = ax[i].get_legend_handles_labels()
    
    fig.legend(handles, labels, loc='upper right', fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    plt.savefig(fname)
    

def plot_forecast(x_pred, x_true, epoch, iter, loss, data):
    b, l, d = x_pred.shape    
    batch_idx = 0
    title = f'ep{epoch} - it{iter} - b{batch_idx} forecast plot (loss = {loss:.4f})'
    fname = f'outputs/forecast/{data}/pl{l}/forecast_plot_it{iter}_ep{epoch}_b{batch_idx}.png'
    x_pred, x_true = x_pred[batch_idx], x_true[batch_idx]

    fig, ax = plt.subplots(d, 1, sharex=True, figsize=(8, 12))
    
    if d==1:
        ax.plot(range(l), x_pred, c='maroon', label='pred', linewidth=0.8)
        ax.plot(range(l), x_true, c='royalblue', label='true', linewidth=0.8)
        handles, labels = ax.get_legend_handles_labels()
    else:
        for i in range(d):
            ax[i].plot(range(l), x_pred[:, i], c='maroon', label='pred', linewidth=0.8)
            ax[i].plot(range(l), x_true[:, i], c='royalblue', label='true', linewidth=0.8)
        handles, labels = ax[i].get_legend_handles_labels()

    fig.legend(handles, labels, loc='upper right', fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    plt.savefig(fname)