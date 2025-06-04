import matplotlib.pyplot as plt




def plot_history_regression(history, xlims=None, ylims=None):
    # Compute default limits based on history if not provided
    if xlims is None:
        xlims = (0, max(history.epoch))
    if ylims is None:
        ylims = (0, max(history.history['loss']))
    print(xlims,ylims)
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[0].set_ylim(ylims)
    ax[0].set_xlim(xlims)

    ax[1].set_title('mse')
    ax[1].plot(history.epoch, history.history["mse"], label="Train mse")
    ax[1].plot(history.epoch, history.history["val_mse"], label="Validation mse")
    ax[1].set_ylim(ylims)
    ax[1].set_xlim(xlims)
    ax[0].legend()
    ax[1].legend()
