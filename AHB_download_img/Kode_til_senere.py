def save_model_output_parameters(model, learning_rate, batch_size, epochs, opimizer, loss_function, ach,
                                 train_losses, val_losses, y_true):
    # save model
    torch.save(model.state_dict(), 'model_weights.pth')

    #save parameters:
    training_metadata = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "optimizer": opimizer,
        "loss_function": loss_function,
        "architecture": ach,
    }
    torch.save(training_metadata, 'training_metadata.pth')


    torch.save({'train_loss': train_losses, 'val_loss': val_losses}, 'loss_curves.pth')
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.savefig('loss_curve.png')

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.savefig('confusion_matrix.png')


    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    torch.save({'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}, 'roc_curve.pth')