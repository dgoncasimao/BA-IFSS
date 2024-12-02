# Define function for early stopping callback
def Callback_EarlyStopping(loss_list, min_delta, patience):

    # Don't allow early stopping for 2*patience epochs
    if len(loss_list) // patience < 2:
        return False

    # Mean loss for last patience epochs and second-last patience epochs
    mean_2nd_last = np.mean(loss_list[::-1][patience:2*patience])
    mean_last = np.mean(loss_list[::-1][:patience])

    # relative change
    delta_abs = np.abs(mean_last - mean_2nd_last)
    delta_rel = np.abs(delta_abs/mean_2nd_last)
    if delta_rel < min_delta:
        print(f"Callback Early Stopping: Loss didn't change much from last epochs, percent change in loss: {delta_rel*100}")
        return True
    else:
        return False

early_stop = Callback_EarlyStopping(val_loss_seq, min_delta=0.1, patience=8)
if early_stop:
    print(f"Callback Early Stopping received at epoch {epoch}, terminating training")
    break

class EarlyStoppingCallback:
    def __init__(self, min_delta, patience):
        self.min_delta = min_delta
        self.patience = patience
        self.loss_list = []
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped_epoch = 0
    
    def on_epoch_end(self, epoch, current_loss):
        self.loss_list.append(current_loss)
        if len(self.loss_list) < 2*self.patience:
            return False
    
        # Mean loss for last patience epochs and second-last patience epochs
        mean_2nd_last = np.mean(self.loss_list[-2*self.patience:-self.patience])
        mean_last = np.mean(self.loss_list[-self.patience:])

        # relative change
        delta_abs = np.abs(mean_last - mean_2nd_last)
        delta_rel = np.abs(delta_abs / mean_2nd_last)

        if delta_rel < self.min_delta:
            print(f"Callback Early Stopping: Loss didn't change much from last epochs, percent change in loss: {delta_rel*100}")
            self.stopped_epoch = epoch
            return True
        else:
            return False

    def get_stopped_epoch(self):
        return self.stopped_epoch

def Callback_ReduceLROnPlateau(loss_list, factor, min_lr, patience, threshold):
    
    current_lr = optimizer.learning_rate.numpy()

    # Don't allow reducing the learning rate for 2*patience epochs
    if len(loss_list) // patience < 2:
        return False

    # Mean loss for last patience epochs and second-last patience epochs
    mean_2nd_last = np.mean(loss_list[::-1][patience:2*patience])
    mean_last = np.mean(loss_list[::-1][:patience])

    # relative change
    delta_abs = np.abs(mean_last - mean_2nd_last)
    delta_rel = np.abs(delta_abs/mean_2nd_last)

    # check if relative change is below the threshold
    if delta_rel < threshold:
        new_lr = max(current_lr*factor, min_lr)
        optimizer.learning_rate.assign(new_lr)
        print(f"Learning rate reduced to: {new_lr}")
        return True
    else:
        return False

reduce_lr = Callback_ReduceLROnPlateau(val_loss_seq, factor=0.1, min_lr=0.00001, patience=5, threshold=0.01)
if reduce_lr:
    print("Learning rate was reduced")