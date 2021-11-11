from typing import Iterable
from torch import nn

def train(model: nn.Module, epochs: int, batches: Iterable, loss: Callable, validation_data, patience) -> None: #callable: function or class

patience = 5
optimizer = ...
    
    for epoch in epochs:
        for batch in batches:
            X, y = prepare_batch(batch)

            y_hat = model.forward(X)
            
            loss = loss_function(y,y_hat)

            #optimizer

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        #early stopping 
        # when performance on validation set stops improving, we want to stop training
        validation_y_hat = model.forward(validation_X)
        validation_loss = loss(val_y, validation_y_hat)
        val_losses.append(validation_loss)

        if validation_loss <  best_validation_loss or best_validation_loss is None:
            best_validation_loss = validation_loss
            #save the model

        is_it_better = for vl in validation_loss[-patience:] if validation_loss >= vl #the last 5
        #
        if len(is_it_better) == patience:
            break

    #model = load_model
    return model



