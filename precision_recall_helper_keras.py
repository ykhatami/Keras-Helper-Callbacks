"""
Helper functions to get precision/recall from Keras. 
"""
from sklearn.metrics import precision_score, recall_score, f1_score


### Use this class with model.fit_generator() to get precision/recall/f1 scores.
### We need the whole validation data to calculate precision/recall/f1. When using generators in Keras, the validation data is not readily available in the model. It is available in small batches.
### So we need a special method to take care of this situation.
### First, we collect all the validation batches, then calculate the scores. If we calculate the scores for each batch, we get incorrect scores.

class GeneratorMetrics(keras.callbacks.Callback):
    """
    How to use this method:
        # Set up metrics
        gen_metric = GeneratorMetrics(
            val_image_generator,
            validation_steps)

        # Train the model
        model.fit_generator(train_image_generator,
                            steps_per_epoch=~~,
                            epochs=~~,
                            validation_data=val_image_generator,
                            validation_steps=~~,
                            callbacks=[gen_metric]
                            )

        The image generator can be any general purpose image generator or use image_generator_ysn(). 
    """
    def on_train_begin(self, logs={}):
        self._data = []

    def get_data(self):
        return self._data

    def __init__(self, validation_generator, validation_steps):
        self.validation_generator = validation_generator
        self.validation_steps = int(validation_steps)
    
    def on_epoch_end(self, batch, logs={}):
        # hold all the validation data
        y_val_all = []
        y_predict_all = []

        # loop through validation data in batches and update y_val_all and y_predict_all
        for batch_index in range(self.validation_steps):
            X_val, y_val = next(self.validation_generator)            
            y_predict = np.asarray(self.model.predict(X_val))

            # convert from one_hot to regular notation
            y_val = np.argmax(y_val, axis=1)
            y_predict = np.argmax(y_predict, axis=1)
            
            y_val_all = y_val_all + list(y_val)
            y_predict_all = y_predict_all + list(y_predict)
        
        # Now all the validation data is collected, let's calculate recall/precision/f1 scores. 
        self._data.append({
            'val_recall': recall_score(y_val_all, y_predict_all, average=None),
            'val_precision': precision_score(y_val_all, y_predict_all, average=None),
            'f1_score': f1_score(y_val_all, y_predict_all, average=None)
        })
        return

### Use this class with model.fit() to get precision/recall scores.
### This is much simpler because we dont use generators. So all the validation data is available to us.
class Metrics(keras.callbacks.Callback):
    """
    Example: 
    prf_metrics = Metrics()
    model.fit(**, callbacks=[prf_metrics])
    """
    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        # Collect validation data
        X_val, y_val = self.validation_data[0], self.validation_data[1]
        y_predict = np.asarray(self.model.predict(X_val))

        # convert from one_hot to regular notation
        y_val = np.argmax(y_val, axis=1)
        y_predict = np.argmax(y_predict, axis=1)

        # Calculate the metrics and return them
        self._data.append({
            'val_recall': recall_score(y_val, y_predict, average=None),
            'val_precision': precision_score(y_val, y_predict, average=None),
            'f1_score': f1_score(y_val, y_predict, average=None)
        })
        return

    def get_data(self):
        return self._data

# Convert the precision/recall array from above to a dataframe with class numbers and epoch numbers
# Special use method to convert precision/recall/f1 data to a pandas datafram. 
# I will complete the documentation later. 
def pr_keras_to_df(pr, n_classes, init_epoch):
    pr_df = pd.DataFrame()
    i = init_epoch
    for prx in pr:
        prx.update({'epoch':i})
        prx.update({'class':np.arange(n_classes)})
        pr_df = pr_df.append(pd.DataFrame(prx))
        i+=1
    if le is not None:
        pr_df['class_desc'] = le.inverse_transform(pr_df['class'])
    return pr_df


