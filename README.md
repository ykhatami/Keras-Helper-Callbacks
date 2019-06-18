# Keras-Helper-Callbacks
Helper functions for precision/recall and generator methods.

This is a collection of helper methods I have written for keras.

List of methods:

GeneratorMetrics: generate precision/recall/f1 score for generators in keras using K callback. To be used with fit_generator()

Metrics: precision/recall/f1 without generator. To be used with model.fit()

image_generator_ysn: a special generator for keras which shifts the image and rolls it back to the beginneing/end in contrast to built in image rotation with repeat, etc. This is useful for time series sensor signals, where we don't know where in time the data was collected.
