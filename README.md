# Keras-Helper-Callbacks
Helper functions to precision/recall and generator methods.

This is a collection of helper methods I have written for keras.

List of methods:

GeneratorMetrics: generate precision/recall/f1 score for generators in keras using K callback

Metrics: precision/recall/d1 without generator. Use with model.fit()

image_generator_ysn: a special generator for keras which shifts the image and rolls it back to the beginneing/end.