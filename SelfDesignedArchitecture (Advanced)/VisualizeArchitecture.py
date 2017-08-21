import keras
Model = keras.models.load_model('Model.h5')
keras.utils.plot_model(Model,to_file='Architecture.png',show_shapes=True,show_layer_names=False)