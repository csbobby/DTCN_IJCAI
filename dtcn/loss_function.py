import keras.backend as K

def penalized_loss(noise):
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true) - K.square(y_true - noise), axis=-1)
    return loss

#unused noise
def normal_mean_squared_error(noise):
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true))
    return loss

def my_penalized_loss(noise,mask,sample_weight):
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true) - mask*sample_weight*K.square(y_true - noise), axis=-1)
    return loss
