import tensorflow as tf
from keras.models import Model
from keras.layers import Lambda, merge

def make_parallel(model, gpu_count, batch_size):
    ##################################################################
    # Function takes a keras model (model) just prior to compilation and distributes it 
    # using data parallelism across a given number of gpus (gpu_count). 
    #
    # Batch size must be divisible by gpu_count.  
    #
    # Example Usage:
    #
    # theShape = (20,3392)
    # training_input = Input(shape=theShape)
    # x = LSTM(200)(training_input)
    # x = Dense(1, activation='sigmoid')(x)
    # model = Model(training_input, x, name='LSTM baseline')
    # #########################
    # model = make_parallel(model=model, gpu_count=4, batch_size=100) # this distributes the model across 4 gpus each processing 25 observations
    # #########################
    # model.compile(loss='binary_crossentropy', optimizer=Adam, metrics=['accuracy'])
    # print(model.summary()) # this should show lambda layers.  If you want to print your overall network structure, just remove the parallelism.
    ##################################################################
    
    assert batch_size % gpu_count == 0, "Batch size (%r) must be divisible by gpu_count (%r)" % (batch_size, gpu_count)

    def chunkify(lst, n):
        return [lst[i::n] for i in range(n)]

    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([shape[:1] // parts, shape[1:]], 0)
        stride = tf.concat([shape[:1] // parts, shape[1:] * 0], 0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    #Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                #Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
                    inputs.append(slice_n)                

                outputs = model(inputs)
                
                if not isinstance(outputs, list):
                    outputs = [outputs]
                
                #Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0))
            
        return Model(input=model.inputs, output=merged)