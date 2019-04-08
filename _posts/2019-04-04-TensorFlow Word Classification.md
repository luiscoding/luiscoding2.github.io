---
layout:     post
title:      TensorFlow Word Classification
subtitle:   
date:       2019-04-04
author:     Lu Zhang
header-img: img/images/img-190403.jpg
catalog: true
tags:
    - Machine Learning
    - Tensorflow
---

## TensorFlow 
Open Source library to train the ML models, Keras,with ML, we can build the deep learning network. 

## NLP with Tensorflow
### Tokenization 

```python 
def Tokenize(comment):
    """Receives a string (comment) and returns array of tokens."""
    
    words = re.split(r"[^a-zA-Z]+",comment)
    words = list(map(lambda x:x.lower(),words))
    words = [word for word in words if len(word)>=2]


    return words

```

### Firstlayer 
Replace the first input layer without using embedding look up 

```python 
def FirstLayer(net, l2_reg_val, is_training):
    """First layer of the neural network.

    Args:
        net: 2D tensor (batch-size, number of vocabulary tokens),
        l2_reg_val: float -- regularization coefficient.
        is_training: boolean tensor.A

    Returns:
        2D tensor (batch-size, 40), where 40 is the hidden dimensionality.
    """
    net = tf.nn.l2_normalize(net,axis=1)
    net = tf.contrib.layers.fully_connected(
            net, 40, activation_fn=None,biases_initializer=None)
    
    define_loss = 2*l2_reg_val*tf.nn.l2_loss(net)
    tf.losses.add_loss(define_loss,tf.GraphKeys.REGULARIZATION_LOSSES)
    net = tf.math.tanh(net)
    net = tf.contrib.layers.batch_norm(net,is_training=is_training)
    return net
```

### Self-Define Regularization Function 

```python 
def EmbeddingL2RegularizationUpdate(embedding_variable, net_input, learn_rate, l2_reg_val):
    """Accepts tf.Variable, tensor (batch_size, vocab size), regularization coef.
    Returns tf op that applies Vone regularization step on embedding_variable."""
    # TODO(student): Change this to something useful. Currently, this is a no-op.

    net_input = tf.linalg.l2_normalize(net_input,axis =1)
    x_tran = tf.transpose(net_input)
    res1 = tf.linalg.matmul(x_tran,net_input)
    res2 = tf.linalg.matmul(res1,embedding_variable)
    #embedding_variable = tf.math.subtract()  
  
    #return embedding_variable.assign(embedding_variable-l2_reg_val*2*learn_rate*res2)
    return tf.assign(embedding_variable,embedding_variable-l2_reg_val*2*learn_rate*res2)

    #return embedding_variable.assign(embedding_variable)
```

### Random dropout some element in sparse matrix 

```python 
def SparseDropout(slice_x, keep_prob=0.5):
    """Sets random (1 - keep_prob) non-zero elements of slice_x to zero.

    Args:
        slice_x: 2D numpy array (batch_size, vocab_size)

    Returns:
        2D numpy array (batch_size, vocab_size)
    """
    #r = numpy.random.uniform(-1, 10, size=10000)
    #data = numpy.random.uniform(0, 1, size=(len(slice_x),len(slice_x[0]))).tolist()

    #data = numpy.rint(data) 
    #for i in range(len(slice_x)):
        #for j in range(len(slice_x[0])):
            #slice_x[i][j] *= data[i][j]
   # a = np.array([True, True, True, False, False])
   # b = np.array([[1,2,3,4,5], [1,2,3,4,5]])
    
   # vectorize the function, speed up to apply on the numpy array 
    def myfunc(a):
       x = int(random.uniform(0,1))
       print(x)
       if(x==0):
           return 0
       else: 
           return a
        
    vfunc = numpy.vectorize(myfunc)                                          
    nonzero_indx = numpy.nonzero(slice_x)
    nx = nonzero_indx[0]
    ny = nonzero_indx[1]
    nx,ny = skshuffle(nx,ny)
    
    # nx == ny so drop the length as nx
    drop_len= int(len(nx)*(1-keep_prob))
    slice_x[nx[:drop_len],ny[:drop_len]]=0
    return slice_x
```

### Visualize the Result and Saveas PDF 
```python 
def savePdf(sess):
    if EMBEDDING_VAR is None:
        print('Cannot visualize embeddings. EMBEDDING_VAR is not set')
        return
    embedding_mat = sess.run(EMBEDDING_VAR)
    tsne_embeddings = ComputeTSNE(embedding_mat)
    class_to_words = {
        'positive': [
                'relaxing', 'upscale', 'luxury', 'luxurious', 'recommend', 'relax',
                'choice', 'best', 'pleasant', 'incredible', 'magnificent', 
                'superb', 'perfect', 'fantastic', 'polite', 'gorgeous', 'beautiful',
                'elegant', 'spacious'
        ],
        'location': [
                'avenue', 'block', 'blocks', 'doorman', 'windows', 'concierge', 'living'
        ],
        'furniture': [
                'bedroom', 'floor', 'table', 'coffee', 'window', 'bathroom', 'bath',
                'pillow', 'couch'
        ],
        'negative': [
                'dirty', 'rude', 'uncomfortable', 'unfortunately', 'ridiculous',
                'disappointment', 'terrible', 'worst', 'mediocre'
        ]
    }

    # TODO(student): Visualize scatter plot of tsne_embeddings, showing only words
    # listed in class_to_words. Words under the same class must be visualized with
    # the same color. Plot both the word text and the tSNE coordinates.
    print('visualization should generate now')
    x=[]
    y=[]
    label=[]
    colors=[]
    
    for k,v in class_to_words.items():
        if k=='positive':
            color='b'
        elif k=='negative':
            color='orange'
        elif k=='location':
            color='g'
        else:
            color='r'    
        cc = [color]*len(v)
        colors.extend(cc)
        label.extend(v)
        
        xx = [tsne_embeddings[TERM_INDEX[wd]][0] for wd in v]
        yy = [tsne_embeddings[TERM_INDEX[wd]][1] for wd in v]
        
        x.extend(xx)
        y.extend(yy)
        
    x = numpy.array(x)
    y = numpy.array(y)
    
    axis = plt.gca()
    axis.scatter(x,y,c=colors)
   
    for ii,ll in enumerate(label):
        axis.annotate(ll,(x[ii],y[ii]))     
   # f = plt.figure()
   # with open("test.txt","w") as f:
        #f.write("this is test")
    plt.savefig("tsne_embeddings.pdf", bbox_inches='tight')
```

### Whole Procedure 
```python 

def main(argv):
    ######### Read dataset
    x_train, y_train, x_test, y_test = GetDataset()

    ######### Neural Network Model
    x = tf.placeholder(tf.float32, [None, x_test.shape[1]], name='x')
    y = tf.placeholder(tf.float32, [None, y_test.shape[1]], name='y')
    is_training = tf.placeholder(tf.bool, [])

    l2_reg_val = 1e-6    # Co-efficient for L2 regularization (lambda)
    net = BuildInferenceNetwork(x, l2_reg_val, is_training)


    ######### Loss Function
    tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=net)

    ######### Training Algorithm
    learning_rate = tf.placeholder_with_default(
            numpy.array(0.01, dtype='float32'), shape=[], name='learn_rate')
    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = tf.contrib.training.create_train_op(tf.losses.get_total_loss(), opt)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    def evaluate(batch_x=x_test, batch_y=y_test):
        probs = sess.run(net, {x: batch_x, is_training: False}) 
        print_f1_measures(probs, batch_y)

    def batch_step(batch_x, batch_y, lr):
            sess.run(train_op, {
                    x: batch_x,
                    y: batch_y,
                    is_training: True, learning_rate: lr,
            })

    def step(lr=0.01, batch_size=100):
        indices = numpy.random.permutation(x_train.shape[0])
        for si in range(0, x_train.shape[0], batch_size):
            se = min(si + batch_size, x_train.shape[0])
            slice_x = x_train[indices[si:se]] + 0    # + 0 to copy slice
            slice_x = SparseDropout(slice_x)
            batch_step(slice_x, y_train[indices[si:se]], lr)


    lr = 0.05
    print('Training model ... ')
    for j in range(300): step(lr)
    for j in range(300): step(lr/2)
    for j in range(300): step(lr/4)
    print('Results from training:')
    evaluate()
    savePdf(sess)
```

