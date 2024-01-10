import numpy as np
import numpy.ma as ma
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tabulate
from recsysNN_utils import *

class ContentBasedRecSys:
    def __init__(self, num_epochs):
        # Load Data, set configuration variables
        (self.item_train, self.user_train, self.y_train, self.item_features, 
         self.user_features, self.item_vecs, self.movie_dict, self.user_to_genre) = load_data()
        
        #Set the number of epochs:
        self.num_epochs = num_epochs
        
        self.num_user_features = self.user_train.shape[1] - 3 # remove userid, rating count and ave rating during training
        self.num_item_features = self.item_train.shape[1] - 1  # remove movie id at train time

        self.uvs = 3  # user genre vector start
        self.ivs = 3  # item genre vector start
        self.u_s = 3  # start of columns to use in training, user
        self.i_s = 1  # start of columns to use in training, items

        self.ScaleAndSplitData()

        #Create Neural Network
        self.num_outputs = 32
        self.CreateNeuralNetworks()

    def ScaleAndSplitData(self):
        # scale training data
        self.item_train_unscaled = self.item_train
        self.user_train_unscaled = self.user_train
        self.y_train_unscaled    = self.y_train

        self.scalerItem = StandardScaler()
        self.scalerItem.fit(self.item_train)
        self.item_train = self.scalerItem.transform(self.item_train)

        self.scalerUser = StandardScaler()
        self.scalerUser.fit(self.user_train)
        self.user_train = self.scalerUser.transform(self.user_train)

        self.scalerTarget = MinMaxScaler((-1, 1))
        self.scalerTarget.fit(self.y_train.reshape(-1, 1))
        self.y_train = self.scalerTarget.transform(self.y_train.reshape(-1, 1))

        self.item_train, self.item_test = train_test_split(self.item_train, train_size=0.80, shuffle=True, random_state=1)
        self.user_train, self.user_test = train_test_split(self.user_train, train_size=0.80, shuffle=True, random_state=1)
        self.y_train, self.y_test       = train_test_split(self.y_train,    train_size=0.80, shuffle=True, random_state=1)

    
    def CreateNeuralNetworks(self):
        tf.random.set_seed(1)
        self.user_NN = tf.keras.models.Sequential([  
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(self.num_outputs)
        ])

        self.item_NN = tf.keras.models.Sequential([  
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(self.num_outputs)
        ])

        # create the user input and point to the base network
        self.input_user = tf.keras.layers.Input(shape=(self.num_user_features))
        self.vu = self.user_NN(self.input_user)
        self.vu = tf.linalg.l2_normalize(self.vu, axis=1)

        # create the item input and point to the base network
        self.input_item = tf.keras.layers.Input(shape=(self.num_item_features))
        self.vm = self.item_NN(self.input_item)
        self.vm = tf.linalg.l2_normalize(self.vm, axis=1)

        # compute the dot product of the two vectors vu and vm
        self.output = tf.keras.layers.Dot(axes=1)([self.vu, self.vm])

        # specify the inputs and output of the model
        self.model = tf.keras.Model([self.input_user, self.input_item], self.output)
       
        #Compile, fit and evaluate our model
        self.CompileModel()
        self.FitModel()
        self.EvaluateModel()
    
    def CompileModel(self):
        tf.random.set_seed(1)
        cost_fn = tf.keras.losses.MeanSquaredError()
        opt = keras.optimizers.Adam(learning_rate=0.01)
        self.model.compile(optimizer=opt,
                    loss=cost_fn)

    def FitModel(self):
        tf.random.set_seed(1)
        self.model.fit([self.user_train[:, self.u_s:], self.item_train[:, self.i_s:]], self.y_train, epochs=self.num_epochs)
    
    def EvaluateModel(self):
        self.model.evaluate([self.user_test[:, self.u_s:], self.item_test[:, self.i_s:]], self.y_test)

    def RecommendMoviesForNewUser(self, user_vec):
        # generate and replicate the user vector to match the number movies in the data set.
        user_vecs = gen_user_vecs(user_vec, len(self.item_vecs))

        # scale our user and item vectors
        suser_vecs = self.scalerUser.transform(user_vecs)
        sitem_vecs = self.scalerItem.transform(self.item_vecs)

        # make a prediction
        y_p = self.model.predict([suser_vecs[:, self.u_s:], sitem_vecs[:, self.i_s:]])

        # unscale y prediction 
        y_pu = self.scalerTarget.inverse_transform(y_p)

        # sort the results, highest prediction first
        sorted_index = np.argsort(-y_pu,axis=0).reshape(-1).tolist()  #negate to get largest rating first
        sorted_ypu   = y_pu[sorted_index]
        sorted_items = self.item_vecs[sorted_index]  #using unscaled vectors for display

        print_pred_movies(sorted_ypu, sorted_items, self.movie_dict, maxcount = 10)

    def RecommendMoviesForExistingUser(self, user_id):
        # form a set of user vectors. This is the same vector, transformed and repeated.
        user_vecs, y_vecs = get_user_vecs(user_id, self.user_train_unscaled, self.item_vecs, self.user_to_genre)

        # scale our user and item vectors
        suser_vecs = self.scalerUser.transform(user_vecs)
        sitem_vecs = self.scalerItem.transform(self.item_vecs)

        # make a prediction
        y_p = self.model.predict([suser_vecs[:, self.u_s:], sitem_vecs[:, self.i_s:]])

        # unscale y prediction 
        y_pu = self.scalerTarget.inverse_transform(y_p)

        # sort the results, highest prediction first
        sorted_index = np.argsort(-y_pu,axis=0).reshape(-1).tolist()  #negate to get largest rating first
        sorted_ypu   = y_pu[sorted_index]
        sorted_items = self.item_vecs[sorted_index]  #using unscaled vectors for display
        sorted_user  = user_vecs[sorted_index]
        sorted_y     = y_vecs[sorted_index]

        #print sorted predictions for movies rated by the user
        print_existing_user(sorted_ypu, sorted_y.reshape(-1,1), sorted_user, sorted_items, self.ivs, self.uvs, self.movie_dict, maxcount = 50)

    def RecommendMovies(self):
        input_item_m = tf.keras.layers.Input(shape=(self.num_item_features)) # input layer
        vm_m = self.item_NN(input_item_m) # use the trained item_NN
        vm_m = tf.linalg.l2_normalize(vm_m, axis=1) # incorporate normalization as was done in the original model
        self.model_m = tf.keras.Model(input_item_m, vm_m)                                
        scaled_item_vecs = self.scalerItem.transform(self.item_vecs)
        vms = self.model_m.predict(scaled_item_vecs[:,self.i_s:])
        
        count = 50  # number of movies to display
        
        dim = len(vms)
        dist = np.zeros((dim,dim))

        for i in range(dim):
            for j in range(dim):
                dist[i,j] = sq_dist(vms[i, :], vms[j, :])
                
        m_dist = ma.masked_array(dist, mask=np.identity(dist.shape[0]))  # mask the diagonal
        disp = [["If you enjoyed:", "You may also like:"]]
        for i in range(count):
            min_idx = np.argmin(m_dist[i])
            movie1_id = int(self.item_vecs[i,0])
            movie2_id = int(self.item_vecs[min_idx,0])
            disp.append( [self.movie_dict[movie1_id]['title'], self.movie_dict[movie2_id]['title']]
                    )
        table = tabulate.tabulate(disp, tablefmt='grid', headers="firstrow")
        print(table)