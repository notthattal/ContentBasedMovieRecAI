import pandas as pd
from RecSysFuncs import *

'''
This function is where you can set your own movie preferences based on genre.
Movies are rated on a scale from 0.5-5.0 inclusive.

Note: You do not need to adjust the user_id, rating_ave or rating_count variables
'''
def CreateNewUser():
    #############################
    #Adjust your ratings here:
    #############################
    action = 4.5
    adventure = 4.0
    animation = 5.0
    childrens = 4.0
    comedy = 5.0
    crime = 4.25
    documentary = 4.0
    drama = 3.0
    fantasy = 5.0
    horror = 0.5
    mystery = 3.5
    romance = 2.5
    scifi = 4.5
    thriller = 3.5
    #############################

    user_id = 5000
    rating_ave = 0.0
    rating_count = 14

    user_vec = np.array([[user_id, rating_count, rating_ave, action, adventure, animation, childrens,
                      comedy, crime, documentary, drama, fantasy, horror, mystery, romance, scifi, thriller]])
    user_vec[0][2] = np.sum(user_vec[0][3:]) / len(user_vec[0][3:])
    return user_vec

'''
You may adjust the number of epochs with 'num_epochs' to train the model

After training, 3 tables are printed out:
  1. The first is a table with recommendations for you based on the rankings you submitted above
  2. The second shows an example of how the model would predict rankings for movies for a specific user. You can select
     a different user to see how the model behaves differently by changing the 'existing_user_id' variable
  3. The last table grabs 50 movies from the list and shows what the model thinks would be the closest recommendations to
     the movie selected
'''
def main():
    pd.set_option("display.precision", 1)
    user_vec = CreateNewUser()
    existing_user_id = 2
    num_epochs = 30
    movie_rec_sys = ContentBasedRecSys(num_epochs)

    movie_rec_sys.RecommendMoviesForNewUser(user_vec)
    movie_rec_sys.RecommendMoviesForExistingUser(existing_user_id)
    movie_rec_sys.RecommendMovies()


if __name__ == "__main__":
    main()