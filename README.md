## Welcome to your very own movie recommendation AI

This AI is built using a content-based filtering approach and uses the "ml-latest-small.zip" dataset from Movielens which can be found here: https://grouplens.org/datasets/movielens/latest/
The dataset has been reduced in size to focus on movies from the years since 2000 and popular genres. The reduced dataset has 397 users, 847 movies and 25521 ratings.

When testing this project for yourself, you can see some pre-selected ratings for movie genres in the CreateNewUser() function. If you would like to add your own ratings, just modify the variables within the comments in that function
to fit your own preferences. Movies are ranked on a scale of 0.5-5, inclusive.

Some variables you may want to adjust for fine-tuning include: 
  - num_epochs: The number of epochs used for training
  - existing_user_id: The user for which you want table-2 to output its predicted recommendations

If all that seems like a lot of work and you just want to see what this code does, don't worry I've included some images below!

### Sample Output

#### Table 1: This table shows recommendations for you based on the rankings you submitted per-genre
![Table1_PersonalMovieRecommendationSample](https://github.com/notthattal/ContentBasedMovieRecAI/assets/61614571/658a1816-4b1d-4b29-aec2-60bfbe174700)

#### Table 2: This table shows an example of how the model would predict rankings for movies for a specific user (this table shows predictions for user 2)
![Table2_ExistingUserPredictedRatingSample](https://github.com/notthattal/ContentBasedMovieRecAI/assets/61614571/520b17c3-21a7-4f31-a463-6ee8ef7791af)

#### Table 3: This last table shows the model's predictions (right column) for the closest recommendations to the movie selected (left column)
![Table3_ContentRecommendationSample](https://github.com/notthattal/ContentBasedMovieRecAI/assets/61614571/ee270c4b-0532-46af-a920-da693f510935)
