import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

##### 1. 필요한 데이터 불러오기 #####
all_data = pd.read_csv('data_new/all_data_final.csv')
users = pd.read_csv('data_new/users_final.csv').set_index('User-ID')
ratings = all[['User-ID', 'ISBN', 'Age-Group', 'Book-Rating']]
book_scaled = pd.read_csv('data_new/book_info_scaled.csv').set_index('ISBN')


##### 2. ISBN_similarity 대각행렬 만들기 #####
ISBN_similarity = cosine_similarity(book_scaled, book_scaled)
ISBN_similarity = pd.DataFrame(ISBN_similarity, index=book_scaled.index, columns=book_scaled.index)


##### 3. train test split #####
train, test, y_train, y_test = train_test_split(ratings, ratings['Book-Rating'], test_size=0.3, random_state=42)


##### 4. 평점평균 계산 #####
# 1) 전체 도서의 평점 평균 구하기
total_mean_rating = train['Book-Rating'].mean()

# 2) ISBN별(도서별) 평점평균 구하기
mean_rating_per_ISBN = train.groupby('ISBN')['Book-Rating'].mean()

# 3) 사용자 연령별 평점평균 구하기
mean_rating_per_AgeGroup = train.groupby('Age-Group')['Book-Rating'].mean()


##### 5. 필요 함수 정의 #####
def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

def score(model):
    id_pairs = zip(test['User-ID'], test['ISBN'])
    y_pred = np.array([model(user_id, book) for (user_id, book) in id_pairs])
    y_true = np.array(test['Book-Rating'])
    return RMSE(y_true, y_pred)

# knn 모델 사용시에 사용해야하는 score 함수
def knn_score(model, k=15):
    id_pairs = zip(test['User-ID'], test['ISBN'])
    y_pred = np.array([model(user_id, book, k) for (user_id, book) in id_pairs])
    y_true = np.array(test['Book-Rating'])
    return RMSE(y_true, y_pred)

# 특정 ISBN과 높은 유사도를 가지는 k개의 (ISBN, 코사인유사도)를 return해주는 함수
def get_knn_ISBNs(ISBN, k=15):
    knn_result = ISBN_similarity.loc[ISBN].sort_values(ascending=False)
    knn_lst = list(zip(knn_result.index, knn_result.values))[1:k+1]
    return knn_lst

# knn으로 구한 상위 15개 ISBN에 대해, (각 ISBN의 평점평균 * consine similarity)한 값들의 평균값을 최종 예측 rating으로 사용하는 함수
# train_set에 해당 ISBN이 없을 경우, default 값으로는 total_mean_rating(전체 평점 평균) 사용
def ISBN_similarity_model(ISBN, k=15):
    
    knn_isbn_lst = get_knn_ISBNs(ISBN, k)
    
    rating_lst = []
    for ISBN, cosine in knn_isbn_lst:
        if ISBN in mean_rating_per_ISBN: # train set에 해당 ISBN이 있는 경우
            rating = mean_rating_per_ISBN[ISBN] * cosine # cosine similarity에 따라 가중평균
        else: # train set에 해당 ISBN이 없는 경우
            rating = total_mean_rating # default 값은 전체 평점 평균인 total_mean_rating 사용
        rating_lst.append(rating)
    rating_array = np.array(rating_lst)
    return rating_array.mean() # 15개의 가중평점들을 평균한 값

# knn으로 구한 상위 15개 ISBN에 대해, (각 ISBN의 평점평균 * consine similarity)한 값들의 평균값을 최종 예측 rating으로 사용하는 함수
# train_set에 해당 ISBN이 없을 경우, default 값으로는 해당 user_id의 연령을 확인 -> mean_rating_per_AgeGroup(나이별 전체 평점 평균) 사용
def ISBN_similarity_age_model(user_id, ISBN, k=15):
    
    knn_isbn_lst = get_knn_ISBNs(ISBN, k)
    
    rating_lst = []
    for ISBN, cosine in knn_isbn_lst:
        if ISBN in mean_rating_per_ISBN: # train set에 해당 ISBN이 있는 경우
            rating = mean_rating_per_ISBN[ISBN] * cosine # cosine similarity에 따라 가중평균
        else: # train set에 해당 ISBN이 없는 경우
            user_age = users.loc[user_id]['Age-Group'] # user_id의 age_group 정보 확인
            rating = mean_rating_per_AgeGroup[user_age] # 해당 age_group의 전체 평점 평균 반환
        rating_lst.append(rating)
    rating_array = np.array(rating_lst) 
    
    return rating_array.mean() # 15개의 가중평점들을 평균한 값