import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from catboost import CatBoostClassifier, Pool
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import pickle

raw_df1 = pd.read_csv('/home/duryan/Documents/Yeti/hackathon/strap/model/data/youtube/CAvideos.csv')
raw_df2 = pd.read_csv('/home/duryan/Documents/Yeti/hackathon/strap/model/data/youtube/USvideos.csv')
raw_df = pd.concat([raw_df1, raw_df2])
raw_df.reset_index(inplace=True)

regex = r"^.*?(?=T)"
publish_date = []
for i in range(len(raw_df['publish_time'])):
    test_str = raw_df['publish_time'][i]
    matches = re.findall(regex, test_str)[0]
    publish_date.append(matches)
    i += 1
raw_df['publish_date'] = publish_date


raw_df['trending_date'] = [x.replace('.', '-') for x in raw_df['trending_date']]
raw_df["publish_date"] = pd.to_datetime(raw_df["publish_date"])
raw_df["trending_date"] = pd.to_datetime(raw_df["trending_date"],format='%y-%d-%m')
raw_df["days_to_trend"] = raw_df["trending_date"] - raw_df["publish_date"]
raw_df["days_to_trend"] = raw_df["days_to_trend"].apply(lambda x: x.days)
raw_df["publish_date"] = raw_df["publish_date"].apply(lambda x: x.weekday())

# create a unified metric based on 5 categories
view_weight = 0.5
likes_weight = 0.25
dislikes_weight = 0.1
comment_weight = 0.15
days_weight = 0.13

raw_df["days_penalty"] = 1/(days_weight*(1+raw_df["days_to_trend"]))
raw_df["metric"] = raw_df["days_penalty"]*(raw_df["views"]*view_weight)+(raw_df["likes"]*likes_weight)+(raw_df["dislikes"]*dislikes_weight)+(raw_df["comment_count"]*comment_weight)
raw_df[["metric"]].head()

view_cut = 1000000 #set cutoff at 1 million views
day_cut = 5 #set cutoff days to trend
likes_cut = raw_df["likes"][raw_df["views"] > view_cut].mean()
dislikes_cut = raw_df["dislikes"][raw_df["views"] > view_cut].mean()
comment_cut = raw_df["comment_count"][raw_df["views"] > view_cut].mean()
cutoff = np.log((1/(days_weight*(1+day_cut)))*((view_cut*view_weight)+(likes_cut*likes_weight)+(dislikes_cut*dislikes_weight)+(comment_cut*comment_weight)))

raw_df[["metric"]] = np.log(raw_df["metric"]) # log transform to approximate normal distribution
raw_df["viral"] = np.where(raw_df["metric"]>=cutoff, 1,0) # create new class column as target to predict virality

raw_df["title_len"]=raw_df["title"].apply(lambda x: len(x)) # extract extra column from title length

raw_df["description"].fillna("", inplace=True)

raw_df["titles"]=raw_df["title"]+" "+raw_df["tags"]+" "+raw_df["description"]

raw_df = raw_df.drop(['index', 'video_id', 'channel_title',
         'publish_time', 'views', 'likes',
         'dislikes', 'comment_count', "description", "trending_date", "title","tags",'thumbnail_link',
         'comments_disabled', 'ratings_disabled', 'days_to_trend', "days_penalty", "trending_date", 'video_error_or_removed', "metric"], axis = 1)

target = 'viral'
X = raw_df.drop(target, axis = 1)
y = raw_df[[target]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

mapper = DataFrameMapper([
    (['publish_date'], [SimpleImputer(), LabelBinarizer()]),
    (['category_id'], [SimpleImputer(), LabelBinarizer()]),
    (['title_len'], [SimpleImputer(), StandardScaler()]),
    ('titles', TfidfVectorizer(stop_words='english', max_features=800, token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b'))
], df_out = True)

Z_train = mapper.fit_transform(X_train)
Z_test = mapper.transform(X_test)


cat = CatBoostClassifier(
    iterations=5000,
    eval_metric="F1",
    random_seed=42,
    learning_rate=0.5,
    early_stopping_rounds=3000
)

train_pool = Pool(data=Z_train,
                  label=y_train)

validation_pool = Pool(data=Z_test,
                       label=y_test)
# cat.fit(
#     train_pool,
#     eval_set=validation_pool,
#     verbose=False
# )

pipe = make_pipeline(mapper, cat.fit(
    train_pool,
    eval_set=validation_pool,
    verbose=False
))

print(f'Model is fitted: {cat.is_fitted()}')
print(cat.best_score_)

pickle.dump(pipe, open('/home/duryan/Documents/Yeti/hackathon/strap/model/pipe.pkl', 'wb'))
