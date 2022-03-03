from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from quotes_for_posts_783.data.quotes_data import get_quotes_data()

class QuotesTrainer():
    def __init__(self, picture_caption):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.picture_caption = picture_caption

   
    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        topic_pipeline = Pipeline([('tfidf', TfidfVectorizer(max_df=0.75, min_df=2, stop_words="english")),
            ('lda', LatentDirichletAllocation(learning_decay=0.5, n_components=5))
])
        self.pipeline = Pipeline([('topics', topic_pipeline)])
        return self.pipeline

    def run(self):
        """set and train the pipeline"""
        self.pipeline = self.set_pipeline().fit(self.X, self.y)
        return self.pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        quotes.iloc[-1] = ['A large crowd is assembled on an outdoor street scene , with toy balloons visible and a woman walking a pink bicyc', 'image','image','5','A large crowd is assembled on an outdoor street scene , with toy balloons visible and a woman walking a pink bicyc','1']
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)
        self.mlflow_log_metric('rmse', rmse)
        return rmse

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        self.pipeline = self.set_pipeline().fit(self.X, self.y)
        return joblib.dump(self.pipeline, 'model.joblib')


if __name__ == "__main__":
    quotes = get_quotes_data()
    df = clean_data(df, test=False)
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    trainer = Trainer(X_train, y_train)

    trainer.run()
    rmse = trainer.evaluate(X_test, y_test)
    trainer.save_model()
