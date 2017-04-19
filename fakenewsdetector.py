import pandas as pd
import os, sys, getopt, cPickle, csv, sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 
from textblob import TextBlob
from Tkinter import *
newsS = pd.read_csv('news.txt', sep='\t', quoting=csv.QUOTE_NONE, names=["label", "news"])
def tokens(news):
    news = unicode(news,'utf8')
    return TextBlob(news).words

def lemmas(news):
    news = unicode(news,'utf8').lower()
    words = TextBlob(news).words
    return [word.lemma for word in words]


def train_multinomial_nb(newss):   
    msg_train, msg_test, label_train, label_test = train_test_split(newss['news'], newss['label'], test_size=0.2)    
    pipeline = Pipeline([('bow', CountVectorizer(analyzer=lemmas)),('tfidf', TfidfTransformer()),('classifier', MultinomialNB())])   
    params = {
    'tfidf__use_idf': (True, False),
    'bow__analyzer': (lemmas, tokens),
    }
    grid = GridSearchCV(
        pipeline,
        params, 
        refit=True, 
        n_jobs=-1,
        scoring='accuracy',
        cv=StratifiedKFold(label_train, n_folds=5),
    )   
    nb_detector = grid.fit(msg_train, label_train)
    predictions = nb_detector.predict(msg_test)
    file_name = 'newsmodel.txt'
    with open(file_name, 'wb') as fout:
        cPickle.dump(nb_detector, fout)
    print 'model written to: ' + file_name



def predict(news):
  nb_detector = cPickle.load(open('newsmodel1.txt'))
  svm_detector = cPickle.load(open('newsmodel1.txt'))
  nb_predict = nb_detector.predict([news])[0]
  svm_predict = svm_detector.predict([news])[0]
  return  'FAKE NEWS DETECTOR DETECTED AS ' + nb_predict


def main():
   root = Tk(className ="FAKE NEWS DETECTOR")
   root.geometry("800x800")
   svalue = StringVar() # defines the widget state as string
   w = Entry(root,textvariable=svalue) # adds a textarea widget
   w.pack()

   def act():
      print "THE NEWS ENTERED"
      print '%s' % svalue.get()
      arg='%s' % svalue.get()
      if(os.path.isfile('newsmodel1.txt') == False):            
              print "Creating Naive Bayes Model....."
              train_multinomial_nb(newsS)
      inputnews = ''   
      prediction = predict(arg)
      print 'This news is predicted by', prediction
      root = Tk()
      root.geometry("400x400")
      w = Label(root, text=prediction)
      w.pack()
      root.mainloop()
   foo = Button(root,text="SUBMIT", command=act)
   foo.pack()
   root.mainloop()
if __name__ == "__main__":
   main()
