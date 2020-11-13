import pandas as pd
from keras.preprocessing import sequence
import warnings
warnings.filterwarnings("ignore")

token = pd.read_pickle("token")
model = pd.read_pickle("model")

print("Predict whether a headline is clickbait or not\nEnter Q to quit\n\n")
q = True
while q:
    title = input("Headline : ")
    if (title=="q" or title=="Q"):
        break
    else:
        title = token.texts_to_sequences([title])
        title = sequence.pad_sequences(title, maxlen=150)
        label = model.predict_classes(title)
        if label[0][0] == 0:
            print("This news is not a clickbait")
        else:
            print("This news is a clickbait")
        print()