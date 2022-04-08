# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
from datetime import datetime

startTime = datetime.now()




filename = "05/model.sv"
model = pickle.load(open(filename,'rb'))
# otwieramy wcześniej wytrenowany model

sex_d = {0:"Male",1:"Female"}
pclass_d = {0:"First",1:"Second", 2:"Third"}
embarked_d = {0:"Cherbourg", 1:"Queenstown", 2:"Southampton"}
# o ile wcześniej kodowaliśmy nasze zmienne, to teraz wprowadzamy etykiety z ich nazewnictwem

def main():

    st.set_page_config(page_title="My pretty app")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    st.image("https://www.top13.net/wp-content/uploads/2015/10/perfectly-timed-funny-cat-pictures-5.jpg")

    with overview:
        st.title("My pretty app")

    with left:
        sex_radio = st.radio( "Sex", list(sex_d.keys()), format_func=lambda x : sex_d[x] )
        embarked_radio = st.radio( "Port embarkation", list(embarked_d.keys()), index=0, format_func= lambda x: embarked_d[x] )
        pclass_radio = st.radio( "Person class", list(pclass_d.keys()), index=1, format_func= lambda x: pclass_d[x] )

    with right:
        age_slider = st.slider("Age", value=25, min_value=1, max_value=100)
        sibsp_slider = st.slider("Number of siblings and / or partner", value=2, min_value=0, max_value=5)
        parch_slider = st.slider("Number of parents and / or children", value=4,min_value=0, max_value=10)
        fare_slider = st.slider("Ticket price", value=1000, min_value=0, max_value=5000, step=1)

    data = [[pclass_radio, sex_radio,  age_slider, sibsp_slider, parch_slider, fare_slider, embarked_radio]]
    survival = model.predict(data)
    s_confidence = model.predict_proba(data)

    with prediction:
        st.subheader("Will person survive?")
        st.subheader(("Yes" if survival[0] == 1 else "No"))
        st.write("Probability: {0:.2f}%".format(s_confidence[0][survival][0] * 100))

if __name__ == "__main__":
    main()
