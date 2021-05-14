
import pickle
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler

model = pickle.load(open("model.pkl","rb"))

scaler = pickle.load(open("scaler.pkl","rb"))
Gender_encoder = pickle.load(open("Gender_encoder.pkl","rb"))
Customer_Type_encoder = pickle.load(open("Customer_Type_encoder.pkl","rb"))
Type_of_Travel_encoder = pickle.load(open("Type_of_Travel_encoder.pkl","rb"))
Class_encoder = pickle.load(open("Class_encoder.pkl","rb"))

 
page_bg_img = '''
<style>
body {
background-image: url("https://wallpaperaccess.com/full/254361.jpg");
background-size: cover;
}
</style>
'''


st.markdown(
    """
<style>

.big-font {
    font-size:40px !important;
    color:red;
    text-align:center;
}
.small-font {
    font-size:20px !important;
    color:black;
    text-align:center;

}
.error_class {
    font-size:30px !important;
    color:black;
    background:#ff6666;
    text-align:center;
    
}
.success_class {
    font-size:30px !important;
    color:black;
    background:#66ff66;
    text-align:center;
}
.reportview-container .markdown-text-container {
    font-family: monospace;
}
.sidebar .sidebar-content {
    background-image: linear-gradient(#2e7bcf,#2e7bcf);
    color: white;
}
.Widget>label {
    color: black;
    font-size:20px !important;
    font-family: monospace;
    background : #ffcccc
}
[class^="st-b"]  {
    color: white;
    font-family: monospace;
}
.slider{
    color:black
}
.st-bb {
    background-color: transparent;
}
.st-at {
    background-color: #ff5050;
}
footer {
    font-family: monospace;
}
.reportview-container .main footer, .reportview-container .main footer a {
    color: #0c0080;
}
header .decoration {
    background-image: none;
}
.val{
    font-size:10px !important;
    color:black;
    text-align:center;
}
</style>
""",
    unsafe_allow_html=True,
)


def input_page():
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.markdown('<p class="big-font"><b>Airline Passenger Satisfaction</b></p>', unsafe_allow_html=True)
    st.markdown('<p class="small-font">Machine Learning Model</p>', unsafe_allow_html=True)
    

    cust_type = st.selectbox('Customer Type', ["Loyal Customer","disloyal Customer"])
        
    age = st.slider('Age', min_value=10, max_value=80)

    
    travel_type= st.selectbox('Type of Travel', ["Personal Travel","Business travel"])
    class_= st.selectbox('Class', ["Business","Eco","Eco Plus"])
    
    flight_dist= st.text_input('Flight Distance', 0)
    
    wifi_service = st.slider('Inflight wifi service', min_value=0, max_value=5)
    ease_ol_book = st.slider('Ease of Online booking', min_value=0, max_value=5)
    food = st.slider('Food and drink', min_value=0, max_value=5)
    ol_boarding = st.slider('Online boarding', min_value=0, max_value=5)
    seat = st.slider('Seat comfort', min_value=0, max_value=5)
    ent = st.slider('Inflight entertainment', min_value=0, max_value=5)
    on_board_service = st.slider('On-board service', min_value=0, max_value=5)
    Leg = st.slider('Leg room service', min_value=0, max_value=5)
    Bagg = st.slider('Baggage handling', min_value=0, max_value=5)
    Checkin = st.slider('Checkin service', min_value=0, max_value=5)
    Inflight_service = st.slider('Inflight service', min_value=0, max_value=5)
    cleanliness = st.slider('Cleanliness', min_value=0, max_value=5)
    
    dept_delay= st.text_input('Departure Delay in Minutes',0)
    arr_delay= st.text_input('Arrival Delay in Minutes', 0)

    
    
    
    

    btn = st.button("Predict")
    
    if btn:
        
        cust_type = Customer_Type_encoder.transform(np.array(cust_type).reshape(-1,1))[0]
        age = scaler.transform(np.array(float(age)).reshape(-1,1))[0]
        travel_type = Type_of_Travel_encoder.transform(np.array(travel_type).reshape(-1,1))[0]
        class_ = Class_encoder.transform(np.array(class_).reshape(-1,1))[0]
        flight_dist = scaler.transform(np.array(float(flight_dist)).reshape(-1,1))
        dept_delay = scaler.transform(np.array(float(dept_delay)).reshape(-1,1))
        arr_delay = scaler.transform(np.array(float(arr_delay)).reshape(-1,1))
    
        test=np.array([[cust_type,age,travel_type,class_,flight_dist,
                       wifi_service,ease_ol_book,food,ol_boarding,seat,
                       ent,on_board_service,Leg,Bagg,Checkin,Inflight_service,
                       cleanliness,dept_delay,arr_delay]])
        result = model.predict(test)[0]
        if result==0:
            st.markdown('<p class="error_class"><b>Not Satisfied</b></p>', unsafe_allow_html=True)

        if result==1:
            st.markdown('<p class="success_class"><b>Satisfied</b></p>', unsafe_allow_html=True)

        

    
    
input_page()      
    
    
                  
