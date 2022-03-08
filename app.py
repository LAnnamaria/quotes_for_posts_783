
from tkinter import DISABLED
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import time
import os
import datetime

import requests



st.markdown(f'<h1 style="color:#FFFFFF;font-size:60px;">{"Quotes For Posts"}</h1>', unsafe_allow_html=True)
st.markdown(f'<h1 style="color:#E1E1E1;font-size:30px;">{"Wondering how to caption your picture? Let me give you a suitable quote as a suggestion 📸"}</h1>', unsafe_allow_html=True)

st.markdown("""---""")
#components.html("""<hr style="height:2px;border:none;color:#333;background-color:#333;" /> """)

st.markdown('#')


st.markdown('### About the project 📚:')
with st.expander("See explanation"):
    st.write("""
         Wondering what caption to use with your picture? Don´t want to spend a lot of time looking for a nice quote online? Get our suggestions within seconds by only one click? If you´re not satisfied, you can give us some help finding the very best quote for you which you can copy and use immediately under your precious memories!
     """)
    st.image("https://pledgeviewer.eu/sites/default/files/2020-05/le-wagon-color.png")








#Sidebar

st.sidebar.markdown("# Control Panel")
st.sidebar.markdown("""---""")

SIDEBAR_OPTION_DEMO_IMAGE = "Select a Demo Image"
SIDEBAR_OPTION_UPLOAD_IMAGE = "Upload an Image"
SIDEBAR_OPTION_JUST_TAGS="Just using tags"
SIDEBAR_OPTION_TEAM="More about our team 💼"

DEMO_PHOTO_SIDEBAR_OPTIONS=["None","Cat","Dance","Galexy","Paris"]
SIDEBAR_OPTIONS = [ SIDEBAR_OPTION_DEMO_IMAGE, SIDEBAR_OPTION_UPLOAD_IMAGE,SIDEBAR_OPTION_JUST_TAGS,SIDEBAR_OPTION_TEAM]
app_mode = st.sidebar.selectbox("Please select from the modes", SIDEBAR_OPTIONS)

quotes_demo=[
"I'm selfish, impatient and a little insecure. I make mistakes, I am out of control and at times hard to handle. But if you can't handle me at my worst, then you sure as hell don't deserve me at my best.  Marilyn Monroe",
"You've gotta dance like there's nobody watching,Love like you'll never be hurt,Sing like there's nobody listening,And live like it's heaven on earth.   William W. Purkey,",
"You know you're in love when you can't fall asleep because reality is finally better than your dreams.   Dr. Seuss",
"A friend is someone who knows all about you and still loves you.   Elbert Hubbard",
"Darkness cannot drive out darkness: only light can do that. Hate cannot drive out hate: only love can do that.   Martin Luther King Jr"
]


##############
# Just Tags  #
##############
if app_mode == SIDEBAR_OPTION_JUST_TAGS:
    st.sidebar.markdown("""---""")
    st.warning('Just Tags Mode')
    st.sidebar.write("- Please give us some tags and see the suitable quotes! 🧐")
    with st.sidebar.form(key="topics", clear_on_submit=True):
        t1=st.text_input("Tag 1")
        t2=st.text_input("Tag 2")
        t3=st.text_input("Tag 3")
        t4=st.text_input("Tag 4")
        t5=st.text_input("Tag 5")
        submitted = st.form_submit_button("Submit Topics & Run!")
    if submitted:
        st.sidebar.write("Your Tags:",t1,t2,t3,t4,t5)
        #Topic list for model
        added_topics=[t1,t2,t3,t4,t5]
        #st.write(added_topics)
        with st.spinner('Wait for it...'):
                        time.sleep(3)
                        st.success('Your Quotes are ready!')
                        with st.container():
                            for count,ele in enumerate(quotes_demo,1):
                                st.write(count,ele)
                                for x in range(5):
                                    st.code(quotes_demo[x])






###########
#   Team  #
###########
if app_mode == SIDEBAR_OPTION_TEAM:
    st.markdown("## Our Team:")
    st.write("- Annamaria Libor [🔗](https://www.linkedin.com/in/annamaria-libor/)")
    st.write("- Morad Younis [🔗](https://github.com/Morad4444)")
    st.write("- Ali Narimani [🔗](https://www.linkedin.com/in/ali-narimani/)")
    st.write("- Mohanakrishnan Guruguhanathan [🔗](https://www.linkedin.com/in/mkguru5101991/)")

###########
#   Demo  #
###########
if app_mode == SIDEBAR_OPTION_DEMO_IMAGE:

        photo_select=st.sidebar.selectbox("Please select the photo", DEMO_PHOTO_SIDEBAR_OPTIONS)
        with st.container():
            st.markdown("""---""")
            st.warning('Demo Mode')

        if photo_select=="Cat":
            image = Image.open('/home/ali/code/quotes_for_posts_783_front_end/quotes_for_posts_783/raw_data/front_end_dir/demo_images/Cat.jpg')
            st.image(image, caption='Cat is playing')

        if photo_select=="Dance":
            image = Image.open('/home/ali/code/quotes_for_posts_783_front_end/quotes_for_posts_783/raw_data/front_end_dir/demo_images/Dance.jpg')
            st.image(image, caption='Dancing in the club')

        if photo_select=="Galexy":
            image = Image.open('/home/ali/code/quotes_for_posts_783_front_end/quotes_for_posts_783/raw_data/front_end_dir/demo_images/Galexy.jpg')
            st.image(image, caption='Amazing Galexy')

        if photo_select=="Paris":
            image = Image.open('/home/ali/code/quotes_for_posts_783_front_end/quotes_for_posts_783/raw_data/front_end_dir/demo_images/Paris.jpg')
            st.image(image, caption='Beauty Paris')

        st.markdown("""---""")

        if photo_select!="None":
            cal_b=st.sidebar.button('Show me the suitable quotes')
            if cal_b:
                with st.spinner('Wait for it...'):
                    time.sleep(3)
                    st.success('Your Quotes are ready')
                    with st.container():
                        for count,ele in enumerate(quotes_demo,1):
                            st.write(count,ele)
                            for x in range(5):
                                    st.code(quotes_demo[x])
        else:
            pass

###########
#  Upload #
###########
#st.session_state.cal_b=False
#st.session_state.topic=False

if app_mode == SIDEBAR_OPTION_UPLOAD_IMAGE:

    st.sidebar.warning('Please upload your desired photo.Choose a file in .JPG and Max Size:5 Mb')
    uploaded_file=st.sidebar.file_uploader("")
    st.markdown("""---""")
    st.warning('Image Upload Mode')
    if uploaded_file is not None:
        with st.container():

            image = Image.open(uploaded_file)
            st.image(image)
            path = os.getcwd()
            with open(f"{path}/tempDir","wb") as f:
                f.write(uploaded_file.getbuffer())
        if 'count' not in st.session_state:
            st.session_state.count = 0
        cal_b=st.sidebar.button('Show me the suitable quotes')
        if cal_b:
            st.session_state.load_topics = True
            with st.spinner('Wait for it...'):
                time.sleep(3)
                st.success('Your Quotes are ready!')
                with st.container():
                    for count,ele in enumerate(quotes_demo,1):
                        st.write(count,ele)
                st.markdown("""---""")
                st.markdown("##### 👈 If the sentiment of the picture is different than the Top 5 quotes and you would like to define it by yourself, please give us some tags and submit! 🧐")


else:
    pass


#####################
#  Not Saticfaction #
#####################
# Making sure topics section is True + mode is upload
if 'load_topics' in st.session_state and app_mode == SIDEBAR_OPTION_UPLOAD_IMAGE:
    st.sidebar.markdown("""---""")
    st.sidebar.write("- Give us your tags and submit!")
    with st.sidebar.form(key="topics", clear_on_submit=True):
        t1=st.text_input("Tag 1")
        t2=st.text_input("Tag 2")
        t3=st.text_input("Tag 3")
        t4=st.text_input("Tag 4")
        t5=st.text_input("Tag 5")
        submitted = st.form_submit_button("Submit Topics & Run!")
    if submitted:
        st.sidebar.write("Your Tags:",t1,t2,t3,t4,t5)
        #Topic list for model
        added_topics=[t1,t2,t3,t4,t5]
        #st.write(added_topics)
        with st.spinner('Wait for it...'):
                        time.sleep(3)
                        st.success('Your Quotes are ready!')
                        with st.container():
                            for count,ele in enumerate(quotes_demo,1):
                                st.write(count,ele)



# st.session_state.input=True
# if st.session_state.topic==True :
#     st.session_state.input=True
#     st.sidebar.write("- If you are not satisfied, do not worry. You can add up to 5 favorite topics. 🧐")
#     t1=st.sidebar.text_input("A")
#     st.sidebar.write(t1)

#more_topic=st.sidebar.button('Add Topics',disabled=True)

# if cal_b :
#      st.sidebar.markdown("""---""")
#      st.sidebar.write("- If you are not satisfied, do not worry. You can add up to 5 favorite topics. 🧐")
#      with st.sidebar.form(key="topics", clear_on_submit=True):
#             t1=st.text_input("Topic 1")
#             t2=st.text_input("Topic 2")
#             t3=st.text_input("Topic 3")
#             t4=st.text_input("Topic 4")
#             t5=st.text_input("Topic 5")

#             submitted = st.form_submit_button("Submit Topics")
#             if submitted:
#                 st.write(t1,t2,t3,t4,t5)
#                 #Topic list for model
#                 added_topics=[t1,t2,t3,t4,t5]

# else:
#     pass


# if cal_b:
#     st.sidebar.markdown("""---""")
#     st.sidebar.write("- If you are not satisfied, do not worry. You can add up to 5 favorite topics. 🧐")
#         #more_topic=st.sidebar.button('Add Topics')
#     t1=st.sidebar.text_input("Topic 1")
#         # t2=st.sidebar.text_input("Topic 2")
#         # t3=st.sidebar.text_input("Topic 3")
#         # t4=st.sidebar.text_input("Topic 4")
#         # t5=st.sidebar.text_input("Topic 5")
#     st.markdown(t1)
