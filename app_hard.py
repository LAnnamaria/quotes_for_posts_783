
#from tkinter import DISABLED
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import time
import os
import datetime
import requests
#import ipdb

st.session_state.wrapped_quotes = {}

st.markdown(f'<h1 style="color:#FFFFFF;font-size:60px;">{"Quotes For Posts"}</h1>', unsafe_allow_html=True)
st.markdown(f'<h1 style="color:#E1E1E1;font-size:30px;">{"Wondering how to caption your picture? Let me give you a suitable quote as a suggestion 📸"}</h1>', unsafe_allow_html=True)

st.markdown("""---""")
#components.html("""<hr style="height:2px;border:none;color:#333;background-color:#333;" /> """)

st.markdown('#')


st.markdown('### About the project 📚:')
with st.expander("See more about the Quotes for your Posts"):
    st.write("""
         Wondering what caption to use with your picture? Don´t want to spend a lot of time looking for a nice quote online? Get our suggestions within seconds by only one click! If you´re not satisfied, you can give us some help finding the very best quote for you which you can copy and use immediately under your precious memories!
     """)
    path = os.getcwd()
    st.image(f"{path}/demo_images/Le Wagon.png")

# def pc.copy(text):
#     command = 'echo ' + text.strip() + '| clip'
#     os.system(command)


#Sidebar

st.sidebar.markdown("# Control Panel")
st.sidebar.markdown("""---""")

SIDEBAR_OPTION_DEMO_IMAGE = "Select a Demo Image 💻 "
SIDEBAR_OPTION_UPLOAD_IMAGE = "Upload an Image 📷"
SIDEBAR_OPTION_JUST_TAGS="Just using tags ✏️"
SIDEBAR_OPTION_TEAM="More about our team 💼"

DEMO_PHOTO_SIDEBAR_OPTIONS=["None","Cat","Dance","Galaxy","Paris"]
SIDEBAR_OPTIONS = [ SIDEBAR_OPTION_DEMO_IMAGE, SIDEBAR_OPTION_UPLOAD_IMAGE,SIDEBAR_OPTION_JUST_TAGS,SIDEBAR_OPTION_TEAM]
app_mode = st.sidebar.selectbox("Please select from the modes", SIDEBAR_OPTIONS)

##############
# Hardcoded  #
##############

cat_q=["What greater gift than the love of a cat.Charles Dickens",
"I felt like an animal, and animals don’t know sin, do they? Jess C. Scott, Wicked Lovely",
"Until one has loved an animal, a part of one's soul remains unawakened. Anatole France",
"If having a soul means being able to feel love and loyalty and gratitude, then animals are better off than a lot of humans. James Herriot, All Creatures Great and Small",
"Love all God’s creation, both the whole and every grain of sand. Love every leaf, every ray of light. Love the animals, love the plants, love each separate thing. If thou love each thing thou wilt perceive the mystery of God in all; and when once thou perceive this, thou wilt thenceforward grow every day to a fuller understanding of it: until thou come at last to love the whole world with a love that will then be all-embracing and universal. Fyodor Dostoyevsky, The Brothers Karamazov"]

paris_q=["Mine was the twilight and the morning. Mine was a world of rooftops and love songs.Roman Payne, Rooftop Soliloquy",
"you’ll have to fall in love at least once in your life, or Paris has failed to rub off on you. E.A. Bucchianeri, Brushstrokes of a Gadfly",
"Yes, it was too late, and Sabina knew she would leave Paris, move on, and on again, because were she to die here they would cover her up with a stone, and in the mind of a woman for whom no place is home the thought of an end to all flight is unbearable. Milan Kundera, The Unbearable Lightness of Being",
"Paris is the only city in the world where starving to death is still considered an art. Carlos Ruiz Zafón, The Shadow of the Wind",
"There is always a city. There is always a civilisation. There is always a barbarian with a pickaxe. Sometimes you are the city, sometimes you are the civilisation, but to become that city, that civilisation, you once took a pickaxe and destroyed what you hated, and what you hated was what you did not understand. Jeanette Winterson, The Powerbook"]

quotes_demo=[
"I'm selfish, impatient and a little insecure. I make mistakes, I am out of control and at times hard to handle. But if you can't handle me at my worst, then you sure as hell don't deserve me at my best.  Marilyn Monroe",
"You've gotta dance like there's nobody watching,Love like you'll never be hurt,Sing like there's nobody listening,And live like it's heaven on earth.   William W. Purkey,",
"You know you're in love when you can't fall asleep because reality is finally better than your dreams.   Dr. Seuss",
"A friend is someone who knows all about you and still loves you.   Elbert Hubbard",
"Darkness cannot drive out darkness: only light can do that. Hate cannot drive out hate: only love can do that.   Martin Luther King Jr"
]

sky_q=["When you do your works sincerely, you put wings to your works and then they will fly all over the world on their own! Mehmet Murat ildan",
"Sometimes you try to fly and you fall. Remember your falls are not fatal, they're just a little painful. Endure the pain, clean the blood stain, you'll surely gain! Get up and fly again!Israelmore Ayivor, Daily Drive 365",
"If only I could freeze this moment, this feeling, because right now, nothing else mattered. I was f***ing flying.Tara Kelly,Encore",
"I was sitting, starring into the sky with a tear in my eye thinking: what a beautiful world that the MAKER has allowed us to enjoy for a moment - not to destroy - to enjoy. So let us remove the strife, seize the moments we are blessed with and love this life. Bobby F. Kimbrough Jr.",
"We are earthbound creatures, Maggie had thought. No matter how tempting the sky. No matter how beautiful the stars. No matter how deep the dream of flight. We are creatures of the earth. Born with legs, not wings, legs that root us to the earth, and hands that allow us to build our homes, hands that bind us to our loved ones within those homes. The glamour, the adrenaline rush, the true adventure, is here, within these homes. The wars, the detente, the coups, the peace treaties, the celebrations, the mournings, the hunger, the sating, all here. Thrity Umrigar"]

sky_q_topics=[ "Children of the same family, the same blood, with the same first associations and habits, have some means of enjoyment in their power, which no subsequent connections can supply. Jane Austen, Mansfield Park"]

cat_q_topics=["Babies have the power to make grumpy people happy because they love you no matter what. Dogs are that way, too.Mariel Hemingway, Invisible Girl"]

just_q=["Life is war, war is game so life is game. Either lose or win. Know it earlier. Oladosu feyikogbon"]


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
                        time.sleep(4)
                        st.success('Your Quotes are ready!')
                        with st.container():
                            for count,ele in enumerate(just_q,1):
                                st.write(count,ele)
                                # columns = st.columns([0.01,1.5])
                                # st.session_state.wrapped_quotes[count] = "\n".join(wrap(ele,width=80))
                                # columns[0].markdown(count)
                                # columns[1].markdown(f'"{st.session_state.wrapped_quotes[count]}"')
                                # if st.button(f"Copy quote {count} to clipboard"):
                                #     pc.copy(st.session_state.wrapped_quotes[count])







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

        path = os.getcwd()

        if photo_select=="Cat":
            image = Image.open(f'{path}/demo_images/Cat.jpg')
            st.image(image, caption='Cat is playing')
            cal_b=st.sidebar.button('Show me the suitable quotes',key="paris")
            if cal_b:
                st.session_state.load_topics_cat = True
                with st.spinner('Wait for it...'):
                    time.sleep(3)
                    st.success('Your Quotes are ready')
                    with st.container():

                        for count,ele in enumerate(cat_q,1):
                          st.write(count,ele)
                    st.markdown("""---""")
                st.markdown("##### 👈 If the sentiment of the picture is different than the Top 5 quotes and you would like to define it by yourself, please give us some tags and submit! 🧐")


        if photo_select=="Dance":
            image = Image.open(f'{path}/demo_images/Dance.jpg')
            st.image(image, caption='Dancing in the club')

        if photo_select=="Galaxy":
            image = Image.open(f'{path}/demo_images/Galaxy.jpg')
            st.image(image, caption='Amazing Galaxy')

        if photo_select=="Paris":
            image = Image.open(f'{path}/demo_images/Paris.jpg')
            st.image(image, caption='Beauty Paris')
            cal_b=st.sidebar.button('Show me the suitable quotes',key="paris")
            if cal_b:
                with st.spinner('Wait for it...'):
                    time.sleep(3)
                    st.success('Your Quotes are ready')
                    with st.container():

                        for count,ele in enumerate(paris_q,1):
                          st.write(count,ele)


        st.markdown("""---""")

        if photo_select=="None":
            cal_b=st.sidebar.button('Show me the suitable quotes')
            if cal_b:
                with st.spinner('Wait for it...'):
                    time.sleep(3)
                    st.success('Your Quotes are ready')
                    with st.container():

                        for count,ele in enumerate(quotes_demo,1):
                          st.write(count,ele)

                            # columns = st.columns([0.01,1.5])
                            # st.session_state.wrapped_quotes[count] = "\n".join(wrap(ele,width=80))
                            # columns[0].markdown(count)
                            # #ipdb.set_trace()
                            # columns[1].markdown(f'"{st.session_state.wrapped_quotes[count]}"')
                            # if st.button(f"Copy quote {count} to clipboard"):
                            #     pc.copy(st.session_state.wrapped_quotes[count])

                            # # if st.button(f"Copy quote {count} to clipboard"):
                            # #     pc.copy(st.session_state.wrapped_quote)
                            # #columns[1].code(f'"{st.session_state.wrapped_quote}"')
                            # #columns[1].code(str("\n".join(wrap(f'"{ele}"',width=80))))


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
                time.sleep(5)
                st.success('Your Quotes are ready!')
                with st.container():
                    for count,ele in enumerate(sky_q,1):
                        st.write(count,ele)
                        #columns = st.columns([0.01,1.5])
                        # wrapped_quote = {count:"\n".join(wrap(ele,width=80))}
                        # columns[0].markdown(count)
                        # columns[1].markdown(f'"{wrapped_quote}"')

                        # if st.button(f"Copy quote {count} to clipboard"):
                        #     pc.copy(wrapped_quote[count])
                        # columns = st.columns([0.01,1.5])
                        # wrapped_quote = {count:"\n".join(wrap(ele,width=80))}
                        # columns[0].markdown(count)
                        # columns[1].markdown(f'"{wrapped_quote}"')

                        # if st.button(f"Copy quote {count} to clipboard"):
                        #     pc.copy(wrapped_quote[count])


                st.markdown("""---""")
                st.markdown("##### 👈 If the sentiment of the picture is different than the Top 5 quotes and you would like to define it by yourself, please give us some tags and submit! 🧐")


else:
    pass


################################
#  Not Saticfaction for upload #
################################
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
                        time.sleep(4)
                        st.success('Your Quotes are ready!')
                        with st.container():
                            for count,ele in enumerate(sky_q_topics,1):
                                st.write(count,ele)

                                # columns = st.columns([0.01,1.5])
                                # wrapped_quote = {count:"\n".join(wrap(ele,width=80))}
                                # columns[0].markdown(count)
                                # columns[1].markdown(f'"{wrapped_quote}"')
                                # if st.button(f"Copy quote {count} to clipboard"):
                                #     pc.copy(wrapped_quote[count])



##################################
#  Not Saticfaction for demo-cat #
##################################
# Making sure topics section is True + mode is upload
if 'load_topics_cat' in st.session_state and app_mode == SIDEBAR_OPTION_DEMO_IMAGE:
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
                        time.sleep(4)
                        st.success('Your Quotes are ready!')
                        with st.container():
                            for count,ele in enumerate(cat_q_topics,1):
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

#EXAMPLE FROM CHALLENGE

# #'''parameters = {"pickup_datetime": pickup_datetime,
#         "pickup_longitude": pickup_longitude,
#         "pickup_latitude": pickup_latitude,
#         "dropoff_longitude": dropoff_longitude,
#         "dropoff_latitude": dropoff_latitude,
#         "passenger_count": passenger_count}
