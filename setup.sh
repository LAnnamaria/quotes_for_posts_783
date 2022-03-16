mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"annamarie@windowslive.com\"\n\
" > ~/.streamlit/credentials.toml

mkdir -p ~/.streamlit/

echo "[theme]
base='dark'
primaryColor='#969696'
backgroundColor='#272835'
secondaryBackgroundColor='#015c5e'
textColor= ‘#424242’
font = ‘sans serif’
[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
