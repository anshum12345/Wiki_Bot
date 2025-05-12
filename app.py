import streamlit as st
from streamlit_chat import message
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
import requests
import bs4
from langchain_community.document_loaders import TextLoader
from langchain.chains import ConversationalRetrievalChain

DB_FAISS_PATH = 'vectorstores\db_faiss'

def load_llm():
    llm = CTransformers(
        model = r"C:\Users\DEVANSH\OneDrive\Desktop\ML_TUT\GenAI\llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type = "llama",
        max_new_tokens = 128,
        temperature = 0.3
    )
    return llm

st.title("Wikipedia Conversational Bot 📖")
st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)
st.subheader('Built with the Llama 2 🦙, Langchain, and Streamlit')
st.markdown('<style>h3{color: pink;  text-align: center;}</style>', unsafe_allow_html=True)

# Expander for app details
with st.expander("About the App"):
    st.write("This app allows you to have conversation with your whole wikipedia blog.")
    st.write("Enter a Wikipedia URL in the input box below and click 'Submit' to start.")

# Input box for YouTube URL
wiki_url = st.text_input("Enter Wikipedia URL")

if st.button("Submit") and wiki_url:

    r = requests.get(wiki_url)

    soup = bs4.BeautifulSoup(r.content, 'html.parser')

    text = ""
    for paragraph in soup.find_all('p'):
        text += paragraph.get_text()
        
    tmp_file_path = "scraped_text.txt"

    # Open the file in write mode and write the text to it
    with open(tmp_file_path, 'w', encoding='utf-8') as file:
        file.write(text)
        
    loader = TextLoader(file_path=tmp_file_path, encoding='utf-8')
    data = loader.load()
    
        
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(data, embeddings)
    db.save_local(DB_FAISS_PATH)
    llm = load_llm()
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

    def conversational_chat(query):
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me anything about given wikipedia content" + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]
        
    #container for the chat history
    response_container = st.container()
    #container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):            
            user_input = st.text_area("Query:", placeholder="Talk to your link here (:", key='input').strip()
            submit_button = st.form_submit_button(label='Send')
            print("User Input:", repr(user_input))
            print('1')
        
        if submit_button and user_input:
            output = conversational_chat(user_input)
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

    