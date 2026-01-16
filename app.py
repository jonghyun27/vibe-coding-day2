import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# 1. 페이지 설정 및 API 키 로드
st.set_page_config(page_title="PDF ChatBot (Gemini 2.5)", layout="wide")
st.title("📄 PDF RAG 챗봇 (Gemini 2.5 Flash)")

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("Streamlit Secrets에 'GEMINI_API_KEY'가 설정되지 않았습니다.")
    st.stop()

# 2. RAG 파이프라인 구축 (캐싱 처리로 속도 향상)
@st.cache_resource
def setup_rag_chain(uploaded_file=None):
    # 파일 저장 (PyPDFLoader는 경로가 필요함)
    temp_file_path = "temp_pdf_storage.pdf"
    
    if uploaded_file:
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    elif os.path.exists("test.pdf"):
        temp_file_path = "test.pdf"
    else:
        return None

    # PDF 로드 및 분할
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # 임베딩 및 벡터 저장소 생성 (Gemini 모델 사용)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vectorstore = FAISS.from_documents(texts, embeddings)

    # 프롬프트 템플릿 설정
    template = """당신은 업로드된 문서의 내용을 바탕으로 답변하는 비서입니다. 
    문서에 없는 내용에 대해 질문하면 "문서에서 해당 내용을 찾을 수 없습니다."라고 답변하세요.
    말투는 정중하고 간결하게 하세요.

    Context: {context}
    Question: {question}
    Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # Gemini 2.5 Flash 모델 설정
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.1
    )

    # QA 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain

# 3. 사이드바 - 파일 업로드
with st.sidebar:
    st.header("설정")
    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")
    if st.button("문서 학습 시작"):
        with st.spinner("문서를 분석 중입니다..."):
            st.session_state.qa_chain = setup_rag_chain(uploaded_file)
            st.success("학습 완료!")

# 4. 채팅 인터페이스 구현
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 메시지 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 채팅 입력창
if prompt := st.chat_input("문서에 대해 궁금한 점을 물어보세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if "qa_chain" in st.session_state and st.session_state.qa_chain:
            response = st.session_state.qa_chain.invoke(prompt)
            answer = response["result"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.warning("먼저 PDF 파일을 업로드하고 '문서 학습 시작' 버튼을 눌러주세요.")
