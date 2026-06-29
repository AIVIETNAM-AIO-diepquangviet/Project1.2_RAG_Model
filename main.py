import streamlit as st
import tempfile, os, time
import pypdf
import chromadb
import ollama

LLM_MODEL = "vicuna:7b-v1.5-q5_1"
EMBED_MODEL = "bge-m3"

PROMPT = """Bạn là trợ lý hỏi đáp. Dùng các đoạn ngữ cảnh dưới đây để trả lời câu hỏi.
Nếu ngữ cảnh không có thông tin, hãy nói là bạn không biết, đừng bịa.
Trả lời ngắn gọn, chính xác, bằng tiếng Việt.

Ngữ cảnh: {context}

Câu hỏi: {question}
Trả lời:"""

# Initiate session state
for k, v in {"collection": None, "pdf_name": "", "chat_history": []}.items():
  st.session_state.setdefault(k, v)

# Core function
def embed(texts):
  """Chuyển text thành vector embedding."""
  return ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"]  # kết quả chỉ trả về embeddings trong các trường id, document và metadata

def chunk_text(text, size=1000, overlap=200):
  """Cắt text thành các chunk nhỏ."""
  paras = [p.strip() for p in text.split("\n") if p.strip()]
  chunks, cur = [], ""
  for p in paras:
    if len(cur) + len(p) + 1 <= size:
      cur += p + "\n"
    else:
      if cur:
        chunks.append(cur.strip())
        cur = (cur[-overlap:] + p + "\n") if overlap else (p + "\n")
  if cur.strip():
    chunks.append(cur.strip()) # append chunk cuối cùng k đủ số lượng size ban đầu
  return chunks

def process_pdf(uploaded_file):
  """Đọc PDF, cắt nhỏ, tạo embedding và lưu vào ChromaDB."""
  # Lưu file upload thành file tạm
  with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
    tmp.write(uploaded_file.getvalue())
    path = tmp.name

  # Đọc nội dung PDF
  text = "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(path).pages)
  os.unlink(path) # Xóa file tạm

  # Cắt nhỏ và lưu vào ChromaDB
  chunks = chunk_text(text)
  client = chromadb.Client()
  col = client.get_or_create_collection(f"rag_{int(time.time())}")
  col.add(
      ids=[str(i) for i in range(len(chunks))],
      documents=chunks,
      embeddings=embed(chunks)
      )
  return col, len(chunks) # vector database

def rag(question, collection, k=4):
  """Hàm RAG: tìm context và hỏi LLM."""
  res = collection.query(query_embeddings=embed([question]), n_results=k) # question đầu vào cũng được embedding như dữ liệu file PDF, so sánh với các vector trong database, dữ liệu gốc nằm trong collection
  context = "\n\n".join(res["documents"][0]) # lầy ra từ văn bản gốc các đoạn liên quan
  resp = ollama.chat(
      model=LLM_MODEL,
      messages=[{"role": "user", "content": PROMPT.format(context=context, question=question)}], # dựa vào PROMT, ngữ cảnh trong văn bản gốc và câu hỏi để sinh câu trả lời
      options={"temperature": 0},
      )
  return resp["message"]["content"]

# Giao diện người dùng UI
# Cấu hình trang
st.set_page_config(page_title="PDF RAG Chatbot", layout="wide",
                   initial_sidebar_state="expanded")
st.title("PDF RAG Assistant: Native")

# Sidebar: upload PDF và nút điều khiển
with st.sidebar:
  st.subheader("Upload tài liệu")
  f = st.file_uploader("Chọn file PDF", type="pdf")
  if f and st.button("Xử lý PDF", use_container_width=True):
    with st.spinner("Đang xử lý..."):
      st.session_state.collection, n = process_pdf(f)
      st.session_state.pdf_name = f.name
      st.session_state.chat_history = []
    st.success(f"{n} chunks")
  st.info(f" {st.session_state.pdf_name}" if st.session_state.pdf_name
          else " Chưa có tài liệu")
  if st.button("Xóa lịch sử chat", use_container_width=True):
    st.session_state.chat_history = []

# Hiển thị lịch sử chat
for m in st.session_state.chat_history:
  with st.chat_message(m["role"]):
    st.write(m["content"])

# Ô nhập câu hỏi
if st.session_state.collection is None:
  st.info("Upload và xử lý PDF trước khi chat.")
  st.chat_input("Nhập câu hỏi...", disabled=True)
else:
  q = st.chat_input("Nhập câu hỏi của bạn...")
  if q:
    st.session_state.chat_history.append({"role": "user", "content": q})
    with st.chat_message("user"):
      st.write(q)
    with st.chat_message("assistant"):
      with st.spinner("Đang suy nghĩ..."):
        ans = rag(q, st.session_state.collection)
        st.write(ans)
    st.session_state.chat_history.append({"role": "assistant", "content": ans})
