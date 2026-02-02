import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Hardcoded Settings
CHUNK_SIZE = 32000
CHUNK_OVERLAP = 200

st.set_page_config(page_title="Custom Text Splitter", layout="wide")
st.title("Text Splitter (32k/200)")

text_input = st.text_area("Paste text here", height=300)

if text_input:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    splits = splitter.split_text(text_input)
    
    st.write(f"Created {len(splits)} chunks.")

    # All splits download
    all_text = "\n\n---NEW CHUNK---\n\n".join(splits)
    st.download_button("Download All Chunks", all_text, file_name="all_splits.txt")

    # Individual splits
    for i, split in enumerate(splits):
        with st.expander(f"Chunk {i+1}"):
            st.code(split, language="text")
            st.download_button(f"Download Chunk {i+1}", split, file_name=f"split_{i+1}.txt")
