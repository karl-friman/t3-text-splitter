import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
import io
import zipfile
import tiktoken

# --- HARDCODED SETTINGS ---
CHUNK_SIZE = 32000
CHUNK_OVERLAP = 200

# Setup token counting
enc = tiktoken.get_encoding("cl100k_base")

def length_function(text: str) -> int:
    return len(enc.encode(text))

st.set_page_config(page_title="Text Splitter (32k Tokens)", layout="wide")

st.title("Text Splitter Explorer")
st.caption(f"Settings: Chunk Size {CHUNK_SIZE} Tokens, Overlap {CHUNK_OVERLAP} Tokens")

# Initialize session state to store splits so they don't disappear on click
if "splits" not in st.session_state:
    st.session_state.splits = []

# 1. Text Input
text_input = st.text_area(
    "Paste your text here",
    height=300,
    placeholder="Enter text to split...",
)

# 2. Split Text Button
if st.button("Split Text", type="primary"):
    if text_input:
        # Initializing splitter with token length function
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=length_function,
        )
        st.session_state.splits = splitter.split_text(text_input)
    else:
        st.error("Please paste some text first.")

# If we have splits, show the download sections
if st.session_state.splits:
    splits = st.session_state.splits
    num_splits = len(splits)
    
    st.divider()
    st.subheader(f"Results: {num_splits} Chunks")

    # --- DOWNLOAD ALL SECTION (ZIP File) ---
    # We create a zip file in memory so "Download All" gives you separate files
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for i, split in enumerate(splits):
            zip_file.writestr(f"split_{i+1}.txt", split)
    
    st.download_button(
        label="📥 Download All Chunks (as .zip)",
        data=zip_buffer.getvalue(),
        file_name="all_splits.zip",
        mime="application/zip",
        help="Downloads a ZIP file containing each split as a separate .txt file"
    )

    # --- INDIVIDUAL DOWNLOAD BUTTONS SECTION ---
    st.write("### Individual Downloads")
    # Using columns to show buttons in a grid so they don't take up too much vertical space
    cols = st.columns(5) 
    for i, split in enumerate(splits):
        col_index = i % 5
        with cols[col_index]:
            st.download_button(
                label=f"File {i+1}",
                data=split,
                file_name=f"split_{i+1}.txt",
                mime="text/plain",
                key=f"btn_{i}"
            )

    st.divider()

    # --- PREVIEW SECTION ---
    st.write("### Text Preview")
    for i, split in enumerate(splits):
        with st.expander(f"Preview Chunk {i+1}", expanded=False):
            st.text(split)
