import streamlit as st
from rag import process_video, ask_question

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Chat with YouTube Video",
    page_icon="🎥",
    layout="wide"
)

# -------------------------------------------------
# Session State Initialization
# -------------------------------------------------
if "video_processed" not in st.session_state:
    st.session_state.video_processed = False

if "video_url" not in st.session_state:
    st.session_state.video_url = None

if "video_id" not in st.session_state:
    st.session_state.video_id = None

if "chain" not in st.session_state:
    st.session_state.chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------------------------
# Sidebar – Video Setup
# -------------------------------------------------
st.sidebar.title("🎥 Video Setup")

input_url = st.sidebar.text_input(
    "YouTube Video URL",
    placeholder="https://www.youtube.com/watch?v=..."
)

if st.sidebar.button("🚀 Process Video"):
    if input_url.strip() == "":
        st.sidebar.warning("Please enter a YouTube URL")

    elif st.session_state.video_processed:
        st.sidebar.info("Video already processed")

    else:
        with st.spinner("Processing video..."):
            result = process_video(input_url)

        st.session_state.video_url = input_url
        st.session_state.video_id = result["video_id"]
        st.session_state.chain = result["chain"]
        st.session_state.video_processed = True
        st.session_state.chat_history = []

        st.sidebar.success("Video processed successfully ✅")


# -------------------------------------------------
# Sidebar – Status & Reset
# -------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Status")

if st.session_state.video_processed:
    st.sidebar.success("Ready for chat")
    st.sidebar.caption(f"Video ID: `{st.session_state.video_id}`")
else:
    st.sidebar.warning("No video processed")

if st.sidebar.button("🔄 Reset Video"):
    st.session_state.video_processed = False
    st.session_state.video_url = None
    st.session_state.video_id = None
    st.session_state.chain = None
    st.session_state.chat_history = []
    st.sidebar.success("Reset complete")

# -------------------------------------------------
# Main UI – Chat
# -------------------------------------------------
st.title("💬 Chat with YouTube Video")
st.caption(
    "Ask questions and get answers grounded **only** in the video transcript."
)

if not st.session_state.video_processed:
    st.info("👈 Paste a YouTube link in the sidebar and process it to begin.")
else:
    user_question = st.chat_input("Ask a question about the video")

    if user_question:
        with st.spinner("Thinking..."):
            answer = ask_question(
                user_question,
                st.session_state.chain
            )

        st.session_state.chat_history.append(("user", user_question))
        st.session_state.chat_history.append(("assistant", answer))

    # Render chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)
