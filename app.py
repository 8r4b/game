import streamlit as st

# Set page configuration
st.set_page_config(page_title="Mohamed's Fun Website", layout="centered")

# Title of the app
st.title("Welcome to Mohamed's Website!")
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f5;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
        }
    </style>
    """, unsafe_allow_html=True)

# Subtitle
st.subheader("God is watching, so give him good show...")


# Add a text input for user interaction
name = st.text_input("What's your name?")
if name:
    st.write(f"Hello, {name}! Nice to meet you!")

# Sidebar for additional options
st.sidebar.title("Navigation")
st.sidebar.markdown("""
    - [Home](#)
    - [About](#)
    - [Contact](#)
""")

# Footer
st.markdown("""
    ---
    Made with ❤️ by **Mohamed**.
    """)