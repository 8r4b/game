import streamlit as st
import random

class Coin:
    def __init__(self):
        self.side = "Heads"

    def flip(self):
        self.side = random.choice(["Heads", "Tails"])
        return self.side 

    def choose(self, choice):
        if choice == self.side:
            return "فزت 👉👈"  # "You won" in Arabic
        else:
            return "خسرت طاحظك"  # "You lost" in Arabic

# Create an instance of the Coin class
game = Coin()

# Streamlit interface
st.title("ميع ميع")  # Title in Arabic
st.write("اختار وجه لو ظهر")  # Instructions in Arabic

user_choice = st.radio("اختار وحدة سيد محمد", ["Heads", "Tails"])  # User choice

if st.button("Flip the coin"):
    result = game.flip()  # Call the flip method on the instance
    st.write(f"the coin landed on {result}")  # Show the result
    if user_choice:
        feedback = game.choose(user_choice)  # Call the choose method
        st.write(feedback)  # Display feedback