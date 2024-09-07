import streamlit as st
from model.train_model import load_model, predict_results


def main():
    # Set header and title
    st.set_page_config(page_title="Iris Prediction", page_icon="👁️")
    st.title("Iris Species Prediction")

    # Take inputs from user
    sep_len = st.number_input(
        "Please enter Sepal Length in cm : ", min_value=0.00, step=0.01
    )
    sep_wid = st.number_input(
        "Please enter Sepal Width in cm : ", min_value=0.00, step=0.01
    )
    pet_len = st.number_input(
        "Please enter Petal Length in cm : ", min_value=0.00, step=0.01
    )
    pet_wid = st.number_input(
        "Please enter Petal Width in cm : ", min_value=0.00, step=0.01
    )

    # Load model
    model = load_model()

    # Predict button
    submit = st.button("Predict")

    # Predict results
    if submit:
        pred, prob = predict_results(model, sep_len, sep_wid, pet_len, pet_wid)
        st.subheader("Predictions are : ")
        st.subheader(f"Predicted Species : {pred}")
        st.subheader(f"\nProbabilities : ")
        for species, pr in prob.items():
            st.write(f"{species} : {pr:.4f}")
            st.progress(pr)


if __name__ == "__main__":
    main()
