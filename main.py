import streamlit as st
from PIL import Image

from eval import load_class_data, predict, prepare_model, preprocess_image


def main():
    class_data = load_class_data('label_num_to_disease_map.json')
    model = prepare_model('INCEPTION')
    st.title("Cassava Disease Classification")

    image_file = st.file_uploader("Upload your image", type=["png", "jpg", "jpeg"])
    if image_file is not None:
        image = Image.open(image_file)

        col1, col2, col3 = st.columns([0.2, 5, 0.2])
        col2.image(image, use_column_width=True)

        predicted_class = predict(preprocess_image(image), model)

        st.markdown(f"**Predicted class:** {class_data[predicted_class][0]}")
        if predicted_class != len(class_data) - 1:
            st.markdown(f"**[Wikipedia page]({class_data[predicted_class][1]}) about disease.**")


if __name__ == "__main__":
    main()
