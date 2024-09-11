# ML Handwriting Detector

This project represents my first exploration into Convolutional Neural Networks (CNNs). The handwriting detector was trained on over 1000 pieces of data created using my own handwriting to automatically translate my handwritten characters into text. Currently, this model achieves **69% accuracy**, with plans to improve the accuracy in the future as well as expanding to handle entire words and sentences in future iterations.

## Technologies Used

* **PyTorch** was used to build and train the CNN model for handwriting recognition.
* **Python** was used throughout the project for running the **Flask** server as well as general preprocessing of data and training of the model.

## How to Use

1. [Clone repository](https://github.com/junhecui/handwriting-detector).
2. Run `python3.11 model/train.py` to train the model.
3. Run `python3.11 server.py` and go to `http://127.0.0.1:5000/upload` in browser.
4. Upload an image of a handwritten character. The model will attempt to recognize the character and display the result.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact Information

For support, please contact `cjunhe05@gmail.com`.
