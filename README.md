# 🔢 Digit Recognizer

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)
![Vercel](https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)

Full-stack application that recognizes handwritten digits in real time. The user draws a number on a canvas, and a Machine Learning model trained on the MNIST dataset predicts which digit it is, along with the full probability distribution across all 10 possible classes.

**🔗 Live demo:** [https://digit-recognizer-frontend.onrender.com](https://digit-recognizer-frontend.onrender.com)
 
> ⚠️ **Note:** the backend is hosted on Render's free tier, which spins down after periods of inactivity. The first request after idle time may take 30–60 seconds to respond while the service wakes up — please be patient on your first prediction.

## 🎯 Project goal

This project started as a hands-on exercise to apply, end to end, the core concepts from a first Machine Learning course: supervised learning, logistic regression, cost functions, and gradient descent. The goal wasn't to build something innovative, but to prove the ability to take those theoretical concepts and turn them into a real, usable, interactive application — from training the model to a UI that anyone can actually try out.

## 🧠 The model

The model is trained using **[scikit-learn](https://scikit-learn.org/)**, specifically `LogisticRegression`, the same logistic regression algorithm covered in the theory, applied here to a **multiclass classification** problem (10 classes, one for each digit from 0 to 9).

Under the hood, `LogisticRegression` solves this as a **softmax regression** problem (the generalization of binary logistic regression to multiple classes), using the `lbfgs` solver — a more efficient variant of gradient descent for this kind of problem, which iteratively adjusts the model's weights by minimizing the cost function.

```python
model = LogisticRegression(solver='lbfgs', max_iter=1000)
```

**Why `max_iter=1000`?** This parameter sets the maximum number of iterations the optimizer can take while searching for the minimum of the cost function. With the default value (100), the model didn't fully converge, and scikit-learn raised a `ConvergenceWarning`, meaning the algorithm was still improving but got cut off too early. Raising the limit to 1000 gives gradient descent enough room to converge properly.

With this configuration, the model reaches **92% accuracy** on the test set.

## 📊 The dataset: MNIST

The model is trained on **MNIST**, one of the most classic and widely used datasets in Machine Learning for image classification tasks. It consists of **70,000 handwritten digit images** (0 through 9), each 28x28 pixels in grayscale, originally collected from forms filled out by US census employees and high school students.

This dataset was chosen for a few reasons:
- It's the standard entry point for anyone starting out in Machine Learning and Computer Vision, making it ideal for a first applied project.
- It's perfectly balanced and clean, which makes it possible to focus on understanding the model and the pipeline instead of dealing with data quality issues.
- Since it's so well known, anyone with ML experience immediately recognizes the problem and can evaluate the result with context.

In this project, MNIST is loaded directly with `fetch_openml('mnist_784')` from scikit-learn, with no need to download it manually.

## 💾 Model persistence with joblib

Once trained, the model isn't retrained on every API run. Instead, it's serialized to disk using **[joblib](https://joblib.readthedocs.io/)**, a library optimized for saving objects that contain large NumPy arrays (like the weights of a scikit-learn model), more efficient than `pickle` for this kind of data.

```python
joblib.dump(model, 'digit_model.joblib')
```

The backend API simply loads that file once on startup (`joblib.load(...)`) and reuses it in memory for every prediction.

## 🖼️ Image preprocessing

This was the most interesting technical challenge in the project: the model was trained on MNIST images (perfectly centered, normalized, and in a specific color format), but a drawing made by a user on an HTML canvas doesn't meet any of those conditions by default. The `preprocess_image` function transforms any hand-drawn image into the exact format the model expects, through the following steps:

1. **Grayscale conversion**: the image is converted to a single color channel.
2. **Automatic polarity detection**: the corners of the image (where there's never any stroke) are analyzed to determine whether the background is light or dark, and colors are only inverted if needed — so the pipeline works with both light-background/dark-stroke images and transparent/dark canvases.
3. **Background noise removal**: very low pixel values (leftovers from compression or antialiasing) are zeroed out.
4. **Bounding box cropping**: the exact region containing the drawn stroke is detected, discarding all the empty space around it.
5. **Resize to 20x20 preserving aspect ratio**: the cropped digit is scaled down while keeping its original aspect ratio, mirroring how MNIST images are actually built.
6. **Geometric centering on a 28x28 canvas**: the resized image is pasted in the center of a black canvas matching the exact size the model expects.
7. **Center-of-mass centering**: the visual "weight" of the stroke is recalculated, and the image is shifted a few pixels so that center of mass lands exactly in the middle — the same centering criterion real MNIST uses, more precise than simple bounding-box centering.
8. **Contrast stretching**: pixel values are rescaled so the brightest part of the stroke reaches pure white (255), compensating for faint strokes that get washed out during resizing.
9. **Normalization**: final values are divided by 255, landing in the [0, 1] range, matching the data used during training.
10. **Flattening**: the 28x28 image is converted into a 784-value vector, the input format the model expects.

## 🎨 Frontend

The frontend is built with **Vite + React + TypeScript**, styled with **TailwindCSS**. Communication with the API is handled through a **custom hook** (`useDigitPrediction`) that encapsulates the prediction state (loading, result, latency, and errors), keeping the presentation components simple and decoupled from the networking logic.

The visual interface was generated with **[v0 by Vercel](https://v0.dev)**, with a robotic/chrome aesthetic inspired by the work of artist Hajime Sorayama.

## 🚀 Full stack

| Layer | Technology |
|---|---|
| Model training | Python, scikit-learn, joblib |
| Backend / API | FastAPI |
| Frontend | Vite, React, TypeScript, TailwindCSS |
| UI design | v0 (Vercel) |

## ▶️ Running it locally

```bash
# 1. Train the model
python train.py

# 2. Start the API
cd api
uvicorn main:app --reload

# 3. Start the frontend
cd frontend
npm install
npm run dev
```