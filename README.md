# DLS-Project: Image Fragment Matching System
Project for 'Deep Learning for Search' course at Innopolis University

This project focuses on developing a **deep learning-based image search system** that identifies original photos from fragmented image inputs. The system is designed for use cases like plagiarism detection, product search, and content verification in social media.

---

## Authors
- Ilia Mistiurin
- Nazgul Salikhova
- Milyausha Shamsutdinova

---

## Links
[Report link](https://github.com/MrIlyaneX/DLS-Project)
[Presentation slides link](https://github.com/MrIlyaneX/DLS-Project)

## 📌 Problem Statement
The system matches an image fragment to its original photo or highly similar images. It leverages advanced machine learning models to achieve high accuracy, low latency, and scalability.

### Core Questions Addressed:
- How to preprocess and represent data for efficient image matching?
- What architecture ensures high performance with minimal computational resources?
- How to evaluate the system's accuracy and reliability?

---

## 💡 Use Cases
1. **Plagiarism Detection**: Identifying unauthorized use of image content.
2. **Content Verification**: Authenticating image content in social media posts.
3. **Retail Search**: Locating products using image fragments.

---

## 🚀 Key Features
- **High Accuracy**: Ensures reliable image matching.
- **Low Latency**: Results delivered in ≤ 5 seconds.
- **Scalability**: Handles large datasets with efficient indexing.

---

## 🔧 System Architecture

### Components:
1. **Image Preprocessing**:
   - Sliding window technique.
   - Object detection using `YOLOv8`.
2. **Embedding Generation**:
   - `Nomic-Embed-Vision-v1.5` for high-quality embeddings.
3. **Database**:
   - Vector search using **Qdrant** (HNSW for indexing).
4. **Search Service**:
   - Fast similarity search with scalable architecture.

### Data Flow:
1. Image fragments are generated using sliding window or object detection.
2. Fragments and original images are embedded.
3. Embeddings are stored and indexed in **Qdrant**.
4. Query fragments are matched with stored embeddings using distance metrics (e.g., Euclidean, cosine).

---

## 📊 Experimental Results
### Embedding Dimensions:
- **Before Reduction**: 768 dimensions (310 MB memory).
- **After PCA Reduction**: 223 dimensions (140 MB memory, 90% variance explained).

### Dataset:
- **Source**: Open Images V7 (class: "Flowers").
- **Size**: 5,000 training images yielding 59,980 fragments.

### Evaluation:
- 100 test images evaluated using fragments from various methods (sliding window, object detection).
- Metrics: Precision, Recall, F1-Score.

---

## 🛠 Technology Stack
- **Programming Language**: Python 3.11
- **Models**: 
  - `YOLOv8` for object detection.
  - `Nomic-Embed-Vision-v1.5` for embedding generation.
- **Database**: Qdrant (HNSW for efficient indexing).
- **Dataset**: Open Images V7.

---

## 💻 Hardware Requirements
- **VRAM**: 1.5 GB (depends on batch size).
- **Disk Space**: 5 GB (dataset storage).
- **RAM**: 1 GB (database memory).

---

## 📈 Metrics
1. **Offline**:
   - Precision, Recall, F1-Score.
   - Computational efficiency (memory, latency).
2. **Online**:
   - User feedback on search accuracy.

---

## 🔮 Future Work
- Enhance database indexing using FAISS.
- Expand dataset to include diverse classes.
- Improve detection model accuracy and efficiency.

---

## 🐳 Deployment
- Containerization with **Docker**.
- Scalable via **Qdrant Clusters**.

---

## 🔗 Repository
[GitHub Link to Project](https://github.com/MrIlyaneX/DLS-Project)

