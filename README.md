# Word2Vec Model on Game of Thrones Dataset

This project trains a **Word2Vec** model using text data from the *Game of Thrones* universe to find relationships between words, characters, and concepts. It uses **NLTK** for text preprocessing and **Gensim** for model training, then visualizes the word vectors in 3D.

---

## 📌 What This Project Does

1. **Loads a dataset** (multiple `.txt` files zipped together).
2. **Preprocesses the text** by:

   * Breaking text into sentences.
   * Tokenizing words.
   * Removing **stopwords** (common words like "the", "is", "and").
3. **Trains a Word2Vec model** to learn word relationships.
4. **Explores the model** by:

   * Finding similar words.
   * Checking which word doesn't match in a group.
   * Calculating similarity between two words.
5. **Visualizes word vectors** in 3D space.

---

## 📦 Requirements

You need these Python libraries:

```bash
pip install numpy pandas matplotlib seaborn gensim nltk plotly scikit-learn
```

---

## 📂 Dataset

* Your dataset should be a **zip file** named `data.zip`.
* It should contain `.txt` files with text data.
* Example: `data.zip` → contains `got1.txt`, `got2.txt`, etc.

---

## ⚙️ Steps in the Code

### 1️⃣ Import Libraries

The code imports **NumPy, Pandas, Matplotlib, Seaborn, NLTK, Gensim, Plotly**, and **Scikit-learn**.

### 2️⃣ Upload & Extract Dataset

* `files.upload()` lets you upload `data.zip` in **Google Colab**.
* `zipfile.ZipFile(...).extractall()` unzips the file into `/content/data`.

### 3️⃣ Download NLTK Resources

```python
nltk.download('punkt')
nltk.download('stopwords')
```

* **punkt** → for sentence & word tokenization.
* **stopwords** → list of common English words to remove.

### 4️⃣ Preprocess Text

```python
raw_sent = sent_tokenize(corpus)  # Break text into sentences
tokens = simple_preprocess(sent)  # Lowercase & remove punctuation
filtered_tokens = [word for word in tokens if word not in stop_words]
story.append(filtered_tokens)
```

* Turns text into **clean word lists** without stopwords.

### 5️⃣ Train Word2Vec Model

```python
model = gensim.models.Word2Vec(window=11, min_count=2)
model.build_vocab(story)
model.train(story, total_examples=model.corpus_count, epochs=model.epochs)
```

* **window=11** → context size (how many words around the target word to consider).
* **min\_count=2** → ignores words that appear less than twice.

### 6️⃣ Explore the Model

```python
model.wv.most_similar('daenerys')     # Similar words to 'daenerys'
model.wv.doesnt_match([...])          # Odd word out in a list
model.wv.similarity('arya', 'sansa')  # Similarity score between two words
```

### 7️⃣ Visualize Word Embeddings

* **PCA (Principal Component Analysis)** reduces vector dimensions to 3D for visualization.
* **Plotly** is used for interactive 3D scatter plots of word vectors.

---

## 📊 Example Output

* **Similar words**: Shows closest matches based on context in text.
* **Odd word out**: Finds which word doesn’t belong in a group.
* **3D Plot**: Displays word relationships visually.

---

## 🚀 How to Run

1. Open **Google Colab**.
2. Upload `data.zip` when prompted.
3. Run all cells from top to bottom.
4. Explore model results and interactive plot.

---

## 📖 Learning Outcome

* Understand **Word2Vec** word embeddings.
* Learn **text preprocessing** with NLTK.
* Use **Gensim** to train a model.
* Visualize **word relationships** in 3D.

-----
Dataset Link:https://www.kaggle.com/datasets/khulasasndh/game-of-thrones-books
