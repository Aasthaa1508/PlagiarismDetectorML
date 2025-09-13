# Plagiarism Detector Using Machine Learning

A Python-based plagiarism detection system that uses TF-IDF and cosine similarity to identify similar documents. This tool helps detect potential plagiarism by comparing text documents and calculating their similarity scores.

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn-orange)
![NLTK](https://img.shields.io/badge/NLP-NLTK-green)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-red)

## Features

- **Text Preprocessing**: Tokenization, stopword removal, and stemming
- **TF-IDF Vectorization**: Convert text documents to numerical features
- **Cosine Similarity**: Measure similarity between documents
- **Multiple Input Methods**: Support for sample documents, custom folders, and manual input
- **Visualization**: Heatmaps and dimensionality reduction plots (PCA, t-SNE)
- **Configurable Threshold**: Adjustable similarity threshold for plagiarism detection
- **N-gram Support**: Enhanced detection using unigrams and bigrams

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/plagiarism-detector.git
cd plagiarism-detector
Install required dependencies:

bash
pip install -r requirements.txt
Download NLTK data resources:

python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
Usage
Running in Jupyter Notebook
Start Jupyter Notebook:

bash
jupyter notebook
Open plagiarism_detector.ipynb

Run all cells or follow the interactive prompts

Input Methods
The detector supports multiple input methods:

Sample Documents: Use built-in sample texts for testing

Folder Input: Load your own text files from a directory

Manual Input: Paste text directly into the interface

Single Document Check: Test one document against existing corpus

Example Usage
python
# Basic usage with sample documents
from plagiarism_detector import detect_plagiarism

# Load documents
documents, filenames = load_documents('my_docs_folder')

# Detect plagiarism
results = detect_plagiarism(documents, filenames, threshold=0.7)

# Check a new document
new_text = "Your text here..."
sources = check_for_plagiarism(new_text, documents, filenames)
How It Works
Text Preprocessing:

Convert to lowercase

Remove punctuation and numbers

Tokenize and remove stopwords

Apply stemming

Feature Extraction:

Convert text to TF-IDF vectors

Optional n-gram processing (1-2 grams)

Similarity Calculation:

Compute cosine similarity between documents

Identify pairs above threshold

Visualization:

Generate similarity heatmaps

Create 2D visualizations using PCA and t-SNE

Project Structure
text
plagiarism-detector/
│
├── plagiarism_detector.ipynb      # Main Jupyter notebook
├── requirements.txt               # Python dependencies
├── sample_docs/                   # Sample documents directory
│   ├── doc_1.txt
│   ├── doc_2.txt
│   └── ...
├── utils/
│   ├── text_processing.py         # Text preprocessing functions
│   ├── similarity.py              # Similarity calculation functions
│   └── visualization.py           # Plotting functions
└── README.md
Results Interpretation
Similarity Score 0.0-0.3: Documents are different

Similarity Score 0.3-0.7: Documents share some similarities

Similarity Score 0.7-1.0: Potential plagiarism detected

Customization
Adjusting Threshold
python
# Set custom threshold (default: 0.7)
results = detect_plagiarism(documents, filenames, threshold=0.6)
Using Different N-gram Ranges
python
# Use only unigrams
vectorizer = TfidfVectorizer(ngram_range=(1, 1))

# Use unigrams and bigrams (default)
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

# Use up to trigrams
vectorizer = TfidfVectorizer(ngram_range=(1, 3))
Example Output
text
Plagiarism Detection Results (threshold = 0.7):
doc_1.txt and doc_2.txt - Similarity: 0.8341
doc_4.txt and doc_5.txt - Similarity: 0.8341
doc_6.txt and doc_7.txt - Similarity: 0.8341

3 plagiarism pairs detected.
Applications
Academic Integrity: Check student submissions for plagiarism

Content Creation: Ensure originality of written content

Research: Identify duplicate publications

Code Plagiarism: Can be adapted for source code comparison

Limitations
Works best with longer documents (>100 words)

May produce false positives with common phrases

Doesn't detect paraphrasing that significantly changes word choice

Language specific (currently optimized for English)

Future Enhancements
Add support for more file formats (PDF, DOCX)

Implement semantic similarity using word embeddings

Add multilingual support

Create web interface

Add database storage for documents

Implement real-time checking API

Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the project

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Built with Scikit-learn, NLTK, and other amazing Python libraries

Inspired by academic plagiarism detection systems

Thanks to the open-source community for contributions

Support
If you have any questions or issues, please open an issue on GitHub or contact [your-email@example.com].

Note: This tool is designed for educational purposes and should be used as part of a comprehensive plagiarism detection approach, not as the sole method for determining academic misconduct.

text

## Additional Files to Create

### requirements.txt
numpy>=1.21.2
pandas>=1.3.3
scikit-learn>=0.24.2
nltk>=3.6.2
matplotlib>=3.4.3
seaborn>=0.11.2
jupyter>=1.0.0

text

### .gitignore
Byte-compiled / optimized / DLL files
pycache/
*.py[cod]
*$py.class

Jupyter Notebook
.ipynb_checkpoints/

Virtual environment
venv/
env/

IDE files
.vscode/
.idea/

Data files
*.csv
*.pkl
*.model

Logs
logs/
*.log

OS files
.DS_Store
Thumbs.db
