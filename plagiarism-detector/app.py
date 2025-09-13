from flask import Flask, render_template, request, redirect, url_for, flash
import os
import numpy as np
import pandas as pd
import re
import nltk
import fitz  # PyMuPDF for PDF processing
import io
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    print("NLTK download may have failed. Using fallback stopwords.")

app = Flask(__name__)
app.secret_key = 'plagiarism_detector_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class PDFPlagiarismDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 
                                 "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
                                 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
                                 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 
                                 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
                                 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
                                 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
                                 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 
                                 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
                                 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
                                 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 
                                 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 
                                 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 
                                 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 
                                 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', 
                                 "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 
                                 'wouldn', "wouldn't"])
        self.documents = {}  # Store document name: text content
        self.results = []
        
    def is_pdf_file(self, file_content, filename):
        """
        Check if file is PDF by checking magic bytes
        """
        # PDF magic bytes: %PDF
        if len(file_content) > 4 and file_content[:4] == b'%PDF':
            return True
        
        # Also check filename extension as backup
        if filename.lower().endswith('.pdf'):
            return True
            
        return False
        
    def extract_text_from_pdf(self, file_content, filename):
        """
        Extract text content from PDF file content
        """
        try:
            # Open the PDF from bytes
            pdf_document = fitz.open(stream=io.BytesIO(file_content), filetype="pdf")
            text = ""
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                text += page.get_text() + " "
            
            pdf_document.close()
            return text.strip()
        except Exception as e:
            print(f"Error reading PDF {filename}: {e}")
            return None
    
    def preprocess_text(self, text):
        """
        Clean and preprocess the text
        """
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize the text
        try:
            tokens = word_tokenize(text)
        except:
            # Fallback simple tokenization if nltk fails
            tokens = text.split()
        
        # Remove stopwords and stem
        filtered_tokens = []
        for token in tokens:
            if token not in self.stop_words:
                try:
                    filtered_tokens.append(self.stemmer.stem(token))
                except:
                    filtered_tokens.append(token)
        
        # Join tokens back to text
        processed_text = ' '.join(filtered_tokens)
        
        return processed_text
    
    def calculate_similarity(self, text1, text2):
        """
        Calculate cosine similarity between two texts
        """
        if not text1 or not text2:
            return 0.0
            
        # Create TF-IDF vectors
        try:
            tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            return similarity_matrix[0][0]
        except:
            return 0.0
    
    def add_document(self, file_content, filename):
        """
        Add a document to the collection
        """
        # Check if file is PDF using our custom method
        if not self.is_pdf_file(file_content, filename):
            return False, f"Error: {filename} is not a PDF file"
            
        text = self.extract_text_from_pdf(file_content, filename)
        if text:
            processed_text = self.preprocess_text(text)
            self.documents[filename] = processed_text
            return True, f"Added document: {filename}"
        return False, f"Failed to extract text from: {filename}"
    
    def check_all_against_all(self, threshold=0.8):
        """
        Compare all documents against each other
        """
        doc_names = list(self.documents.keys())
        documents = list(self.documents.values())
        
        if len(doc_names) < 2:
            return []
        
        # Create similarity matrix
        try:
            tfidf_matrix = self.vectorizer.fit_transform(documents)
            similarity_matrix = cosine_similarity(tfidf_matrix)
        except:
            return []
        
        # Create results list (not DataFrame)
        results = []
        for i in range(len(documents)):
            for j in range(i+1, len(documents)):
                similarity = similarity_matrix[i][j]
                is_plagiarism = similarity >= threshold
                results.append({
                    'Document 1': doc_names[i],
                    'Document 2': doc_names[j],
                    'Similarity Score': f"{similarity:.2%}",
                    'Plagiarism Detected': 'YES' if is_plagiarism else 'NO'
                })
        
        self.results = results
        return results
    
    def check_specific_document(self, source_doc, threshold=0.8):
        """
        Check a specific document against all others
        """
        if source_doc not in self.documents:
            return []
            
        source_text = self.documents[source_doc]
        results = []
        
        for doc_name, doc_text in self.documents.items():
            if doc_name != source_doc:
                similarity = self.calculate_similarity(source_text, doc_text)
                is_plagiarism = similarity >= threshold
                results.append({
                    'Source Document': source_doc,
                    'Compared Document': doc_name,
                    'Similarity Score': f"{similarity:.2%}",
                    'Plagiarism Detected': 'YES' if is_plagiarism else 'NO'
                })
        
        self.results = results
        return results

# Global detector instance
detector = PDFPlagiarismDetector()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if files were uploaded
        if 'files' not in request.files:
            flash('No files selected')
            return redirect(request.url)
        
        files = request.files.getlist('files')
        threshold = float(request.form.get('threshold', 0.8))
        
        # Process each file
        for file in files:
            if file and file.filename != '':
                filename = secure_filename(file.filename)
                file_content = file.read()
                success, message = detector.add_document(file_content, filename)
                if not success:
                    flash(message)
        
        if len(detector.documents) < 2:
            flash('Need at least 2 PDF documents to compare')
            return redirect(request.url)
        
        # Perform comparison - returns list, not DataFrame
        results = detector.check_all_against_all(threshold)
        
        # Check for 100% plagiarism
        plagiarism_100 = []
        for row in results:
            if row['Similarity Score'] == '100.00%':
                plagiarism_100.append({
                    'doc1': row['Document 1'],
                    'doc2': row['Document 2']
                })
        
        return render_template('result.html', 
                             results=results,  # Pass the list directly
                             plagiarism_100=plagiarism_100,
                             threshold=threshold,
                             doc_count=len(detector.documents))
    
    return render_template('index.html', documents=list(detector.documents.keys()))

@app.route('/clear', methods=['POST'])
def clear_documents():
    detector.documents = {}
    detector.results = []
    flash('All documents cleared')
    return redirect(url_for('index'))

if __name__ == '__main__':
    print("Starting plagiarism detector application...")
    app.run(debug=True, host='0.0.0.0', port=5000)
