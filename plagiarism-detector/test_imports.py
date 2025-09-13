try:
    import flask
    print(f"✓ Flask version: {flask.__version__}")
except ImportError as e:
    print(f"✗ Flask import failed: {e}")

try:
    import numpy
    print(f"✓ NumPy version: {numpy.__version__}")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")

try:
    import pandas
    print(f"✓ Pandas version: {pandas.__version__}")
except ImportError as e:
    print(f"✗ Pandas import failed: {e}")

try:
    import sklearn
    print(f"✓ Scikit-learn version: {sklearn.__version__}")
except ImportError as e:
    print(f"✗ Scikit-learn import failed: {e}")

try:
    import nltk
    print(f"✓ NLTK version: {nltk.__version__}")
except ImportError as e:
    print(f"✗ NLTK import failed: {e}")

try:
    import pdfplumber
    print(f"✓ PDFPlumber version: {pdfplumber.__version__}")
except ImportError as e:
    print(f"✗ PDFPlumber import failed: {e}")

try:
    import werkzeug
    print(f"✓ Werkzeug version: {werkzeug.__version__}")
except ImportError as e:
    print(f"✗ Werkzeug import failed: {e}")