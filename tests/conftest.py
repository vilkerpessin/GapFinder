import warnings

# Suppress PyMuPDF SWIG deprecation warnings (internal to library)
warnings.filterwarnings(
    "ignore",
    message="builtin type Swig.*",
    category=DeprecationWarning
)
warnings.filterwarnings(
    "ignore",
    message="builtin type swigvarlink.*",
    category=DeprecationWarning
)
