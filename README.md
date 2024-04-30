# SVM-DistilBERT-Anorexia-Detection
This repository compares the classification results of SVM and DistilBERT models for anorexia detection in text.

## Overview

This code implements a text classification pipeline for the early detection of signs of anorexia using Support Vector Machine (SVM) with TF-IDF vectorization and DistilBERT. The pipeline includes data preprocessing, model training, evaluation, and comparison between SVM and DistilBERT.

## Author Information

- **Author:** Sarah Kayembe
- **Email:** sarah.kayembe@maine.edu
- **Date:** 29 April 2024
- **Lab:** eRisks CLEF LAB 2024

## Requirements

- Python 3.x
- Libraries: `pandas`, `numpy`, `scikit-learn`, `beautifulsoup4`, `transformers`, `torch`

## Usage

1. Clone the repository or download the code files.

2. Ensure that all required libraries are installed. You can install them using pip:

   ```
   pip install pandas numpy scikit-learn beautifulsoup4 transformers torch
   ```

3. Run the main script:

   ```
   python text_classification_pipeline.py
   ```

4. The script will perform the following tasks:

   - Data preprocessing: Extracting text from XML files and preparing the data for training and validation.
   - Model training: Training SVM with TF-IDF and DistilBERT models using the training data.
   - Model evaluation: Evaluating the trained models on the validation data and calculating various metrics such as accuracy, precision, recall, F1-score, ERDE@N, latency, and latency-weighted F1.
   - Comparison: Comparing the performance of SVM with TF-IDF and DistilBERT models.
   - Predictions: Making predictions on the test data using both models and writing the predictions to a file.

5. After execution, the script will generate output files containing evaluation metrics and predictions.

## Directory Structure

- **positive_examples:** Directory containing XML files with positive examples.
- **negative_examples:** Directory containing XML files with negative examples.
- **test:** Directory containing XML files for testing.

## Additional Notes

- This code assumes that the XML files contain text data related to signs of anorexia.
- Adjustments may be needed based on the specific structure and format of the input data.
- The code includes extensive comments and documentation to aid understanding and customization.

## Citation

If you use this code or find it helpful in your research, please consider citing:

```
@misc{kayembe2024anorexia,
  title={Text Classification Pipeline for Early Detection of Signs of Anorexia},
  author={Kayembe, Sarah},
  year={2024},
  month={April},
  note={GitHub repository},
  howpublished={\url{https://github.com/yourusername/yourrepository}}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

