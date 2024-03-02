# Generative Machine Learning for Precision Medicine

## Project Description

### Introduction

Predicting patient outcomes is crucial in clinical decision-making, particularly in time-sensitive scenarios like acute stroke. This Master's thesis project focuses on enhancing the accuracy of clinical decision-support systems (CDSS) through the use of generative machine learning (ML) models, specifically variational autoencoders (VAEs) and generative adversarial imputation networks (GAINs).

### Project Objectives

1. **Design and Implementation of Generative Models:**
    - Explore VAEs in combination with multiclass classifiers for patient outcome prediction.
    - Optionally explore GAINs for the same purpose.
    - Implement methods to incorporate generative models into a prognostic tool that is robust to incomplete inputs.

2. **Evaluation of Generative Model Fidelity:**
    - Measure imputation quality on a simulated dataset with artificially removed data items.
    - Evaluate metrics expressing uncertainty in imputed values.
    - Investigate the impact of generative model fidelity on the accuracy of the overall prognostic pipeline.
    - Assess trade-offs between model fidelity and computational efficiency in real-time clinical settings.

### Data Sources

Utilize patient information datasets like AmsterdamUMCdb, HIRID, eICU, MIMIC, and acute stroke data from the annotated MRI dataset in Baltimore.

### Methodology

1. Develop a generative model based on either "Variational Autoencoder with Arbitrary Conditioning" or "Generative Adversarial Imputation Network (GAIN)" to impute missing entries in patient data.
   
2. Integrate imputed data into a predictive model to enhance prognosis accuracy with incomplete inputs.
   
3. Determine the best procedure to calibrate the resulting predictive model.

### Expected Outcomes

- Improved precision in patient outcome predictions through a generative model-based pipeline.
- A notion of uncertainty due to missing data.
- Insights into the influence of generative model fidelity on clinical decision support system accuracy.
- Validation of VAEs' potential in addressing missing data challenges in clinical datasets.

### Significance of the Project

- Creation of a prognostic ML-based tool for CDSS with potential extended functionalities.
- Enhanced clinical decision-making for better patient outcomes and improved quality of life.
- Contribution to robust methods for handling missing data in high-stakes healthcare scenarios.
- Practical application insights of generative models in precision medicine.

### Getting Started

To get started with the project, follow these steps:

1. Clone this repository to your local machine.
2. Before running the code, pleases init and update necessary submodules by running `git submodule init` and `git submodule update` in terminal.

---

**References:**
- [Variational Autoencoder with Arbitrary Conditioning](https://doi.org/10.48550/arXiv.1806.02382)
- [GAIN: Missing Data Imputation Using Generative Adversarial Nets](https://doi.org/10.48550/arXiv.1806.02920)
- [Annotated Clinical MRIs and Linked Metadata of Patients with Acute Stroke](https://doi.org/10.3886/ICPSR38464.v5)
