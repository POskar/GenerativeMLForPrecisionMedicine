# ACIT5900 - Short Master's Thesis
The accuracy of machine learning models used in clinical decision making has a direct impact on a patient's chances of recovery. Missing data pose a challenge and generative models can assist overcome it.

Clinical decision support systems have the potential to create tangible improvements in the chances of recovery and quality of life of the patients. They can also contribute to save lives. To achieve this goal, improving the precision of machine learning models for outcome prediction is crucial. Ideally, models perform best when all information about a patient is available, but in real-world cases (especially in highly-time constrained cases such as acute stroke) they must operate with limited data. In this project, we will explore the use of variational autoencoders to accurately model the dependencies in a clinical dataset, evaluate their benefit on the precision of the predictions, and ultimately their potential to maximize the quality of life of patients.

## Goal
1. Design and implement methods for patient outcome prediction using generative models in combination with multiclass classifiers
2. Evaluate the influence of the fidelity of the generative model on the final system's accuracy