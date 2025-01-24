# gPC_Image_Classification

# Sensitivity Analysis of Image Classification Models using Generalized Polynomial Chaos

This repository contains the code for the research paper titled "Sensitivity Analysis of Image Classification Models using Generalized Polynomial Chaos", which is currently published as a preprint on arXiv. Here, we provide all the necessary information and resources to reproduce the results presented in our paper.

## Abstract

Integrating advanced communication protocols in production has accelerated the adoption of data-driven predictive quality methods, notably machine learning (ML) models.
However, ML models in image classification often face significant uncertainties arising from model, data, and domain shifts. These uncertainties lead to overconfidence in the classification models regarding the training dataset. To better understand these models, sensitivity analysis can help practitioners analyze the relative influence of input parameters on the output.
Our work investigates the sensitivity of image classification models used for predictive quality. We propose modeling the distributional domain shifts of inputs with random variables and quantifying their impact on outputs using Sobol indices computed via generalized polynomial chaos (gPC). This approach is validated through a case study involving a welding defect classification problem, utilizing a fine-tuned ResNet18 model and an emblem classification model used in BMW Group production facilities.

## Citation

If you find this work useful for your research, please cite our paper:

```
TODO: Arxive Link
```

## Repository Structure

```
├── data/                   # Example FMEA (as csv)
├── .gitignore              # The gitignore file
├── requirements.txt        # The requirements file
├── create_data.py          # The requirements file
├── fine_tune_model.py      # The requirements file
├── run_gpc.py              # The requirements file
├── LICENSE                 # The license file
└── README.md               # The README file (this file)
```

## Setup and Installation

Clone the repository and install the required packages:
1. Clone the repository and install the required Python packages
```bash
# Example setup commands
git clone https://github.com/lpossner/gPC_Image_Classification.git
cd gPC_Image_Classification
pip install -r requirements.txt
```
2. Download the tig-aluminium-5083 dataset from Kaggle [instance](https://www.kaggle.com/datasets/danielbacioiu/tig-aluminium-5083/) and place it in the data folder.

## Usage

Instructions on how to start the backend service.

```bash
# Example usage commands
python fine_tune_model.py
python create_data.py
python run_gpc.py
```

## Additional Resources

- [Link to preprint](TODO: Arxive Link)
- [tig-aluminium-5083 dataset](https://www.kaggle.com/datasets/danielbacioiu/tig-aluminium-5083/)

## Contributing

Please feel free to contact one of the authors in case you wish to contribute.

## License

This project is licensed under the MIT License - see the [MIT License](https://github.com/lukasbahr/kg-rag-fmea/blob/main/LICENSE) file for details.

## Contact Information

For any queries regarding the paper or the code, please open an issue on this repository or contact the authors directly at:

- [Lukas Bahr](mailto:lukas.bahr@bmw.de)