# Evaluating the Robustness of Explainable AI in Medical Image Recognition Under Natural and Adversarial Data Corruption

This repository contains the official PyTorch implementation for the paper:** Evaluating the Robustness of Explainable AI in Medical Image Recognition Under Natural and Adversarial Data Corruption**. 


## Dependencies and Reproducibility

In order to improve the reproducibility of our experiments, we released our anaconda environment, containing all dependencies and corresponding SW versions. 
The environment can be installed by running the following command: 

```shell
conda env create -f environment.yml
```
Once the environment is created, we can use it by typing `conda activate XAI-for-Healthcare`.

## Code Folding

The code is structured as follows: 


- **models/finetuning**: Scripts and utilities for finetuning models.
- **model_explainability/ActivationsAndGradients**: A modified version of pytorch-grad-cam, adapted to allow differentiable computation of explanations.
- **model_explainability/agreement**: Useful function to compute similarity between explanations.
- **model_explainability/custom_XAI**: Core utilities for applying various explanation (XAI) techniques.
- **model_explainability/ExplainableModels**: Functions to load datasets and models, generate explanations, and execute adversarial attacks.
- **models_explainability/MedDataset.py**: Code for creating and handling a custom data loader for medmnist.
- **models_explainability/patches**: A helpet function to support explanation techniques on transformer-based architectures.
- **models_explainability/test_agreements.py**: Main functions for evaluating explanation robustness under natural data corruption.
- **models_explainability/MedViT.py**: Useful functions to built the MedViT model ( Manzari, Omid Nejati, et al. "MedViT: a robust vision transformer for generalized medical image classification." Computers in biology and medicine 157 (2023): 106791.) in the timm library. 
- **models_explainability/ViT.py**: Adapted version of the ViT model from the timm library, tailored to integrate seamlessly with explanation methods.
- **fine_tune.py**: Main script to download datasets (clean and corrupted) and finetuning models.
- **run_attack.py**: Main script to launch adversarial attacks targeting model explanations.
- **run_natural_noise.py**: Main script to launch natural noise experiments.


## Running Experiments 
To perform an attack to a model, with a specific budget, a specific explanation method and a specific dataset, you can use:

```shell
python run_attack.py --model_name model --train_data_name dataset --n_classes classes --data_name dataset --algorithm XAI_technique --epsilon budget
```
After having executed the main function, a folder structure inside **models_explainability/attack**" will be created containing
a json file with all results.

To evaluate robustness under natural corruptions, use:

```shell
python run_natural_noise.py --run_single --model_name model --n_classes classes --data_name dataset --corruption corruption
```

or

```shell
python run_natural_noise.py --run_all
```

to run all experiments.

After having executed the main function, a folder structure inside **models_explainability/corruption_agreement** will be created containing csv files with results.


## Acknowledgements
The authors would like to thank the contributors of [captum](https://github.com/pytorch/captum) and [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) for having facilitated the development of this project.

This project has been partially developed with the support of European Union’s [ELSA – European Lighthouse on Secure and Safe AI](https://elsa-ai.eu), Horizon Europe, grant agreement No. 101070617, and [Sec4AI4Sec - Cybersecurity for AI-Augmented Systems](https://www.sec4ai4sec-project.eu), Horizon Europe, grant agreement No. 101120393.

<img src="git_images/sec4AI4sec.png" alt="sec4ai4sec" style="width:70px;"/> &nbsp;&nbsp; 
<img src="git_images/elsa.png" alt="elsa" style="width:70px;"/> &nbsp;&nbsp; 
<img src="git_images/FundedbyEU.png" alt="europe" style="width:240px;" />
