# Evaluating the Robustness of Explainable AI in Medical Image Recognition Under Natural and Adversarial Data Corruption

This code is the official PyTorch implementation of the **Evaluating the Robustness of Explainable AI in Medical Image Recognition Under Natural and Adversarial Data Corruption**. 


## Dependencies and Reproducibility

- Python ≥ 3.12.*
- PyTorch ≥ 2.4.*
- torchvision ≥ 0.19.*
- timm ≥ 1.0.15

In order to improve the reproducibility of our experiments, we released our anaconda environment, containing all dependencies and corresponding SW versions. 
The environment can be installed by running the following command: 

```shell
conda env create -f environment.yml
```
Once the environment is created, we can use it by typing `conda activate MedViT`.

## Code Folding

The code is structured as follows: 

- **models/finetuning**, contains code to finetune the models.
- **model_explainability/ActivationsAndGradients**, contains a modified version of the same function of pytorch-grad-cam, with some            modifications to make computation of explanations  differentiable
- **model_explainability/agreement**, contains useful function to compute similarity between explanations.
- **model_explainability/custom_XAI**,contains useful function to use explanation methods.
-**model_explainability/custom_XAI**,contains useful function to use load the data and the model and to compute the explanation and to perform the attack.
- **models_explainability/MedDataset.py**, contains useful code to built the dataloader
- **models_explainability/Metrics**, 
- **models_explainability/patches**
- **models_explainability/test_agreements.py**, contains code to perform the evaluation of robustness against natural noise.
- **models_explainability/ViT.py**, containts code to modify the official ViT code of timm to use in a easier way explanation methods
- **fine_tune.py**, used to finetune models.
- **run_attack.py**, used to perform the attack.
- **run_attributions_comp**, use to compute and store explanations (non so se ha senso tenerlo a cancellarlo)


### Running Experiments 
To perform an attach to a model, with a specific budget, a specifict explanation method and a specific dataset, you can use:

```shell
python run_attack.py --model_name=model --train_data_name = dataset --data_name=dataset --epsilon budget
```
After having executed the main function, a folder structure inside **models_explainability/attack**" will be created containing
a json file with all results.

To evaluate natural noise robustness, you can use the function full_corruption agreement() from **models_explainability/test_agreement**".
After having executed the main function, a folder structure inside **models_explainability/corruption_agreement**" will be created containing
a csv file with all results.


## Acknowledgements
The authors would like to thank the contributors of [captum](https://github.com/pytorch/captum) and [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) for having facilitated the development of this project.

This project has been partially developed with the support of European Union’s [ELSA – European Lighthouse on Secure and Safe AI](https://elsa-ai.eu), Horizon Europe, grant agreement No. 101070617, and [Sec4AI4Sec - Cybersecurity for AI-Augmented Systems](https://www.sec4ai4sec-project.eu), Horizon Europe, grant agreement No. 101120393.

<img src="git_images/sec4AI4sec.png" alt="sec4ai4sec" style="width:70px;"/> &nbsp;&nbsp; 
<img src="git_images/elsa.jpg" alt="elsa" style="width:70px;"/> &nbsp;&nbsp; 
<img src="git_images/FundedbytheEU.png" alt="europe" style="width:240px;" />