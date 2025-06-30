import os
import argparse

from models_explainability.ExplainableModels import ExplainableModel

if __name__ == "__main__":
    
    new_cwd = os.path.join(os.getcwd(), "models_explainability")
    os.chdir(new_cwd)

    print("working in: ",os.getcwd())
    
    argparser = argparse.ArgumentParser(description='Run attack on model')
    argparser.add_argument('--model_name', type=str, help='model name', required=True)
    
    argparser.add_argument('--train_data_name', type=str, help='data name',required=True)
    argparser.add_argument('--n_classes', type=int, help='number of classes',required=True)
    
    argparser.add_argument("--data_name", type=str, help='data name to run the attack on', required=True)
    argparser.add_argument("--algorithm", type=str, help='algorithm to use. Default: calculate all attributions.', default=None)
    
    argparser.add_argument("--batch_size", type=int, help='batch size for the attack', required=True)
    
    argparser.add_argument('--new_process', action='store_true', help='run in a new process')
    
    args = argparser.parse_args()
    
    explainableModel = ExplainableModel(model_name=args.model_name,
                                                train_data_name=args.train_data_name,
                                                n_classes=args.n_classes)
    
    if args.algorithm is None:
        explainableModel.calculate_all_attributions(data_name=args.data_name,
                                                    data_split="test")
    else:
        explainableModel.explain_dataset(algorithm=args.algorithm,
                                        data_name=args.data_name,
                                        data_split="test",
                                        batch_size=args.batch_size,
                                        new_process=args.new_process)