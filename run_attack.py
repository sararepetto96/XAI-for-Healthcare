import os
import argparse

from models_explainability.ExplainableModels import ExplainableModel

if __name__ == "__main__":
    
    new_cwd = os.path.join(os.getcwd(), "models_explainability")
    os.chdir(new_cwd)

    print("working in: ",os.getcwd())

    
    argparser = argparse.ArgumentParser(description='Run attack on model')
    argparser.add_argument('--model_name', type=str, help='model name')
    
    argparser.add_argument('--train_data_name', type=str, help='data name')
    argparser.add_argument('--n_classes', type=int, help='number of classes')
    
    argparser.add_argument("--data_name", type=str, help='data name to run the attack on')
    argparser.add_argument("--algorithm", type=str, help='algorithm to use')
    
    argparser.add_argument("--epsilon", type=int, help='epsilon for the attack')
    argparser.add_argument("--batch_size", type=int, help='batch size for the attack',default=8)
    argparser.add_argument("--n_steps", type=int, help='number of steps for the attack', default=100)
    argparser.add_argument('--dataset_subset_size', type=int, default=100, help='size of the dataset subset to use for the attack')
    
    argparser.add_argument('--new_process', action='store_true', help='run in a new process')


    args = argparser.parse_args()
    
    explainableModel = ExplainableModel(model_name = args.model_name,
                                    train_data_name = args.train_data_name,
                                     n_classes = args.n_classes)
    try:
        explainableModel.attack(algorithm=args.algorithm,
                                                    data_name=args.data_name,
                                                    batch_size=args.batch_size,
                                                    ε = args.epsilon,
                                                    n_steps=args.n_steps,
                                                    new_process=args.new_process,
                                                    dataset_subset_size=args.dataset_subset_size)
    except Exception as e:
        print(f"ERROR algorithm: {args.algorithm}, {args.epsilon}, model_name:{args.model_name}, train_data_name: {args.train_data_name}: {e}")
        raise e