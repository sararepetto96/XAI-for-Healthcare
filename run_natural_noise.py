import os
import argparse
from models_explainability.test_agreements import full_corruption_agreement, corruption_agreement


if __name__ == "__main__":

    new_cwd = os.path.join(os.getcwd(), "models_explainability")
    os.chdir(new_cwd)

    print("working in: ",os.getcwd())

    parser = argparse.ArgumentParser(description='Run natural noise experiments')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run_single", action="store_true", help="Run natural noise experiments on a single model, dataset and corruption")
    group.add_argument("--run_all", action="store_true", help="Run all natural noise experiments")

    parser.add_argument('--model_name', type=str, help='model name')
    parser.add_argument('--n_classes', type=int, help='number of classes')
    parser.add_argument("--data_name", type=str, help='data name to run the natural noise on', choices=['dermamnist', 'octmnist', 'pneumoniamnist'])
    parser.add_argument("--corruption", type=int, help='corruption to run experiment on.')

    args = parser.parse_args()

    # Validation: if --run_all is used, disallow all the other params
    if args.run_all:
        forbidden_args = ['model_name', 'n_classes', 'data_name', 'corruption']
        for arg in forbidden_args:
            if getattr(args, arg) is not None:
                parser.error(f"--{arg} is not allowed when using --run_all")

    # Optional: check that required args for run_single are present
    if args.run_single:
        required_args = ['model_name', 'data_name', 'n_classes', "corruption"]
        for arg in required_args:
            if getattr(args, arg) is None:
                parser.error(f"--{arg} is required when using --run_single")

    if args.run_single:
        print(f"Running natural noise experiment for model: {args.model_name}, dataset: {args.data_name}, corruption: {args.corruption}")
        corruption_agreement(model_name=args.model_name, 
                            data_name=args.data_name+"_224", 
                            data_name_corrupted=f"{args.data_name}_corrupted_{args.corruption}_224",
                            data_split="test",
                            print_mode="csv",
                            best_to_show=0,
                            worst_to_show=0)
        
    elif args.run_all:
        full_corruption_agreement()
    
    else:
        print("No valid option selected. Use --run_single or --run_all.")
        parser.print_help()
        exit(1)
