import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    parser = argparse.ArgumentParser(description="ST-CDF Main Pipeline")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["train_teacher", "train_student_fed", "eval", "eval_et0",
                                 "counterfactual", "baselines", "ablation"],
                        help="Execution mode")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lambda_phy", type=float, default=0.1, help="Physics loss weight")
    parser.add_argument("--clients", type=int, default=5, help="Number of federated clients")
    parser.add_argument("--missing_ratio", type=float, default=0.5, help="Missing ratio for eval")
    parser.add_argument("--missing_type", type=str, default="random", choices=["random", "block"])
    parser.add_argument("--target_node", type=int, default=0, help="Target node for counterfactual")
    parser.add_argument("--simulate_duration", type=int, default=3, help="Simulation duration (hours)")
    parser.add_argument("--model", type=str, default="itransformer", help="Baseline model name")
    parser.add_argument("--remove_gat", action="store_true", help="Ablation: remove GAT")

    args = parser.parse_args()
    logging.info(f"Starting ST-CDF pipeline in mode: {args.mode}")

    if args.mode == "train_teacher":
        from train.train_teacher import train_teacher_pipeline
        train_teacher_pipeline(args)

    elif args.mode == "train_student_fed":
        from train.fed_cgd import federated_cgd_pipeline
        federated_cgd_pipeline(args)

    elif args.mode == "eval":
        from eval.eval_imputation import evaluate_imputation
        evaluate_imputation(args)

    elif args.mode == "eval_et0":
        from eval.eval_et0_downstream import evaluate_et0_downstream
        evaluate_et0_downstream(args)

    elif args.mode == "counterfactual":
        from eval.counterfactual_sim import run_counterfactual_simulation
        run_counterfactual_simulation(args)

    elif args.mode == "baselines":
        logging.info(f"Running baseline: {args.model}")
        if args.model == "itransformer":
            from baselines.run_itransformer import run_baseline
            run_baseline(args)
        elif args.model == "imputeformer":
            from baselines.run_imputeformer import run_baseline
            run_baseline(args)
        else:
            logging.error(f"Baseline '{args.model}' not available in this release.")

    elif args.mode == "ablation":
        from baselines.ablation_study import run_ablation
        run_ablation(args)

    else:
        logging.error("Invalid mode")


if __name__ == "__main__":
    main()
