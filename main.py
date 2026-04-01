from titanic.main import main as run_titanic_main
from house_prices.main import main as run_house_prices_main


def main() -> None:
    """Запускает оба проекта и печатает краткую сводку."""
    print("=" * 100)
    print("Running Titanic pipeline...")
    print("=" * 100)
    titanic_result = run_titanic_main()

    print()
    print("=" * 100)
    print("Running House Prices pipeline...")
    print("=" * 100)
    house_result = run_house_prices_main()

    print()
    print("=" * 100)
    print("Summary")
    print("=" * 100)

    print(f"Project: {titanic_result['project']}")
    print(f"Model:   {titanic_result['model']}")
    print(f"Mean CV Accuracy: {titanic_result['mean_cv_accuracy']:.4f}")
    print(f"Mean CV F1:       {titanic_result['mean_cv_f1']:.4f}")
    print(f"Mean CV ROC AUC:  {titanic_result['mean_cv_auc']:.4f}")
    print(f"Submission:       {titanic_result['submission_path']}")
    print(f"Metrics:          {titanic_result['metrics_path']}")
    print("-" * 100)

    print(f"Project: {house_result['project']}")
    print(f"Model:   {house_result['model']}")
    print(f"Mean CV RMSE: {house_result['mean_cv_rmse']:.4f}")
    print(f"OOF RMSE:     {house_result['oof_rmse']:.4f}")
    print(f"Submission:   {house_result['submission_path']}")
    print(f"Metrics:      {house_result['metrics_path']}")
    print("=" * 100)


if __name__ == "__main__":
    main()
