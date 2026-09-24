from sklearn.pipeline import Pipeline
from pickle import load

if __name__ == '__main__':
    print(""" Model options:
1. Rulebased Model
2. Rulebased Model (trained on grouped data)
3. Decision Tree model
4. Decision Tree model (trained on grouped data)
5. SVM model
6. SVM model (trained on grouped data)
""")
    choice = None
    model = None

    while not choice:
        choice = input("Choose model: ")
        if choice == "1":
            with open("./models/model_rule_based_classifier_regular_data.pkl", "rb") as f:
                model: Pipeline = load(f)
        elif choice == "2":
            with open("./models/model_rule_based_classifier_grouped_data.pkl", "rb") as f:
                model: Pipeline = load(f)
        elif choice == "3":
            with open("./models/model_decision_tree_regular_data.pkl", "rb") as f:
                model: Pipeline = load(f)
        elif choice == "4":
            with open("./models/model_decision_tree_grouped_data.pkl", "rb") as f:
                model: Pipeline = load(f)
        elif choice == "5":
            with open("./models/model_support_vector_machine_regular_data.pkl", "rb") as f:
                model: Pipeline = load(f)
        elif choice == "6":
            with open("./models/model_support_vector_machine_grouped_data.pkl", "rb") as f:
                model: Pipeline = load(f)
        else:
            print("Invalid choice, choose again...")
            print(""" Model options:
            1. Rulebased Model
            2. Decision Tree model
            3. SVM model
            """)
            choice = None
    while True:
        utterence = input("Enter utterence to classify: ").lower()
        print(f"Label: {model[-1].label_encoder_.inverse_transform(model.predict([utterence]))[0]}")

