from DecisionTree import DecisionTree

dt = DecisionTree(
  training_datafile = "data/dataset.csv",
  csv_class_column_index = 2,
  csv_columns_for_features = [1,2,3,4,5],
  entropy_threshold = 0.01,
  max_depth_desired = 8,
  symbolic_to_numeric_cardinality_threshold = 10,
)

dt.get_training_data()
dt.calculate_first_order_probabilities()
dt.calculate_class_priors()
dt.show_training_data()
root_node = dt.construct_decision_tree_classifier()
root_node.display_decision_tree("   ")

test_sample  = ['ColorDeCabello = Negro',
                'Altura = Alto',
                'Peso = Alto',
                'Proteccion = No',
                'Quemadura = Si']
classification = dt.classify(root_node, test_sample)
print "Classification: ", classification