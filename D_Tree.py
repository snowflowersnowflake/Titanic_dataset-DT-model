from collections import Counter
from Node import Node

""" stop_condition takes in list of labels and returns True if all labels are the same, False otherwise """

def stop_condition(labels):
	x = Counter(labels)
	if len(x) > 1: 
		return False
	else:
		return True

""" best_split takes as input 'records' : 2D list of records
    returns attribute with lowest gini split as a numeric value in the range of 0 to # of attributes - 1 """

def best_split(records):
	# n : number of attributes
	# gsplits: list of calculated gini splits for all attributes
	n = len(records[0]) - 1
	d = len(records)
	label_vals = [records[y][-1] for y in range(len(records))] # get label values 
	gsplits = []
	
	# iterate over all attributes to calculate their gsplits
	for i in range(n):
		att_vals = [records[j][i] for j in range(len(records))] # get attribute values of ith attribute
		C = Counter(att_vals)  # get counts of all unique attribute values
		gini_vals = [] # list to store gini index values of all values of a particular attribute
		nr_subsets = [] # list to store number of records in each subset of attribute values
		
		# partition the data by each unique attribute value while getting labels into a list of 2D lists where elements are [att_value, label_value]
	# calculate gini values of each attribute value
		for attribute in C.keys():
			att_subset = [ [att_vals[u], label_vals[u]] for u in range(len(att_vals)) if att_vals[u] == attribute ]
			nr = len(att_subset)
			nr_subsets.append(nr)
			labels_of_subset = [ att_subset[u][1] for u in range(len(att_subset)) ]
			local_c = Counter(labels_of_subset)
			gini = 1 - sum((v/nr)**2 for v in local_c.values())  # p(label) = v/nr
			gini_vals.append(gini)
		
		gs = sum((nr_subsets[x]/d) * gini_vals[x] for x in range(len(nr_subsets)))
		gsplits.append(gs)
	
	return gsplits.index(min(gsplits))


""" Function Build_Tree takes in list of records(type 2D list) and list of 
    attributes(type 1D list) and returns a root of decision tree (type Node) """

def build_tree(records, attributes):
	root = Node()
	labels = [r[-1] for r in records]
	
	if stop_condition(labels):
		root.set_data("Label " + str(labels[0]))
		return root
	
	if attributes.count(0) == 0: # majority voting
		c = Counter(labels)
		majority = c.most_common(1)[0][0] # since most_common returns a list of (element,count) tuples
		root.set_data("Label " + str(majority))
		return root
	
	split_attribute = best_split(records)
	root.set_data("Attribute " + str(split_attribute))
	attributes[split_attribute] = 1
	split_attribute_vals = [records[i][split_attribute] for i in range(len(records))]
	val_counts = Counter(split_attribute_vals)
	for val in val_counts.keys():
		partition = [records[i] for i in range(len(records)) if records[i][split_attribute] == val]
		if len(partition) == 0:
			c = Counter(labels)
			majority = c.most_common(1)[0][0]
			leaf = Node("Label " + str(majority))
			root.add_child(leaf)
		else:
			child = build_tree(partition, attributes)
			root.add_child(child)
	
	return root
	





