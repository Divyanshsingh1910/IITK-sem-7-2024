# You are allowed to import any modules whatsoever (not even numpy, sklearn etc)
# The use of file IO is forbidden. Your code should not read from or write onto files

# SUBMIT YOUR CODE AS TWO PYTHON (.PY) FILES INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILES MUST BE index.py and execute.py

# DO NOT CHANGE THE NAME OF THE METHODS my_execute BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to create indices or statistics

################################
# Non Editable Region Starting #
################################
def my_execute( clause, idx ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to take a WHERE clause specification
	# and return results of the resulting query
	# clause is a list containing either one or two predicates
	# Each predicate is itself a list of 3 objects, column name, comparator and value
	# idx contains the packaged variable returned by the my_index method
	
	# THE METHOD MUST RETURN A SINGLE LIST OF INDICES INTO THE DISK MAP
	return diskloc_list