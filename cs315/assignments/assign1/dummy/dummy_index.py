# You are allowed to import any modules whatsoever (not even numpy, sklearn etc)
# The use of file IO is forbidden. Your code should not read from or write onto files

# SUBMIT YOUR CODE AS TWO PYTHON (.PY) FILES INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILES MUST BE index.py and execute.py

# DO NOT CHANGE THE NAME OF THE METHODS my_index BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to create indices or statistics

################################
# Non Editable Region Starting #
################################
def my_index( tuples ):
################################
#  Non Editable Region Ending  #
################################
	ids = [ t[ 0 ] for t in tuples ]
	years = [ t[ 2 ] for t in tuples ]
	# My plan is to store tuples in increasing order of id and also in 
	# decreasing order of id since it may reduce seek time later on
	disk = sorted( ids ) + sorted( ids, reverse = True )
	# I am too lazy to create an actual index -- lets just list all disk indices
	my_index = list( range( len( disk ) ) )
	# I am too lazy to calculate any useful stats so lets just have some dummy stats
	my_stats = { "count": len( years ), "min": min( years ), "max": max( years ) }
	
	# Combine index and stats in a single list object
	idx_stat = [ my_index, my_stats ]
	
	return disk, idx_stat