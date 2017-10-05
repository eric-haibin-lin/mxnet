#################################
# christos faloutsos, Sept. 2017
#################################

##################################
# GOAL: generates (i,j) entries for a dense matrix
#       so that it has the specified blocks,
#       with the specified densities
#       Made for Haibin, to test whether DBNs can spot dense blocks
# INPUT: block_dims vector ( nrows, ncols, density) for every block
# OUTPUT: dense matrix with '1.0' at specified positions
##################################

import random as R
import numpy as np
import sys
verbose = 1
dry_run = False

def gen(block_dims):
  # block_dims = blocks
  
  if len(block_dims) == 0:
    print "WARNING: empty set of blocks given"
    return
  sba = sum(np.asarray(block_dims))
  nrows = sba[0]
  ncols = sba[1]
  # create an empty, dense, array of appropriate dimensions
  out_matrix = np.zeros( ( int(nrows), int(ncols) ) )
  if verbose > 2:
    print out_matrix

  nr_start = 0 # starting row
  nc_start = 0 # starting column
  
  for block in block_dims:
    if(verbose > 0):
      print "#    ----- block: ", block
    nr_current = block[0]
    nc_current = block[1]
    density = block[2]
  
    nr_end = nr_start + nr_current
    nc_end = nc_start + nc_current
    for i in range(nr_start,nr_end):
      for j in range(nc_start, nc_end):
  
        r = R.random()
        if (r < density):
          if dry_run:
            pass
          else:
            # print i, j
            out_matrix[i][j] = 1.0
    # update limits, for next iteration
    nr_start = nr_end
    nc_start = nc_end
  return out_matrix

if __name__ == '__main__':
  block_dims = [ [30, 40, 0.8], [20, 25, 0.85], [10, 15, 0.9], [250, 350, 0.06]]
  # block_dims = [ [5,8, 0.9], [3, 7, 0.5] ]
  dims = []
  for dim in block_dims:
      for d in dim:
          dims.append(d)
  post_fix = '_'.join(map(str, dims))
  print(post_fix)
  result_matrix = gen(block_dims)

  if verbose > 2:
    print "working on block_dims = ", block_dims
    print result_matrix

  np.save("matrix_" + post_fix + ".npy", result_matrix)
