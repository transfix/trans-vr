README file describing the single seed point outer capsid segmentation algorithm found in file capsid_segment_simple.cpp

In determining whether we should keep a voxel in our data set or remove it, for all voxels that we wish to keep we set their density values to be negative.  For all voxels we wish to remove we leave their density values as positive.  The last step in our algorithm is to then set the remaining positive values to be zero, thereby removing them.  And we set the negative values to be positive.

We take as input from the user a threshold, tlow, which gives a minimum density value.  All voxels below this threshold are discarded.  We also take an initial seed voxel that must be on the outer shell capsid.

Our algorithm creates a stack of voxels with our initial seed voxel pushed on first.  We then repeatedly call SetSeedIndexSimple() on the voxel currently on top the stack, pop the top off the stack, and then push onto the stack all voxels near by that are above tlow.  We continue untill the stack is empty.

SetSeedIndexSimple() does most of the work in our algorithm.  It takes the voxel we’re currently evaluating and if it’s above tlow we set its density to be negative.  We then evaluate several other voxels gotten by rotating our voxel about each axis of 5-fold symmetry.  In detail, suppose we have a voxe V1 that we have just rotated about axis A1 and we now wish to rotate about another axis A2.  We first find the cross product of A1XA2 and the angle between A1 and A2.  We rotate the voxel V1 about A1XA2 by the angle and we get a corresponding voxel V2 in relation to A2.  We can then rotate V2 about A2.  We then test if the resulting voxels are above tlow, if they are then we set their densities to be negative.

When we eventually empty the stack we then search through the entire data set and if we find a voxel whose density is above tlow and there is at least one voxel near by whose density is negative, then we set the vexel’s density to be negative.  Once we finish this we then go through the entire data set again and set all remaining positive voxels to be zero and all negative voxels to be positive.
