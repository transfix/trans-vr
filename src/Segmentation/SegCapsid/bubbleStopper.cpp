#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <memory.h>


#define max(x, y) ((x>y) ? (x):(y))
#define min(x, y) ((x<y) ? (x):(y))
#define IndexVect(i,j,k) ((k)*XDIM*YDIM + (j)*XDIM + (i))
#define MAX_TIME 9999999.0f

namespace SegCapsid {

void bubbleStopper(int XDIM,int YDIM,int ZDIM,unsigned char *segmentIndex, unsigned int MaxBubbleSize){
	int i, j, k, n, m, p, b;
	int tempBorder_i, tempBorder_j, tempBorder_k;
	int thisIndex;
	unsigned char borderIndex;
	int borderSize;
	int bubbleSize;
	int **borderIndices;
	int *bubbleIndices;
	bool *visited;


	borderIndices = (int **)malloc(sizeof(int*)*3);
	borderIndices[0] = (int *)malloc(sizeof(int)*XDIM);
	borderIndices[1] = (int *)malloc(sizeof(int)*YDIM);
	borderIndices[2] = (int *)malloc(sizeof(int)*ZDIM);
	//we add 2 to bubble for safety.
	bubbleIndices = (int *)malloc(sizeof(int)*(MaxBubbleSize+2));
	visited = (bool *)malloc(sizeof(bool)*XDIM*YDIM*ZDIM);
	for (k=0; k<ZDIM; k++){
		for (j=0; j<YDIM; j++){
			for (i=0; i<XDIM; i++){
				visited[IndexVect(i,j,k)] = false;
			}
		}
	}

	for (k=0; k<ZDIM; k++){
		for (j=0; j<YDIM; j++){
			for (i=0; i<XDIM; i++){
				if(visited[IndexVect(i,j,k)] == false){
					visited[IndexVect(i,j,k)] = true;
					thisIndex = segmentIndex[IndexVect(i,j,k)];
					borderIndex = segmentIndex[IndexVect(i,j,k)];
					bubbleSize = 1;
					borderSize = 1;
					bubbleIndices[0] = IndexVect(i,j,k);
					borderIndices[0][0] = i;
					borderIndices[0][1] = j;
					borderIndices[0][2] = k;
					while(borderSize > 0){
						borderSize--;
						tempBorder_i = borderIndices[borderSize][0];
						tempBorder_j = borderIndices[borderSize][1];
						tempBorder_k = borderIndices[borderSize][2];
						for(n=max(0,tempBorder_i-1); n<min(XDIM,tempBorder_i+1); n++){
							for(m=max(0,tempBorder_j-1); m<min(YDIM,tempBorder_j+1); m++){
								for(p=max(0,tempBorder_k-1); p<min(ZDIM,tempBorder_k+1); p++){
									if(n!=tempBorder_i || m!=tempBorder_j || p!=tempBorder_k){
										if(segmentIndex[IndexVect(n,m,p)] == thisIndex){
											if( !visited[IndexVect(n,m,p)] ){
												visited[IndexVect(n,m,p)] = true;
												bubbleIndices[bubbleSize] = IndexVect(n,m,p);
												borderIndices[borderSize][0] = n;
												borderIndices[borderSize][1] = m;
												borderIndices[borderSize][2] = p;
												bubbleSize++;
												borderSize++;
												if(bubbleSize > MaxBubbleSize){
													borderIndex = thisIndex;
													//stop growing the bubble
													p = ZDIM;
													m = YDIM;
													n = XDIM;
													borderSize = 0;
												}
											}
										}else{//segmentIndex[IndexVect(n,m,p)] != thisIndex)
											if(borderIndex == thisIndex){
												borderIndex = segmentIndex[IndexVect(n,m,p)];
											}else if(borderIndex != segmentIndex[IndexVect(n,m,p)]){
												borderIndex = thisIndex;
												//stop growing the bubble
												p = ZDIM;
												m = YDIM;
												n = XDIM;
												borderSize = 0;
											}
										}
									}
								}
							}
						}
					}
					/********************************************************************************
					At this point, we're done growing the bubble.  If borderIndex != thisIndex then
					we have a bubble and we set the segmentIndex of every voxel in the bubble 
					to match the index of the surrounding voxels.  If borderIndex = thisIndex then we 
					do not have a bubble and we do nothing.*/
					if(borderIndex != thisIndex){
						for(b=0; b<bubbleSize; b++){
							segmentIndex[bubbleIndices[b]] = borderIndex;
						}
					}
				}
			}
		}
	}
	
	free(borderIndices[0]);
	free(borderIndices[1]);
	free(borderIndices[2]);
	free(borderIndices);
	free(bubbleIndices);
	free(visited);
}

}
