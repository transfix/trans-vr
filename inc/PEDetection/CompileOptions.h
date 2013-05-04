/*
  Copyright 2006 The University of Texas at Austin

        Authors: Sangmin Park <smpark@ices.utexas.edu>
        Advisor: Chandrajit Bajaj <bajaj@cs.utexas.edu>

  This file is part of PEDetection.

  PEDetection is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License version 2.1 as published by the Free Software Foundation.

  PEDetection is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#ifndef FILE_COMPILEOPTIONS_H
#define FILE_COMPILEOPTIONS_H

#define		FALSE	0
#define		TRUE	1

// Compile Options
// The Options for Saving Results
//#define SAVE_ORIGINAL_IMAGE		// for OCGS
//#define SAVE_CLASSIFICATION		// for OCGS and ClassOnly
//#define SAVE_Histogram_File		// for OCGS and ClassOnly
//#define SAVE_MEMBERSHIP_VALUES	// for OCGS and ClassOnly
//#define BILATERAL_FILTER_FOR_DATA
#define ANISOTROPIC_DIFFUSION_FOR_DATA		// for OCGS, PED, ClassOnly, Evaluation

#define COMPUTE_GRADIENT_VEC_MAG			// for OCGS, PED, ClassOnly, Evaluation
	//#define	MLCA_SMOOTHING					// for Marching Cubes
	// Smooth fileters for gradient 
	//#define BILATERAL_FILTER_FOR_GRADIENT
	// It can be used instead of BILATERAL_FILTER_FOR_GRADIENT
	#define	COMPUTE_GRADIENT_MAG			// for OCGS, PED, ClassOnly, Evaluation
	#define	GRADIENT_VECTOR_DIFFUSION_3D	// for OCGS, PED, ClassOnly, Evaluation
	//#define SAVE_GRADIENT_IMAGE		// for OCGS

#define	COMPUTE_SECOND_DERIVATIVE			// for OCGS, PED, ClassOnly, Evaluation
	//#define BILATERAL_FILTER_FOR_SECOND_DERIVATIVE
	#define ANISOTROPIC_DIFFUSION_FOR_SECONDD	// for OCGS, PED, ClassOnly, Evaluation
	//#define SAVE_SECOND_DERIVATIVE		// for OCGS

// GVF_DIFFUSION_3D diffuse the vectors of gradient magnitude
// It is in GVF.cpp
#define	GVF_DIFFUSION_3D 
//#define RETURN_GVF_IMAGE

// Initialization Options
#define		AUTOMATIC_INITIALIZATION	// for OCGS, PED, ClassOnly, Evaluation

// Functions
//#define		OCTREE
//#define		MARCHING_CUBES_INTENSITY	// NMJ
//#define		MEMBRANE_SEGMENTATION
//#define		CLASSIFICATION_EVALUATION	// for Evaluation


#define 		PE_DETECTION
	#define			VESSELSEG	// PED
		// Saving the computation results and re-use them at the next execution

		#define		SAVE_VOLUME_2ndD_NoFragments	// VesselSeg.cpp
		#define		SAVE_VOLUME_Distance			// VesselSeg.cpp
		#define		SAVE_VOLUME_Stuffed_Lungs		// VesselSeg.cpp	(It takes around 6 minutes)
		//#define		SAVE_VOLUME_Data_Diffusion		// Control.cpp
		//#define		SAVE_VOLUME_Gvec_Diffusion		// Control.cpp		(It takes around 8 minutes)
		#define		SAVE_VOLUME_SecondD				// TFGeneration.cpp

		
	//#define			SKELETON // Penalized-Distance 
	//#define			SKELETON_SEEDPTS // Using Seed Pts
	//#define 			SAVE_SEED_PTS_LINES // Save seed pts of line structures
	//#define			THINNING // 
	//#define			MARCHING_CUBES_SECONDD
	
	//#define			TF_GENERATION	// Distance-based TF Generation, TFGeneration.cpp
		//#define					SAVE_DISTANCE_GMHIT_RAWV
		//#define					SAVE_DISTANCE_GMHIT_RAWIV
		//#define				TF_GENERATION_COMPUTE_DOT_PRODUCT
	
	//#define			TWO_DIM_R3_IMAGE
	
// Control.cpp


//#define			ZERO_CELLS_GRADMAG


//#define	DEBUG
//#define	DEBUG_GVF
//#define 	DEBUG_GVF_AddOutsideBoundary
//#define 	DEBUG_GVF_AddInsideAndOnBoundary
//#define	DEBUG_GVF_LEVEL2
//#define	DEBUG_WATERFLOW
//#define	DEBUG_INITIALIZATION
//#define 	DEBUG_EVALUATION
//#define 	DEBUG_CONTROL
//#define 	DEBUG_TF
//#define 	DEBUG_TF_ZC
//#define	DEBUG_ZEROLOC_SEARCHAGAIN
//#define	DEBUG_TF_SIGMA
//#define 	DEBUG_TF_GRADVEC_INTERPOLATION
//#define 	DEBUG_TF_GRAD_INTERPOLATION
//#define 	DEBUG_TF_CCVOLUME
//#define 	DEBUG_PED_NEAREST_BOUNDARY
//#define 	DEBUG_PED_VT // Automatic Vessel Tracking
//#define 	DEBUG_PED_BEXT // Boundary extraction
//#define		DEBUG_MC_GEOM
//#define		DEBUG_CC_ZERO_CELLS

//#define	DEBUG_THINNING
//#define		DEBUG_MC_NORMAL

//#define INTENSITY_GRAPH
// Control.cpp

//#define GRADIENT_GRAPH
// Control.cpp

//#define SAVE_RANGES_HISTOGRAM
//#define SAVE_MATERIAL_VOLUME_RAWIV
//#define SAVE_MATERIAL_VOLUME_RAWV
//#define SAVE_ORIGINAL_SLICES_AND_INITIAL_VALUES
//#define SAVE_ZERO_CROSSING_SECOND_DERIVATIVE // Save Only Zero 2nd Derivative Voxels
//#define SAVE_ZERO_CROSSING_VOLUME
//#define SAVE_THREE_COLOR_VOLUME // for thinning
//#define SAVE_SEED_PTS // Save seed pts with original images




/*

HISTOGRAM_BASED_EM
 Format of the Probability 
 Data #    Mat1  Mat2  Mat3
   0	   0.1   0.9   0.0
   1	   0.0   0.1   0.9
  ...	   ...
  
NON-HISTOGRAM_BASED_EM
 Format of the Probability 
 Mat1	  x*y*z
 Mat2	  x*y*z
 Mat3	  x*y*z
  ...	   ...

*/

#endif

