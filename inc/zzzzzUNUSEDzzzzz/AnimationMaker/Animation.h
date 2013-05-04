/******************************************************************************
				Copyright   

This code is developed within the Computational Visualization Center at The 
University of Texas at Austin.

This code has been made available to you under the auspices of a Lesser General 
Public License (LGPL) (http://www.ices.utexas.edu/cvc/software/license.html) 
and terms that you have agreed to.

Upon accepting the LGPL, we request you agree to acknowledge the use of use of 
the code that results in any published work, including scientific papers, 
films, and videotapes by citing the following references:

C. Bajaj, Z. Yu, M. Auer
Volumetric Feature Extraction and Visualization of Tomographic Molecular Imaging
Journal of Structural Biology, Volume 144, Issues 1-2, October 2003, Pages 
132-143.

If you desire to use this code for a profit venture, or if you do not wish to 
accept LGPL, but desire usage of this code, please contact Chandrajit Bajaj 
(bajaj@ices.utexas.edu) at the Computational Visualization Center at The 
University of Texas at Austin for a different license.
******************************************************************************/

// Animation.h: interface for the Animation class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_ANIMATION_H__450D6D82_80AE_4641_A1D7_BC41258ECDA3__INCLUDED_)
#define AFX_ANIMATION_H__450D6D82_80AE_4641_A1D7_BC41258ECDA3__INCLUDED_

#include <AnimationMaker/AnimationNode.h>

///\class Animation Animation.h
///\author Anthony Thane
///\brief The Animation class encapsulates a sequence of ViewState objects. It can be read from or
/// written to a file. You can get a ViewState for any time value (in milliseconds) between 0 and
/// whatever getEndTime() returns.
class Animation  
{
public:
///\fn Animation(const ViewState& viewState)
///\brief The constructor
///\param viewState The initial ViewState for the animation
	Animation(const ViewState& viewState);
	virtual ~Animation();

///\fn void addKeyFrame(const ViewState& viewState, unsigned int time)
///\brief Adds a key-frame to the animation
///\param viewState The ViewState for this frame
///\param time The time that this frame occurs
	void addKeyFrame(const ViewState& viewState, unsigned int time);
///\fn void getFrame(ViewState& frame, unsigned int time)
///\brief Fills in a ViewState object for some arbitrary time using linear interpolation to interpolate
/// between the key-frames.
///\param frame The ViewState
///\param time The time
	void getFrame(ViewState& frame, unsigned int time);
///\fn void getCubicFrame(ViewState& frame, unsigned int time)
///\brief Fills in a ViewState object for some arbitrary time using cubic interpolation to interpolate
/// between the key-frames.
///\param frame The ViewState
///\param time The time
	void getCubicFrame(ViewState& frame, unsigned int time);
///\fn unsigned int getEndTime() const
///\brief Returns the time stamp of the last key-frame
///\return A time in milliseconds
	unsigned int getEndTime() const;

///\fn void readAnimation(FILE* fp)
///\brief Reads an Animation from a file that has been opened for reading.
///\param fp A pointer to a FILE that is open for reading
	void readAnimation(FILE* fp); 
///\fn void writeAnimation(FILE* fp)
///\brief Writes an Animation to a file that has been opened for writing
///\param fp A pointer to a FILE that is open for writing
	void writeAnimation(FILE* fp); 

    AnimationNode* m_Head;
    AnimationNode* m_Tail;
    unsigned int m_Size;
};

#endif // !defined(AFX_ANIMATION_H__450D6D82_80AE_4641_A1D7_BC41258ECDA3__INCLUDED_)
