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

// AnimationNode.h: interface for the AnimationNode class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_ANIMATIONNODE_H__35B256EB_389A_4D4F_9DD0_85E13524205B__INCLUDED_)
#define AFX_ANIMATIONNODE_H__35B256EB_389A_4D4F_9DD0_85E13524205B__INCLUDED_

#include <AnimationMaker/ViewState.h>

///\class AnimationNode AnimationNode.h
///\author Anthony Thane
///\brief An AnimationNode is a single key-frame of an Animation. It holds a ViewState object and a
/// time.
class AnimationNode  
{
public:
///\fn AnimationNode(const ViewState& state, unsigned int time, AnimationNode* prev, AnimationNode* next)
///\brief The constructor
///\param state A ViewState object
///\param time A time (in milliseconds)
///\param prev The AnimationNode preceeding this node
///\param next The AnimationNode following this node
	AnimationNode(const ViewState& state, unsigned int time, AnimationNode* prev, AnimationNode* next);
	virtual ~AnimationNode();


	ViewState m_ViewState;
    unsigned int m_Time;

    AnimationNode* m_Next;
    AnimationNode* m_Prev;
};

#endif // !defined(AFX_ANIMATIONNODE_H__35B256EB_389A_4D4F_9DD0_85E13524205B__INCLUDED_)
