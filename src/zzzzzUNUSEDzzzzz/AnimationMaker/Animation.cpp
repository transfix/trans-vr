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

// Animation.cpp: implementation of the Animation class.
//
//////////////////////////////////////////////////////////////////////

#include <AnimationMaker/Animation.h>
#include <AnimationMaker/ViewState.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

Animation::Animation(const ViewState& viewState)
{
	m_Size = 1;
	m_Head = new AnimationNode(viewState, 0, 0, 0);
	m_Tail = m_Head;
}

Animation::~Animation()
{
	delete m_Head;
}

void Animation::addKeyFrame(const ViewState& viewState, unsigned int time)
{
	AnimationNode* current = m_Head;

	while (current!=NULL && time>=current->m_Time) {
		current = current->m_Next;
	}
	AnimationNode* newNode;
	//if (current && time==current->m_Time) {
	//	current->m_ViewState = viewState;
	//}
	//else {
		if (current) {
			newNode = new AnimationNode(viewState, time, current->m_Prev, current);
		}
		else {
			newNode = new AnimationNode(viewState, time, m_Tail, 0);
			m_Tail = newNode;
		}
		m_Size++;
	//}
}

void Animation::getFrame(ViewState& frame, unsigned int time)
{
	AnimationNode* current = m_Head;


	while (current!=NULL && time>current->m_Time) {
		current = current->m_Next;
	}
	double alpha;
	if (current!=NULL) {
		if (time==current->m_Time) {
			frame = current->m_ViewState;
		}
		else {
			//if (current->m_Prev->m_Time == 5157) 
			//	alpha = 1.0;
			alpha = (double)(time-current->m_Prev->m_Time)/
				(double)(current->m_Time-current->m_Prev->m_Time); 
			frame = interpolate(current->m_Prev->m_ViewState, current->m_ViewState, alpha);
		}
	}
	else {
		frame = m_Tail->m_ViewState;
	}
}

void Animation::getCubicFrame(ViewState& frame, unsigned int time)
{
	AnimationNode* current = m_Head;


	while (current!=NULL && time>current->m_Time) {
		current = current->m_Next;
	}
	double alpha;
	if (current!=NULL) {
		if (time==current->m_Time) {
			frame = current->m_ViewState;
		}
		else {
			//if (current->m_Prev->m_Time == 5157) 
			//	alpha = 1.0;
			alpha = (double)(time-current->m_Prev->m_Time)/
				(double)(current->m_Time-current->m_Prev->m_Time); 
			if (current->m_Prev->m_Prev && current->m_Next) 
				frame = interpolate(current->m_Prev->m_Prev->m_ViewState, current->m_Prev->m_ViewState, current->m_ViewState, current->m_Next->m_ViewState, alpha);
			else if (current->m_Prev->m_Prev) 
				frame = interpolate(current->m_Prev->m_Prev->m_ViewState, current->m_Prev->m_ViewState, current->m_ViewState, current->m_ViewState, alpha);
			else if (current->m_Next) 
				frame = interpolate(current->m_Prev->m_ViewState, current->m_Prev->m_ViewState, current->m_ViewState, current->m_Next->m_ViewState, alpha);
			else
				frame = interpolate(current->m_Prev->m_ViewState, current->m_ViewState, alpha);

		}
	}
	else {
		frame = m_Tail->m_ViewState;
	}
}

unsigned int Animation::getEndTime() const
{
	return m_Tail->m_Time;
}

void Animation::readAnimation(FILE* fp)
{
	delete m_Head;
	m_Head = NULL;
	m_Tail = NULL;
	unsigned int length;
	// read in the number of key frames
	fscanf(fp, "Number of Frames: %u\n", &length);


	unsigned int time;
	ViewState state;

	// read in the first frame
	fscanf(fp, "Time: %u\n", &time);
	state.readFromFile(fp);
	m_Head = new AnimationNode(state, time, 0, 0);
	m_Tail = m_Head;
	m_Size = 1;
	unsigned int count;
	for (count = 1; count < length; count++) {
		// read in the rest of the frames
		fscanf(fp, "Time: %u\n", &time);
		state.readFromFile(fp);
		addKeyFrame(state, time);
	}
}

void Animation::writeAnimation(FILE* fp)
{
	// write out the number of frames
	fprintf(fp, "Number of Frames: %u\n", m_Size);

	AnimationNode* current = m_Head;
	while (current) {
		// write the time
		fprintf(fp, "Time: %u\n", current->m_Time);
		// write the frame
		current->m_ViewState.writeToFile(fp);
		// next!
		current = current->m_Next;
	}
}

