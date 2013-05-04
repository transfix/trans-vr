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

#include <PEDetection/Stack.h>

template <class _DataType>
cStack<_DataType>::cStack()
{
	MaxSize_mi=10;
	Buffer_m = new _DataType [MaxSize_mi]; 
	CurrPt_mi = 0;
}


template <class _DataType>
cStack<_DataType>::~cStack()
{ 
	delete [] Buffer_m; 
}

template <class _DataType>
void cStack<_DataType>::Push(_DataType Value)
{ 
	fflush(stdout);
	Buffer_m[CurrPt_mi++] = Value; 
	if (CurrPt_mi>=MaxSize_mi) {
		_DataType	*BackupBuffer = new _DataType [MaxSize_mi*2];
		for (int i=0; i<CurrPt_mi; i++) BackupBuffer[i] = Buffer_m[i];
		delete [] Buffer_m;
		Buffer_m = BackupBuffer;
		MaxSize_mi *= 2;
	}
}

template <class _DataType>
int cStack<_DataType>::Pop()
{
	if (CurrPt_mi==0) return false; // The buffer is empty
	CurrPt_mi--;
	return true;
}

template <class _DataType>
int cStack<_DataType>::Pop(_DataType& Value_Ret)
{
	if (CurrPt_mi==0) return false; // The buffer is empty
	CurrPt_mi--;
	Value_Ret = Buffer_m[CurrPt_mi];
	return true;
}


template <class _DataType>
int cStack<_DataType>::getTopElement(_DataType& Value_Ret)
{
	if (CurrPt_mi==0) return false; // The buffer is empty
	Value_Ret = Buffer_m[CurrPt_mi];
	return true;
}

template <class _DataType>
int cStack<_DataType>::IthValue(int ith, _DataType& Value_Ret)
{
	if (ith >= CurrPt_mi || ith < 0) return false; // Out of range
	Value_Ret = Buffer_m[ith];
	return true;
}

template <class _DataType>
int cStack<_DataType>::setIthValue(int ith, _DataType NewValue)
{
	if (ith >= CurrPt_mi || ith < 0) return false; // Out of range
	Buffer_m[ith] = NewValue;
	return true;
}

template <class _DataType>
int cStack<_DataType>::IsEmpty()
{
	if (CurrPt_mi==0) return true;
	else return false;
}


template <class _DataType>
int cStack<_DataType>::DoesExist(_DataType AnElement)
{
	for (int i=0; i<CurrPt_mi; i++) {
		if (AnElement==this->Buffer_m[i]) return true;
	}
	return false;
}


template <class _DataType>
int cStack<_DataType>::Size()
{
	return CurrPt_mi;
}

template <class _DataType>
void cStack<_DataType>::Clear()
{
	CurrPt_mi = 0;
	delete [] Buffer_m;
	
	MaxSize_mi = 10; 
	Buffer_m = new _DataType [MaxSize_mi]; 	
}


template <class _DataType>
void cStack<_DataType>::setDataPointer(int ith)
{
	CurrPt_mi = ith;
}

template <class _DataType>
void cStack<_DataType>::RemoveFirstNElements(int NumElements)
{
	if (NumElements >= CurrPt_mi) CurrPt_mi = 0;
	else CurrPt_mi -= NumElements;
}

template <class _DataType>
void cStack<_DataType>::RemoveIthElement(int ith)
{
	if (ith<0) return;
	for (int i=ith; i<CurrPt_mi-1; i++) {
		Buffer_m[i] = Buffer_m[i+1];
	}
	CurrPt_mi--;
}

template <class _DataType>
void cStack<_DataType>::RemoveTheElement(_DataType AData)
{
	int		i, Ith = -1;

	for (i=0; i<CurrPt_mi; i++) {
		if (AData==Buffer_m[i]) Ith = i;
	}
	if (Ith<0) return;
	for (int i=Ith; i<CurrPt_mi-1; i++) {
		Buffer_m[i] = Buffer_m[i+1];
	}
	CurrPt_mi--;
}


template <class _DataType>
void cStack<_DataType>::Copy(cStack &Stack2)
{
	_DataType	Data;

	for (int k=0; k<Stack2.Size(); k++) {
		Stack2.IthValue(k, Data);
		this->Push(Data);
	}
}

template <class _DataType>
void cStack<_DataType>::Copy(cStack *Stack2)
{
	_DataType	Data;

	for (int k=0; k<Stack2->Size(); k++) {
		Stack2->IthValue(k, Data);
		this->Push(Data);
	}
}

template <class _DataType>
void cStack<_DataType>::Merge(cStack *Stack2)
{
	_DataType	Data;

	while(Stack2->Size()>0) {
		Stack2->Pop(Data);
		this->Push(Data);
	}
}


template <class _DataType>
void cStack<_DataType>::Display()
{
	int		i;


	printf ("cStack: Size/Capacity = %d/%d ", CurrPt_mi, MaxSize_mi);
	for (i=0; i<CurrPt_mi; i++) {
		printf ("%.2f ", (float)Buffer_m[i]);
	}
	printf ("\n"); fflush (stdout);

}

template <class _DataType>
void cStack<_DataType>::Display(int DisplayNum1, int DisplayNum2)
{
	int		i, ActualDisplayNum1, ActualDisplayNum2;


	printf ("Size/Capacity = %d/%d ", CurrPt_mi, MaxSize_mi);
	if (DisplayNum1>DisplayNum2) {
		ActualDisplayNum1 = 0;
		ActualDisplayNum2 = (CurrPt_mi>10) ? 10:CurrPt_mi;
	}
	else {
		ActualDisplayNum1 = (DisplayNum1>CurrPt_mi) ? 0:DisplayNum1;
		ActualDisplayNum2 = (DisplayNum2>CurrPt_mi) ? CurrPt_mi:DisplayNum2;
	}
	printf ("Range = (%d,%d) ", ActualDisplayNum1, ActualDisplayNum2);
	
	for (i=ActualDisplayNum1; i<ActualDisplayNum2; i++) {
		printf ("%.2f ", (float)Buffer_m[i]);
	}
	printf ("\n"); fflush (stdout);

}

template <class _DataType>
void cStack<_DataType>::Destroy()
{
	delete [] Buffer_m;
	Buffer_m = NULL;
}


template class cStack<unsigned char>;
template class cStack<unsigned short>;
template class cStack<int>;
template class cStack<float>;
