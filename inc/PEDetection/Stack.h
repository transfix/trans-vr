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
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
  USA
*/

#ifndef FILE_STACK_H
#define FILE_STACK_H

#include <stdio.h>

template <class _DataType> class cStack {

protected:
  int MaxSize_mi, CurrPt_mi;
  _DataType *Buffer_m;

public:
  cStack();
  ~cStack();

  void Push(_DataType Value);
  int Pop();
  int Pop(_DataType &Value_Ret);
  int getTopElement(_DataType &Value_Ret);

  int IthValue(int ith, _DataType &Value_Ret);
  int DoesExist(_DataType AnElement);

  int IsEmpty();
  int Size();
  void Clear();

  void setDataPointer(int ith);
  int setIthValue(int ith, _DataType NewValue);
  void RemoveFirstNElements(int NumElements);
  void RemoveIthElement(int ith);
  void RemoveTheElement(_DataType AData);

  void Copy(cStack &Stack2);
  void Copy(cStack *Stack2);
  void Merge(cStack *Stack2);
  _DataType *getBufferPointer() { return Buffer_m; };
  void setBufferPointer(_DataType *BPt) { Buffer_m = BPt; };

  void Display();
  void Display(int DisplayNum1, int DisplayNum2);
  void Destroy();
};

#endif
