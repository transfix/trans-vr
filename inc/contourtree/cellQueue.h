//------------------------------------------------------------------------
//
// cellQueue.h - queue of cell identifiers.  The circular queue dyanmically
//               resizes itself when full.  Elements in the queue are of
//               type and size specified by the user.  Not even template'd
//               yet, just using a void *.
//
// Copyright (c) 1997 Dan Schikore
//------------------------------------------------------------------------
#ifndef CELL_QUEUE_H
#define CELL_QUEUE_H

#include <math.h>
#include <stdio.h>
#if 0
#include <memory.h>
#ifndef MACOS_X
#include <malloc.h>
#endif
#endif
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

typedef unsigned int u_int;

class CellQueue1 {
public:
  // constructor/destructor
  inline CellQueue1(int size = 100);
  ~CellQueue1();

  // add item to the queue
  inline void Add(u_int cell);

  // remove and return the first item in queue
  inline int Get(int &cell);

  // return the first item in queue
  inline int Peek(int &cell);

  // remove the first item in queue
  inline void Pop();

  // reset to empty
  void Reset(void) { nel = 0; }

  // check if queue is empty
  int Empty(void) { return (nel == 0); }

protected:
private:
  int nel;
  int cellsize; /* # of elements in cell array   */
  int start;
  u_int *cells;
};

//------------------------------------------------------------------------
//
// CellQueue1() - create a new cell queue with elements of specified size
//
//------------------------------------------------------------------------
inline CellQueue1::CellQueue1(int size) {
  nel = 0;
  start = 0;
  cellsize = size;
  cells = (u_int *)malloc(sizeof(u_int) * cellsize);
}

//------------------------------------------------------------------------
//
// ~CellQueue1() - free storage
//
//------------------------------------------------------------------------
inline CellQueue1::~CellQueue1() {
  if (cells != NULL)
    free(cells);
}

//------------------------------------------------------------------------
//
// Add() - add an item to the queue
//
//------------------------------------------------------------------------
inline void CellQueue1::Add(u_int c) {
  int n;
  int oldsize;
  int atend;

  n = nel++;

  // resize the queue if needed
  if (nel > cellsize) {
    oldsize = cellsize;
    cellsize *= 2;
    cells = (u_int *)realloc(cells, sizeof(int) * cellsize);

    // move everything from 'start' to the end
    if (start != 0) {
      atend = oldsize - start;
      memmove(&cells[cellsize - atend], &cells[start], sizeof(u_int) * atend);
      start = cellsize - atend;
    }
  }

  n += start;
  if (n >= cellsize)
    n -= cellsize;

  cells[n] = c;
}

//------------------------------------------------------------------------
//
// Get() - return the top item from the queue
//
//------------------------------------------------------------------------
inline int CellQueue1::Get(int &c) {
  if (Peek(c) == -1)
    return (-1);
  Pop();
  return (1);
}

//------------------------------------------------------------------------
//
// Peek() - return the top item, but don't remove it
//
//------------------------------------------------------------------------
inline int CellQueue1::Peek(int &c) {
  if (nel == 0)
    return (-1);

  c = cells[start];

  return (1);
}

//------------------------------------------------------------------------
//
// Pop() - delete the top item in the queue
//
//------------------------------------------------------------------------
inline void CellQueue1::Pop(void) {
  start++;
  if (start == cellsize)
    start = 0;
  nel--;
}

#endif
