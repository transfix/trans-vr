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

#ifndef FILE_MC_CONFIGURATION_H
#define FILE_MC_CONFIGURATION_H

int		ConfigurationTable[256][16][2] = {
{ { 0, 0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //   0
{ { 1, 0}, {0,1},{0,2},{0,4}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //   1
{ { 1, 0}, {0,1},{1,5},{1,3}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //   2
{ { 2, 0}, {1,3},{0,2},{0,4}, {1,3},{0,4},{1,5}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //   3
{ { 1, 0}, {2,3},{2,6},{0,2}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //   4
{ { 2, 0}, {0,1},{2,3},{2,6}, {0,1},{2,6},{0,4}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //   5
{ { 2, 2}, {2,3},{2,6},{0,2}, {0,1},{1,5},{1,3}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //   6
{ { 3, 0}, {2,3},{2,6},{1,3}, {1,3},{2,6},{1,5}, {2,6},{0,4},{1,5}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //   7
{ { 1, 0}, {3,7},{2,3},{1,3}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //   8
{ { 2, 2}, {0,2},{0,4},{0,1}, {1,3},{3,7},{2,3}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //   9
{ { 2, 0}, {2,3},{0,1},{1,5}, {2,3},{1,5},{3,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  10
{ { 3, 0}, {0,2},{0,4},{2,3}, {2,3},{0,4},{3,7}, {0,4},{1,5},{3,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  11
{ { 2, 0}, {3,7},{2,6},{0,2}, {3,7},{0,2},{1,3}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  12
{ { 3, 0}, {1,3},{3,7},{0,1}, {0,1},{3,7},{0,4}, {3,7},{2,6},{0,4}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  13
{ { 3, 0}, {0,1},{1,5},{0,2}, {0,2},{1,5},{2,6}, {1,5},{3,7},{2,6}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  14
{ { 2, 0}, {1,5},{3,7},{0,4}, {0,4},{3,7},{2,6}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  15
{ { 1, 0}, {4,5},{0,4},{4,6}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  16
{ { 2, 0}, {4,5},{0,1},{0,2}, {4,5},{0,2},{4,6}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  17
{ { 2, 2}, {0,4},{4,6},{4,5}, {1,5},{1,3},{0,1}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  18
{ { 3, 0}, {1,5},{1,3},{4,5}, {4,5},{1,3},{4,6}, {1,3},{0,2},{4,6}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  19
{ { 2, 2}, {4,6},{4,5},{0,4}, {0,2},{2,3},{2,6}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  20
{ { 3, 0}, {4,6},{4,5},{2,6}, {2,6},{4,5},{2,3}, {4,5},{0,1},{2,3}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  21
{ { 3, 5}, {4,5},{0,4},{4,6}, {1,5},{1,3},{0,1}, {2,6},{0,2},{2,3}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  22
{ { 4, 0}, {4,5},{2,6},{4,6}, {4,5},{2,3},{2,6}, {4,5},{1,5},{2,3}, {1,5},{1,3},{2,3}, {0,0},{0,0},{0,0}, }, //  23
{ { 2, 2}, {4,5},{0,4},{4,6}, {1,3},{3,7},{2,3}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  24
{ { 3, 3}, {2,3},{1,3},{3,7}, {0,2},{4,6},{4,5}, {0,2},{4,5},{0,1}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  25 
{ { 3, 3}, {4,5},{0,4},{4,6}, {1,5},{3,7},{2,3}, {1,5},{2,3},{0,1}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  26
{ { 4, 0}, {4,5},{0,2},{4,6}, {4,5},{3,7},{0,2}, {4,5},{1,5},{3,7}, {0,2},{3,7},{2,3}, {0,0},{0,0},{0,0}, }, //  27
{ { 3, 3}, {0,4},{4,6},{4,5}, {0,2},{1,3},{3,7}, {0,2},{3,7},{2,6}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  28
{ { 4, 0}, {4,6},{3,7},{2,6}, {4,6},{0,1},{3,7}, {4,6},{4,5},{0,1}, {3,7},{0,1},{1,3}, {0,0},{0,0},{0,0}, }, //  29
{ { 4, 6}, {4,5},{0,4},{4,6}, {0,1},{1,5},{0,2}, {0,2},{1,5},{2,6}, {1,5},{3,7},{2,6}, {0,0},{0,0},{0,0}, }, //  30
{ { 3, 0}, {4,6},{4,5},{2,6}, {4,5},{1,5},{2,6}, {2,6},{1,5},{3,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  31
{ { 1, 0}, {1,5},{4,5},{5,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  32
{ { 2, 2}, {0,1},{0,2},{0,4}, {4,5},{5,7},{1,5}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  33
{ { 2, 0}, {0,1},{4,5},{5,7}, {0,1},{5,7},{1,3}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  34
{ { 3, 0}, {4,5},{5,7},{0,4}, {0,4},{5,7},{0,2}, {5,7},{1,3},{0,2}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  35
{ { 2, 2}, {2,3},{2,6},{0,2}, {5,7},{1,5},{4,5}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  36
{ { 3, 3}, {1,5},{4,5},{5,7}, {0,1},{2,3},{2,6}, {0,1},{2,6},{0,4}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  37
{ { 3, 3}, {0,2},{2,3},{2,6}, {0,1},{4,5},{5,7}, {0,1},{5,7},{1,3}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  38
{ { 4, 0}, {4,5},{2,6},{0,4}, {4,5},{1,3},{2,6}, {4,5},{5,7},{1,3}, {2,6},{1,3},{2,3}, {0,0},{0,0},{0,0}, }, //  39
{ { 2, 2}, {1,5},{4,5},{5,7}, {3,7},{2,3},{1,3}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  40
{ { 3, 5}, {2,3},{1,3},{3,7}, {0,2},{0,4},{0,1}, {5,7},{1,5},{4,5}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  41
{ { 3, 0}, {3,7},{2,3},{5,7}, {5,7},{2,3},{4,5}, {2,3},{0,1},{4,5}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  42
{ { 4, 0}, {0,4},{2,3},{0,2}, {0,4},{3,7},{2,3}, {0,4},{4,5},{3,7}, {4,5},{5,7},{3,7}, {0,0},{0,0},{0,0}, }, //  43
{ { 3, 3}, {5,7},{1,5},{4,5}, {3,7},{2,6},{0,2}, {3,7},{0,2},{1,3}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  44
{ { 4, 6}, {5,7},{1,5},{4,5}, {1,3},{3,7},{0,1}, {0,1},{3,7},{0,4}, {3,7},{2,6},{0,4}, {0,0},{0,0},{0,0}, }, //  45
{ { 4, 0}, {5,7},{0,1},{4,5}, {5,7},{2,6},{0,1}, {5,7},{3,7},{2,6}, {0,1},{2,6},{0,2}, {0,0},{0,0},{0,0}, }, //  46
{ { 3, 0}, {4,5},{5,7},{0,4}, {5,7},{3,7},{0,4}, {0,4},{3,7},{2,6}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  47
{ { 2, 0}, {1,5},{0,4},{4,6}, {1,5},{4,6},{5,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  48
{ { 3, 0}, {0,1},{0,2},{1,5}, {1,5},{0,2},{5,7}, {0,2},{4,6},{5,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  49
{ { 3, 0}, {0,4},{4,6},{0,1}, {0,1},{4,6},{1,3}, {4,6},{5,7},{1,3}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  50
{ { 2, 0}, {5,7},{1,3},{4,6}, {4,6},{1,3},{0,2}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  51
{ { 3, 3}, {2,6},{0,2},{2,3}, {4,6},{5,7},{1,5}, {4,6},{1,5},{0,4}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  52
{ { 4, 0}, {1,5},{4,6},{5,7}, {1,5},{2,3},{4,6}, {1,5},{0,1},{2,3}, {4,6},{2,3},{2,6}, {0,0},{0,0},{0,0}, }, //  53
{ { 4, 6}, {2,6},{0,2},{2,3}, {0,4},{4,6},{0,1}, {0,1},{4,6},{1,3}, {4,6},{5,7},{1,3}, {0,0},{0,0},{0,0}, }, //  54
{ { 3, 0}, {2,3},{2,6},{1,3}, {2,6},{4,6},{1,3}, {1,3},{4,6},{5,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  55
{ { 3, 3}, {1,3},{3,7},{2,3}, {1,5},{0,4},{4,6}, {1,5},{4,6},{5,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  56
{ { 4, 6}, {2,3},{1,3},{3,7}, {0,1},{0,2},{1,5}, {1,5},{0,2},{5,7}, {0,2},{4,6},{5,7}, {0,0},{0,0},{0,0}, }, //  57
{ { 4, 0}, {0,4},{2,3},{0,1}, {0,4},{5,7},{2,3}, {0,4},{4,6},{5,7}, {2,3},{5,7},{3,7}, {0,0},{0,0},{0,0}, }, //  58
{ { 3, 0}, {3,7},{2,3},{5,7}, {2,3},{0,2},{5,7}, {5,7},{0,2},{4,6}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  59
{ { 4, 5}, {5,7},{1,5},{0,4}, {5,7},{0,4},{4,6}, {3,7},{2,6},{1,3}, {2,6},{0,2},{1,3}, {0,0},{0,0},{0,0}, }, //  60
{ { 5, 3}, {3,7},{2,6},{1,3}, {2,6},{0,1},{1,3}, {5,7},{1,5},{4,6}, {4,6},{1,5},{0,1}, {4,6},{0,1},{2,6}, }, //  61
{ { 5, 3}, {4,6},{5,7},{0,4}, {5,7},{0,1},{0,4}, {2,6},{0,2},{3,7}, {3,7},{0,2},{0,1}, {3,7},{0,1},{5,7}, }, //  62
{ { 2, 0}, {5,7},{3,7},{4,6}, {4,6},{3,7},{2,6}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  63
{ { 1, 0}, {6,7},{4,6},{2,6}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  64
{ { 2, 2}, {0,4},{0,1},{0,2}, {2,6},{6,7},{4,6}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  65
{ { 2, 2}, {6,7},{4,6},{2,6}, {1,5},{1,3},{0,1}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  66
{ { 3, 3}, {4,6},{2,6},{6,7}, {0,4},{1,5},{1,3}, {0,4},{1,3},{0,2}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  67
{ { 2, 0}, {2,3},{6,7},{4,6}, {2,3},{4,6},{0,2}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  68
{ { 3, 0}, {0,4},{0,1},{4,6}, {4,6},{0,1},{6,7}, {0,1},{2,3},{6,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  69
{ { 3, 3}, {1,3},{0,1},{1,5}, {2,3},{6,7},{4,6}, {2,3},{4,6},{0,2}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  70
{ { 4, 0}, {4,6},{2,3},{6,7}, {4,6},{1,5},{2,3}, {4,6},{0,4},{1,5}, {2,3},{1,5},{1,3}, {0,0},{0,0},{0,0}, }, //  71
{ { 2, 2}, {6,7},{4,6},{2,6}, {2,3},{1,3},{3,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  72
{ { 3, 5}, {0,1},{0,2},{0,4}, {1,3},{3,7},{2,3}, {4,6},{2,6},{6,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  73
{ { 3, 3}, {2,6},{6,7},{4,6}, {2,3},{0,1},{1,5}, {2,3},{1,5},{3,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  74
{ { 4, 6}, {4,6},{2,6},{6,7}, {0,2},{0,4},{2,3}, {2,3},{0,4},{3,7}, {0,4},{1,5},{3,7}, {0,0},{0,0},{0,0}, }, //  75
{ { 3, 0}, {6,7},{4,6},{3,7}, {3,7},{4,6},{1,3}, {4,6},{0,2},{1,3}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  76
{ { 4, 0}, {0,1},{4,6},{0,4}, {0,1},{6,7},{4,6}, {0,1},{1,3},{6,7}, {1,3},{3,7},{6,7}, {0,0},{0,0},{0,0}, }, //  77
{ { 4, 0}, {0,1},{4,6},{0,2}, {0,1},{3,7},{4,6}, {0,1},{1,5},{3,7}, {4,6},{3,7},{6,7}, {0,0},{0,0},{0,0}, }, //  78
{ { 3, 0}, {6,7},{4,6},{3,7}, {4,6},{0,4},{3,7}, {3,7},{0,4},{1,5}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  79
{ { 2, 0}, {0,4},{2,6},{6,7}, {0,4},{6,7},{4,5}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  80
{ { 3, 0}, {2,6},{6,7},{0,2}, {0,2},{6,7},{0,1}, {6,7},{4,5},{0,1}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  81
{ { 3, 3}, {0,1},{1,5},{1,3}, {0,4},{2,6},{6,7}, {0,4},{6,7},{4,5}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  82
{ { 4, 0}, {1,5},{6,7},{4,5}, {1,5},{0,2},{6,7}, {1,5},{1,3},{0,2}, {6,7},{0,2},{2,6}, {0,0},{0,0},{0,0}, }, //  83
{ { 3, 0}, {0,2},{2,3},{0,4}, {0,4},{2,3},{4,5}, {2,3},{6,7},{4,5}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  84
{ { 2, 0}, {4,5},{0,1},{6,7}, {6,7},{0,1},{2,3}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  85
{ { 4, 6}, {1,3},{0,1},{1,5}, {0,2},{2,3},{0,4}, {0,4},{2,3},{4,5}, {2,3},{6,7},{4,5}, {0,0},{0,0},{0,0}, }, //  86
{ { 3, 0}, {1,5},{1,3},{4,5}, {1,3},{2,3},{4,5}, {4,5},{2,3},{6,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  87
{ { 3, 3}, {3,7},{2,3},{1,3}, {6,7},{4,5},{0,4}, {6,7},{0,4},{2,6}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  88
{ { 4, 6}, {3,7},{2,3},{1,3}, {2,6},{6,7},{0,2}, {0,2},{6,7},{0,1}, {6,7},{4,5},{0,1}, {0,0},{0,0},{0,0}, }, //  89
{ { 4, 5}, {4,5},{0,4},{2,6}, {4,5},{2,6},{6,7}, {1,5},{3,7},{0,1}, {3,7},{2,3},{0,1}, {0,0},{0,0},{0,0}, }, //  90
{ { 5, 3}, {6,7},{4,5},{2,6}, {4,5},{0,2},{2,6}, {3,7},{2,3},{1,5}, {1,5},{2,3},{0,2}, {1,5},{0,2},{4,5}, }, //  91
{ { 4, 0}, {3,7},{0,2},{1,3}, {3,7},{4,5},{0,2}, {3,7},{6,7},{4,5}, {0,2},{4,5},{0,4}, {0,0},{0,0},{0,0}, }, //  92
{ { 3, 0}, {1,3},{3,7},{0,1}, {3,7},{6,7},{0,1}, {0,1},{6,7},{4,5}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  93
{ { 5, 3}, {1,5},{3,7},{0,1}, {3,7},{0,2},{0,1}, {4,5},{0,4},{6,7}, {6,7},{0,4},{0,2}, {6,7},{0,2},{3,7}, }, //  94
{ { 2, 0}, {3,7},{6,7},{1,5}, {1,5},{6,7},{4,5}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  95
{ { 2, 2}, {4,6},{2,6},{6,7}, {5,7},{1,5},{4,5}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  96
{ { 3, 5}, {6,7},{4,6},{2,6}, {5,7},{1,5},{4,5}, {0,2},{0,4},{0,1}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  97
{ { 3, 3}, {6,7},{4,6},{2,6}, {5,7},{1,3},{0,1}, {5,7},{0,1},{4,5}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, //  98
{ { 4, 6}, {6,7},{4,6},{2,6}, {4,5},{5,7},{0,4}, {0,4},{5,7},{0,2}, {5,7},{1,3},{0,2}, {0,0},{0,0},{0,0}, }, //  99
{ { 3, 3}, {4,5},{5,7},{1,5}, {4,6},{0,2},{2,3}, {4,6},{2,3},{6,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 100
{ { 4, 6}, {1,5},{4,5},{5,7}, {0,4},{0,1},{4,6}, {4,6},{0,1},{6,7}, {0,1},{2,3},{6,7}, {0,0},{0,0},{0,0}, }, // 101
{ { 4, 5}, {6,7},{4,6},{0,2}, {6,7},{0,2},{2,3}, {5,7},{1,3},{4,5}, {1,3},{0,1},{4,5}, {0,0},{0,0},{0,0}, }, // 102
{ { 5, 3}, {5,7},{1,3},{4,5}, {1,3},{0,4},{4,5}, {6,7},{4,6},{2,3}, {2,3},{4,6},{0,4}, {2,3},{0,4},{1,3}, }, // 103
{ { 3, 5}, {4,5},{5,7},{1,5}, {4,6},{2,6},{6,7}, {1,3},{3,7},{2,3}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 104
{ { 4,10}, {6,7},{4,6},{2,6}, {5,7},{1,5},{4,5}, {3,7},{1,3},{2,3}, {0,4},{0,1},{0,2}, {0,0},{0,0},{0,0}, }, // 105
{ { 4, 6}, {2,6},{6,7},{4,6}, {3,7},{2,3},{5,7}, {5,7},{2,3},{4,5}, {2,3},{0,1},{4,5}, {0,0},{0,0},{0,0}, }, // 106
{ { 5, 5}, {4,6},{6,7},{2,6}, {0,4},{4,5},{3,7}, {0,4},{3,7},{2,3}, {0,4},{2,3},{0,2}, {4,5},{5,7},{3,7}, }, // 107
{ { 4, 6}, {4,5},{5,7},{1,5}, {6,7},{4,6},{3,7}, {3,7},{4,6},{1,3}, {4,6},{0,2},{1,3}, {0,0},{0,0},{0,0}, }, // 108
{ { 5, 5}, {1,5},{5,7},{4,5}, {0,1},{1,3},{6,7}, {0,1},{6,7},{4,6}, {0,1},{4,6},{0,4}, {1,3},{3,7},{6,7}, }, // 109
{ { 5, 3}, {4,6},{0,2},{6,7}, {0,2},{3,7},{6,7}, {4,5},{5,7},{0,1}, {0,1},{5,7},{3,7}, {0,1},{3,7},{0,2}, }, // 110
{ { 4, 2}, {0,4},{4,5},{5,7}, {0,4},{5,7},{3,7}, {0,4},{3,7},{6,7}, {0,4},{6,7},{4,6}, {0,0},{0,0},{0,0}, }, // 111
{ { 3, 0}, {5,7},{1,5},{6,7}, {6,7},{1,5},{2,6}, {1,5},{0,4},{2,6}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 112
{ { 4, 0}, {6,7},{0,2},{2,6}, {6,7},{0,1},{0,2}, {6,7},{5,7},{0,1}, {5,7},{1,5},{0,1}, {0,0},{0,0},{0,0}, }, // 113
{ { 4, 0}, {6,7},{0,4},{2,6}, {6,7},{1,3},{0,4}, {6,7},{5,7},{1,3}, {0,4},{1,3},{0,1}, {0,0},{0,0},{0,0}, }, // 114
{ { 3, 0}, {2,6},{6,7},{0,2}, {6,7},{5,7},{0,2}, {0,2},{5,7},{1,3}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 115
{ { 4, 0}, {5,7},{2,3},{6,7}, {5,7},{0,4},{2,3}, {5,7},{1,5},{0,4}, {2,3},{0,4},{0,2}, {0,0},{0,0},{0,0}, }, // 116
{ { 3, 0}, {5,7},{1,5},{6,7}, {1,5},{0,1},{6,7}, {6,7},{0,1},{2,3}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 117
{ { 5, 3}, {2,3},{6,7},{0,2}, {6,7},{0,4},{0,2}, {1,3},{0,1},{5,7}, {5,7},{0,1},{0,4}, {5,7},{0,4},{6,7}, }, // 118
{ { 2, 0}, {6,7},{5,7},{2,3}, {2,3},{5,7},{1,3}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 119
{ { 4, 6}, {1,3},{3,7},{2,3}, {5,7},{1,5},{6,7}, {6,7},{1,5},{2,6}, {1,5},{0,4},{2,6}, {0,0},{0,0},{0,0}, }, // 120
{ { 5, 5}, {3,7},{1,3},{2,3}, {6,7},{5,7},{0,1}, {6,7},{0,1},{0,2}, {6,7},{0,2},{2,6}, {5,7},{1,5},{0,1}, }, // 121
{ { 5, 3}, {2,3},{0,1},{3,7}, {0,1},{5,7},{3,7}, {2,6},{6,7},{0,4}, {0,4},{6,7},{5,7}, {0,4},{5,7},{0,1}, }, // 122
{ { 4, 2}, {0,2},{2,6},{6,7}, {0,2},{6,7},{5,7}, {0,2},{5,7},{3,7}, {0,2},{3,7},{2,3}, {0,0},{0,0},{0,0}, }, // 123
{ { 5, 3}, {1,5},{0,4},{5,7}, {0,4},{6,7},{5,7}, {1,3},{3,7},{0,2}, {0,2},{3,7},{6,7}, {0,2},{6,7},{0,4}, }, // 124
{ { 4, 2}, {6,7},{5,7},{1,5}, {6,7},{1,5},{0,1}, {6,7},{0,1},{1,3}, {6,7},{1,3},{3,7}, {0,0},{0,0},{0,0}, }, // 125
{ { 2, 2}, {0,1},{0,4},{0,2}, {3,7},{6,7},{5,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 126
{ { 1, 0}, {6,7},{5,7},{3,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 127
{ { 1, 0}, {6,7},{3,7},{5,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 128
{ { 2, 2}, {0,1},{0,2},{0,4}, {3,7},{5,7},{6,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 129
{ { 2, 2}, {5,7},{6,7},{3,7}, {1,3},{0,1},{1,5}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 130
{ { 3, 3}, {3,7},{5,7},{6,7}, {1,3},{0,2},{0,4}, {1,3},{0,4},{1,5}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 131
{ { 2, 2}, {2,6},{0,2},{2,3}, {3,7},{5,7},{6,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 132
{ { 3, 3}, {6,7},{3,7},{5,7}, {2,6},{0,4},{0,1}, {2,6},{0,1},{2,3}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 133
{ { 3, 5}, {6,7},{3,7},{5,7}, {2,6},{0,2},{2,3}, {1,5},{1,3},{0,1}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 134
{ { 4, 6}, {6,7},{3,7},{5,7}, {2,3},{2,6},{1,3}, {1,3},{2,6},{1,5}, {2,6},{0,4},{1,5}, {0,0},{0,0},{0,0}, }, // 135
{ { 2, 0}, {6,7},{2,3},{1,3}, {6,7},{1,3},{5,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 136
{ { 3, 3}, {0,1},{0,2},{0,4}, {1,3},{5,7},{6,7}, {1,3},{6,7},{2,3}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 137
{ { 3, 0}, {5,7},{6,7},{1,5}, {1,5},{6,7},{0,1}, {6,7},{2,3},{0,1}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 138
{ { 4, 0}, {5,7},{0,4},{1,5}, {5,7},{2,3},{0,4}, {5,7},{6,7},{2,3}, {0,4},{2,3},{0,2}, {0,0},{0,0},{0,0}, }, // 139
{ { 3, 0}, {2,6},{0,2},{6,7}, {6,7},{0,2},{5,7}, {0,2},{1,3},{5,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 140
{ { 4, 0}, {0,1},{2,6},{0,4}, {0,1},{5,7},{2,6}, {0,1},{1,3},{5,7}, {2,6},{5,7},{6,7}, {0,0},{0,0},{0,0}, }, // 141
{ { 4, 0}, {6,7},{1,5},{5,7}, {6,7},{0,1},{1,5}, {6,7},{2,6},{0,1}, {2,6},{0,2},{0,1}, {0,0},{0,0},{0,0}, }, // 142
{ { 3, 0}, {5,7},{6,7},{1,5}, {6,7},{2,6},{1,5}, {1,5},{2,6},{0,4}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 143
{ { 2, 2}, {4,5},{0,4},{4,6}, {6,7},{3,7},{5,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 144
{ { 3, 3}, {5,7},{6,7},{3,7}, {4,5},{0,1},{0,2}, {4,5},{0,2},{4,6}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 145
{ { 3, 5}, {0,1},{1,5},{1,3}, {0,4},{4,6},{4,5}, {3,7},{5,7},{6,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 146
{ { 4, 6}, {3,7},{5,7},{6,7}, {1,5},{1,3},{4,5}, {4,5},{1,3},{4,6}, {1,3},{0,2},{4,6}, {0,0},{0,0},{0,0}, }, // 147
{ { 3, 5}, {0,4},{4,6},{4,5}, {0,2},{2,3},{2,6}, {5,7},{6,7},{3,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 148
{ { 4, 6}, {5,7},{6,7},{3,7}, {4,6},{4,5},{2,6}, {2,6},{4,5},{2,3}, {4,5},{0,1},{2,3}, {0,0},{0,0},{0,0}, }, // 149
{ { 4,10}, {4,5},{0,4},{4,6}, {1,5},{1,3},{0,1}, {5,7},{3,7},{6,7}, {0,2},{2,3},{2,6}, {0,0},{0,0},{0,0}, }, // 150
{ { 5, 5}, {5,7},{3,7},{6,7}, {4,5},{1,5},{2,3}, {4,5},{2,3},{2,6}, {4,5},{2,6},{4,6}, {1,5},{1,3},{2,3}, }, // 151
{ { 3, 3}, {4,6},{4,5},{0,4}, {6,7},{2,3},{1,3}, {6,7},{1,3},{5,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 152
{ { 4, 5}, {0,1},{0,2},{4,6}, {0,1},{4,6},{4,5}, {1,3},{5,7},{2,3}, {5,7},{6,7},{2,3}, {0,0},{0,0},{0,0}, }, // 153
{ { 4, 6}, {4,6},{4,5},{0,4}, {5,7},{6,7},{1,5}, {1,5},{6,7},{0,1}, {6,7},{2,3},{0,1}, {0,0},{0,0},{0,0}, }, // 154
{ { 5, 3}, {6,7},{2,3},{5,7}, {2,3},{1,5},{5,7}, {4,6},{4,5},{0,2}, {0,2},{4,5},{1,5}, {0,2},{1,5},{2,3}, }, // 155
{ { 4, 6}, {0,4},{4,6},{4,5}, {2,6},{0,2},{6,7}, {6,7},{0,2},{5,7}, {0,2},{1,3},{5,7}, {0,0},{0,0},{0,0}, }, // 156
{ { 5, 3}, {4,5},{0,1},{4,6}, {0,1},{2,6},{4,6}, {5,7},{6,7},{1,3}, {1,3},{6,7},{2,6}, {1,3},{2,6},{0,1}, }, // 157
{ { 5, 5}, {4,6},{0,4},{4,5}, {6,7},{2,6},{0,1}, {6,7},{0,1},{1,5}, {6,7},{1,5},{5,7}, {2,6},{0,2},{0,1}, }, // 158
{ { 4, 2}, {2,6},{4,6},{4,5}, {2,6},{4,5},{1,5}, {2,6},{1,5},{5,7}, {2,6},{5,7},{6,7}, {0,0},{0,0},{0,0}, }, // 159
{ { 2, 0}, {3,7},{1,5},{4,5}, {3,7},{4,5},{6,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 160
{ { 3, 3}, {0,4},{0,1},{0,2}, {4,5},{6,7},{3,7}, {4,5},{3,7},{1,5}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 161
{ { 3, 0}, {1,3},{0,1},{3,7}, {3,7},{0,1},{6,7}, {0,1},{4,5},{6,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 162
{ { 4, 0}, {0,4},{1,3},{0,2}, {0,4},{6,7},{1,3}, {0,4},{4,5},{6,7}, {1,3},{6,7},{3,7}, {0,0},{0,0},{0,0}, }, // 163
{ { 3, 3}, {2,3},{2,6},{0,2}, {3,7},{1,5},{4,5}, {3,7},{4,5},{6,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 164
{ { 4, 5}, {2,3},{2,6},{0,4}, {2,3},{0,4},{0,1}, {3,7},{1,5},{6,7}, {1,5},{4,5},{6,7}, {0,0},{0,0},{0,0}, }, // 165
{ { 4, 6}, {0,2},{2,3},{2,6}, {1,3},{0,1},{3,7}, {3,7},{0,1},{6,7}, {0,1},{4,5},{6,7}, {0,0},{0,0},{0,0}, }, // 166
{ { 5, 3}, {2,6},{0,4},{2,3}, {0,4},{1,3},{2,3}, {6,7},{3,7},{4,5}, {4,5},{3,7},{1,3}, {4,5},{1,3},{0,4}, }, // 167
{ { 3, 0}, {1,5},{4,5},{1,3}, {1,3},{4,5},{2,3}, {4,5},{6,7},{2,3}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 168
{ { 4, 6}, {0,4},{0,1},{0,2}, {1,5},{4,5},{1,3}, {1,3},{4,5},{2,3}, {4,5},{6,7},{2,3}, {0,0},{0,0},{0,0}, }, // 169
{ { 2, 0}, {6,7},{2,3},{4,5}, {4,5},{2,3},{0,1}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 170
{ { 3, 0}, {0,2},{0,4},{2,3}, {0,4},{4,5},{2,3}, {2,3},{4,5},{6,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 171
{ { 4, 0}, {2,6},{4,5},{6,7}, {2,6},{1,3},{4,5}, {2,6},{0,2},{1,3}, {4,5},{1,3},{1,5}, {0,0},{0,0},{0,0}, }, // 172
{ { 5, 3}, {4,5},{6,7},{1,5}, {6,7},{1,3},{1,5}, {0,4},{0,1},{2,6}, {2,6},{0,1},{1,3}, {2,6},{1,3},{6,7}, }, // 173
{ { 3, 0}, {2,6},{0,2},{6,7}, {0,2},{0,1},{6,7}, {6,7},{0,1},{4,5}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 174
{ { 2, 0}, {0,4},{4,5},{2,6}, {2,6},{4,5},{6,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 175
{ { 3, 0}, {6,7},{3,7},{4,6}, {4,6},{3,7},{0,4}, {3,7},{1,5},{0,4}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 176
{ { 4, 0}, {6,7},{0,2},{4,6}, {6,7},{1,5},{0,2}, {6,7},{3,7},{1,5}, {0,2},{1,5},{0,1}, {0,0},{0,0},{0,0}, }, // 177
{ { 4, 0}, {0,1},{3,7},{1,3}, {0,1},{6,7},{3,7}, {0,1},{0,4},{6,7}, {0,4},{4,6},{6,7}, {0,0},{0,0},{0,0}, }, // 178
{ { 3, 0}, {6,7},{3,7},{4,6}, {3,7},{1,3},{4,6}, {4,6},{1,3},{0,2}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 179
{ { 4, 6}, {2,3},{2,6},{0,2}, {6,7},{3,7},{4,6}, {4,6},{3,7},{0,4}, {3,7},{1,5},{0,4}, {0,0},{0,0},{0,0}, }, // 180
{ { 5, 3}, {3,7},{1,5},{6,7}, {1,5},{4,6},{6,7}, {2,3},{2,6},{0,1}, {0,1},{2,6},{4,6}, {0,1},{4,6},{1,5}, }, // 181
{ { 5, 5}, {0,2},{2,6},{2,3}, {0,1},{0,4},{6,7}, {0,1},{6,7},{3,7}, {0,1},{3,7},{1,3}, {0,4},{4,6},{6,7}, }, // 182
{ { 4, 2}, {4,6},{6,7},{3,7}, {4,6},{3,7},{1,3}, {4,6},{1,3},{2,3}, {4,6},{2,3},{2,6}, {0,0},{0,0},{0,0}, }, // 183
{ { 4, 0}, {4,6},{1,5},{0,4}, {4,6},{2,3},{1,5}, {4,6},{6,7},{2,3}, {1,5},{2,3},{1,3}, {0,0},{0,0},{0,0}, }, // 184
{ { 5, 3}, {0,2},{4,6},{0,1}, {4,6},{1,5},{0,1}, {2,3},{1,3},{6,7}, {6,7},{1,3},{1,5}, {6,7},{1,5},{4,6}, }, // 185
{ { 3, 0}, {0,4},{4,6},{0,1}, {4,6},{6,7},{0,1}, {0,1},{6,7},{2,3}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 186
{ { 2, 0}, {2,3},{0,2},{6,7}, {6,7},{0,2},{4,6}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 187
{ { 5, 3}, {0,2},{1,3},{2,6}, {1,3},{6,7},{2,6}, {0,4},{4,6},{1,5}, {1,5},{4,6},{6,7}, {1,5},{6,7},{1,3}, }, // 188
{ { 2, 2}, {6,7},{2,6},{4,6}, {1,5},{0,1},{1,3}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 189
{ { 4, 2}, {0,1},{0,4},{4,6}, {0,1},{4,6},{6,7}, {0,1},{6,7},{2,6}, {0,1},{2,6},{0,2}, {0,0},{0,0},{0,0}, }, // 190
{ { 1, 0}, {6,7},{2,6},{4,6}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 191
{ { 2, 0}, {5,7},{4,6},{2,6}, {5,7},{2,6},{3,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 192
{ { 3, 3}, {0,2},{0,4},{0,1}, {2,6},{3,7},{5,7}, {2,6},{5,7},{4,6}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 193
{ { 3, 3}, {1,5},{1,3},{0,1}, {5,7},{4,6},{2,6}, {5,7},{2,6},{3,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 194
{ { 4, 5}, {4,6},{2,6},{3,7}, {4,6},{3,7},{5,7}, {0,4},{1,5},{0,2}, {1,5},{1,3},{0,2}, {0,0},{0,0},{0,0}, }, // 195
{ { 3, 0}, {3,7},{5,7},{2,3}, {2,3},{5,7},{0,2}, {5,7},{4,6},{0,2}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 196
{ { 4, 0}, {3,7},{0,1},{2,3}, {3,7},{4,6},{0,1}, {3,7},{5,7},{4,6}, {0,1},{4,6},{0,4}, {0,0},{0,0},{0,0}, }, // 197
{ { 4, 6}, {1,5},{1,3},{0,1}, {3,7},{5,7},{2,3}, {2,3},{5,7},{0,2}, {5,7},{4,6},{0,2}, {0,0},{0,0},{0,0}, }, // 198
{ { 5, 3}, {5,7},{4,6},{3,7}, {4,6},{2,3},{3,7}, {1,5},{1,3},{0,4}, {0,4},{1,3},{2,3}, {0,4},{2,3},{4,6}, }, // 199
{ { 3, 0}, {2,3},{1,3},{2,6}, {2,6},{1,3},{4,6}, {1,3},{5,7},{4,6}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 200
{ { 4, 6}, {0,1},{0,2},{0,4}, {2,3},{1,3},{2,6}, {2,6},{1,3},{4,6}, {1,3},{5,7},{4,6}, {0,0},{0,0},{0,0}, }, // 201
{ { 4, 0}, {2,6},{5,7},{4,6}, {2,6},{0,1},{5,7}, {2,6},{2,3},{0,1}, {5,7},{0,1},{1,5}, {0,0},{0,0},{0,0}, }, // 202
{ { 5, 3}, {0,4},{1,5},{0,2}, {1,5},{2,3},{0,2}, {4,6},{2,6},{5,7}, {5,7},{2,6},{2,3}, {5,7},{2,3},{1,5}, }, // 203
{ { 2, 0}, {1,3},{5,7},{0,2}, {0,2},{5,7},{4,6}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 204
{ { 3, 0}, {0,4},{0,1},{4,6}, {0,1},{1,3},{4,6}, {4,6},{1,3},{5,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 205
{ { 3, 0}, {0,1},{1,5},{0,2}, {1,5},{5,7},{0,2}, {0,2},{5,7},{4,6}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 206
{ { 2, 0}, {1,5},{5,7},{0,4}, {0,4},{5,7},{4,6}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 207
{ { 3, 0}, {4,5},{0,4},{5,7}, {5,7},{0,4},{3,7}, {0,4},{2,6},{3,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 208
{ { 4, 0}, {5,7},{2,6},{3,7}, {5,7},{0,1},{2,6}, {5,7},{4,5},{0,1}, {2,6},{0,1},{0,2}, {0,0},{0,0},{0,0}, }, // 209
{ { 4, 6}, {0,1},{1,5},{1,3}, {4,5},{0,4},{5,7}, {5,7},{0,4},{3,7}, {0,4},{2,6},{3,7}, {0,0},{0,0},{0,0}, }, // 210
{ { 5, 3}, {1,3},{0,2},{1,5}, {0,2},{4,5},{1,5}, {3,7},{5,7},{2,6}, {2,6},{5,7},{4,5}, {2,6},{4,5},{0,2}, }, // 211
{ { 4, 0}, {2,3},{0,4},{0,2}, {2,3},{4,5},{0,4}, {2,3},{3,7},{4,5}, {3,7},{5,7},{4,5}, {0,0},{0,0},{0,0}, }, // 212
{ { 3, 0}, {3,7},{5,7},{2,3}, {5,7},{4,5},{2,3}, {2,3},{4,5},{0,1}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 213
{ { 5, 5}, {1,3},{1,5},{0,1}, {2,3},{3,7},{4,5}, {2,3},{4,5},{0,4}, {2,3},{0,4},{0,2}, {3,7},{5,7},{4,5}, }, // 214
{ { 4, 2}, {4,5},{1,5},{1,3}, {4,5},{1,3},{2,3}, {4,5},{2,3},{3,7}, {4,5},{3,7},{5,7}, {0,0},{0,0},{0,0}, }, // 215
{ { 4, 0}, {2,3},{0,4},{2,6}, {2,3},{5,7},{0,4}, {2,3},{1,3},{5,7}, {0,4},{5,7},{4,5}, {0,0},{0,0},{0,0}, }, // 216
{ { 5, 3}, {1,3},{5,7},{2,3}, {5,7},{2,6},{2,3}, {0,1},{0,2},{4,5}, {4,5},{0,2},{2,6}, {4,5},{2,6},{5,7}, }, // 217
{ { 5, 3}, {0,4},{2,6},{4,5}, {2,6},{5,7},{4,5}, {0,1},{1,5},{2,3}, {2,3},{1,5},{5,7}, {2,3},{5,7},{2,6}, }, // 218
{ { 2, 2}, {2,3},{0,2},{2,6}, {5,7},{4,5},{1,5}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 219
{ { 3, 0}, {4,5},{0,4},{5,7}, {0,4},{0,2},{5,7}, {5,7},{0,2},{1,3}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 220
{ { 2, 0}, {0,1},{1,3},{4,5}, {4,5},{1,3},{5,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 221
{ { 4, 2}, {0,2},{0,1},{1,5}, {0,2},{1,5},{5,7}, {0,2},{5,7},{4,5}, {0,2},{4,5},{0,4}, {0,0},{0,0},{0,0}, }, // 222
{ { 1, 0}, {1,5},{5,7},{4,5}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 223
{ { 3, 0}, {4,6},{2,6},{4,5}, {4,5},{2,6},{1,5}, {2,6},{3,7},{1,5}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 224
{ { 4, 6}, {0,2},{0,4},{0,1}, {4,6},{2,6},{4,5}, {4,5},{2,6},{1,5}, {2,6},{3,7},{1,5}, {0,0},{0,0},{0,0}, }, // 225
{ { 4, 0}, {4,6},{0,1},{4,5}, {4,6},{3,7},{0,1}, {4,6},{2,6},{3,7}, {0,1},{3,7},{1,3}, {0,0},{0,0},{0,0}, }, // 226
{ { 5, 3}, {2,6},{3,7},{4,6}, {3,7},{4,5},{4,6}, {0,2},{0,4},{1,3}, {1,3},{0,4},{4,5}, {1,3},{4,5},{3,7}, }, // 227
{ { 4, 0}, {2,3},{4,6},{0,2}, {2,3},{1,5},{4,6}, {2,3},{3,7},{1,5}, {4,6},{1,5},{4,5}, {0,0},{0,0},{0,0}, }, // 228
{ { 5, 3}, {0,1},{2,3},{0,4}, {2,3},{4,6},{0,4}, {1,5},{4,5},{3,7}, {3,7},{4,5},{4,6}, {3,7},{4,6},{2,3}, }, // 229
{ { 5, 3}, {0,1},{4,5},{1,3}, {4,5},{3,7},{1,3}, {0,2},{2,3},{4,6}, {4,6},{2,3},{3,7}, {4,6},{3,7},{4,5}, }, // 230
{ { 2, 2}, {4,5},{4,6},{0,4}, {1,3},{2,3},{3,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 231
{ { 4, 0}, {4,5},{1,3},{1,5}, {4,5},{2,3},{1,3}, {4,5},{4,6},{2,3}, {4,6},{2,6},{2,3}, {0,0},{0,0},{0,0}, }, // 232
{ { 5, 5}, {0,4},{0,2},{0,1}, {4,5},{4,6},{2,3}, {4,5},{2,3},{1,3}, {4,5},{1,3},{1,5}, {4,6},{2,6},{2,3}, }, // 233
{ { 3, 0}, {4,6},{2,6},{4,5}, {2,6},{2,3},{4,5}, {4,5},{2,3},{0,1}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 234
{ { 4, 2}, {4,5},{4,6},{2,6}, {4,5},{2,6},{2,3}, {4,5},{2,3},{0,2}, {4,5},{0,2},{0,4}, {0,0},{0,0},{0,0}, }, // 235
{ { 3, 0}, {1,5},{4,5},{1,3}, {4,5},{4,6},{1,3}, {1,3},{4,6},{0,2}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 236
{ { 4, 2}, {4,6},{0,4},{0,1}, {4,6},{0,1},{1,3}, {4,6},{1,3},{1,5}, {4,6},{1,5},{4,5}, {0,0},{0,0},{0,0}, }, // 237
{ { 2, 0}, {4,5},{4,6},{0,1}, {0,1},{4,6},{0,2}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 238
{ { 1, 0}, {4,5},{4,6},{0,4}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 239
{ { 2, 0}, {3,7},{1,5},{2,6}, {2,6},{1,5},{0,4}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 240
{ { 3, 0}, {0,1},{0,2},{1,5}, {0,2},{2,6},{1,5}, {1,5},{2,6},{3,7}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 241
{ { 3, 0}, {1,3},{0,1},{3,7}, {0,1},{0,4},{3,7}, {3,7},{0,4},{2,6}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 242
{ { 2, 0}, {3,7},{1,3},{2,6}, {2,6},{1,3},{0,2}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 243
{ { 3, 0}, {0,2},{2,3},{0,4}, {2,3},{3,7},{0,4}, {0,4},{3,7},{1,5}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 244
{ { 2, 0}, {2,3},{3,7},{0,1}, {0,1},{3,7},{1,5}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 245
{ { 4, 2}, {0,4},{0,2},{2,3}, {0,4},{2,3},{3,7}, {0,4},{3,7},{1,3}, {0,4},{1,3},{0,1}, {0,0},{0,0},{0,0}, }, // 246
{ { 1, 0}, {3,7},{1,3},{2,3}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 247
{ { 3, 0}, {2,3},{1,3},{2,6}, {1,3},{1,5},{2,6}, {2,6},{1,5},{0,4}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 248
{ { 4, 2}, {2,6},{2,3},{1,3}, {2,6},{1,3},{1,5}, {2,6},{1,5},{0,1}, {2,6},{0,1},{0,2}, {0,0},{0,0},{0,0}, }, // 249
{ { 2, 0}, {0,1},{0,4},{2,3}, {2,3},{0,4},{2,6}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 250
{ { 1, 0}, {2,3},{0,2},{2,6}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 251
{ { 2, 0}, {1,3},{1,5},{0,2}, {0,2},{1,5},{0,4}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 252
{ { 1, 0}, {0,1},{1,3},{1,5}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 253
{ { 1, 0}, {0,1},{0,4},{0,2}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 254
{ { 0, 0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, {0,0},{0,0},{0,0}, }, // 255
};

int	RelativeLocByIndex[8][3] = {
	{0, 1, 0},	// 0
	{1, 1, 0},	// 1
	{0, 0, 0},	// 2
	{1, 0, 0},	// 3
	
	{0, 1, 1},	// 4
	{1, 1, 1},	// 5
	{0, 0, 1},	// 6
	{1, 0, 1},	// 7
};

int	CubeEdgeDir[8][8][3] = {
	{ { 0, 0, 0}, { 1, 0, 0}, { 0,-1, 0}, { 0, 0, 0}, { 0, 0, 1}, { 0, 0, 0}, { 0, 0, 0}, { 0, 0, 0}, }, 
	{ {-1, 0, 0}, { 0, 0, 0}, { 0, 0, 0}, { 0,-1, 0}, { 0, 0, 0}, { 0, 0, 1}, { 0, 0, 0}, { 0, 0, 0}, } ,
	{ { 0, 1, 0}, { 0, 0, 0}, { 0, 0, 0}, { 1, 0, 0}, { 0, 0, 0}, { 0, 0, 0}, { 0, 0, 1}, { 0, 0, 0}, } ,
	{ { 0, 0, 0}, { 0, 1, 0}, {-1, 0, 0}, { 0, 0, 0}, { 0, 0, 0}, { 0, 0, 0}, { 0, 0, 0}, { 0, 0, 1}, } ,
	{ { 0, 0,-1}, { 0, 0, 0}, { 0, 0, 0}, { 0, 0, 0}, { 0, 0, 0}, { 1, 0, 0}, { 0,-1, 0}, { 0, 0, 0}, } ,
	{ { 0, 0, 0}, { 0, 0,-1}, { 0, 0, 0}, { 0, 0, 0}, {-1, 0, 0}, { 0, 0, 0}, { 0, 0, 0}, { 0,-1, 0}, } ,
	{ { 0, 0, 0}, { 0, 0, 0}, { 0, 0,-1}, { 0, 0, 0}, { 0, 1, 0}, { 0, 0, 0}, { 0, 0, 0}, { 1, 0, 0}, } ,
	{ { 0, 0, 0}, { 0, 0, 0}, { 0, 0, 0}, { 0, 0,-1}, { 0, 0, 0}, { 0, 1, 0}, {-1, 0, 0}, { 0, 0, 0}, } ,

};


// -1 = there is no edge
int	EdgeTable[8][8] = {
	{-1,  0,  3, -1,  8, -1, -1, -1, }, 
	{ 0, -1, -1,  1, -1,  9, -1, -1, }, 
	{ 3, -1, -1,  2, -1, -1, 11, -1, }, 
	{-1,  1,  2, -1, -1, -1, -1, 10, }, 
	{ 8, -1, -1, -1, -1,  4,  7, -1, }, 
	{-1,  9, -1, -1,  4, -1, -1,  5, }, 
	{-1, -1, 11, -1,  7, -1, -1,  6, }, 
	{-1, -1, -1, 10, -1,  5,  6, -1, }
};


int LevelOneUnConfiguration_gi[256] = {
	0, 1, 1, 2, 1, 2, 3, 5, 1, 3, 2, 5, 2, 5, 5, 8, 1, 2, 3, 5, 3, 5, 7, 9, 4, 6, 6, 
	11, 6, 14, 12, 17, 1, 3, 2, 5, 4, 6, 6, 14, 3, 7, 5, 9, 6, 12, 11, 17, 2, 5, 5, 8, 6, 11, 
	12, 17, 6, 12, 14, 17, 10, 16, 16, 20, 1, 3, 4, 6, 2, 5, 6, 11, 3, 7, 6, 12, 5, 9, 14, 17, 2, 
	5, 6, 14, 5, 8, 12, 17, 6, 12, 10, 16, 11, 17, 16, 20, 3, 7, 6, 12, 6, 12, 10, 16, 7, 13, 12, 15, 
	12, 15, 16, 19, 5, 9, 11, 17, 14, 17, 16, 20, 12, 15, 16, 19, 16, 19, 18, 21, 1, 4, 3, 6, 3, 6, 7, 
	12, 2, 6, 5, 14, 5, 11, 9, 17, 3, 6, 7, 12, 7, 12, 13, 15, 6, 10, 12, 16, 12, 16, 15, 19, 2, 6, 
	5, 11, 6, 10, 12, 16, 5, 12, 8, 17, 14, 16, 17, 20, 5, 14, 9, 17, 12, 16, 15, 19, 11, 16, 17, 20, 16, 
	18, 19, 21, 2, 6, 6, 10, 5, 14, 12, 16, 5, 12, 11, 16, 8, 17, 17, 20, 5, 11, 12, 16, 9, 17, 15, 19, 
	14, 16, 16, 18, 17, 20, 19, 21, 5, 12, 14, 16, 11, 16, 16, 18, 9, 15, 17, 19, 17, 19, 20, 21, 8, 17, 17, 
	20, 17, 20, 19, 21, 17, 19, 20, 21, 20, 21, 21, 22};
	

int	SurfaceIndex[6][4] = {
	// back			right			front		left			top			bottom
	{3, 2, 1, 0}, {2, 6, 5, 1}, {6, 7, 4, 5}, {7, 3, 0, 4}, {2, 3, 7, 6}, {5, 4, 0, 1} 
};

	
/*
// FindBoundaryConnectedNeighbors_EdgeIntersections() of "TFGeneration.cpp" 
int DirectionToEdge[2][2][3] = {
	{ {{ 2, 2}, { 2, 3}, { 3, 3}, }, { { 0, 2}, {-1,-1}, { 1, 3}, }, { { 0, 0}, { 0, 1}, { 1, 1}, }, },
	{ {{ 2, 6}, {-1,-1}, { 3, 7}, }, { {-1,-1}, {-1,-1}, {-1,-1}, }, { { 0, 4}, {-1,-1}, { 1, 5}, }, },
	{ {{ 6, 6}, { 6, 7}, { 7, 7}, }, { { 4, 6}, {-1,-1}, { 5, 7}, }, { { 4, 4}, { 4, 5}, { 5, 5}, }, }
};
*/

#endif

