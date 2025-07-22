/*
   Licensed to the Apache Software Foundation (ASF) under one
   or more contributor license agreements.  See the NOTICE file
   distributed with this work for additional information
   regarding copyright ownership.  The ASF licenses this file
   to you under the Apache License, Version 2.0 (the
   "License"); you may not use this file except in compliance
   with the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
 */


#pragma once

namespace hippoLBM
{
  enum Area
  {
    Local,
    Global
  };

  enum Side
  {
    Left, 
    Right
  };

  enum Traversal
  {
    All, ///< All points into a grid
    Real, ///< All points - ghost layer
    Inside, ///< All points - ghost layer - 1 layer of size 1
    Edge, ///< Read whithout Inside
    Ghost_Edge, ///< All without Inside
    Plan_xy_0,
    Plan_xy_l,
    Plan_xz_0,
    Plan_xz_l,
    Plan_yz_0,
    Plan_yz_l,
    Extend ///< used for paraview and test if the grid have a point
  };

  constexpr int DIMX = 0;
  constexpr int DIMY = 1;
  constexpr int DIMZ = 2;
  constexpr int DIM_MAX = 3;
}

#define FLUIDE_ -1
#define WALL_ -2
